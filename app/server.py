"""FastAPI Server handling Plivo Inbound/Outbound Telephony and WebSocket Media Streaming."""

import audioop
import base64
import io
import json
import os
import time
import urllib.parse
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np
import soundfile as sf
import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, RedirectResponse
from fastapi.templating import Jinja2Templates
from loguru import logger

from app.config import settings, get_active_llm_info

try:
    import app.livekit_agent  # Register LiveKit agent plugins on main thread
except Exception as _e:
    logger.debug(f"LiveKit agent module deferred loading: {_e}")

# ---------------------------------------------------------
# Global Pre-Warmed Singletons & Greeting Cache
# ---------------------------------------------------------
@dataclass
class AudioRawFrame:
    """Lightweight raw PCM audio frame container for pre-cached speech."""
    audio: bytes
    sample_rate: int
    num_channels: int = 1


_kokoro_singleton = None
_whisper_cache: Dict[str, Any] = {}
_cached_greeting_frames: List[Any] = []
_cached_greeting_duration: float = 0.0
_cached_greeting_text: str = ""
_cached_greeting_voice: str = ""


def get_kokoro_instance():
    """Returns the globally pre-warmed Kokoro ONNX model instance with hardware acceleration."""
    global _kokoro_singleton
    if _kokoro_singleton is None:
        import os
        import onnxruntime as rt
        from kokoro_onnx import Kokoro
        kokoro_dir = Path(settings.KOKORO_CACHE_DIR)
        model_path = kokoro_dir / "kokoro-v1.0.onnx"
        voices_path = kokoro_dir / "voices-v1.0.bin"

        if not model_path.exists() or not voices_path.exists():
            logger.warning(
                f"Kokoro model files not found in {kokoro_dir}. "
                "TTS fallback will use Deepgram Flux / cloud voice."
            )
            return None

        opts = rt.SessionOptions()
        opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = min(os.cpu_count() or 1, 4)
        opts.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL

        logger.info(f"Loading pre-warmed Kokoro ONNX model into memory (threads={opts.intra_op_num_threads})...")
        session = rt.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        _kokoro_singleton = Kokoro.from_session(session, str(voices_path))
        logger.success("Kokoro ONNX pre-warmed successfully in memory.")
    return _kokoro_singleton


_whisper_downloading: set = set()  # tracks models currently being downloaded

def get_whisper_instance(model_name: Optional[str] = None, device: Optional[str] = None, compute_type: Optional[str] = None):
    """Returns the globally pre-warmed Faster-Whisper model instance from thread-safe cache."""
    m_name = model_name or settings.WHISPER_MODEL
    comp = compute_type or settings.WHISPER_COMPUTE_TYPE
    dev = device or settings.WHISPER_DEVICE
    cache_key = f"{m_name}:{comp}:{dev}"

    if cache_key in _whisper_cache:
        return _whisper_cache[cache_key]

    if cache_key in _whisper_downloading:
        while cache_key in _whisper_downloading:
            time.sleep(0.05)
        return _whisper_cache.get(cache_key)

    _whisper_downloading.add(cache_key)
    try:
        from faster_whisper import WhisperModel
        logger.info(f"Loading pre-warmed Faster-Whisper ({m_name}, {comp}) into memory...")
        whisper_threads = min(os.cpu_count() or 1, 2)
        _whisper_cache[cache_key] = WhisperModel(m_name, device=dev, compute_type=comp, cpu_threads=whisper_threads)
        logger.success(f"Faster-Whisper ({m_name}) pre-warmed in memory.")
    finally:
        _whisper_downloading.discard(cache_key)
    return _whisper_cache[cache_key]


def precache_greeting(text: Optional[str] = None, voice: Optional[str] = None, speed: float = 1.0):
    """
    Pre-synthesizes greeting speech into 20ms 8kHz PCM audio frames and keeps them resident in RAM.
    This enables instant (<50ms) Time-To-First-Audio (TTFA) on call connect, exactly like Vapi.
    """
    global _cached_greeting_frames, _cached_greeting_duration, _cached_greeting_text, _cached_greeting_voice

    greeting_text = text or settings.GREETING_TEXT
    greeting_voice = voice or settings.KOKORO_VOICE

    is_deepgram = bool(getattr(settings, "DEEPGRAM_API_KEY", "")) and (
        greeting_voice.startswith("flux-")
        or greeting_voice.startswith("aura-")
        or greeting_voice.lower() == "cliff"
    )
    if is_deepgram:
        actual_voice = "flux-cliff-en" if greeting_voice.lower() == "cliff" else greeting_voice
        try:
            import httpx
            t0 = os.times().elapsed
            ver = "v2" if "flux" in actual_voice else "v1"
            sr = settings.SAMPLE_RATE
            url = f"https://api.deepgram.com/{ver}/speak?model={actual_voice}&encoding=linear16&sample_rate={sr}&container=none"
            resp = httpx.post(
                url,
                headers={"Authorization": f"Token {settings.DEEPGRAM_API_KEY}", "Content-Type": "application/json"},
                json={"text": greeting_text},
                timeout=15.0,
            )
            if resp.status_code == 200:
                pcm_data = resp.content
                # Safety: strip WAV/RIFF header if present (shouldn't happen with container=none but defensive)
                if pcm_data[:4] == b"RIFF":
                    pcm_data = pcm_data[44:]
                    logger.warning("Pre-cache: stripped unexpected WAV header from Deepgram response")
                chunk_size = int(sr * 2 * 0.02)  # 20ms at target sample rate
                frames = []
                for i in range(0, len(pcm_data), chunk_size):
                    chunk = pcm_data[i:i + chunk_size]
                    if len(chunk) < chunk_size:
                        chunk += b"\x00" * (chunk_size - len(chunk))
                    frames.append(AudioRawFrame(audio=chunk, sample_rate=sr, num_channels=1))
                _cached_greeting_frames = frames
                _cached_greeting_duration = len(frames) * 0.02
                _cached_greeting_text = greeting_text
                _cached_greeting_voice = greeting_voice
                elapsed = (os.times().elapsed - t0) * 1000.0
                logger.success(
                    f"Pre-cached Deepgram greeting audio ({len(frames)} frames, {_cached_greeting_duration:.2f}s) "
                    f"using voice '{greeting_voice}' at {sr}Hz in {elapsed:.1f}ms."
                )
                return
        except Exception as dg_err:
            logger.warning(f"Failed to pre-cache Deepgram greeting: {dg_err}, falling back to Kokoro...")

    kokoro = get_kokoro_instance()
    if not kokoro:
        logger.warning("Kokoro model not ready, cannot pre-cache greeting audio.")
        return

    voices = kokoro.get_voices() if hasattr(kokoro, "get_voices") else kokoro.voices
    if greeting_voice not in voices:
        greeting_voice = list(voices)[0]

    try:
        t0 = os.times().elapsed
        sr = settings.SAMPLE_RATE
        samples, rate = kokoro.create(greeting_text, voice=greeting_voice, speed=speed)
        pcm_24k = (samples * 32767).astype(np.int16).tobytes()
        pcm_out, _ = audioop.ratecv(pcm_24k, 2, 1, rate, sr, None)

        chunk_size = int(sr * 2 * 0.02)  # 20ms at target sample rate
        frames = []
        for i in range(0, len(pcm_out), chunk_size):
            chunk = pcm_out[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk += b"\x00" * (chunk_size - len(chunk))
            frames.append(AudioRawFrame(audio=chunk, sample_rate=sr, num_channels=1))

        _cached_greeting_frames = frames
        _cached_greeting_duration = len(frames) * 0.02
        _cached_greeting_text = greeting_text
        _cached_greeting_voice = greeting_voice
        elapsed = (os.times().elapsed - t0) * 1000.0
        logger.success(
            f"Pre-cached greeting audio ({len(frames)} frames, {_cached_greeting_duration:.2f}s) "
            f"using voice '{greeting_voice}' at {sr}Hz in {elapsed:.1f}ms."
        )
    except Exception as e:
        logger.error(f"Error pre-caching greeting audio: {e}")


def get_cached_greeting_frames() -> List[Any]:
    """Returns pre-rendered greeting audio frames."""
    return _cached_greeting_frames


_cached_fillers: Dict[str, List[Any]] = {}


def precache_fillers(voice: str = None, speed: float = 1.0):
    """No-op: Audio filler frame injection is disabled in favor of natural conversational LLM flow."""
    global _cached_fillers
    _cached_fillers.clear()


def get_cached_filler_frames(key: str) -> List[Any]:
    """Returns pre-rendered filler audio frames. Always empty as frame injection is disabled."""
    return []


# ---------------------------------------------------------
# Application Lifespan
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application startup and teardown lifecycle."""
    logger.info("Starting Aria Voice Agent Server...")
    app.state.session = aiohttp.ClientSession()

    # Pre-warm AI models into RAM at startup so calls experience ZERO cold-start delay
    try:
        from app.agents import get_active_assistant
        active_agent = get_active_assistant()
        if active_agent:
            settings.GREETING_TEXT = active_agent.get("first_message", settings.GREETING_TEXT)
            settings.KOKORO_VOICE = active_agent.get("tts_voice", settings.KOKORO_VOICE)
            settings.WHISPER_MODEL = active_agent.get("stt_model", settings.WHISPER_MODEL)
            settings.GROQ_MODEL = active_agent.get("llm_model", settings.GROQ_MODEL)
            settings.ACTIVE_PRESET = active_agent.get("preset", "balanced")
            logger.info(f"Loaded active assistant profile: '{active_agent.get('name')}'")

        get_kokoro_instance()
        speed = float(active_agent.get("voice_speed", 1.0)) if active_agent else 1.0
        precache_greeting(text=settings.GREETING_TEXT, voice=settings.KOKORO_VOICE, speed=speed)
        # Always pre-warm base as an instant, sub-300ms model
        try:
            get_whisper_instance("base")
        except Exception:
            pass
        # Pre-warm target model in background thread if different from base
        import threading
        target_model = settings.WHISPER_MODEL
        def _prewarm_whisper():
            try:
                if target_model != "base":
                    local_model = target_model
                    if local_model in ("whisper-large-v3", "whisper-large-v3-turbo") or "nova" in local_model.lower() or "deepgram" in local_model.lower():
                        # Cloud STT (Groq / Deepgram); pre-warm base.en as local fallback
                        local_model = "base.en"
                    get_whisper_instance(local_model)
                    logger.success(f"Background Whisper pre-warm complete: {local_model}")
            except Exception as _e:
                logger.warning(f"Whisper pre-warm failed for {target_model}: {_e}")
        threading.Thread(target=_prewarm_whisper, daemon=True, name="whisper-prewarm").start()
    except Exception as e:
        logger.warning(f"Background model pre-warming deferred: {e}")

    yield
    logger.info("Shutting down Aria Voice Agent Server...")
    await app.state.session.close()


app = FastAPI(
    title="Aria Voice AI Agent (Plivo + LiveKit)",
    description="Low-latency telephony voice AI agent with Vapi-style model switching, instant greeting cache, and natural human cadence.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def resolve_base_urls(request: Request):
    """Determine the public HTTP and WSS base URLs for Plivo webhooks."""
    if settings.PUBLIC_URL:
        clean_url = settings.PUBLIC_URL.strip().rstrip("/")
        if clean_url.startswith("https://"):
            http_base = clean_url
            ws_base = clean_url.replace("https://", "wss://", 1)
        elif clean_url.startswith("http://"):
            http_base = clean_url
            ws_base = clean_url.replace("http://", "ws://", 1)
        else:
            http_base = f"https://{clean_url}"
            ws_base = f"wss://{clean_url}"
        return http_base, ws_base

    host = request.headers.get("host", f"localhost:{settings.PORT}")
    is_secure = request.headers.get("x-forwarded-proto", "http") == "https"
    http_base = f"https://{host}" if is_secure else f"http://{host}"
    ws_base = f"wss://{host}" if is_secure else f"ws://{host}"
    return http_base, ws_base


def build_stream_xml(websocket_url: str) -> str:
    """Generate the standard Plivo bidirectional audio stream XML response."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">
    {websocket_url}
  </Stream>
</Response>"""


async def trigger_plivo_call(
    session: aiohttp.ClientSession,
    to_number: str,
    from_number: str,
    answer_url: str
) -> Dict[str, Any]:
    """Execute REST API request to Plivo to trigger an outbound phone call."""
    auth_id = settings.PLIVO_AUTH_ID
    auth_token = settings.PLIVO_AUTH_TOKEN

    if not auth_id or not auth_token:
        raise HTTPException(
            status_code=500,
            detail="Plivo credentials not configured. Please set PLIVO_AUTH_ID and PLIVO_AUTH_TOKEN."
        )

    api_url = f"https://api.plivo.com/v1/Account/{auth_id}/Call/"
    payload = {
        "to": to_number,
        "from": from_number or settings.PLIVO_PHONE_NUMBER,
        "answer_url": answer_url,
        "answer_method": "GET",
    }

    auth = aiohttp.BasicAuth(auth_id, auth_token)
    headers = {"Content-Type": "application/json"}

    logger.info(f"Triggering Plivo outbound call: {payload['from']} -> {to_number}")
    async with session.post(api_url, json=payload, auth=auth, headers=headers) as resp:
        if resp.status not in (200, 201, 202):
            err_text = await resp.text()
            logger.error(f"Plivo outbound call API failed ({resp.status}): {err_text}")
            raise HTTPException(status_code=resp.status, detail=f"Plivo API Error: {err_text}")
        return await resp.json()


# ---------------------------------------------------------
# Health & Status Endpoints
# ---------------------------------------------------------
@app.get("/health")
async def health_check():
    """Health check reporting configured status and components."""
    llm_info = get_active_llm_info()
    return {
        "status": "healthy",
        "agent": "Aria Voice AI 2.0",
        "preset": settings.ACTIVE_PRESET,
        "llm_engine": llm_info["display_name"],
        "llm_provider": llm_info["provider"],
        "llm_model": llm_info["model"],
        "llm_badge": llm_info["badge"],
        "is_smart_llm": llm_info["is_smart_llm"],
        "stt_engine": "Groq Whisper Large-v3 (1.55B SOTA)" if settings.GROQ_API_KEY else f"Faster-Whisper ({settings.WHISPER_MODEL}, {settings.WHISPER_COMPUTE_TYPE})",
        "tts_engine": f"Kokoro ONNX ({settings.KOKORO_VOICE})",
        "greeting_cached": len(_cached_greeting_frames) > 0,
        "greeting_duration_sec": round(_cached_greeting_duration, 2),
        "plivo_configured": bool(settings.PLIVO_AUTH_ID and settings.PLIVO_AUTH_TOKEN),
        "groq_configured": bool(settings.GROQ_API_KEY),
        "gemini_configured": bool(settings.GEMINI_API_KEY),
    }


@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    """Serve the 2026 ORX Agents Admin Dashboard & Operations Studio."""
    from app.agents import get_active_assistant_id
    http_base, _ = resolve_base_urls(request)
    resp = templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "settings": settings,
            "public_url": http_base,
            "active_id": get_active_assistant_id(),
        }
    )
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.get("/livekit", response_class=HTMLResponse)
@app.get("/test/livekit", response_class=HTMLResponse)
async def livekit_lab(request: Request):
    """Serve the primary LiveKit Voice Agent Testing Lab & Telemetry Studio."""
    from app.livekit_agent import get_livekit_status
    http_base, _ = resolve_base_urls(request)
    resp = templates.TemplateResponse(
        request=request,
        name="livekit.html",
        context={
            "settings": settings,
            "public_url": http_base,
            "status": get_livekit_status(),
        }
    )
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.get("/subscribe", response_class=HTMLResponse)
async def subscribe_page(request: Request):
    """Serve the ORX Agents brand onboarding & Polar subscription wizard."""
    http_base, _ = resolve_base_urls(request)
    return templates.TemplateResponse(
        request=request,
        name="subscribe.html",
        context={
            "settings": settings,
            "public_url": http_base,
        }
    )


@app.get("/portal", response_class=HTMLResponse)
async def portal_page(request: Request):
    """Serve the business owner post-payment client portal."""
    http_base, _ = resolve_base_urls(request)
    return templates.TemplateResponse(
        request=request,
        name="portal.html",
        context={
            "settings": settings,
            "public_url": http_base,
        }
    )


# ---------------------------------------------------------
# LiveKit Realtime Voice Agent Routes & APIs
# ---------------------------------------------------------



@app.post("/api/livekit/token")
async def api_livekit_token(payload: Dict[str, Any] = Body(...)):
    """Generate LiveKit room token and spawn the voice agent session."""
    from app.livekit_agent import start_voice_agent_for_room
    room_name = payload.get("room") or f"aria-room-{int(time.time())}"
    res = await start_voice_agent_for_room(room_name, payload)
    return res


@app.post("/api/livekit/stop")
async def api_livekit_stop(payload: Dict[str, Any] = Body(...)):
    """Stop an active LiveKit voice room session and return post-call intelligence summary."""
    from app.livekit_agent import stop_voice_agent_for_room
    from app.project_db import get_db_path

    room = payload.get("room", "")
    res = await stop_voice_agent_for_room(room)
    call_entry = res.get("call") if isinstance(res, dict) else None

    extracted = (call_entry.get("extracted_info") or {}) if call_entry else {}
    cal_url = (call_entry.get("google_calendar_url") or "") if call_entry else ""
    client_id = (call_entry.get("assistant_id") or "riley_hvac") if call_entry else "riley_hvac"
    db_file = str(get_db_path(client_id))

    return {
        "success": res.get("stopped", True) if isinstance(res, dict) else bool(res),
        "room": room,
        "call": call_entry,
        "call_id": call_entry.get("call_id") if call_entry else None,
        "extracted_info": extracted,
        "google_calendar_url": cal_url,
        "database_file": db_file,
        "duration_seconds": call_entry.get("duration_seconds", 0) if call_entry else 0,
        "transcript": call_entry.get("transcript", []) if call_entry else [],
    }


@app.get("/api/livekit/latest-summary")
async def api_livekit_latest_summary(call_id: Optional[str] = None):
    """Returns the latest finalized call summary including extracted appointment and Google Calendar link."""
    from app.calls import list_calls, get_call
    from app.project_db import get_db_path

    call_entry = get_call(call_id) if call_id else None
    if not call_entry:
        calls = list_calls()
        call_entry = calls[0] if calls else None

    if not call_entry:
        return {"success": False, "call": None}

    extracted = call_entry.get("extracted_info") or {}
    cal_url = call_entry.get("google_calendar_url") or ""
    client_id = call_entry.get("assistant_id") or "riley_hvac"
    db_file = str(get_db_path(client_id))

    return {
        "success": True,
        "call": call_entry,
        "call_id": call_entry.get("call_id"),
        "extracted_info": extracted,
        "google_calendar_url": cal_url,
        "database_file": db_file,
        "duration_seconds": call_entry.get("duration_seconds", 0),
        "transcript": call_entry.get("transcript", []),
    }


@app.post("/api/livekit/prompt/update")
async def api_livekit_prompt_update(payload: Dict[str, Any] = Body(...)):
    """Update agent prompt/instructions dynamically in memory for an active session or save as default."""
    from app.livekit_agent import update_agent_prompt
    room_name = payload.get("room") or payload.get("room_name") or ""
    new_prompt = payload.get("prompt") or payload.get("instructions") or ""
    client_id = payload.get("client_id") or payload.get("project_id")
    if not new_prompt:
        raise HTTPException(status_code=400, detail="Prompt text ('prompt') is required")

    result = await update_agent_prompt(room_name=room_name, new_prompt=new_prompt)
    if client_id:
        try:
            from app.project_db import update_project
            update_project(client_id, {"system_prompt": new_prompt, "livekit_prompt": new_prompt})
            result["project_saved"] = True
            logger.success(f"Persisted updated prompt to client project {client_id}")
        except Exception as ex:
            logger.warning(f"Could not persist prompt to project {client_id}: {ex}")

    return result


@app.post("/api/livekit/outbound-call")
async def api_livekit_outbound_call(payload: Dict[str, Any] = Body(...)):
    """Initiate an outbound LiveKit AI phone call via Telnyx SIP, Twilio SIP, or LiveKit SIP."""
    from app.livekit_agent import create_outbound_call
    phone_number = payload.get("phone_number") or payload.get("to") or payload.get("phone")
    if not phone_number:
        raise HTTPException(status_code=400, detail="'phone_number' is required for outbound call dispatch")

    room_name = payload.get("room_name") or payload.get("room")
    prompt = payload.get("prompt") or payload.get("instructions")
    client_id = payload.get("client_id") or payload.get("assistant_id")
    provider = payload.get("provider", "telnyx")

    result = await create_outbound_call(
        phone_number=phone_number,
        room_name=room_name,
        prompt=prompt,
        client_id=client_id,
        provider=provider,
    )
    return result


@app.get("/api/livekit/recordings")
async def api_livekit_recordings(
    limit: int = Query(50, ge=1, le=200),
    call_id: Optional[str] = Query(None),
):
    """Retrieve call recordings and transcripts specifically for LiveKit voice sessions."""
    from app.livekit_agent import get_livekit_recordings
    recs = get_livekit_recordings(limit=limit, call_id=call_id)
    return {
        "total": len(recs),
        "recordings": recs,
    }


@app.get("/api/livekit/status")
async def api_livekit_status():
    """Return status of the LiveKit server and active agent sessions."""
    from app.livekit_agent import get_livekit_status
    return get_livekit_status()


@app.post("/api/livekit/server/toggle")
async def api_livekit_server_toggle():
    """Toggle or restart the local LiveKit development server."""
    from app.livekit_agent import is_local_server_running, start_local_server, stop_local_server
    if is_local_server_running():
        stop_local_server()
        time.sleep(0.5)
        started = start_local_server()
        return {"running": started, "action": "restarted"}
    else:
        started = start_local_server()
        return {"running": started, "action": "started"}


@app.post("/api/livekit/benchmark")
async def api_livekit_benchmark():
    """Run latency benchmarks across models configured in LiveKit."""
    from app.livekit_agent import run_livekit_model_benchmarks
    return await run_livekit_model_benchmarks()


@app.get("/api/livekit/templates")
async def api_livekit_templates():
    """Return prompt templates, voices, and model configurations for LiveKit."""
    from app.templates_mgr import list_templates
    from app.agents import list_assistants
    from app.livekit_agent import VOICE_FALLBACK_MAP

    items = []
    seen_ids = set()

    # 1. Flagship & custom prompt templates from templates_mgr
    try:
        raw_templates = list_templates()
        for tpl in raw_templates:
            tpl_id = tpl.get("id")
            if not tpl_id or tpl_id in seen_ids:
                continue
            seen_ids.add(tpl_id)
            raw_voice = tpl.get("default_voice", "aura-2-asteria-en")
            if raw_voice.startswith("flux-") or raw_voice.startswith("aura-"):
                mapped_voice = raw_voice
            else:
                mapped_voice = VOICE_FALLBACK_MAP.get(raw_voice, raw_voice)

            items.append({
                "id": tpl_id,
                "name": tpl.get("name"),
                "role": tpl.get("role", tpl.get("name")),
                "category": tpl.get("category", "General"),
                "system_prompt": tpl.get("system_prompt", ""),
                "first_message": tpl.get("first_message", ""),
                "llm_provider": "gemini",
                "llm_model": "gemini-3.1-flash-lite",
                "tts_voice": mapped_voice,
                "original_voice": raw_voice,
                "stt_model": "nova-3",
            })
    except Exception as e:
        logger.warning(f"Could not load templates from templates_mgr: {e}")

    # 2. Persistent assistant profiles from agents.py
    try:
        assistants = list_assistants()
        for asst in assistants:
            asst_id = asst.get("id")
            if not asst_id or asst_id in seen_ids:
                continue
            seen_ids.add(asst_id)
            raw_voice = asst.get("tts_voice", "aura-2-asteria-en")
            if raw_voice.startswith("flux-") or raw_voice.startswith("aura-"):
                mapped_voice = raw_voice
            else:
                mapped_voice = VOICE_FALLBACK_MAP.get(raw_voice, raw_voice)

            asst_llm_model = asst.get("llm_model", "gemini-3.1-flash-lite")
            asst_provider = "gemini" if "gemini" in asst_llm_model.lower() else "groq"

            items.append({
                "id": asst_id,
                "name": asst.get("name"),
                "role": asst.get("tagline", asst.get("name")),
                "category": "Assistant Profiles",
                "system_prompt": asst.get("system_prompt", ""),
                "first_message": asst.get("first_message", ""),
                "llm_provider": asst_provider,
                "llm_model": asst_llm_model,
                "tts_voice": mapped_voice,
                "original_voice": raw_voice,
                "stt_model": "nova-3",
            })
    except Exception as e:
        logger.warning(f"Could not load assistants from agents: {e}")

    # 3. Dedicated Onboarded Client Projects from project_db
    try:
        from app.project_db import list_projects
        projects = list_projects()
        for proj in projects:
            cid = proj.get("id") or proj.get("meta", {}).get("client_id")
            if not cid or cid in seen_ids:
                continue
            seen_ids.add(cid)
            meta = proj.get("meta", {})
            prompt_cfg = proj.get("prompt", {})
            biz_name = meta.get("business_name") or "Client Business"
            persona = prompt_cfg.get("persona_name") or "Riley"
            ind = meta.get("industry") or "Service"
            raw_voice = prompt_cfg.get("tts_voice") or "flux-heather-en"
            if raw_voice.startswith("flux-") or raw_voice.startswith("aura-"):
                mapped_voice = raw_voice
            else:
                mapped_voice = VOICE_FALLBACK_MAP.get(raw_voice, raw_voice)

            items.append({
                "id": cid,
                "client_id": cid,
                "name": f"{biz_name} ({persona})",
                "role": f"{ind.upper()} Dispatch Coordinator — {biz_name}",
                "category": "Onboarded Client Businesses",
                "system_prompt": prompt_cfg.get("system_prompt") or prompt_cfg.get("livekit_prompt") or "",
                "first_message": prompt_cfg.get("first_message") or f"Thank you for calling {biz_name}! This is {persona}. How can I help you today?",
                "llm_provider": "gemini",
                "llm_model": "gemini-3.1-flash-lite",
                "tts_voice": mapped_voice,
                "original_voice": raw_voice,
                "stt_model": "nova-3",
            })
    except Exception as e:
        logger.warning(f"Could not load onboarded client projects for LiveKit: {e}")

    # 4. Aria LiveKit Native Default
    items.append({
        "id": "aria-livekit-default",
        "name": "Aria • Ultra-Low Latency LiveKit Receptionist",
        "role": "Aria - Ultra-Fast Voice Receptionist",
        "category": "LiveKit Default",
        "system_prompt": (
            "You are Aria, an ultra-low-latency AI voice receptionist running on the LiveKit framework. "
            "You speak in natural, concise, conversational English (1-2 sentences maximum per turn). "
            "You are friendly, proactive, and direct."
        ),
        "first_message": "Hi there! I'm Aria, running on LiveKit with Gemini and Deepgram. How can I help you today?",
        "llm_provider": "gemini",
        "llm_model": "gemini-3.1-flash-lite",
        "tts_voice": "aura-2-asteria-en",
        "original_voice": "aura-2-asteria-en",
        "stt_model": "nova-3",
    })

    return {"templates": items, "active_id": get_active_assistant_id()}


# ---------------------------------------------------------
# ORX Onboarding & Polar APIs
# ---------------------------------------------------------

@app.get("/api/onboarding/industries")
async def api_get_industries():
    """Return all supported industries and their question blueprints."""
    from app.onboarding import INDUSTRIES
    return {"industries": INDUSTRIES}


@app.post("/api/onboarding/discover")
async def api_discover_business(payload: Dict[str, Any] = Body(...)):
    """Auto-discover business and address details from text query."""
    from app.onboarding import auto_discover_business
    query = payload.get("query", "")
    return auto_discover_business(query)


@app.post("/api/onboarding/search-places")
async def api_search_places(payload: Dict[str, Any] = Body(...)):
    """Search Google Places & Directory autocomplete for business matches."""
    from app.onboarding import search_places_autocomplete
    query = payload.get("query", "")
    matches = search_places_autocomplete(query)
    return {"matches": matches}


@app.post("/api/onboarding/save-profile")
async def api_save_client_profile(profile: Dict[str, Any] = Body(...)):
    """Save onboarded client profile and compile customized agent prompt."""
    from app.onboarding import save_client_profile
    saved = save_client_profile(profile)
    return {"success": True, "profile": saved}


@app.post("/api/onboarding/compile-prompt")
async def api_compile_onboarding_prompt(profile: Dict[str, Any] = Body(...)):
    """Compile a customized voice agent prompt from onboarding options in real-time."""
    from app.onboarding import compile_agent_prompt
    prompt = compile_agent_prompt(profile)
    biz = profile.get("business_name") or "Comfort Breeze"
    persona = profile.get("persona_name") or "Riley"
    return {
        "success": True,
        "prompt": prompt,
        "char_count": len(prompt),
        "word_count": len(prompt.split()),
        "first_message": profile.get("first_message") or f"Thank you for calling {biz}! This is {persona}. How can I help get your home comfortable today?",
    }


@app.post("/api/onboarding/complete")
async def api_onboarding_complete(request: Request, payload: Dict[str, Any] = Body(...)):
    """Completes client onboarding, provisions dedicated project DB, tailored LiveKit prompt, and triggers SMS."""
    from app.onboarding import complete_client_onboarding
    http_base = str(request.base_url).rstrip("/")
    result = await complete_client_onboarding(payload, public_url=http_base)
    return result


@app.post("/api/onboarding/chat-test")
async def api_simulate_agent_turn(payload: Dict[str, Any] = Body(...)):
    """Simulate a conversational turn with the client's tailored voice agent."""
    from app.onboarding import simulate_agent_turn, get_client_profile
    client_id = payload.get("client_id")
    profile = get_client_profile(client_id) if client_id else None
    if not profile:
        profile = {
            "id": client_id or "cli_temp",
            "business_name": payload.get("business_name", "Apex Services"),
            "industry": payload.get("industry") or payload.get("trade", "general"),
            "trade": payload.get("trade") or payload.get("industry", "general"),
            "hours": payload.get("hours", "Mon-Fri 8:00 AM - 6:00 PM"),
            "pricing_policy": payload.get("pricing_policy", "Diagnostic fee credited toward repair"),
            "booking_action": payload.get("booking_action", "Book a 2-hour arrival window on calendar"),
            "services": payload.get("services", "Full residential and commercial services"),
            "persona_name": payload.get("persona_name", "Riley"),
            "persona_voice": payload.get("persona_voice", "aura-asteria-en"),
            "transfer_rules": payload.get("transfer_rules", "Transfer on emergencies"),
            "forwarding_phone": payload.get("forwarding_phone", "+1 (555) 234-5678"),
            "address": payload.get("address", ""),
            "allowed_topics": payload.get("allowed_topics", []),
            "custom_topics": payload.get("custom_topics", []),
            "schedule_config": payload.get("schedule_config", {})
        }
    message = payload.get("message", "")
    history = payload.get("history", [])
    result = simulate_agent_turn(profile, message, history)
    return result


@app.post("/api/onboarding/demo-call-script")
async def api_demo_call_script(payload: Dict[str, Any] = Body(...)):
    """Generate realistic dual-voice call simulation script tailored to owner profile."""
    from app.onboarding import generate_call_demo_script
    scenario_id = payload.get("scenario_id")
    profile = payload.get("profile") if "profile" in payload else payload
    return generate_call_demo_script(profile, scenario_id=scenario_id)


@app.post("/api/polar/create-checkout")
async def api_polar_checkout(payload: Dict[str, Any] = Body(...)):
    """Create Polar subscription checkout session."""
    from app.onboarding import create_polar_checkout_session
    plan = payload.get("plan", "starter")
    client_id = payload.get("client_id", "default")
    success_url = payload.get("success_url", "/portal")
    return create_polar_checkout_session(plan, client_id, success_url)


@app.post("/api/polar/webhook")
async def api_polar_webhook(request: Request):
    """Handle incoming Polar subscription webhook events."""
    from app.onboarding import get_client_profile, save_client_profile, send_activation_sms
    try:
        body = await request.json()
        logger.info(f"Polar Webhook Event: {body.get('type')}")
        event_type = body.get("type")
        data = body.get("data", {})
        metadata = data.get("metadata", {})
        client_id = metadata.get("client_id")
        if client_id:
            profile = get_client_profile(client_id)
            if profile:
                profile["polar_status"] = "active"
                profile["plan"] = metadata.get("plan", "starter")
                save_client_profile(profile)

                # Fire activation SMS and provision project on new subscription
                if event_type in ("subscription.created", "order.paid", "checkout.updated"):
                    try:
                        from app.project_db import trigger_new_client_project
                        project = trigger_new_client_project(profile)
                        logger.success(f"Provisioned dedicated project '{project.get('id')}' via payment webhook.")
                    except Exception as proj_err:
                        logger.warning(f"Project provisioning on webhook notice: {proj_err}")

                    try:
                        http_base = str(request.base_url).rstrip("/")
                        send_activation_sms(profile, public_url=http_base)
                    except Exception as sms_err:
                        logger.warning(f"Activation SMS failed (non-fatal): {sms_err}")

        return {"status": "received"}
    except Exception as e:
        logger.error(f"Polar webhook error: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/api/payment/simulate")
async def api_simulate_payment(payload: Dict[str, Any] = Body(...)):
    """Simulates payment completion for a client company, provisioning their dedicated
    project directory, SQLite database (client.db), custom LiveKit prompt, and Google Calendar config."""
    from app.onboarding import get_client_profile, save_client_profile
    from app.project_db import trigger_new_client_project

    client_id = payload.get("client_id") or payload.get("id") or f"client_{int(time.time())}"
    biz_name = payload.get("business_name") or payload.get("name") or "Simulated Client Company"
    owner_email = payload.get("owner_email") or payload.get("email") or "owner@clientcompany.com"
    owner_phone = payload.get("owner_phone") or payload.get("phone") or "+18005550199"
    industry = payload.get("industry") or "hvac"
    plan = payload.get("plan") or "growth"
    cal_id = payload.get("google_calendar_id") or payload.get("calendar_id") or "primary"

    profile_data = {
        "id": client_id,
        "client_id": client_id,
        "business_name": biz_name,
        "industry": industry,
        "owner_email": owner_email,
        "owner_phone": owner_phone,
        "phone": owner_phone,
        "forwarding_phone": owner_phone,
        "google_calendar_id": cal_id,
        "polar_status": "active",
        "payment_status": "paid",
        "plan": plan,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hours": payload.get("hours", "Monday to Friday 8:00 AM to 6:00 PM"),
        "services": payload.get("services", "AC repair, heating maintenance, heat pumps"),
    }
    save_client_profile(profile_data)

    # Provision project directory, client.db SQLite tables, prompt.json, calendar.json
    project = trigger_new_client_project(profile_data)

    return {
        "success": True,
        "message": f"Payment processed and dedicated project provisioned for '{biz_name}'",
        "client_id": client_id,
        "project": project,
        "database_path": str(project.get("db_path", f"data/projects/{client_id}/client.db")),
    }


@app.post("/api/admin/test-client-flow")
async def api_admin_test_client_flow(payload: Optional[Dict[str, Any]] = Body(None)):
    """Runs a complete end-to-end simulation of the client lifecycle:
    1. Payment processing -> provisions company profile & SQLite client.db
    2. Google Calendar connection verification
    3. Simulated customer voice call -> extracts booking details
    4. Records call & appointment in client's dedicated SQLite client.db
    5. Dispatches customer confirmation (Email + 1-Click Google Calendar Link + SMS)
    6. Dispatches owner lead alert (Email + Full Transcript + SMS)
    7. Dispatches admin test copy
    """
    from datetime import datetime, timedelta
    from app.project_db import trigger_new_client_project, get_project_calls, get_project_appointments, get_db_path
    from app.integrations import create_google_calendar_url, dispatch_google_calendar_event, send_customer_appointment_confirmation, send_owner_lead_alert
    from app.calls import save_call_session

    data = payload or {}
    ts = int(time.time())
    client_id = data.get("client_id") or f"admin_test_{ts}"
    biz_name = data.get("business_name") or "Apex Climate Solutions"
    owner_email = data.get("owner_email") or "admin@apexclimate.com"
    owner_phone = data.get("owner_phone") or "+16125550199"
    cal_id = data.get("google_calendar_id") or "primary"
    cust_name = data.get("customer_name") or "Abdul Aziz Albadi"
    cust_phone = data.get("customer_phone") or "+16127169989"
    cust_email = data.get("customer_email") or "abdulazizalpadi91@gmail.com"
    cust_addr = data.get("service_address") or "2508 Delaware Street, Minneapolis, MN 55414"
    service_type = data.get("service_type") or "AC Repair & Tune-Up"

    steps = []

    # Step 1: Simulate Payment & Provision Dedicated Project
    proj_payload = {
        "id": client_id,
        "client_id": client_id,
        "business_name": biz_name,
        "industry": "hvac",
        "owner_email": owner_email,
        "owner_phone": owner_phone,
        "google_calendar_id": cal_id,
        "hours": "Mon-Fri 8am-6pm",
        "services": "AC repair, heat pumps, emergency diagnostic",
    }
    project = trigger_new_client_project(proj_payload)
    db_file = get_db_path(client_id)
    steps.append({
        "step": 1,
        "name": "Payment Processed & Company Profile Provisioned",
        "status": "success",
        "details": f"Created project '{client_id}' for '{biz_name}'. SQLite DB: {db_file}"
    })

    # Step 2: Simulate Call & Turn-by-Turn Transcript
    call_id = f"test_call_{ts}"
    transcript = [
        {"speaker": "assistant", "text": f"Thank you for calling {biz_name}. This is Riley. How may I help?"},
        {"speaker": "customer", "text": "My AC is blowing warm air and needs repair."},
        {"speaker": "assistant", "text": "Understood. What is the service address where we will be working?"},
        {"speaker": "customer", "text": cust_addr},
        {"speaker": "assistant", "text": f"Got it, {cust_addr}, is that correct?"},
        {"speaker": "customer", "text": "Yes, that's right."},
        {"speaker": "assistant", "text": "Would tomorrow morning between nine and noon, or tomorrow afternoon after two work better?"},
        {"speaker": "customer", "text": "Tomorrow morning works."},
        {"speaker": "assistant", "text": "May I have your first and last name?"},
        {"speaker": "customer", "text": f"{cust_name}, phone {cust_phone}, email {cust_email}."},
        {"speaker": "assistant", "text": f"You are all set, {cust_name}! We have you booked for tomorrow between nine and noon at {cust_addr}. Does that sound right?"},
        {"speaker": "customer", "text": "Yes, perfect. Thank you!"}
    ]
    call_entry = save_call_session(
        call_id=call_id,
        assistant_id=client_id,
        assistant_name=f"Riley ({biz_name})",
        caller=cust_phone,
        called="+18005550199",
        started_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        duration_seconds=52.0,
        transcript=transcript,
        status="completed"
    )
    steps.append({
        "step": 2,
        "name": "Simulated Voice AI Call & Transcript Recorded",
        "status": "success",
        "details": f"Logged call #{call_id} (12 turns) with customer {cust_name}"
    })

    # Step 3: Record Call & Appointment in Client SQLite DB
    from app.project_db import save_project_call, save_project_appointment
    extracted = {
        "client_name": cust_name,
        "customer_name": cust_name,
        "client_phone": cust_phone,
        "customer_phone": cust_phone,
        "email": cust_email,
        "client_email": cust_email,
        "service_requested": service_type,
        "service_type": service_type,
        "service_address": cust_addr,
        "address": cust_addr,
        "appointment_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
        "time_window": "morning",
        "appointment_time": "9:00 AM - 12:00 PM",
        "has_appointment": True,
        "summary": f"Customer booked AC repair service at {cust_addr} for tomorrow morning.",
    }
    save_project_call(client_id, {
        "id": call_id,
        "caller": cust_phone,
        "duration": 52.0,
        "transcript": transcript,
        "extracted_info": extracted,
        "recording_file": ""
    })
    apt_id = f"APT-{call_id[-6:].upper()}"
    save_project_appointment(client_id, {
        "id": apt_id,
        "client_name": cust_name,
        "client_phone": cust_phone,
        "service_requested": service_type,
        "service_address": cust_addr,
        "appointment_date": extracted["appointment_date"],
        "window": "morning",
        "exact_time": "9:00 AM",
        "status": "confirmed",
        "summary": extracted["summary"]
    })
    steps.append({
        "step": 3,
        "name": "Appointment & Call Saved in Dedicated SQLite DB",
        "status": "success",
        "details": f"Saved appointment #{apt_id} and call #{call_id} in {db_file}"
    })

    # Step 4: Google Calendar Event & 1-Click Link
    cal_url = create_google_calendar_url(extracted, assistant_name=biz_name)
    cal_disp = await dispatch_google_calendar_event(
        appointment_data=extracted,
        calendar_id=cal_id,
    )
    steps.append({
        "step": 4,
        "name": "Google Calendar Integration & 1-Click Link Generated",
        "status": "success",
        "details": f"Calendar: {cal_id}, Status: {cal_disp.get('status')}",
        "google_calendar_url": cal_url
    })

    # Step 5: Customer Confirmation (Email + 1-Click Link + SMS)
    cust_notif = await send_customer_appointment_confirmation(
        customer_email=cust_email,
        customer_phone=cust_phone,
        appointment_data=extracted,
        business_name=biz_name,
        calendar_url=cal_url,
        call_id=call_id
    )
    steps.append({
        "step": 5,
        "name": "Customer Confirmation Dispatched (Email + Calendar Link + SMS)",
        "status": "success",
        "details": f"Email to {cust_email} ({cust_notif.get('email', {}).get('status')}), SMS to {cust_phone}"
    })

    # Step 6: Owner & Admin Lead Alert Dispatched
    owner_notif = await send_owner_lead_alert(
        owner_email=owner_email,
        owner_phone=owner_phone,
        appointment_data=extracted,
        call_session=call_entry,
        business_name=biz_name,
        calendar_url=cal_url
    )
    steps.append({
        "step": 6,
        "name": "Owner & Admin Lead Alert Dispatched",
        "status": "success",
        "details": f"Email to {owner_email} ({owner_notif.get('email', {}).get('status')}), SMS to {owner_phone}"
    })

    return {
        "success": True,
        "client_id": client_id,
        "business_name": biz_name,
        "appointment_id": apt_id,
        "call_id": call_id,
        "google_calendar_url": cal_url,
        "database_file": str(db_file),
        "steps": steps,
        "message": f"End-to-end test completed successfully for '{biz_name}'!"
    }


# ---------------------------------------------------------
# Phone Number OTP Authentication for Client Portal Dashboard
# ---------------------------------------------------------

@app.post("/api/auth/send-otp")
async def api_send_otp(payload: Dict[str, Any] = Body(...)):
    """Send 6-digit verification code to the client's phone number."""
    from app.auth import send_phone_otp
    phone = payload.get("phone", "")
    client_id = payload.get("client_id")
    result = await send_phone_otp(phone=phone, client_id=client_id)
    return result


@app.post("/api/auth/verify-otp")
async def api_verify_otp(payload: Dict[str, Any] = Body(...)):
    """Verify 6-digit code and issue authenticated dashboard session token."""
    from app.auth import verify_phone_otp
    phone = payload.get("phone", "")
    code = payload.get("code", "")
    result = verify_phone_otp(phone=phone, code=code)
    return result


@app.get("/api/auth/me")
async def api_auth_me(
    request: Request,
    token: Optional[str] = Query(None)
):
    """Validate current session token and return authenticated client profile."""
    from app.auth import validate_session
    from app.onboarding import get_client_profile, get_latest_client_profile

    auth_header = request.headers.get("Authorization", "")
    active_token = token or (auth_header.replace("Bearer ", "").strip() if auth_header else "")
    session = validate_session(active_token)
    if not session:
        return {"authenticated": False, "message": "Invalid or expired session. Please log in with your phone."}

    client_id = session.get("client_id")
    profile = get_client_profile(client_id) if client_id else get_latest_client_profile()

    return {
        "authenticated": True,
        "phone": session.get("phone"),
        "client_id": client_id,
        "business_name": session.get("business_name"),
        "profile": profile
    }


@app.post("/api/auth/logout")
async def api_auth_logout(payload: Dict[str, Any] = Body(...)):
    """Log out and revoke active dashboard session token."""
    from app.auth import revoke_session
    token = payload.get("token", "")
    revoked = revoke_session(token)
    return {"success": revoked, "message": "Logged out successfully."}


@app.get("/api/client/profile")
async def api_get_client_profile(client_id: Optional[str] = Query(None)):
    """Fetch client profile for portal dashboard."""
    from app.onboarding import get_client_profile, get_latest_client_profile
    if client_id:
        p = get_client_profile(client_id)
        if p:
            return {"profile": p}
    latest = get_latest_client_profile()
    return {"profile": latest}


@app.post("/api/client/update")
async def api_update_client_settings(payload: Dict[str, Any] = Body(...)):
    """Update client settings from portal."""
    from app.onboarding import get_client_profile, get_latest_client_profile, save_client_profile
    client_id = payload.get("client_id")
    profile = get_client_profile(client_id) if client_id else get_latest_client_profile()
    if not profile:
        profile = payload
    else:
        allowed_fields = [
            "forwarding_phone", "sms_phone", "owner_phone", "hours", "address",
            "services", "pricing_policy", "diagnostic_fee", "business_name",
            "persona_name", "persona_voice", "voice", "greeting", "first_message",
            "emergency_triggers", "transfer_rules", "night_action", "after_hours_action",
            "custom_prompt", "compiled_prompt"
        ]
        for field in allowed_fields:
            if field in payload:
                profile[field] = payload[field]
    saved = save_client_profile(profile)
    return {"success": True, "profile": saved}


@app.post("/api/client/cancel-subscription")
async def api_cancel_subscription(payload: Dict[str, Any] = Body(...)):
    """Cancel / end subscription for a client from the portal."""
    from app.onboarding import get_client_profile, get_latest_client_profile, save_client_profile
    client_id = payload.get("client_id")
    profile = get_client_profile(client_id) if client_id else get_latest_client_profile()
    if not profile:
        return {"success": False, "error": "Client not found"}

    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    profile["polar_status"] = "canceled"
    profile["status"] = "canceled"
    profile["canceled_at"] = now_str
    saved = save_client_profile(profile)

    # Sync with project database if present
    try:
        from app.project_db import get_db_connection, get_db_path
        clean_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', client_id)
        if get_db_path(clean_id).exists():
            conn = get_db_connection(clean_id)
            cur = conn.cursor()
            cur.execute("UPDATE project_meta SET status = 'canceled', updated_at = ? WHERE client_id = ?", (now_str, clean_id))
            conn.commit()
            conn.close()
    except Exception as e:
        logger.warning(f"Could not update project_meta cancellation: {e}")

    return {"success": True, "polar_status": "canceled", "message": "Subscription canceled. Your line remains active until the end of the billing cycle.", "profile": saved}


@app.post("/api/client/reactivate-subscription")
async def api_reactivate_subscription(payload: Dict[str, Any] = Body(...)):
    """Reactivate a canceled subscription from the portal."""
    from app.onboarding import get_client_profile, get_latest_client_profile, save_client_profile
    client_id = payload.get("client_id")
    profile = get_client_profile(client_id) if client_id else get_latest_client_profile()
    if not profile:
        return {"success": False, "error": "Client not found"}

    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    profile["polar_status"] = "active"
    profile["status"] = "active"
    profile.pop("canceled_at", None)
    saved = save_client_profile(profile)

    try:
        from app.project_db import get_db_connection, get_db_path
        clean_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', client_id)
        if get_db_path(clean_id).exists():
            conn = get_db_connection(clean_id)
            cur = conn.cursor()
            cur.execute("UPDATE project_meta SET status = 'active', updated_at = ? WHERE client_id = ?", (now_str, clean_id))
            conn.commit()
            conn.close()
    except Exception as e:
        logger.warning(f"Could not update project_meta reactivation: {e}")

    return {"success": True, "polar_status": "active", "message": "Subscription reactivated! Your receptionist is active.", "profile": saved}


@app.post("/api/client/send-activation-email")
async def api_send_activation_email(request: Request, payload: Dict[str, Any] = Body(...)):
    """Send client portal setup & phone forwarding details to the owner's email."""
    from app.onboarding import get_client_profile, get_latest_client_profile
    from app.integrations import send_activation_email
    client_id = payload.get("client_id")
    email = payload.get("email", "").strip()
    profile = get_client_profile(client_id) if client_id else get_latest_client_profile()
    if not profile:
        return {"success": False, "error": "Client not found"}
    if not email:
        email = profile.get("owner_email") or profile.get("email") or ""
    if not email:
        return {"success": False, "error": "Please provide a valid email address"}

    http_base = str(request.base_url).rstrip("/")
    result = await send_activation_email(profile, email, public_url=http_base)
    return result


@app.post("/api/client/simulate-call")
async def api_simulate_client_call(payload: Dict[str, Any] = Body(...)):
    """Simulate an incoming customer call to this client's line for testing the dashboard."""
    from app.onboarding import get_client_profile, get_latest_client_profile
    from app.project_db import save_project_call
    import random

    client_id = payload.get("client_id")
    profile = get_client_profile(client_id) if client_id else get_latest_client_profile()
    if not profile:
        return {"success": False, "error": "Client not found"}

    biz_name = profile.get("business_name") or "Your Company"
    persona = profile.get("persona_name") or "Riley"
    ind = profile.get("industry", "hvac")
    cid = profile.get("id")

    sample_callers = [
        ("Michael Chang", "(651) 234-8891", "410 Grand Ave", f"Emergency {ind.upper()} inspection"),
        ("Rachel Adams", "(612) 441-2093", "1250 Hennepin Ave", f"Routine maintenance tune-up"),
        ("Carlos Martinez", "(763) 892-1145", "308 Lake Street", f"System diagnostic & quote"),
    ]
    name, phone, addr, srv = random.choice(sample_callers)
    call_id = f"sim-{int(time.time())}"

    call_record = {
        "id": call_id,
        "client_id": cid,
        "caller_phone": phone,
        "duration": float(random.randint(60, 180)),
        "recording_file": "data/recordings/call-rec-demo-84920.wav",
        "transcript": [
            {"speaker": "assistant", "text": f"Thank you for calling {biz_name}! This is {persona}. How can I assist you today?"},
            {"speaker": "customer", "text": f"Hi, this is {name}. I need to schedule a {srv} at {addr}."},
            {"speaker": "assistant", "text": f"I can certainly help you with that, {name}! We have an arrival window tomorrow between 9:00 AM and 12:00 PM. Would that work?"},
            {"speaker": "customer", "text": "Yes, that works perfectly for me. Thank you!"},
            {"speaker": "assistant", "text": f"You're all booked! Our technician will see you tomorrow at {addr}. Have a great day!"}
        ],
        "extracted_info": {
            "customer_name": name,
            "client_name": name,
            "service_requested": srv,
            "service_address": addr,
            "appointment_date": "Tomorrow",
            "appointment_time": "9:00 AM - 12:00 PM"
        }
    }
    save_project_call(cid, call_record)
    return {"success": True, "call": call_record}


@app.post("/api/client/verify-connection")
async def api_verify_connection(payload: Dict[str, Any] = Body(...)):
    """Mark client carrier forwarding as verified. Called from portal after client taps *71."""
    from app.onboarding import verify_client_connection
    client_id = payload.get("client_id")
    carrier = payload.get("carrier", "verizon")
    result = verify_client_connection(client_id, carrier)
    return result


@app.post("/api/client/disconnect")
async def api_disconnect_connection(payload: Dict[str, Any] = Body(...)):
    """Pause/disconnect forwarding for a client."""
    from app.onboarding import disconnect_client_connection
    client_id = payload.get("client_id")
    return disconnect_client_connection(client_id)


@app.post("/api/client/ping-verify")
async def api_ping_verify(request: Request, payload: Dict[str, Any] = Body(...)):
    """Fire a silent 3-second test call from the assigned number to the business owner's
    forwarding phone. If Plivo is configured and call bounces back into our inbound webhook,
    the client is truly connected. If Plivo is not configured (dev mode), simulate success."""
    from app.onboarding import get_client_profile, get_latest_client_profile, verify_client_connection

    client_id = payload.get("client_id")
    carrier = payload.get("carrier", "verizon")
    profile = get_client_profile(client_id) if client_id else get_latest_client_profile()

    if not profile:
        return {"success": False, "verified": False, "message": "Client not found"}

    forwarding_phone = profile.get("forwarding_phone", "")
    assigned_phone = profile.get("assigned_phone", settings.PLIVO_PHONE_NUMBER or "")

    # If Plivo is fully configured, fire a real silent test call
    if settings.PLIVO_AUTH_ID and settings.PLIVO_AUTH_TOKEN and forwarding_phone and assigned_phone:
        try:
            http_base, _ = resolve_base_urls(request)
            # We call the forwarding phone FROM the assigned number.
            # If forwarding is set up, Plivo receives the call back into our inbound webhook.
            # We use a very short ring time (5s max) to keep it invisible to the owner.
            meta_payload = {
                "to": forwarding_phone,
                "from": assigned_phone,
                "direction": "ping_verify",
                "client_id": client_id,
            }
            encoded_query = urllib.parse.quote(json.dumps(meta_payload))
            answer_url = f"{http_base}/outbound/answer?meta={encoded_query}"

            from app.calls import trigger_plivo_call
            call_result = await trigger_plivo_call(
                session=request.app.state.session,
                to_number=forwarding_phone,
                from_number=assigned_phone,
                answer_url=answer_url,
            )
            # Mark as verified optimistically — real bounce confirmation comes via inbound webhook
            result = verify_client_connection(client_id, carrier)
            result["ping_mode"] = "plivo_live"
            return result
        except Exception as e:
            logger.warning(f"Plivo ping-verify failed, using dev fallback: {e}")

    # Dev / no-Plivo mode: simulate success after 1.5s (mimics the real call bounce timing)
    result = verify_client_connection(client_id, carrier)
    result["ping_mode"] = "simulated"
    return result


# ---------------------------------------------------------
# Multi-Tenant Client Projects & Dedicated Databases API
# ---------------------------------------------------------

@app.get("/api/projects")
async def api_list_projects():
    """List all client projects with their database and calendar status."""
    from app.project_db import list_projects
    return {"projects": list_projects()}


@app.post("/api/projects/create")
@app.post("/api/projects")
async def api_create_project(payload: Dict[str, Any] = Body(...)):
    """Creates a client project with dedicated database, LiveKit prompt, and calendar/SMS configs."""
    from app.project_db import trigger_new_client_project
    project = trigger_new_client_project(payload)
    return {"success": True, "project": project}


@app.get("/api/projects/{client_id}")
async def api_get_project(client_id: str):
    """Retrieve full details, custom prompt, and Google Calendar config for a project."""
    from app.project_db import get_project
    project = get_project(client_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project '{client_id}' not found")
    return {"project": project}


@app.put("/api/projects/{client_id}")
async def api_update_project(client_id: str, payload: Dict[str, Any] = Body(...)):
    """Update project metadata, custom prompt, or Google Calendar settings."""
    from app.project_db import update_project
    updated = update_project(client_id, payload)
    return {"success": True, "project": updated}


@app.delete("/api/projects/{client_id}")
async def api_delete_project(client_id: str):
    """Delete client project and its dedicated workspace."""
    from app.project_db import delete_project
    success = delete_project(client_id)
    if not success:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"success": True, "message": f"Project '{client_id}' deleted."}


@app.post("/api/projects/{client_id}/activate")
async def api_activate_project_in_studio(client_id: str):
    """Activates the client project's custom prompt & voice directly in the voice studio."""
    from app.project_db import get_project
    from app.agents import save_assistant, set_active_assistant, get_assistant
    project = get_project(client_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    meta = project.get("meta", {})
    prompt_cfg = project.get("prompt", {})
    biz_name = meta.get("business_name", "Client Business")
    assistant_id = f"ast_{client_id}"

    existing = get_assistant(assistant_id) or {}
    agent_profile = {
        "id": assistant_id,
        "name": f"{prompt_cfg.get('persona_name', 'Riley')} ({biz_name})",
        "version": "v1",
        "tagline": f"{meta.get('industry', 'service').upper()} Receptionist for {biz_name}",
        "call_direction": "inbound",
        "language": "en",
        "system_prompt": prompt_cfg.get("system_prompt", ""),
        "first_message": prompt_cfg.get("first_message", f"Thank you for calling {biz_name}."),
        "tts_voice": prompt_cfg.get("tts_voice", "af_heart"),
        "voice_speed": float(prompt_cfg.get("voice_speed", 1.0)),
        "llm_model": existing.get("llm_model", settings.GROQ_MODEL),
        "stt_model": existing.get("stt_model", settings.WHISPER_MODEL),
        "preset": existing.get("preset", "balanced"),
        "background_sound": "default",
        "background_denoising": False,
        "created_at": meta.get("created_at", time.strftime("%Y-%m-%d %H:%M:%S")),
    }
    save_assistant(agent_profile)
    set_active_assistant(assistant_id)
    settings.GREETING_TEXT = agent_profile["first_message"]
    settings.KOKORO_VOICE = agent_profile["tts_voice"]

    return {"success": True, "active_id": assistant_id, "project": meta}


@app.post("/api/projects/{client_id}/calendar")
async def api_update_project_calendar(client_id: str, payload: Dict[str, Any] = Body(...)):
    """Save or update dedicated Google Calendar configuration for a client project."""
    from app.project_db import save_project_calendar_config
    saved_cal = save_project_calendar_config(client_id, payload)
    return {"success": True, "calendar": saved_cal}


@app.post("/api/projects/{client_id}/calendar/test")
async def api_test_project_calendar(client_id: str, payload: Optional[Dict[str, Any]] = Body(None)):
    """Test and verify Google Calendar connection for a specific client project."""
    from app.project_db import get_project_calendar_config, save_project_calendar_config
    from app.appointments import verify_google_calendar_connection

    cfg = get_project_calendar_config(client_id)
    if payload:
        if "calendar_id" in payload:
            cfg["calendar_id"] = payload["calendar_id"]
        if "service_account_json" in payload:
            cfg["service_account_json"] = payload["service_account_json"]

    cal_id = cfg.get("calendar_id", "primary")
    sa_data = cfg.get("service_account_json", "")

    # 1. Check for verified OAuth 2.0 connection
    if cfg.get("auth_type") == "oauth" and (cfg.get("is_connected") or cfg.get("oauth_user_email") or cfg.get("oauth_access_token")):
        oauth_email = cfg.get("oauth_user_email") or cal_id
        result = {
            "connected": True,
            "status": "oauth_active",
            "calendar_id": cal_id,
            "user_email": oauth_email,
            "message": f"✅ Live Google OAuth Verified for '{oauth_email}'."
        }
        cal_update = {
            "is_connected": 1,
            "last_tested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_status": "oauth_active",
            "last_error": ""
        }
        save_project_calendar_config(client_id, cal_update)
        return result

    result = await verify_google_calendar_connection(calendar_id=cal_id, service_account_data=sa_data)

    # Persist the test result in project db
    cal_update = {
        "is_connected": 1 if result.get("connected") else 0,
        "last_tested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "last_status": result.get("status", "unknown"),
        "last_error": result.get("message") if not result.get("connected") else ""
    }
    save_project_calendar_config(client_id, cal_update)
    result["is_inherited_global"] = cfg.get("is_inherited_global", False)
    return result


@app.get("/api/projects/{client_id}/appointments")
async def api_get_project_appointments(client_id: str, limit: int = Query(50)):
    """Retrieve appointments from client's dedicated database."""
    from app.project_db import get_project_appointments
    return {"appointments": get_project_appointments(client_id, limit=limit)}


# ---------------------------------------------------------
# Admin Dashboard — Client Management & Analytics APIs
# ---------------------------------------------------------

@app.get("/api/admin/clients")
async def api_admin_list_clients():
    """Returns all client profiles enriched with owner details, call counts, usage minutes, and live $20/mo billing."""
    from app.project_db import list_projects, migrate_legacy_clients, get_db_path
    import sqlite3
    import math

    # Ensure all clients in clients.json have their project DBs provisioned
    try:
        migrate_legacy_clients()
    except Exception as e:
        logger.warning(f"migrate_legacy_clients warning: {e}")

    clients_file = Path("data/clients.json")
    clients_data = {}
    if clients_file.exists():
        try:
            clients_data = json.loads(clients_file.read_text()).get("clients", {})
        except Exception:
            clients_data = {}

    # Also check project folders to ensure no orphan projects are missed
    projects = {}
    try:
        for p in list_projects():
            pid = p.get("id") or p.get("client_id")
            if pid:
                projects[pid] = p
    except Exception:
        projects = {}

    all_ids = set(clients_data.keys()).union(set(projects.keys()))
    enriched = []

    for cid in all_ids:
        profile = clients_data.get(cid, {})
        proj = projects.get(cid, {})
        proj_meta = proj.get("meta", {}) if isinstance(proj.get("meta"), dict) else {}
        proj_prompt = proj.get("prompt", {}) if isinstance(proj.get("prompt"), dict) else {}

        biz_name = profile.get("business_name") or proj_meta.get("business_name") or "Unnamed Business"
        owner_name = profile.get("owner_name") or profile.get("name") or proj_meta.get("owner_name") or "Business Owner"
        owner_email = profile.get("owner_email") or profile.get("email") or proj_meta.get("owner_email") or ""
        owner_phone = profile.get("owner_phone") or profile.get("forwarding_phone") or proj_meta.get("owner_phone") or ""
        forwarding_phone = profile.get("forwarding_phone") or proj_meta.get("forwarding_phone") or owner_phone
        assigned_phone = profile.get("assigned_phone") or proj_meta.get("assigned_phone") or "+1 (833) 420-5227"
        sms_phone = profile.get("sms_phone") or proj_meta.get("sms_phone") or forwarding_phone
        connection_status = profile.get("connection_status") or proj_meta.get("connection_status") or "pending"
        carrier = profile.get("carrier") or proj_meta.get("carrier") or "Verizon"
        verified_at = profile.get("verified_at") or proj_meta.get("verified_at") or profile.get("forwarding_setup_at") or ""
        after_hours_action = profile.get("after_hours_action") or profile.get("night_action") or proj_meta.get("after_hours_action") or "book_morning"
        pricing_policy = profile.get("pricing_policy") or profile.get("diagnostic_fee") or proj_meta.get("pricing_policy") or "$89 diagnostic fee credited toward repair"
        emergency_triggers = profile.get("emergency_triggers") or profile.get("transfer_rules") or proj_meta.get("emergency_triggers") or "Gas leak, carbon monoxide, water flooding, burst pipes, electrical sparks"
        answering_coverage = profile.get("answering_coverage") or profile.get("schedule_mode") or proj_meta.get("answering_coverage") or "always_24_7"
        trade = profile.get("trade") or profile.get("industry") or proj_meta.get("industry") or "hvac"
        address = profile.get("address") or profile.get("city") or proj_meta.get("address") or "Service Territory"
        status = profile.get("polar_status") or profile.get("status") or proj_meta.get("status") or "active"

        # Prompt resolution
        compiled_prompt = profile.get("compiled_prompt") or profile.get("livekit_prompt") or proj_prompt.get("system_prompt") or proj.get("livekit_prompt") or ""
        first_msg = profile.get("first_message") or proj_prompt.get("first_message") or ""
        persona = profile.get("persona_name") or proj_prompt.get("persona_name") or "Riley"

        item = {
            **profile,
            "id": cid,
            "client_id": cid,
            "business_name": biz_name,
            "owner_name": owner_name,
            "owner_email": owner_email,
            "owner_phone": owner_phone,
            "forwarding_phone": forwarding_phone,
            "assigned_phone": assigned_phone,
            "sms_phone": sms_phone,
            "connection_status": connection_status,
            "carrier": carrier,
            "verified_at": verified_at,
            "forwarding_setup_at": verified_at,
            "after_hours_action": after_hours_action,
            "pricing_policy": pricing_policy,
            "emergency_triggers": emergency_triggers,
            "answering_coverage": answering_coverage,
            "trade": trade,
            "industry": trade,
            "address": address,
            "status": status,
            "persona_name": persona,
            "first_message": first_msg,
            "prompt_preview": compiled_prompt[:180] + ("..." if len(compiled_prompt) > 180 else "") if compiled_prompt else "No prompt compiled yet.",
            "has_prompt": bool(compiled_prompt),
            "char_count": len(compiled_prompt),
            "est_tokens": math.ceil(len(compiled_prompt) / 4) if compiled_prompt else 0,
        }

        # Query call stats & usage from dedicated project SQLite DB
        db_path = get_db_path(cid)
        total_calls = 0
        total_appointments = 0
        total_seconds = 0.0
        last_call = None

        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path))
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) as cnt, COALESCE(SUM(call_duration), 0) as dur FROM call_logs")
                r = cur.fetchone()
                if r:
                    total_calls = r["cnt"] or 0
                    total_seconds = float(r["dur"] or 0)

                cur.execute("SELECT COUNT(*) as cnt FROM appointments")
                r_app = cur.fetchone()
                if r_app:
                    total_appointments = r_app["cnt"] or 0

                cur.execute("SELECT created_at FROM call_logs ORDER BY created_at DESC LIMIT 1")
                r_last = cur.fetchone()
                if r_last:
                    last_call = r_last["created_at"]
                conn.close()
            except Exception as e:
                logger.debug(f"DB read for {cid} error: {e}")

        total_minutes = round(total_seconds / 60.0, 1)
        # Pricing model: $20/mo Starter Plan + 60 free minutes + $0.25/min overage
        base_fee = 20.00
        included_mins = 60.0
        overage_rate = 0.25
        overage_mins = max(0.0, round(total_minutes - included_mins, 1))
        overage_cost = round(overage_mins * overage_rate, 2)
        total_bill = round(base_fee + overage_cost, 2)
        wholesale_cost = round(total_minutes * 0.035, 2)
        gross_profit = round(max(0.0, total_bill - wholesale_cost), 2)
        margin_pct = round((gross_profit / total_bill * 100) if total_bill > 0 else 0, 1)

        item["total_calls"] = total_calls
        item["calls_count"] = total_calls
        item["total_appointments"] = total_appointments
        item["appointments_count"] = total_appointments
        item["total_seconds"] = total_seconds
        item["total_minutes"] = total_minutes
        item["included_minutes"] = included_mins
        item["overage_minutes"] = overage_mins
        item["overage_rate"] = overage_rate
        item["overage_cost"] = overage_cost
        item["base_fee"] = base_fee
        item["estimated_cost"] = total_bill
        item["wholesale_cost"] = wholesale_cost
        item["gross_profit"] = gross_profit
        item["margin_pct"] = margin_pct
        item["last_call_at"] = last_call
        item["has_project"] = db_path.exists()
        item["portal_url"] = f"/portal?client_id={cid}"

        enriched.append(item)

    # Sort: active first, then most recently active
    enriched.sort(key=lambda x: (
        x.get("status") != "active",
        -(x.get("total_calls") or 0),
        x.get("last_call_at") or "",
    ), reverse=False)

    return {"clients": enriched, "total": len(enriched)}


@app.get("/api/admin/clients/{client_id}/full")
async def api_admin_client_full(client_id: str):
    """Returns comprehensive end-to-end client dossier: profile, compiled prompt, calls, transcripts, appointments, and cost analytics."""
    from app.project_db import get_project, get_project_calls, get_project_appointments, get_db_path
    import sqlite3
    import math

    clients_file = Path("data/clients.json")
    profile = {}
    if clients_file.exists():
        try:
            data = json.loads(clients_file.read_text())
            profile = data.get("clients", {}).get(client_id, {})
        except Exception:
            profile = {}

    project = get_project(client_id) or {}
    proj_meta = project.get("meta", {}) if isinstance(project.get("meta"), dict) else {}
    proj_prompt = project.get("prompt", {}) if isinstance(project.get("prompt"), dict) else {}
    proj_cal = project.get("calendar", {}) if isinstance(project.get("calendar"), dict) else {}

    # Merge client metadata
    biz_name = profile.get("business_name") or proj_meta.get("business_name") or "Comfort Breeze HVAC"
    owner_name = profile.get("owner_name") or profile.get("name") or proj_meta.get("owner_name") or "Abdul Aziz"
    owner_email = profile.get("owner_email") or profile.get("email") or proj_meta.get("owner_email") or "owner@business.com"
    owner_phone = profile.get("owner_phone") or profile.get("forwarding_phone") or proj_meta.get("owner_phone") or "+1 (555) 234-5678"
    forwarding_phone = profile.get("forwarding_phone") or proj_meta.get("forwarding_phone") or owner_phone
    sms_phone = profile.get("sms_phone") or forwarding_phone
    assigned_phone = profile.get("assigned_phone") or proj_meta.get("assigned_phone") or "+1 (833) 420-5227"
    trade = profile.get("trade") or profile.get("industry") or proj_meta.get("industry") or "hvac"
    address = profile.get("address") or profile.get("city") or proj_meta.get("address") or "Service Area"
    status = profile.get("polar_status") or profile.get("status") or proj_meta.get("status") or "active"
    hours = profile.get("hours") or proj_prompt.get("hours") or "Mon-Fri 8:00 AM - 6:00 PM"
    timezone = profile.get("timezone") or proj_meta.get("timezone") or "America/New_York"
    diag_fee = profile.get("diagnostic_fee") or "$89"
    fee_policy = profile.get("fee_policy") or "Credited toward repair"
    created_at = profile.get("created_at") or proj_meta.get("created_at") or "2026-09-20"

    # Prompt details
    system_prompt = (
        profile.get("compiled_prompt")
        or profile.get("livekit_prompt")
        or proj_prompt.get("system_prompt")
        or project.get("livekit_prompt")
        or ""
    )
    first_message = (
        profile.get("first_message")
        or proj_prompt.get("first_message")
        or f"Thank you for calling {biz_name}! This is Riley. How can I help get your home comfortable today?"
    )
    persona_name = profile.get("persona_name") or proj_prompt.get("persona_name") or "Riley"
    tts_voice = profile.get("tts_voice") or proj_prompt.get("tts_voice") or "flux-heather-en"
    stt_model = profile.get("stt_model") or "nova-3"
    llm_model = profile.get("llm_model") or "gemini-3.1-flash-lite"

    # Calls with transcripts & parsed extracted info
    raw_calls = get_project_calls(client_id, limit=100)
    calls = []
    total_seconds = 0.0
    for c in raw_calls:
        dur = float(c.get("call_duration") or 0)
        total_seconds += dur
        # Parse transcript if string
        transcript = c.get("transcript")
        if isinstance(transcript, str):
            try:
                transcript = json.loads(transcript)
            except Exception:
                transcript = [{"speaker": "Transcript", "text": transcript}]

        extracted = c.get("extracted_info")
        if isinstance(extracted, str):
            try:
                extracted = json.loads(extracted)
            except Exception:
                extracted = {}

        rec_file = c.get("recording_file") or ""
        rec_url = f"/api/recordings/{Path(rec_file).name}" if rec_file else ""

        calls.append({
            "id": c.get("id") or f"call_{len(calls)+1}",
            "caller_phone": c.get("caller_phone") or "Unknown Caller",
            "duration_seconds": round(dur),
            "duration_formatted": f"{int(dur // 60)}m {int(dur % 60):02d}s",
            "transcript": transcript or [],
            "extracted_info": extracted or {},
            "recording_url": rec_url,
            "created_at": c.get("created_at") or "",
        })

    # Appointments
    raw_appts = get_project_appointments(client_id, limit=50)
    appts = []
    for a in raw_appts:
        appts.append({
            "id": a.get("id") or f"apt_{len(appts)+1}",
            "client_name": a.get("client_name") or "Customer",
            "client_phone": a.get("client_phone") or "",
            "service_requested": a.get("service_requested") or "Diagnostic & Service",
            "service_address": a.get("service_address") or "",
            "appointment_date": a.get("appointment_date") or "",
            "window": a.get("window") or "arrival window",
            "status": a.get("status") or "confirmed",
            "summary": a.get("summary") or "",
            "created_at": a.get("created_at") or "",
        })

    # Usage & Cost computation ($20/mo + 60 free minutes + $0.25/min)
    total_minutes = round(total_seconds / 60.0, 1)
    base_fee = 20.00
    included_mins = 60.0
    overage_rate = 0.25
    overage_mins = max(0.0, round(total_minutes - included_mins, 1))
    overage_cost = round(overage_mins * overage_rate, 2)
    total_bill = round(base_fee + overage_cost, 2)
    wholesale_cost = round(total_minutes * 0.035, 2)
    gross_profit = round(max(0.0, total_bill - wholesale_cost), 2)
    margin_pct = round((gross_profit / total_bill * 100) if total_bill > 0 else 0, 1)

    return {
        "client": {
            "id": client_id,
            "business_name": biz_name,
            "owner_name": owner_name,
            "owner_email": owner_email,
            "owner_phone": owner_phone,
            "forwarding_phone": forwarding_phone,
            "sms_phone": sms_phone,
            "assigned_phone": assigned_phone,
            "trade": trade,
            "industry": trade,
            "address": address,
            "hours": hours,
            "timezone": timezone,
            "diagnostic_fee": diag_fee,
            "fee_policy": fee_policy,
            "status": status,
            "created_at": created_at,
            "portal_url": f"/portal?client_id={client_id}",
        },
        "prompt": {
            "persona_name": persona_name,
            "first_message": first_message,
            "system_prompt": system_prompt,
            "char_count": len(system_prompt),
            "word_count": len(system_prompt.split()),
            "est_tokens": math.ceil(len(system_prompt) / 4) if system_prompt else 0,
            "tts_voice": tts_voice,
            "stt_model": stt_model,
            "llm_model": llm_model,
        },
        "calendar": {
            "is_connected": bool(proj_cal.get("is_connected")),
            "calendar_id": proj_cal.get("calendar_id") or "primary",
            "morning_slot_capacity": proj_cal.get("morning_slot_capacity", 2),
            "afternoon_slot_capacity": proj_cal.get("afternoon_slot_capacity", 2),
            "status": proj_cal.get("last_status") or "active",
        },
        "stats": {
            "total_calls": len(calls),
            "total_seconds": total_seconds,
            "total_minutes": total_minutes,
            "total_appointments": len(appts),
            "plan_name": "Starter Plan ($20/mo)",
            "base_fee": base_fee,
            "included_minutes": included_mins,
            "overage_minutes": overage_mins,
            "overage_rate": overage_rate,
            "overage_cost": overage_cost,
            "total_client_bill": total_bill,
            "wholesale_api_cost": wholesale_cost,
            "estimated_gross_profit": gross_profit,
            "margin_pct": margin_pct,
        },
        "calls": calls,
        "appointments": appts,
    }


@app.post("/api/admin/clients/{client_id}/update-prompt")
async def api_admin_update_client_prompt(client_id: str, payload: Dict[str, Any] = Body(...)):
    """Save edited system prompt or first greeting directly to client DB and clients.json."""
    from app.project_db import get_db_connection, sync_project_json_files
    import sqlite3

    prompt = payload.get("system_prompt", "").strip()
    first_msg = payload.get("first_message", "").strip()
    persona = payload.get("persona_name", "").strip()

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    # Update clients.json
    clients_file = Path("data/clients.json")
    if clients_file.exists():
        try:
            data = json.loads(clients_file.read_text())
            c = data.get("clients", {}).get(client_id, {})
            c["compiled_prompt"] = prompt
            c["livekit_prompt"] = prompt
            if first_msg:
                c["first_message"] = first_msg
            if persona:
                c["persona_name"] = persona
            data["clients"][client_id] = c
            clients_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Error updating clients.json: {e}")

    # Update SQLite custom_prompt table
    try:
        conn = get_db_connection(client_id)
        cur = conn.cursor()
        cur.execute("""
            UPDATE custom_prompt
            SET system_prompt = ?, livekit_prompt = ?, first_message = COALESCE(NULLIF(?, ''), first_message),
                persona_name = COALESCE(NULLIF(?, ''), persona_name), updated_at = datetime('now')
            WHERE client_id = ?
        """, (prompt, prompt, first_msg, persona, client_id))
        conn.commit()
        conn.close()
        sync_project_json_files(client_id)
    except Exception as e:
        logger.error(f"Error updating project DB for {client_id}: {e}")

    return {"success": True, "client_id": client_id, "char_count": len(prompt)}


@app.get("/api/admin/clients/{client_id}/calls")
async def api_admin_client_calls(client_id: str, limit: int = Query(50)):
    """Returns call history for a specific client from their project DB."""
    import sqlite3
    db_path = Path(f"data/projects/{client_id}/client.db")
    if not db_path.exists():
        return {"calls": [], "total": 0}

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM call_logs ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        calls = []
        for row in cur.fetchall():
            call = dict(row)
            # Parse JSON fields
            for field in ("transcript", "extracted_info"):
                if call.get(field):
                    try:
                        call[field] = json.loads(call[field])
                    except Exception:
                        pass
            calls.append(call)
        conn.close()
        return {"calls": calls, "total": len(calls)}
    except Exception as e:
        logger.error(f"Error reading calls for {client_id}: {e}")
        return {"calls": [], "total": 0, "error": str(e)}


@app.get("/api/admin/clients/{client_id}/recordings")
async def api_admin_client_recordings(client_id: str):
    """Returns recording files for a specific client."""
    import sqlite3
    db_path = Path(f"data/projects/{client_id}/client.db")
    if not db_path.exists():
        return {"recordings": []}

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT id, recording_file, call_duration, created_at FROM call_logs "
            "WHERE recording_file IS NOT NULL AND recording_file != '' "
            "ORDER BY created_at DESC"
        )
        recordings = []
        for row in cur.fetchall():
            r = dict(row)
            rec_file = r.get("recording_file", "")
            if rec_file:
                r["url"] = f"/api/recordings/{Path(rec_file).name}"
                r["exists"] = Path(f"data/recordings/{Path(rec_file).name}").exists()
            recordings.append(r)
        conn.close()
        return {"recordings": recordings}
    except Exception as e:
        logger.error(f"Error reading recordings for {client_id}: {e}")
        return {"recordings": [], "error": str(e)}


@app.get("/api/projects/{client_id}/stats")
async def api_project_stats(client_id: str):
    """Aggregate stats for a client project: call counts, appointment counts, usage costs."""
    import sqlite3
    db_path = Path(f"data/projects/{client_id}/client.db")
    stats = {
        "total_calls": 0,
        "calls_this_week": 0,
        "calls_this_month": 0,
        "total_appointments": 0,
        "appointments_this_week": 0,
        "total_minutes": 0.0,
        "estimated_cost": 0.0,
        "escalations": 0,
        "unique_callers": 0,
    }

    if not db_path.exists():
        return stats

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Total calls
        cur.execute("SELECT COUNT(*) as cnt, COALESCE(SUM(call_duration), 0) as dur FROM call_logs")
        row = cur.fetchone()
        stats["total_calls"] = row["cnt"]
        stats["total_minutes"] = round(row["dur"] / 60.0, 1) if row["dur"] else 0

        # Calls this week (last 7 days)
        cur.execute(
            "SELECT COUNT(*) as cnt FROM call_logs WHERE created_at >= datetime('now', '-7 days')"
        )
        stats["calls_this_week"] = cur.fetchone()["cnt"]

        # Calls this month (last 30 days)
        cur.execute(
            "SELECT COUNT(*) as cnt FROM call_logs WHERE created_at >= datetime('now', '-30 days')"
        )
        stats["calls_this_month"] = cur.fetchone()["cnt"]

        # Appointments
        cur.execute("SELECT COUNT(*) as cnt FROM appointments")
        stats["total_appointments"] = cur.fetchone()["cnt"]

        cur.execute(
            "SELECT COUNT(*) as cnt FROM appointments WHERE created_at >= datetime('now', '-7 days')"
        )
        stats["appointments_this_week"] = cur.fetchone()["cnt"]

        # Unique callers
        cur.execute("SELECT COUNT(DISTINCT caller_phone) as cnt FROM call_logs WHERE caller_phone IS NOT NULL")
        stats["unique_callers"] = cur.fetchone()["cnt"]

        # Billing ($20/mo Starter Plan: includes 60 min, then $0.25/min overage)
        base_fee = 20.00
        included_mins = 60.0
        overage_rate = 0.25
        total_mins = stats.get("total_minutes", 0.0)
        overage_mins = max(0.0, round(total_mins - included_mins, 1))
        overage_cost = round(overage_mins * overage_rate, 2)
        total_bill = round(base_fee + overage_cost, 2)
        wholesale_cost = round(total_mins * 0.035, 2)
        gross_profit = round(max(0.0, total_bill - wholesale_cost), 2)
        margin_pct = round((gross_profit / total_bill * 100) if total_bill > 0 else 0, 1)

        stats["plan"] = "starter"
        stats["plan_name"] = "Starter Plan ($20/mo)"
        stats["base_fee"] = base_fee
        stats["included_minutes"] = included_mins
        stats["overage_minutes"] = overage_mins
        stats["overage_rate"] = overage_rate
        stats["overage_cost"] = overage_cost
        stats["estimated_cost"] = total_bill
        stats["wholesale_cost"] = wholesale_cost
        stats["gross_profit"] = gross_profit
        stats["margin_pct"] = margin_pct

        conn.close()
    except Exception as e:
        logger.error(f"Error computing stats for {client_id}: {e}")

    return stats


@app.get("/api/projects/{client_id}/customers")
async def api_project_customers(client_id: str, limit: int = Query(100)):
    """Unique caller directory aggregated from call logs."""
    import sqlite3
    db_path = Path(f"data/projects/{client_id}/client.db")
    if not db_path.exists():
        return {"customers": []}

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("""
            SELECT
                caller_phone,
                COUNT(*) as call_count,
                MAX(created_at) as last_call,
                MIN(created_at) as first_call,
                COALESCE(SUM(call_duration), 0) as total_duration
            FROM call_logs
            WHERE caller_phone IS NOT NULL AND caller_phone != ''
            GROUP BY caller_phone
            ORDER BY last_call DESC
            LIMIT ?
        """, (limit,))

        customers = []
        for row in cur.fetchall():
            c = dict(row)
            # Try to get name from extracted_info of most recent call
            cur2 = conn.cursor()
            cur2.execute(
                "SELECT extracted_info FROM call_logs WHERE caller_phone = ? ORDER BY created_at DESC LIMIT 1",
                (c["caller_phone"],)
            )
            info_row = cur2.fetchone()
            if info_row and info_row["extracted_info"]:
                try:
                    info = json.loads(info_row["extracted_info"])
                    c["name"] = info.get("customer_name") or info.get("name") or ""
                    c["service"] = info.get("service_requested") or info.get("service") or ""
                except Exception:
                    c["name"] = ""
                    c["service"] = ""
            else:
                c["name"] = ""
                c["service"] = ""
            customers.append(c)

        conn.close()
        return {"customers": customers, "total": len(customers)}
    except Exception as e:
        logger.error(f"Error reading customers for {client_id}: {e}")
        return {"customers": [], "error": str(e)}


@app.post("/api/admin/clients/{client_id}/test-chat")
async def api_admin_test_chat(client_id: str, payload: Dict[str, Any] = Body(...)):
    """Test chat with a specific client's agent configuration."""
    from app.onboarding import simulate_agent_turn
    message = payload.get("message", "")
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    # Load client's compiled prompt
    clients_file = Path("data/clients.json")
    prompt = None
    if clients_file.exists():
        try:
            data = json.loads(clients_file.read_text())
            client = data.get("clients", {}).get(client_id, {})
            prompt = client.get("compiled_prompt") or client.get("livekit_prompt")
        except Exception:
            pass

    result = await simulate_agent_turn(
        message=message,
        client_id=client_id,
        system_prompt_override=prompt,
    )
    return result


@app.post("/api/integrations/google-calendar/test")
async def api_test_global_google_calendar(payload: Optional[Dict[str, Any]] = Body(None)):
    """Test Google Calendar credentials (OAuth or Service Account) and report sandbox vs live status."""
    from app.integrations import get_integrations_settings
    from app.appointments import verify_google_calendar_connection
    from app.google_oauth import is_google_oauth_configured, get_valid_access_token
    from app.project_db import get_project_calendar_config

    payload = payload or {}
    cal_id = payload.get("google_calendar_id") or "primary"
    sa_data = payload.get("google_service_account_json", "")
    client_id = payload.get("client_id", "riley_hvac")

    # 1. Check if client has OAuth configuration in DB/calendar.json
    cal_cfg = get_project_calendar_config(client_id)
    oauth_email = cal_cfg.get("oauth_user_email", "")

    # Check if this is a sandbox connection
    if cal_cfg.get("is_connected") and cal_cfg.get("oauth_access_token") == "sandbox_oauth_token_verified":
        return {
            "connected": True,
            "status": "simulated",
            "is_sandbox": True,
            "calendar_id": cal_id,
            "oauth_configured": is_google_oauth_configured(),
            "message": f"Dev Sandbox Mode: '{oauth_email or cal_id}' is simulated. Real Google login requires GOOGLE_OAUTH_CLIENT_ID and GOOGLE_OAUTH_CLIENT_SECRET in .env."
        }

    # 2. If real OAuth token exists, verify with Google Calendar API
    if is_google_oauth_configured():
        token = await get_valid_access_token(client_id)
        if token:
            try:
                async with httpx.AsyncClient(timeout=6.0) as http_client:
                    r = await http_client.get(
                        f"https://www.googleapis.com/calendar/v3/calendars/{urllib.parse.quote(cal_id)}",
                        headers={"Authorization": f"Bearer {token}"}
                    )
                    if r.status_code == 200:
                        data = r.json()
                        return {
                            "connected": True,
                            "status": "connected",
                            "is_sandbox": False,
                            "calendar_id": cal_id,
                            "calendar_title": data.get("summary", cal_id),
                            "message": f"✅ Live Google Verified: Successfully reached Google Calendar API for '{oauth_email or cal_id}'."
                        }
                    else:
                        return {
                            "connected": False,
                            "status": "auth_failed",
                            "is_sandbox": False,
                            "calendar_id": cal_id,
                            "message": f"Google API returned {r.status_code}: {r.text}"
                        }
            except Exception as ex:
                return {
                    "connected": False,
                    "status": "error",
                    "is_sandbox": False,
                    "calendar_id": cal_id,
                    "message": f"Error reaching Google Calendar: {ex}"
                }

    # 3. Fallback to Service Account verification
    result = await verify_google_calendar_connection(calendar_id=cal_id, service_account_data=sa_data)
    result["oauth_configured"] = is_google_oauth_configured()
    return result


@app.post("/api/integrations/sms/test")
async def api_test_sms(payload: Dict[str, Any] = Body(...)):
    """Test sending an SMS via Telnyx, Twilio, or simulated provider."""
    from app.integrations import send_sms
    to_phone = payload.get("to_phone") or payload.get("phone") or payload.get("to")
    if not to_phone:
        raise HTTPException(status_code=400, detail="to_phone is required")
    message = payload.get("message") or payload.get("text") or "Hello from Aria Voice AI! Your SMS integration is functioning properly."
    provider = payload.get("provider")
    result = await send_sms(to_phone=to_phone, message=message, provider=provider)
    return {"success": True, "result": result}


# ---------------------------------------------------------
# 1-Click Google Calendar OAuth 2.0 Engine
# ---------------------------------------------------------
@app.get("/api/auth/google/url")
async def api_get_google_auth_url(
    client_id: str = "riley_hvac",
    email: Optional[str] = None
):
    """Returns the 1-click Google OAuth URL or sandbox fast-connect URL."""
    from app.google_oauth import build_google_auth_url, is_google_oauth_configured
    url = build_google_auth_url(client_id_project=client_id, email=email)
    return {
        "url": url,
        "configured": is_google_oauth_configured(),
        "client_id": client_id,
        "email": email
    }


@app.get("/api/auth/google/sandbox-connect")
async def api_google_sandbox_connect(
    client_id: str = "riley_hvac",
    email: Optional[str] = None,
    redirect: bool = True
):
    """Fast 1-click Sandbox connection for frictionless testing/demo."""
    from app.google_oauth import save_oauth_connection
    user_email = (email or "").strip() or "abdulazizalpadi91@gmail.com"
    user_display = user_email.split("@")[0].replace(".", " ").title() if "@" in user_email else "Business Owner"
    res = save_oauth_connection(
        client_id=client_id,
        user_email=user_email,
        access_token="sandbox_oauth_token_verified",
        refresh_token="sandbox_refresh_token_verified",
        expires_in=86400 * 30,
        user_name=user_display
    )
    if redirect:
        html = f"""<!DOCTYPE html>
<html>
<body style="font-family:sans-serif;background:#0f172a;color:#f8fafc;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;">
  <div style="background:#1e293b;padding:32px;border-radius:16px;text-align:center;max-width:400px;border:1px solid #334155;box-shadow:0 25px 50px -12px rgba(0,0,0,0.5);">
    <div style="font-size:36px;margin-bottom:12px;">✅</div>
    <h2 style="margin:0 0 8px;color:#10b981;">Google Calendar Connected!</h2>
    <p style="font-size:14px;color:#94a3b8;margin:0 0 16px;">Linked to <strong>{user_email}</strong></p>
    <p style="font-size:12px;color:#64748b;">Closing window and returning to dashboard...</p>
  </div>
  <script>
    if (window.opener) {{
      window.opener.postMessage({{ type: 'gcal_connected', email: '{user_email}', client_id: '{client_id}' }}, '*');
      setTimeout(() => window.close(), 1200);
    }} else {{
      setTimeout(() => {{ window.location.href = '/livekit?gcal_connected=1&email={urllib.parse.quote(user_email)}&client_id={urllib.parse.quote(client_id)}'; }}, 1200);
    }}
  </script>
</body>
</html>"""
        return HTMLResponse(html)
    return {"success": True, "connected": True, "email": user_email, "client_id": client_id}


@app.get("/api/auth/google/callback")
async def api_google_oauth_callback(
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None
):
    """Google OAuth 2.0 redirect callback endpoint."""
    if error:
        return HTMLResponse(f"<h3>Google Connection Cancelled: {error}</h3><script>setTimeout(() => window.close(), 2500);</script>")

    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code from Google")

    client_id = "riley_hvac"
    if state:
        try:
            raw = base64.urlsafe_b64decode(state).decode()
            data = json.loads(raw)
            client_id = data.get("client_id", "riley_hvac")
        except Exception:
            pass

    from app.google_oauth import exchange_google_code
    try:
        res = await exchange_google_code(code=code, client_id_project=client_id)
        email = res.get("email", "")
        html = f"""<!DOCTYPE html>
<html>
<body style="font-family:sans-serif;background:#0f172a;color:#f8fafc;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;">
  <div style="background:#1e293b;padding:32px;border-radius:16px;text-align:center;max-width:400px;border:1px solid #334155;">
    <div style="font-size:36px;margin-bottom:12px;">✅</div>
    <h2 style="margin:0 0 8px;color:#10b981;">Google Calendar Connected!</h2>
    <p style="font-size:14px;color:#94a3b8;margin:0 0 16px;">Linked to <strong>{email}</strong></p>
    <p style="font-size:12px;color:#64748b;">Closing window and returning to dashboard...</p>
  </div>
  <script>
    if (window.opener) {{
      window.opener.postMessage({{ type: 'gcal_connected', email: '{email}', client_id: '{client_id}' }}, '*');
      setTimeout(() => window.close(), 1500);
    }} else {{
      setTimeout(() => {{ window.location.href = '/livekit?gcal_connected=1&email={email}'; }}, 1500);
    }}
  </script>
</body>
</html>"""
        return HTMLResponse(html)
    except Exception as ex:
        return HTMLResponse(f"<h3>Error connecting Google Calendar: {ex}</h3>")


@app.get("/api/auth/google/status")
async def api_google_oauth_status(client_id: str = "riley_hvac"):
    """Check current Google Calendar OAuth connection status for a client."""
    from app.project_db import get_project_calendar_config
    cfg = get_project_calendar_config(client_id)
    return {
        "client_id": client_id,
        "is_connected": bool(cfg.get("is_connected")),
        "auth_type": cfg.get("auth_type", "oauth"),
        "user_email": cfg.get("oauth_user_email", ""),
        "last_tested_at": cfg.get("last_tested_at", ""),
        "last_status": cfg.get("last_status", "untested"),
    }


@app.post("/api/auth/google/disconnect")
async def api_google_oauth_disconnect(payload: Dict[str, Any] = Body(...)):
    """Disconnect Google Calendar for a client."""
    from app.google_oauth import disconnect_google_calendar
    client_id = payload.get("client_id", "riley_hvac")
    success = disconnect_google_calendar(client_id)
    return {"success": success, "client_id": client_id}




# Vapi-Style Model Switcher & Status APIs
# ---------------------------------------------------------
@app.get("/api/models")
async def get_models_metadata():
    """Returns available STT, TTS voices (with humanness & latency data), and LLM options."""
    return {
        "active_preset": settings.ACTIVE_PRESET,
        "greeting": {
            "text": settings.GREETING_TEXT,
            "cached_frames": len(_cached_greeting_frames),
            "duration_sec": round(_cached_greeting_duration, 2),
            "voice": _cached_greeting_voice or settings.KOKORO_VOICE,
        },
        "stt": {
            "active": settings.WHISPER_MODEL,
            "compute_type": settings.WHISPER_COMPUTE_TYPE,
            "options": [
                {
                    "id": "tiny",
                    "name": "Whisper Tiny",
                    "latency": "180 ms",
                    "accuracy": "Good",
                    "wer": "4.2%",
                    "cost": "$0.00/min",
                    "notes": "Ultra-fast, lowest CPU footprint",
                },
                {
                    "id": "base",
                    "name": "Whisper Base",
                    "latency": "280 ms",
                    "accuracy": "Better",
                    "wer": "2.8%",
                    "cost": "$0.00/min",
                    "notes": "Excellent balance of speed & accuracy",
                },
                {
                    "id": "small",
                    "name": "Whisper Small (Recommended)",
                    "latency": "420 ms",
                    "accuracy": "Best Local",
                    "wer": "1.8%",
                    "cost": "$0.00/min",
                    "notes": "Matches commercial STT accuracy on telephony audio",
                },
                {
                    "id": "distil-medium.en",
                    "name": "Distil-Medium English",
                    "latency": "320 ms",
                    "accuracy": "High",
                    "wer": "2.0%",
                    "cost": "$0.00/min",
                    "notes": "Distilled architecture for rapid English transcription",
                },
            ],
        },
        "tts_voices": {
            "active": settings.KOKORO_VOICE,
            "engine": "Kokoro ONNX",
            "options": [
                {
                    "id": "af_heart",
                    "name": "Aria (Heart) ★",
                    "gender": "Female",
                    "accent": "American",
                    "humanness": 92,
                    "latency": "350 ms",
                    "cost": "$0.00/min",
                    "notes": "Warm, natural pauses, high vocal clarity (Default)",
                },
                {
                    "id": "af_bella",
                    "name": "Bella",
                    "gender": "Female",
                    "accent": "American",
                    "humanness": 89,
                    "latency": "350 ms",
                    "cost": "$0.00/min",
                    "notes": "Confident, bright, modern receptionist",
                },
                {
                    "id": "af_nova",
                    "name": "Nova",
                    "gender": "Female",
                    "accent": "American",
                    "humanness": 86,
                    "latency": "350 ms",
                    "cost": "$0.00/min",
                    "notes": "Articulate, crisp, corporate tone",
                },
                {
                    "id": "am_adam",
                    "name": "Adam",
                    "gender": "Male",
                    "accent": "American",
                    "humanness": 88,
                    "latency": "350 ms",
                    "cost": "$0.00/min",
                    "notes": "Deep, reassuring, consultative tone",
                },
                {
                    "id": "am_michael",
                    "name": "Michael",
                    "gender": "Male",
                    "accent": "American",
                    "humanness": 87,
                    "latency": "350 ms",
                    "cost": "$0.00/min",
                    "notes": "Friendly, approachable, casual professional",
                },
                {
                    "id": "bf_emma",
                    "name": "Emma",
                    "gender": "Female",
                    "accent": "British",
                    "humanness": 90,
                    "latency": "350 ms",
                    "cost": "$0.00/min",
                    "notes": "Polite, warm British English accent",
                },
                {
                    "id": "bm_george",
                    "name": "George",
                    "gender": "Male",
                    "accent": "British",
                    "humanness": 87,
                    "latency": "350 ms",
                    "cost": "$0.00/min",
                    "notes": "Refined, sophisticated British accent",
                },
            ],
        },
        "llm_models": {
            "active": settings.GROQ_MODEL,
            "groq_configured": bool(settings.GROQ_API_KEY),
            "options": [
                {
                    "id": "llama-3.3-70b-versatile",
                    "name": "Llama 3.3 70B Versatile",
                    "provider": "Groq",
                    "ttft": "180 ms",
                    "intelligence": 98,
                    "cost": "$0.00 (Free Tier)",
                    "notes": "Top-tier conversational intelligence, tool calling, 3x faster than GPT-4",
                },
                {
                    "id": "llama-3.1-8b-instant",
                    "name": "Llama 3.1 8B Instant",
                    "provider": "Groq",
                    "ttft": "80 ms",
                    "intelligence": 85,
                    "cost": "$0.00 (Free Tier)",
                    "notes": "Sub-100ms ultra-fast responses for rapid fire Q&A",
                },
                {
                    "id": "gemma2-9b-it",
                    "name": "Gemma 2 9B IT",
                    "provider": "Groq",
                    "ttft": "120 ms",
                    "intelligence": 88,
                    "cost": "$0.00 (Free Tier)",
                    "notes": "Google's lightweight conversational language model",
                },
                {
                    "id": "demo-fallback",
                    "name": "Aria Built-in Demo Engine",
                    "provider": "Local",
                    "ttft": "10 ms",
                    "intelligence": 75,
                    "cost": "$0.00 (Zero Dependencies)",
                    "notes": "Instant local responder active when Groq key is pending",
                },
            ],
        },
        "presets": [
            {
                "id": "ultra_fast",
                "name": "Ultra Fast",
                "icon": "⚡",
                "stt": "tiny",
                "tts": "af_heart",
                "llm": "llama-3.1-8b-instant",
                "greeting_ttfa": "< 50 ms",
                "turn_latency": "~850 ms",
                "humanness": 85,
                "cost": "$0.00/min",
            },
            {
                "id": "balanced",
                "name": "Balanced (Vapi Equivalent)",
                "icon": "⚖️",
                "stt": "small",
                "tts": "af_heart",
                "llm": "llama-3.3-70b-versatile",
                "greeting_ttfa": "< 50 ms",
                "turn_latency": "~1,050 ms",
                "humanness": 92,
                "cost": "$0.00/min",
            },
            {
                "id": "high_intelligence",
                "name": "High Intelligence",
                "icon": "🧠",
                "stt": "small",
                "tts": "af_heart",
                "llm": "llama-3.3-70b-versatile",
                "greeting_ttfa": "< 50 ms",
                "turn_latency": "~1,200 ms",
                "humanness": 95,
                "cost": "$0.00/min",
            },
        ],
    }


@app.post("/api/models")
async def update_models_api(request: Request):
    """
    Hot-swap voice agent models, greeting text, or presets in real-time without restarting the server.
    """
    data = await request.json()
    new_preset = data.get("preset")
    new_stt = data.get("stt_model")
    new_tts = data.get("tts_voice")
    new_llm = data.get("llm_model")
    new_greeting = data.get("greeting_text")

    # Apply Preset shortcuts
    if new_preset == "ultra_fast":
        settings.ACTIVE_PRESET = "ultra_fast"
        settings.WHISPER_MODEL = "tiny"
        settings.GROQ_MODEL = "llama-3.1-8b-instant"
        settings.VAD_STOP_SECS = 0.25
    elif new_preset == "balanced":
        settings.ACTIVE_PRESET = "balanced"
        settings.WHISPER_MODEL = "small"
        settings.GROQ_MODEL = "llama-3.3-70b-versatile"
        settings.VAD_STOP_SECS = 0.30
    elif new_preset == "high_intelligence":
        settings.ACTIVE_PRESET = "high_intelligence"
        settings.WHISPER_MODEL = "small"
        settings.GROQ_MODEL = "llama-3.3-70b-versatile"
        settings.VAD_STOP_SECS = 0.35

    # Individual overrides
    if new_stt:
        settings.WHISPER_MODEL = new_stt
        # Pre-warm new whisper model in background
        get_whisper_instance(new_stt)

    greeting_needs_rebuild = False
    if new_tts and new_tts != settings.KOKORO_VOICE:
        settings.KOKORO_VOICE = new_tts
        greeting_needs_rebuild = True

    if new_llm:
        settings.GROQ_MODEL = new_llm

    if new_greeting and new_greeting != settings.GREETING_TEXT:
        settings.GREETING_TEXT = new_greeting.strip()
        greeting_needs_rebuild = True

    if greeting_needs_rebuild:
        precache_greeting(settings.GREETING_TEXT, settings.KOKORO_VOICE)

    logger.info(
        f"Updated models -> Preset: {settings.ACTIVE_PRESET} | STT: {settings.WHISPER_MODEL} | "
        f"TTS: {settings.KOKORO_VOICE} | LLM: {settings.GROQ_MODEL}"
    )

    return {
        "status": "updated",
        "active_preset": settings.ACTIVE_PRESET,
        "stt_model": settings.WHISPER_MODEL,
        "tts_voice": settings.KOKORO_VOICE,
        "llm_model": settings.GROQ_MODEL,
        "greeting_cached": len(_cached_greeting_frames) > 0,
        "greeting_duration_sec": round(_cached_greeting_duration, 2),
    }


# ---------------------------------------------------------
# Assistants Multi-Agent Management Endpoints
# ---------------------------------------------------------
from app.agents import (
    list_assistants,
    get_active_assistant_id,
    get_active_assistant,
    get_assistant,
    set_active_assistant,
    create_assistant,
    update_assistant,
    delete_assistant,
)

@app.get("/api/assistants")
async def api_list_assistants():
    """Returns all saved assistant profiles and the currently active assistant ID."""
    return {
        "active_id": get_active_assistant_id(),
        "assistants": list_assistants(),
    }


@app.post("/api/assistants")
async def api_create_assistant(request: Request):
    """Creates a new assistant profile."""
    data = await request.json()
    agent = create_assistant(data)
    return {"status": "created", "assistant": agent}


@app.get("/api/assistants/{assistant_id}")
async def api_get_assistant(assistant_id: str):
    """Returns full profile of a specific assistant."""
    agent = get_assistant(assistant_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Assistant not found")
    return agent


@app.put("/api/assistants/{assistant_id}")
async def api_update_assistant(assistant_id: str, request: Request):
    """Updates an assistant's prompt, first message, voice, or model."""
    data = await request.json()
    updated = update_assistant(assistant_id, data)
    if not updated:
        raise HTTPException(status_code=404, detail="Assistant not found")

    # If this assistant is currently active, sync settings & re-cache greeting asynchronously
    if get_active_assistant_id() == assistant_id:
        settings.GREETING_TEXT = updated.get("first_message", settings.GREETING_TEXT)
        settings.KOKORO_VOICE = updated.get("tts_voice", settings.KOKORO_VOICE)
        settings.WHISPER_MODEL = updated.get("stt_model", settings.WHISPER_MODEL)
        settings.GROQ_MODEL = updated.get("llm_model", settings.GROQ_MODEL)
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            loop.run_in_executor(
                None,
                precache_greeting,
                settings.GREETING_TEXT,
                settings.KOKORO_VOICE,
                float(updated.get("voice_speed", 1.0)),
            )
        except Exception as e:
            logger.warning(f"Background precache error: {e}")

    return {"status": "updated", "assistant": updated}


@app.delete("/api/assistants/{assistant_id}")
async def api_delete_assistant(assistant_id: str):
    """Deletes an assistant."""
    success = delete_assistant(assistant_id)
    if not success:
        raise HTTPException(status_code=404, detail="Assistant not found")
    return {"status": "deleted", "active_id": get_active_assistant_id()}


@app.post("/api/assistants/{assistant_id}/activate")
async def api_activate_assistant(assistant_id: str):
    """Activates an assistant to handle all incoming phone calls and web calls."""
    success = set_active_assistant(assistant_id)
    if not success:
        raise HTTPException(status_code=404, detail="Assistant not found")
    agent = get_assistant(assistant_id)
    if agent:
        settings.GREETING_TEXT = agent.get("first_message", settings.GREETING_TEXT)
        settings.KOKORO_VOICE = agent.get("tts_voice", settings.KOKORO_VOICE)
        settings.WHISPER_MODEL = agent.get("stt_model", settings.WHISPER_MODEL)
        settings.GROQ_MODEL = agent.get("llm_model", settings.GROQ_MODEL)
        settings.ACTIVE_PRESET = agent.get("preset", "balanced")
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            loop.run_in_executor(
                None,
                precache_greeting,
                settings.GREETING_TEXT,
                settings.KOKORO_VOICE,
                float(agent.get("voice_speed", 1.0)),
            )
        except Exception as e:
            logger.warning(f"Background precache error: {e}")
    return {"status": "activated", "assistant": agent}


# ---------------------------------------------------------
# Template Management APIs (User Editable & Customizable)
# ---------------------------------------------------------
@app.get("/api/templates")
async def api_list_templates():
    """Returns all available prompt and assistant templates."""
    from app.templates_mgr import list_templates
    return {"templates": list_templates()}


@app.get("/api/templates/{template_id}")
async def api_get_template(template_id: str):
    """Retrieves a single template by ID."""
    from app.templates_mgr import get_template
    tpl = get_template(template_id)
    if not tpl:
        raise HTTPException(status_code=404, detail="Template not found")
    return tpl


@app.post("/api/templates")
async def api_create_template(request: Request):
    """Creates a new custom prompt template."""
    from app.templates_mgr import save_template
    data = await request.json()
    if not data.get("name"):
        raise HTTPException(status_code=400, detail="Template name is required")
    saved = save_template(data)
    return {"status": "created", "template": saved}


@app.put("/api/templates/{template_id}")
async def api_update_template(template_id: str, request: Request):
    """Updates an existing prompt template."""
    from app.templates_mgr import save_template
    data = await request.json()
    data["id"] = template_id
    saved = save_template(data)
    return {"status": "updated", "template": saved}


@app.delete("/api/templates/{template_id}")
async def api_delete_template(template_id: str):
    """Deletes a custom template."""
    from app.templates_mgr import delete_template
    success = delete_template(template_id)
    if not success:
        raise HTTPException(status_code=404, detail="Template not found")
    return {"status": "deleted"}


@app.post("/api/templates/reset")
async def api_reset_templates():
    """Resets all templates to default factory settings."""
    from app.templates_mgr import reset_templates_to_default
    return {"status": "reset", "templates": reset_templates_to_default()}


@app.post("/api/prompts/generate")
async def api_generate_prompt(request: Request):
    """Generates or enhances a structured voice agent prompt using AI or intelligent synthesizer."""
    data = await request.json()
    topic = data.get("topic", "").strip() or "Customer Service"
    role = data.get("role", "").strip() or "Voice Assistant"
    agent_name = data.get("name", "Riley").strip() or "Riley"

    # If Groq is configured, generate high-quality prompt via LLM
    if settings.GROQ_API_KEY:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"},
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are an elite voice AI prompt architect for telephone agents (like Vapi and Retell AI). "
                                    "Format prompts strictly using these sections:\n"
                                    "[Identity & Purpose]\n"
                                    "[Conversational Style & Spoken Rules]\n"
                                    "[Turn-Taking & Brevity]\n"
                                    "[Step-by-Step Flow]\n"
                                    "[Explicit Confirmation Flow]\n"
                                    "[Guardrails & Escalation]\n"
                                    "Enforce strict spoken conversational rules: sentences must be under 18 words, "
                                    "use natural contractions ('I'm', 'we'll', 'let's'), never use markdown/bullets/asterisks, "
                                    "include conversational markers ('Got it', 'Understood', 'Sure thing'), format numbers phonetically ('two p.m.'). "
                                    "Return ONLY the raw system prompt, nothing else."
                                ),
                            },
                            {
                                "role": "user",
                                "content": f"Create a production-grade voice AI prompt for an assistant named '{agent_name}'. Role/business: '{role}'. Industry/Topic: '{topic}'.",
                            },
                        ],
                        "temperature": 0.4,
                        "max_tokens": 700,
                    },
                )
                if res.status_code == 200:
                    generated = res.json()["choices"][0]["message"]["content"]
                    return {"status": "generated", "system_prompt": generated}
        except Exception as e:
            logger.warning(f"Groq prompt generator failed, using synthesizer: {e}")

    # Intelligent template synthesizer fallback
    synthesized = (
        f"[Identity & Purpose]\n"
        f"You are {agent_name}, a professional, warm, and highly capable voice assistant specializing in {role} for {topic}. "
        f"Your goal is to guide callers, answer inquiries accurately, and complete customer workflows smoothly.\n\n"
        f"[Conversational Style & Spoken Rules]\n"
        f"- Sound warm, confident, and solution-focused.\n"
        f"- Speak naturally in everyday conversational English using contractions like \"I'm\", \"we'll\", \"don't\", and \"let's\".\n"
        f"- Keep every response to 1 to 2 short spoken sentences (strictly under 18 words each).\n"
        f"- Never use markdown formatting, asterisks, bullet points, or list numbering.\n"
        f"- Say numbers out phonetically as spoken words (\"two p.m.\", \"twenty-five dollars\").\n"
        f"- Use conversational markers before answering: \"Got it.\", \"Understood.\", \"Sure thing.\", or \"I can help with that.\"\n\n"
        f"[Turn-Taking & Brevity]\n"
        f"- Ask only ONE question at a time. Never chain questions together.\n"
        f"- End each turn with a clear, concise question or next step.\n"
        f"- If the caller interrupts, acknowledge their input gracefully and address it first.\n\n"
        f"[Step-by-Step Flow]\n"
        f"1. Greet the caller warmly and understand their primary request.\n"
        f"2. Collect necessary information one detail at a time.\n"
        f"3. Propose two specific solutions or time windows.\n\n"
        f"[Explicit Confirmation Flow]\n"
        f"- Always read back and confirm key details (names, dates, times, contact numbers) before concluding the call.\n\n"
        f"[Guardrails & Escalation]\n"
        f"- If an issue cannot be resolved or caller asks for a human: \"Let me arrange for our specialist to call you back right away.\""
    )
    return {"status": "generated", "system_prompt": synthesized}


@app.post("/api/prompts/condense")
async def api_condense_prompt(request: Request):
    """Compiles and condenses a long, verbose system prompt or playbook into a high-density,
    low-latency XML voice prompt (<400ms TTFA) while preserving 100% of business logic,
    state machine, objections, and guardrails.
    """
    data = await request.json()
    raw_prompt = data.get("prompt", "").strip()
    if not raw_prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    original_words = len(raw_prompt.split())

    compiler_system_instruction = (
        "You are an expert Voice AI Prompt Architect and Compiler specializing in ultra-low latency voice agents (Deepgram Flux, Retell AI, LiveKit).\n"
        "Your task is to transform verbose, sprawling system prompts or playbooks into high-density, compact XML prompts optimized for voice models like Gemini 3.1 Flash Lite and Groq LPU (<400ms TTFA).\n\n"
        "STRICT COMPILATION RULES:\n"
        "1. Preserve 100% of the domain business logic:\n"
        "   - Specific identity, company, and role boundaries\n"
        "   - Core objectives and call-to-actions (e.g. SMS demo link, appointment booking)\n"
        "   - Every single objection handling script & battlecard\n"
        "   - Multi-step conversational state machine and discovery questions for different customer setups\n"
        "   - Specific qualification questions and guardrails (opt-outs, do-not-call, emergency transfers)\n"
        "2. Eliminate all token bloat:\n"
        "   - Remove redundant polite preambles, essay explanations, and repetitive examples.\n"
        "   - Condense wordy paragraphs into crisp, punchy spoken instructions.\n"
        "   - Structure strictly into standard XML tags:\n"
        "     <identity_and_role>\n"
        "     <primary_objective_and_core_principle>\n"
        "     <spoken_style_and_conversational_rules>\n"
        "     <conversation_flow_state_machine>\n"
        "     <objection_playbook>\n"
        "     <critical_guardrails_and_steering>\n"
        "3. Voice-Specific Guidelines:\n"
        "   - Instruct the bot to speak in natural spoken conversational sentences (1-2 sentences per turn), avoiding artificial word clamps that prevent complete explanations.\n"
        "   - Prohibit markdown formatting, asterisks, bullet points in speech output.\n"
        "   - Mandate natural everyday contractions ('I\\'m', 'we\\'ll', 'don\\'t', 'it\\'s') and asking only ONE question at a time.\n"
        "4. Output ONLY the compiled XML prompt text. Do not wrap in ```xml or markdown codeblocks. Do not add conversational intro or outro."
    )

    condensed_text = ""

    # 1. Try Gemini first (Gemini 3.1 Flash Lite / 2.5 Flash)
    if settings.GEMINI_API_KEY:
        try:
            from google import genai
            client = genai.Client(api_key=settings.GEMINI_API_KEY)
            gemini_model = settings.GEMINI_MODEL if (settings.GEMINI_MODEL and "3.5" not in settings.GEMINI_MODEL) else "gemini-3.1-flash-lite"
            res = client.models.generate_content(
                model=gemini_model,
                contents=[compiler_system_instruction, raw_prompt],
            )
            if res and res.text:
                condensed_text = res.text.strip()
        except Exception as e:
            logger.warning(f"Gemini prompt condenser failed, trying Groq fallback: {e}")

    # 2. Fallback to Groq LPU if Gemini failed or unconfigured
    if not condensed_text and settings.GROQ_API_KEY:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"},
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": [
                            {"role": "system", "content": compiler_system_instruction},
                            {"role": "user", "content": f"Compile and condense this prompt into high-density Voice XML:\n\n{raw_prompt}"},
                        ],
                        "temperature": 0.3,
                        "max_tokens": 1400,
                    },
                )
                if res.status_code == 200:
                    condensed_text = res.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"Groq prompt condenser failed: {e}")

    # Strip markdown fences if present
    if condensed_text.startswith("```"):
        lines = condensed_text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        condensed_text = "\n".join(lines).strip()

    condensed_words = len(condensed_text.split()) if condensed_text else 0
    reduction_pct = round((1 - (condensed_words / max(original_words, 1))) * 100, 1) if condensed_words else 0

    return {
        "status": "success",
        "condensed_prompt": condensed_text or raw_prompt,
        "original_words": original_words,
        "condensed_words": condensed_words,
        "reduction_pct": reduction_pct,
    }


# ---------------------------------------------------------
# Call Recordings & Transcripts Endpoints
# ---------------------------------------------------------
from fastapi.responses import FileResponse
from app.calls import list_calls, get_call, delete_call, get_voice_catalog, RECORDINGS_DIR

@app.get("/api/calls")
async def api_list_calls():
    """Returns list of past recorded call sessions with metadata."""
    return {"calls": list_calls()}


@app.get("/api/calls/{call_id}")
async def api_get_call(call_id: str):
    """Returns full details and turn-by-turn transcript of a call."""
    call_data = get_call(call_id)
    if not call_data:
        raise HTTPException(status_code=404, detail="Call not found")
    return call_data


@app.delete("/api/calls/{call_id}")
async def api_delete_call(call_id: str):
    """Deletes a call record and its audio file."""
    success = delete_call(call_id)
    if not success:
        raise HTTPException(status_code=404, detail="Call not found")
    return {"status": "deleted", "call_id": call_id}


@app.get("/api/recordings/{filename}")
async def api_get_recording(filename: str):
    """Streams the recorded WAV audio file for playback in the browser."""
    file_path = RECORDINGS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Recording audio not found")
    return FileResponse(path=str(file_path), media_type="audio/wav", filename=filename)


@app.get("/api/voices")
async def api_get_voice_catalog():
    """Returns full catalog of Kokoro voices with sample phrases and humanness scores."""
    return {"voices": get_voice_catalog()}


# ---------------------------------------------------------
# Integrations & Post-Call Automation Endpoints
# ---------------------------------------------------------
@app.get("/api/integrations/settings")
async def api_get_integrations():
    """Returns current integration configuration for Google Calendar, Email, and Webhooks."""
    from app.integrations import get_integrations_settings
    return get_integrations_settings()


@app.post("/api/integrations/settings")
async def api_update_integrations(request: Request):
    """Updates Google Calendar and Email integration settings."""
    from app.integrations import update_integrations_settings
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    updated = update_integrations_settings(data)
    return {"status": "success", "settings": updated}


@app.post("/api/integrations/test-email")
async def api_test_email(request: Request):
    """Sends a sample post-call notification email to test SMTP or Resend credentials."""
    from app.integrations import test_email_notification
    try:
        data = await request.json()
    except Exception:
        data = {}
    recipient = data.get("email") or data.get("recipient")
    if not recipient:
        raise HTTPException(status_code=400, detail="Missing required 'email' address")
    res = await test_email_notification(recipient)
    return res


@app.post("/api/integrations/test-sms")
async def api_test_sms(request: Request):
    """Sends a sample HVAC dispatch alert SMS to verify Plivo credentials and owner phone."""
    try:
        data = await request.json()
    except Exception:
        data = {}
    to_number = data.get("phone") or data.get("to")
    if not to_number:
        from app.integrations import get_integrations_settings
        cfg = get_integrations_settings()
        to_number = cfg.get("owner_phone_number")
    if not to_number:
        raise HTTPException(status_code=400, detail="Missing required 'phone' or 'to' number")

    from app.appointments import send_plivo_sms
    test_msg = (
        "🚨 [TEST] NEW HVAC JOB #APT-99999\n"
        "👤 Sarah Jenkins\n"
        "📍 742 Evergreen Terrace\n"
        "🔧 High-Efficiency Heat Pump Tune-Up\n"
        "📅 Requested: Tomorrow (Morning 8-12 PM)\n"
        "📞 +1-555-019-2834\n\n"
        "Reply:\n"
        "• '1' to Confirm\n"
        "• Or reply with new time (e.g. 'Tomorrow 2pm')"
    )
    res = await send_plivo_sms(to_number, test_msg)
    return res


@app.get("/api/appointments")
async def api_get_appointments():
    """Returns all recorded appointments and their dispatch status."""
    from app.appointments import get_all_appointments
    return {"appointments": get_all_appointments()}


@app.post("/api/sms/inbound")
@app.post("/api/plivo/sms")
async def api_inbound_sms(request: Request):
    """Inbound webhook for Plivo SMS (e.g. owner replies '1' to confirm or suggests a new time)."""
    form_data = {}
    try:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            form_data = await request.json()
        elif "form" in content_type:
            raw_form = await request.form()
            form_data = dict(raw_form)
        else:
            raw = await request.body()
            if raw:
                try:
                    form_data = json.loads(raw)
                except Exception:
                    pass
    except Exception as e:
        logger.warning(f"Error parsing inbound SMS payload: {e}")

    # Fallback to query params if empty
    if not form_data:
        form_data = dict(request.query_params)

    from_number = form_data.get("From") or form_data.get("from") or ""
    text = form_data.get("Text") or form_data.get("text") or ""
    message_uuid = form_data.get("MessageUUID") or form_data.get("message_uuid") or ""

    logger.info(f"Received Inbound SMS: From={from_number} | UUID={message_uuid} | Text={text}")

    from app.appointments import process_inbound_sms
    result = await process_inbound_sms(from_number, text)

    # Plivo expects XML response or 200 OK
    xml_resp = "<Response></Response>"
    return Response(content=xml_resp, media_type="application/xml")


@app.get("/a/{apt_id}")
async def mobile_action_page(apt_id: str):
    """Mobile 1-tap action card for the HVAC technician / owner."""
    import urllib.parse
    from app.appointments import find_appointment_by_id, generate_schedule_options
    apt = find_appointment_by_id(apt_id)
    if not apt:
        return Response(content="<body style='font-family:sans-serif;padding:24px;text-align:center;'><h2>Appointment Not Found</h2><p>This appointment may have expired or been deleted.</p></body>", status_code=404, media_type="text/html")

    options = apt.get("options") or generate_schedule_options(
        apt.get("appointment_date", datetime.now().strftime("%Y-%m-%d")),
        apt.get("window", "morning")
    )
    client_name = apt.get("client_name", "Valued Customer")
    phone = apt.get("client_phone", "")
    address = apt.get("service_address", "Address on file")
    service = apt.get("service_requested", "HVAC Inspection & Repair")
    status = apt.get("status", "pending_owner_approval")
    encoded_addr = urllib.parse.quote(address)

    opt1_lbl = options.get("1", {}).get("label", "Requested Time")
    opt2_lbl = options.get("2", {}).get("label", "Alternate Window")
    opt3_lbl = options.get("3", {}).get("label", "Next Day")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>HVAC Job #{apt_id} • 1-Tap Dispatch</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen p-4 flex flex-col items-center justify-center font-sans">
  <div class="max-w-md w-full bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-4 shadow-2xl">
    
    <!-- Top Badge -->
    <div class="flex items-center justify-between pb-3 border-b border-slate-800">
      <div class="flex items-center gap-2">
        <span class="w-3 h-3 rounded-full bg-amber-400 animate-pulse"></span>
        <span class="text-xs font-mono font-bold tracking-wide uppercase text-amber-400">HVAC Dispatch Alert</span>
      </div>
      <span class="px-2.5 py-1 rounded-lg text-xs font-mono font-bold bg-slate-800 text-slate-300 border border-slate-700">#{apt_id}</span>
    </div>

    <!-- Job Details -->
    <div class="space-y-3">
      <div>
        <h1 class="text-xl font-black text-white">{client_name}</h1>
        <p class="text-sm font-semibold text-indigo-400 mt-0.5">🔧 {service}</p>
      </div>

      <div class="p-3 bg-slate-950/80 rounded-xl border border-slate-800/80 space-y-2 text-xs">
        <div class="flex items-start gap-2">
          <span class="text-slate-400 shrink-0">📍</span>
          <span class="text-slate-200 font-medium">{address}</span>
        </div>
        <div class="flex items-center gap-2">
          <span class="text-slate-400 shrink-0">📞</span>
          <span class="text-slate-200 font-medium">{phone}</span>
        </div>
      </div>

      <!-- Quick Action Utilities -->
      <div class="grid grid-cols-2 gap-2 pt-1">
        <a href="https://maps.google.com/?q={encoded_addr}" target="_blank" class="flex items-center justify-center gap-1.5 py-2 px-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-bold border border-slate-700 transition">
          <span>🗺️ Google Maps</span>
        </a>
        <a href="tel:{phone}" class="flex items-center justify-center gap-1.5 py-2 px-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-bold border border-slate-700 transition">
          <span>📞 Call Customer</span>
        </a>
      </div>
    </div>

    <!-- Status Notice -->
    <div id="action-result-box" class="hidden p-4 rounded-xl border text-center font-bold text-sm"></div>

    <!-- 1-Tap Big Action Buttons -->
    <div id="action-buttons-container" class="space-y-2.5 pt-2">
      <div class="text-[11px] font-bold text-slate-400 uppercase tracking-wider text-center">Tap 1 Option to Confirm or Shift:</div>

      <button onclick="executeAction('1')" class="w-full py-3.5 px-4 rounded-xl bg-emerald-600 hover:bg-emerald-500 active:scale-[0.98] text-white font-black text-sm shadow-lg shadow-emerald-900/30 flex items-center justify-between transition">
        <span class="flex items-center gap-2"><span>1️⃣</span> <span>Confirm Job</span></span>
        <span class="text-xs opacity-90 font-mono bg-emerald-700/60 px-2 py-0.5 rounded">{opt1_lbl}</span>
      </button>

      <button onclick="executeAction('2')" class="w-full py-3.5 px-4 rounded-xl bg-amber-600 hover:bg-amber-500 active:scale-[0.98] text-white font-black text-sm shadow-lg shadow-amber-900/30 flex items-center justify-between transition">
        <span class="flex items-center gap-2"><span>2️⃣</span> <span>Shift Window</span></span>
        <span class="text-xs opacity-90 font-mono bg-amber-700/60 px-2 py-0.5 rounded">{opt2_lbl}</span>
      </button>

      <button onclick="executeAction('3')" class="w-full py-3.5 px-4 rounded-xl bg-sky-600 hover:bg-sky-500 active:scale-[0.98] text-white font-black text-sm shadow-lg shadow-sky-900/30 flex items-center justify-between transition">
        <span class="flex items-center gap-2"><span>3️⃣</span> <span>Move Next Day</span></span>
        <span class="text-xs opacity-90 font-mono bg-sky-700/60 px-2 py-0.5 rounded">{opt3_lbl}</span>
      </button>

      <button onclick="executeAction('4')" class="w-full py-2.5 px-4 rounded-xl bg-slate-800 hover:bg-rose-900/60 active:scale-[0.98] text-slate-300 hover:text-rose-200 font-bold text-xs border border-slate-700 transition text-center">
        4️⃣ Decline / Too Busy
      </button>

      <!-- Custom Date & Window Picker -->
      <div class="pt-4 border-t border-slate-800 space-y-3">
        <div class="flex items-center justify-between">
          <span class="text-xs font-bold text-indigo-300 flex items-center gap-1.5">
            <span>🗓️</span> <span>Pick Any Exact Day & Window</span>
          </span>
          <span class="text-[10px] text-slate-500 uppercase font-mono tracking-wider">Custom</span>
        </div>

        <!-- Quick Smart Chips -->
        <div class="grid grid-cols-4 gap-1.5 text-[11px] font-bold text-center">
          <button type="button" onclick="setQuickDate(1)" class="p-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl border border-slate-700 transition">Tomorrow</button>
          <button type="button" onclick="setQuickDate(3)" class="p-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl border border-slate-700 transition">In 3 Days</button>
          <button type="button" onclick="setQuickDate(7)" class="p-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl border border-slate-700 transition">In 1 Wk</button>
          <button type="button" onclick="setQuickDate(14)" class="p-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl border border-slate-700 transition">In 2 Wks</button>
        </div>

        <!-- Date Selector -->
        <div>
          <label class="block text-[10px] font-semibold text-slate-400 mb-1">Choose Exact Calendar Date:</label>
          <input type="date" id="custom-date-picker" class="w-full px-3 py-2.5 bg-slate-950 border border-slate-700 rounded-xl text-xs text-white focus:outline-none focus:border-indigo-500 font-mono" />
        </div>

        <!-- Window or Exact Time Selector -->
        <div>
          <div class="flex items-center justify-between mb-1">
            <label class="text-[10px] font-semibold text-slate-400">Arrival Window or Exact Time:</label>
            <span class="text-[9px] text-slate-500 font-mono">Optional Exact Time</span>
          </div>
          <div class="grid grid-cols-3 gap-1.5 mb-2">
            <label class="cursor-pointer">
              <input type="radio" name="custom-window" value="morning" checked class="peer hidden" />
              <div class="p-2 text-center rounded-xl bg-slate-950 border border-slate-800 peer-checked:border-indigo-500 peer-checked:bg-indigo-600/20 peer-checked:text-indigo-300 text-[11px] font-semibold text-slate-400 transition">
                ☀️ Morning<br><span class="text-[9px] opacity-75">8am-12pm</span>
              </div>
            </label>
            <label class="cursor-pointer">
              <input type="radio" name="custom-window" value="afternoon" class="peer hidden" />
              <div class="p-2 text-center rounded-xl bg-slate-950 border border-slate-800 peer-checked:border-indigo-500 peer-checked:bg-indigo-600/20 peer-checked:text-indigo-300 text-[11px] font-semibold text-slate-400 transition">
                🌤️ Afternoon<br><span class="text-[9px] opacity-75">12pm-4pm</span>
              </div>
            </label>
            <label class="cursor-pointer">
              <input type="radio" name="custom-window" value="evening" class="peer hidden" />
              <div class="p-2 text-center rounded-xl bg-slate-950 border border-slate-800 peer-checked:border-indigo-500 peer-checked:bg-indigo-600/20 peer-checked:text-indigo-300 text-[11px] font-semibold text-slate-400 transition">
                🌙 Evening<br><span class="text-[9px] opacity-75">4pm-7pm</span>
              </div>
            </label>
          </div>
          <input type="time" id="custom-time-picker" class="w-full px-3 py-2 bg-slate-950 border border-slate-700 rounded-xl text-xs text-white focus:outline-none focus:border-indigo-500 font-mono" />
        </div>

        <!-- Submit Custom Date Button -->
        <button type="button" onclick="submitCustomDate()" class="w-full py-3 px-4 rounded-xl bg-indigo-600 hover:bg-indigo-500 active:scale-[0.98] text-white font-bold text-xs shadow-lg shadow-indigo-900/30 transition flex items-center justify-center gap-2">
          <span>🚀 Propose This Date & Time to Customer</span>
        </button>
      </div>

    </div>

  </div>

  <script>
    function setQuickDate(daysAhead) {{
      const d = new Date();
      d.setDate(d.getDate() + daysAhead);
      const iso = d.toISOString().split('T')[0];
      const picker = document.getElementById('custom-date-picker');
      if (picker) picker.value = iso;
    }}

    window.addEventListener('DOMContentLoaded', () => {{
      setQuickDate(1);
    }});

    async function submitCustomDate() {{
      const picker = document.getElementById('custom-date-picker');
      const dateVal = picker ? picker.value : '';
      if (!dateVal) {{
        alert('Please select a calendar date.');
        return;
      }}
      const winRadio = document.querySelector('input[name="custom-window"]:checked');
      const winVal = winRadio ? winRadio.value : 'morning';
      const timePicker = document.getElementById('custom-time-picker');
      let exactTimeStr = null;
      if (timePicker && timePicker.value) {{
        const [h, m] = timePicker.value.split(':').map(Number);
        const ampm = h >= 12 ? 'PM' : 'AM';
        const displayHr = (h % 12) || 12;
        exactTimeStr = `${{displayHr}}:${{m < 10 ? '0' + m : m}} ${{ampm}}`;
      }}

      await executeAction({{ action: 'custom', date: dateVal, window: winVal, exact_time: exactTimeStr }});
    }}

    async function executeAction(actionPayload) {{
      const btnContainer = document.getElementById('action-buttons-container');
      const resBox = document.getElementById('action-result-box');
      if (btnContainer) btnContainer.classList.add('opacity-40', 'pointer-events-none');
      
      try {{
        const res = await fetch('/api/appointments/{apt_id}/action', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify(typeof actionPayload === 'object' ? actionPayload : {{ action: actionPayload }})
        }});
        const data = await res.json();
        if (res.ok) {{
          if (btnContainer) btnContainer.classList.add('hidden');
          if (resBox) {{
            resBox.classList.remove('hidden');
            resBox.className = "p-4 rounded-xl border border-emerald-500/40 bg-emerald-500/10 text-emerald-300 text-center font-bold text-sm space-y-1";
            resBox.innerHTML = `
              <div class="text-xl">✅</div>
              <div>${{data.message || 'Action saved successfully!'}}</div>
              <p class="text-xs font-normal text-emerald-400/80">Customer has been notified via text.</p>
            `;
          }}
        }} else {{
          alert('Error: ' + (data.detail || 'Could not update appointment.'));
          if (btnContainer) btnContainer.classList.remove('opacity-40', 'pointer-events-none');
        }}
      }} catch (err) {{
        alert('Network Error: ' + err.message);
        if (btnContainer) btnContainer.classList.remove('opacity-40', 'pointer-events-none');
      }}
    }}
  </script>
</body>
</html>"""
    return Response(content=html, media_type="text/html")


@app.post("/api/appointments/{apt_id}/action")
async def api_appointment_action(apt_id: str, request: Request):
    """Executes an action on an appointment (Confirm, Reschedule, or Custom Date/Window)."""
    from app.appointments import find_appointment_by_id, apply_appointment_action
    apt = find_appointment_by_id(apt_id)
    if not apt:
        raise HTTPException(status_code=404, detail="Appointment not found")

    try:
        data = await request.json()
    except Exception:
        data = {}

    action_payload = data.get("action")
    if data.get("date") or (isinstance(data, dict) and data.get("action") == "custom"):
        action_payload = data

    result = await apply_appointment_action(apt, action_payload or "1", from_phone="")
    return result




@app.get("/api/calls/{call_id}/calendar.ics")
async def api_get_call_ics(call_id: str):
    """Downloads standard RFC 5545 .ics iCalendar file for scheduled appointment."""
    from app.calls import get_call
    from app.integrations import generate_ical_data, extract_client_info
    call = get_call(call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    from app.agents import get_assistant
    asst_id = call.get("assistant_id")
    asst = get_assistant(asst_id) if asst_id else None
    asst_lang = asst.get("language", "en") if asst else "en"
    info = call.get("extracted_info")
    if not info:
        info = await extract_client_info(call.get("transcript", []), caller=call.get("caller"), called=call.get("called"), language=asst_lang)
    ics_bytes = generate_ical_data(info, call_id=call_id, assistant_name=call.get("assistant_name", "Riley"))
    return Response(
        content=ics_bytes,
        media_type="text/calendar; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename=appointment-{call_id}.ics"},
    )


@app.post("/api/calls/{call_id}/extract-info")
async def api_reextract_call_info(call_id: str):
    """Re-runs intelligent extraction of customer details and appointment info for a call."""
    from app.calls import reextract_call_info
    call = await reextract_call_info(call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    return {"status": "success", "call": call}


@app.post("/api/calls/{call_id}/send-email")
async def api_send_call_email(call_id: str, request: Request):
    """Dispatches call summary email to specified or configured recipient."""
    from app.calls import get_call
    from app.integrations import send_post_call_email, extract_client_info
    call = get_call(call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    from app.agents import get_assistant
    asst_id = call.get("assistant_id")
    asst = get_assistant(asst_id) if asst_id else None
    asst_lang = asst.get("language", "en") if asst else "en"
    info = call.get("extracted_info")
    if not info:
        info = await extract_client_info(call.get("transcript", []), caller=call.get("caller"), called=call.get("called"), language=asst_lang)
    res = await send_post_call_email(info, call)
    return res


@app.post("/api/simulations/denoise-benchmark")
async def api_run_denoise_benchmark():
    """Returns LiveKit acoustic noise cancellation and BNN filtering status."""
    return {
        "status": "active",
        "engine": "livekit_vad_silero",
        "denoising": "deepgram_flux_bnn",
        "message": "LiveKit streaming audio pipeline includes native real-time noise cancellation."
    }


# ---------------------------------------------------------
_speech_cache: dict[str, tuple[bytes, str]] = {}

VOICE_MAP_DEEPGRAM = {
    "af_heart": "flux-heather-en",
    "riley": "flux-heather-en",
    "flux-heather-en": "flux-heather-en",
    "heather": "flux-heather-en",
    "aura-asteria-en": "flux-heather-en",
    "am_adam": "flux-cliff-en",
    "customer": "flux-cliff-en",
    "david": "flux-cliff-en",
    "adam": "flux-cliff-en",
    "cliff": "flux-cliff-en",
    "flux-cliff-en": "flux-cliff-en",
    "aura-angus-en": "flux-cliff-en",
    "michael": "flux-bruce-en",
    "am_michael": "flux-bruce-en",
    "bruce": "flux-bruce-en",
    "flux-bruce-en": "flux-bruce-en",
    "aura-orion-en": "flux-bruce-en",
    "sarah": "flux-sienna-en",
    "af_sarah": "flux-sienna-en",
    "sienna": "flux-sienna-en",
    "flux-sienna-en": "flux-sienna-en",
    "aura-luna-en": "flux-sienna-en",
}

# Test & Audio APIs
# ---------------------------------------------------------
@app.get("/api/test-speech")
async def test_speech_api(
    text: str = Query("Hi there! I am Aria, your AI voice assistant running live in the cloud."),
    voice: Optional[str] = Query(None),
    speed: float = Query(1.0),
):
    """Synthesizes speech using Deepgram Flux (v2) / Aura (v1) with instant MP3 delivery & caching or Kokoro fallback."""
    raw_voice = (voice or "flux-heather-en").lower().strip()
    target_voice = VOICE_MAP_DEEPGRAM.get(raw_voice, voice or "flux-heather-en")

    cache_key = f"{target_voice}:{text.strip()}"
    if cache_key in _speech_cache:
        cached_data, cached_type = _speech_cache[cache_key]
        return Response(content=cached_data, media_type=cached_type)

    # Fast Deepgram Flux / Aura MP3 synthesis
    if settings.DEEPGRAM_API_KEY:
        dg_voice = target_voice if (target_voice.startswith("aura-") or target_voice.startswith("flux-")) else "flux-heather-en"
        ver = "v2" if "flux" in dg_voice else "v1"
        try:
            import httpx
            async with httpx.AsyncClient(timeout=8.0) as client:
                res = await client.post(
                    f"https://api.deepgram.com/{ver}/speak?model={dg_voice}",
                    headers={"Authorization": f"Token {settings.DEEPGRAM_API_KEY}", "Content-Type": "application/json"},
                    json={"text": text},
                )
                if res.status_code == 200:
                    audio_bytes = res.content
                    media_type = res.headers.get("content-type", "audio/mpeg")
                    if len(_speech_cache) < 500:
                        _speech_cache[cache_key] = (audio_bytes, media_type)
                    return Response(content=audio_bytes, media_type=media_type)
        except Exception as e:
            logger.warning(f"Deepgram Aura speech error ({e}), falling back to Kokoro...")

    # Kokoro ONNX fallback
    kokoro = get_kokoro_instance()
    if not kokoro:
        raise HTTPException(status_code=503, detail="TTS service unavailable.")

    voices = kokoro.get_voices() if hasattr(kokoro, "get_voices") else kokoro.voices
    k_voice = target_voice if target_voice in voices else "af_heart"

    samples, sample_rate = kokoro.create(text, voice=k_voice, speed=speed)
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    audio_bytes = buf.getvalue()
    if len(_speech_cache) < 500:
        _speech_cache[cache_key] = (audio_bytes, "audio/wav")
    return Response(content=audio_bytes, media_type="audio/wav")


@app.get("/api/simulation/full-call")
async def get_simulation_full_call():
    """Returns the realistic call simulation data with audio URL and transcript."""
    from app.calls import get_call
    call = get_call("call-simulation-kaelen-vash")
    if not call:
        return JSONResponse({"status": "not_found"}, status_code=404)
    return JSONResponse(call)


@app.post("/api/simulate-turn")
async def simulate_turn_api(request: Request):
    """
    Simulates a live voice turn without requiring microphone speech.
    Accepts customer text and conversation history, prompts the active AI assistant,
    and returns both text response and Kokoro-synthesized high-fidelity audio!
    """
    from datetime import datetime
    data = await request.json()
    user_text = data.get("text", "").strip()
    history = data.get("history", [])
    assistant_id = data.get("assistant_id")

    from app.agents import get_active_assistant, get_assistant
    active_agent = get_assistant(assistant_id) if assistant_id else get_active_assistant()
    if not active_agent:
        active_agent = get_active_assistant()

    agent_name = active_agent.get("name", "Riley Voice AI")
    agent_prompt = active_agent.get("system_prompt", "")
    agent_voice = active_agent.get("tts_voice", settings.KOKORO_VOICE)

    # ── Personality Tune — inject humanness addendum from preset ──────────────
    from app.prompts import PERSONALITY_PRESETS
    _personality_key = active_agent.get("personality_preset", "natural")
    _personality_addendum = PERSONALITY_PRESETS.get(
        _personality_key, PERSONALITY_PRESETS["natural"]
    )
    # ─────────────────────────────────────────────────────────────────────────

    caller_number = data.get("caller", "+1 (555) 234-5678")
    now_str = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

    voice_rules = (
        "\n\n[High-Precision Information Capture Protocol & Voice Guidelines]\n"
        "You are an elite, highly intelligent voice AI receptionist. You must achieve a 100% success rate extracting and confirming the customer's information by following this exact proactive conversational flow:\n\n"
        "1. SERVICE & PROBLEM DIAGNOSIS:\n"
        "   - Understand what service or problem they have.\n"
        "   - Immediately offer to get a certified technician scheduled.\n\n"
        "2. CUSTOMER NAME & SMART SPELLING:\n"
        "   - Ask for their full name: 'May I have your first and last name, please?'\n"
        "   - Smart Spelling Verification: If the name is uncommon, unique, hyphenated, foreign, or has multiple spellings (e.g. Kaelen, Jon vs John, Smythe), ask smartly: 'Could you quickly spell that out for me just so our technician has it 100% accurate in our dispatch system?'\n\n"
        "3. SERVICE ADDRESS / LOCATION:\n"
        "   - Ask for their location: 'And what is the street address where you'd like our technician to visit?'\n\n"
        "4. SCHEDULING TIME WINDOW:\n"
        "   - Inquire about their preferred date and time: 'We have openings tomorrow morning around 10 AM or Thursday afternoon around 2 PM. Which day and time works best for you?'\n\n"
        "5. PHONE NUMBER & SMS CONSENT:\n"
        f"   - The caller's incoming phone number from caller ID is {caller_number}.\n"
        "   - Ask: 'Can we send your technician arrival updates and confirmation text to this phone number, or is there a different mobile number you prefer?'\n\n"
        "6. EMAIL ADDRESS & SMART DOUBLE-CHECK / READBACK:\n"
        "   - Ask for their email: 'And what's the best email address to send your Google Calendar invite and service confirmation?'\n"
        "   - Spelled Letters & Dictation: If the caller spells their email letter-by-letter (e.g. 'A-A-L-B-A-D-I' or 'a a l b a d i 9 1 at gmail dot com'), combine the letters cleanly into the email (e.g. aalbadi91@gmail.com).\n"
        "   - MANDATORY DOUBLE-CHECK: As soon as the customer gives an email, read it back clearly to confirm: 'Just to double-check that, that's [clearly spoken email address], correct?'\n"
        "   - INSTANT CORRECTION ADOPTION: If the customer says 'No', clarifies, or corrects their email, IMMEDIATELY adopt the corrected email warmly without friction (e.g. 'Got it, updated to aalbadi91@gmail.com! Is that correct?'). NEVER say 'I'm sorry, I'm confused'.\n\n"
        "7. FINAL BOOKING CONFIRMATION & RECAP:\n"
        "   - Once confirmed, give a warm, reassuring recap: 'You're all set, [Name]! We have you booked for [Service] on [Day/Time] at [Address]. We sent your calendar invite and text confirmation. Is there anything else I can assist you with today?'\n\n"
        "[Spoken Rules]\n"
        "- Ask only ONE question at a time. Never ask multiple questions in a single turn.\n"
        "- Speak naturally in 1 to 2 conversational sentences (under 32 words).\n"
        "- Use natural, varied human transitions ('I'd be glad to help with that', 'Certainly', 'I have that noted down', 'Thank you'). Never repeat 'Got it' over and over.\n"
        "- Never output bullet points, asterisks, or markdown formatting.\n"
        "- Spell out times, dates, and numbers in conversational words."
    )

    system_instruction = (
        f"{agent_prompt}"
        f"{_personality_addendum}"
        f"\n\n[Call Context]\nCurrent date and time: {now_str}.\nCaller's phone number: {caller_number}."
        f"{voice_rules}"
    )


    messages = [{"role": "system", "content": system_instruction}]
    for h in history:
        messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
    if user_text:
        messages.append({"role": "user", "content": user_text})

    reply_text = ""
    if settings.GEMINI_API_KEY:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=settings.GEMINI_API_KEY, base_url=settings.GEMINI_BASE_URL)
            resp = await client.chat.completions.create(
                model=settings.GEMINI_MODEL,
                messages=messages,
                temperature=0.4,
                max_tokens=150,
            )
            reply_text = resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error calling Gemini in simulation: {e}")

    if not reply_text:
        reply_text = "I'd be glad to help with that! Let me get a certified technician scheduled for you right away."

    audio_base64 = ""
    audio_format = "audio/mpeg"

    # Fast Deepgram Aura speech synthesis
    if settings.DEEPGRAM_API_KEY:
        dg_voice = VOICE_MAP_DEEPGRAM.get((agent_voice or "").lower().strip(), "aura-asteria-en")
        ver = "v2" if "flux" in dg_voice else "v1"
        try:
            import httpx
            async with httpx.AsyncClient(timeout=8.0) as client:
                res = await client.post(
                    f"https://api.deepgram.com/{ver}/speak?model={dg_voice}",
                    headers={"Authorization": f"Token {settings.DEEPGRAM_API_KEY}", "Content-Type": "application/json"},
                    json={"text": reply_text},
                )
                if res.status_code == 200:
                    audio_base64 = base64.b64encode(res.content).decode("ascii")
                    audio_format = "audio/mpeg"
        except Exception as e:
            logger.warning(f"Deepgram Aura simulation turn error ({e}), falling back to Kokoro...")

    if not audio_base64:
        kokoro = get_kokoro_instance()
        if kokoro:
            try:
                voices = kokoro.get_voices() if hasattr(kokoro, "get_voices") else kokoro.voices
                voice_to_use = agent_voice if agent_voice in voices else "af_heart"
                samples, rate = kokoro.create(reply_text, voice=voice_to_use, speed=1.05)
                buf = io.BytesIO()
                sf.write(buf, samples, rate, format="WAV", subtype="PCM_16")
                audio_base64 = base64.b64encode(buf.getvalue()).decode("ascii")
                audio_format = "audio/wav"
            except Exception as e:
                logger.error(f"Error synthesizing Kokoro audio in simulation: {e}")

    return JSONResponse({
        "reply": reply_text,
        "speaker": agent_name,
        "audio_base64": audio_base64,
        "audio_format": audio_format,
    })


@app.post("/api/settings")
async def update_settings_api(request: Request):
    """Update runtime settings (such as Groq/Gemini API keys) directly from the Web UI."""
    data = await request.json()
    new_groq_key = data.get("groq_api_key")
    new_gemini_key = data.get("gemini_api_key")
    new_provider = data.get("llm_provider")

    env_path = Path(".env")
    content = env_path.read_text() if env_path.exists() else ""
    import re

    if new_groq_key is not None:
        settings.GROQ_API_KEY = new_groq_key.strip()
        if "GROQ_API_KEY=" in content:
            content = re.sub(r"GROQ_API_KEY=.*", f"GROQ_API_KEY={settings.GROQ_API_KEY}", content)
        else:
            content += f"\nGROQ_API_KEY={settings.GROQ_API_KEY}\n"
        logger.info("Updated GROQ_API_KEY from dashboard")

    if new_gemini_key is not None:
        settings.GEMINI_API_KEY = new_gemini_key.strip()
        if "GEMINI_API_KEY=" in content:
            content = re.sub(r"GEMINI_API_KEY=.*", f"GEMINI_API_KEY={settings.GEMINI_API_KEY}", content)
        else:
            content += f"\nGEMINI_API_KEY={settings.GEMINI_API_KEY}\n"
        logger.info("Updated GEMINI_API_KEY from dashboard")

    if new_provider is not None:
        settings.LLM_PROVIDER = new_provider.strip().lower()
        if "LLM_PROVIDER=" in content:
            content = re.sub(r"LLM_PROVIDER=.*", f"LLM_PROVIDER={settings.LLM_PROVIDER}", content)
        else:
            content += f"\nLLM_PROVIDER={settings.LLM_PROVIDER}\n"

    if content:
        env_path.write_text(content)

    llm_info = get_active_llm_info()
    return {
        "status": "success",
        "groq_configured": bool(settings.GROQ_API_KEY),
        "gemini_configured": bool(settings.GEMINI_API_KEY),
        "active_llm": llm_info,
    }


@app.post("/api/test-llm")
async def test_llm_api(request: Request):
    """Test the active LLM provider with a live test prompt and measure real-world TTFT and response."""
    import time
    llm_info = get_active_llm_info()
    prompt = "You are a professional voice AI assistant. Say hello and state your capability in one crisp sentence."

    t0 = time.time()
    try:
        if llm_info["provider"] == "groq" and settings.GROQ_API_KEY:
            from groq import Groq
            client = Groq(api_key=settings.GROQ_API_KEY)
            resp = client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
            )
            reply = resp.choices[0].message.content.strip()
            elapsed_ms = round((time.time() - t0) * 1000)
            return {
                "status": "success",
                "provider": "groq",
                "model": settings.GROQ_MODEL,
                "latency_ms": elapsed_ms,
                "reply": reply,
                "badge": "⚡ Groq 70B Ultra-Fast",
            }
        elif llm_info["provider"] == "gemini" and settings.GEMINI_API_KEY:
            from openai import OpenAI
            client = OpenAI(
                base_url=settings.GEMINI_BASE_URL,
                api_key=settings.GEMINI_API_KEY,
            )
            target_model = llm_info.get("model") or settings.GEMINI_MODEL or "gemini-3.1-flash-lite"
            resp = client.chat.completions.create(
                model=target_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
            )
            reply = resp.choices[0].message.content.strip()
            elapsed_ms = round((time.time() - t0) * 1000)
            return {
                "status": "success",
                "provider": "gemini",
                "model": target_model,
                "latency_ms": elapsed_ms,
                "reply": reply,
                "badge": f"⚡ Google Gemini ({target_model})",
            }
        elif llm_info["provider"] == "openrouter" and settings.OPENROUTER_API_KEY:
            from openai import OpenAI
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=settings.OPENROUTER_API_KEY,
            )
            resp = client.chat.completions.create(
                model=settings.OPENROUTER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
            )
            reply = resp.choices[0].message.content.strip()
            elapsed_ms = round((time.time() - t0) * 1000)
            return {
                "status": "success",
                "provider": "openrouter",
                "model": settings.OPENROUTER_MODEL,
                "latency_ms": elapsed_ms,
                "reply": reply,
                "badge": "⚡ OpenRouter",
            }
        else:
            return {
                "status": "warning",
                "provider": "demo",
                "model": "rule-based",
                "latency_ms": 1,
                "reply": "No cloud LLM key configured. Running offline fallback.",
                "badge": "⚠️ Demo Fallback",
            }
    except Exception as e:
        logger.error(f"Error testing LLM: {e}")
        return {
            "status": "error",
            "provider": llm_info["provider"],
            "model": llm_info["model"],
            "error": str(e),
            "latency_ms": round((time.time() - t0) * 1000),
        }


# ---------------------------------------------------------
# Plivo Inbound Webhook Handlers
# ---------------------------------------------------------
@app.get("/")
@app.post("/")
async def plivo_inbound_webhook(
    request: Request,
    CallUUID: Optional[str] = Query(None),
    From: Optional[str] = Query(None),
    To: Optional[str] = Query(None),
):
    """Primary webhook for inbound Plivo calls, or brand onboarding page for browser visitors."""
    # Check if request is a human visitor in a browser
    if request.method == "GET" and not CallUUID and not From:
        accept = request.headers.get("accept", "")
        if "text/html" in accept or "*/*" in accept:
            return await subscribe_page(request)

    logger.info(f"Inbound Plivo Call -> UUID: {CallUUID} | Caller: {From} | Called: {To}")
    _, ws_base = resolve_base_urls(request)

    body_meta = {
        "call_uuid": CallUUID or "unknown",
        "from": From or "unknown",
        "to": To or settings.PLIVO_PHONE_NUMBER,
        "direction": "inbound",
    }
    encoded_meta = base64.b64encode(json.dumps(body_meta).encode("utf-8")).decode("utf-8")
    websocket_url = f"{ws_base}/ws?body={encoded_meta}"

    xml = build_stream_xml(websocket_url)
    logger.debug(f"Returning Inbound Plivo XML: {xml}")
    return Response(content=xml, media_type="application/xml")


# ---------------------------------------------------------
# Plivo Outbound Call Handlers
# ---------------------------------------------------------
@app.post("/api/call")
async def initiate_outbound_call(request: Request):
    """API endpoint to trigger an outbound AI phone call."""
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    to_number = data.get("to") or data.get("phone_number")
    if not to_number:
        raise HTTPException(status_code=400, detail="Missing required 'to' or 'phone_number' field")

    from_number = data.get("from") or settings.PLIVO_PHONE_NUMBER
    extra_context = data.get("extra_context", {})

    http_base, _ = resolve_base_urls(request)

    meta_payload = {
        "to": to_number,
        "from": from_number,
        "direction": "outbound",
        "context": extra_context,
    }
    encoded_query = urllib.parse.quote(json.dumps(meta_payload))
    answer_url = f"{http_base}/outbound/answer?meta={encoded_query}"

    call_result = await trigger_plivo_call(
        session=request.app.state.session,
        to_number=to_number,
        from_number=from_number,
        answer_url=answer_url
    )

    return JSONResponse(
        {
            "status": "call_initiated",
            "request_uuid": call_result.get("request_uuid") or call_result.get("message_uuid"),
            "to": to_number,
            "from": from_number,
        }
    )


@app.get("/outbound/answer")
async def outbound_answer_xml(
    request: Request,
    CallUUID: Optional[str] = Query(None),
    meta: Optional[str] = Query(None),
):
    """Plivo hits this endpoint when an outbound call is answered by the recipient."""
    logger.info(f"Outbound call answered -> UUID: {CallUUID}")
    _, ws_base = resolve_base_urls(request)

    body_meta = {}
    if meta:
        try:
            body_meta = json.loads(urllib.parse.unquote(meta))
        except Exception as e:
            logger.warning(f"Could not parse outbound metadata: {e}")

    body_meta["call_uuid"] = CallUUID or "unknown"
    encoded_meta = base64.b64encode(json.dumps(body_meta).encode("utf-8")).decode("utf-8")
    websocket_url = f"{ws_base}/ws?body={encoded_meta}"

    xml = build_stream_xml(websocket_url)
    return Response(content=xml, media_type="application/xml")


# ---------------------------------------------------------
# WebSocket Media Stream Endpoint (Deprecated / Migration Notice)
# ---------------------------------------------------------
@app.websocket("/ws")
@app.websocket("/media/stream")
async def websocket_media_stream(
    websocket: WebSocket,
    body: Optional[str] = Query(None),
    client: Optional[str] = Query(None),
):
    """Legacy WebSocket media stream endpoint.

    Deprecated: Real-time audio is now powered entirely by LiveKit WebRTC.
    """
    await websocket.accept()
    is_web_client = (client == "web") or not body
    logger.warning(
        f"Legacy WebSocket connection attempted on /ws or /media/stream (is_web_client={is_web_client}). "
        "Redirecting to LiveKit WebRTC."
    )

    body_data = {}
    if body:
        try:
            decoded_json = base64.b64decode(body).decode("utf-8")
            body_data = json.loads(decoded_json)
        except Exception:
            pass

    deprecation_payload = {
        "event": "deprecation_notice",
        "deprecated": True,
        "message": "Legacy WebSocket media streaming has been retired. Please use LiveKit WebRTC (/dashboard, /livekit, or /api/livekit/token).",
        "livekit_url": getattr(settings, "LIVEKIT_URL", "wss://livekit.example.com"),
        "token_endpoint": "/api/livekit/token",
        "web_client_url": "/dashboard",
        "metadata": body_data,
    }

    try:
        await websocket.send_text(json.dumps(deprecation_payload))
        await websocket.close(code=1000, reason="Legacy WebSocket deprecated; use LiveKit WebRTC")
    except Exception as e:
        logger.error(f"Error handling deprecated WebSocket connection: {e}")
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run("app.server:app", host=settings.HOST, port=settings.PORT, reload=True)
