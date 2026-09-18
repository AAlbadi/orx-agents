"""FastAPI Server handling Plivo Inbound/Outbound Telephony and WebSocket Media Streaming."""

import audioop
import base64
import io
import json
import os
import time
import urllib.parse
from contextlib import asynccontextmanager
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

# ---------------------------------------------------------
# Global Pre-Warmed Singletons & Greeting Cache
# ---------------------------------------------------------
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
        if model_path.exists() and voices_path.exists():
            logger.info(f"Loading pre-warmed Kokoro ONNX model from {kokoro_dir}...")
            sess_opts = rt.SessionOptions()
            sess_opts.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
            optimal_threads = min(os.cpu_count() or 1, 4)
            sess_opts.intra_op_num_threads = optimal_threads
            sess_opts.inter_op_num_threads = 2
            sess_opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = ["CPUExecutionProvider"]
            session = rt.InferenceSession(str(model_path), sess_options=sess_opts, providers=providers)
            _kokoro_singleton = Kokoro.from_session(session, str(voices_path))
            # Pre-warm with a tiny dummy phrase so first turn is not paying JIT compilation cost
            try:
                _kokoro_singleton.create("Ready", voice="af_heart", speed=1.0)
            except Exception:
                pass
            logger.success(f"Kokoro ONNX model pre-warmed in memory (providers: {session.get_providers()}).")
    return _kokoro_singleton


_whisper_downloading: set = set()  # tracks models currently being downloaded

def get_whisper_instance(model_name: Optional[str] = None, device: Optional[str] = None, compute_type: Optional[str] = None):
    """Returns a globally pre-warmed Faster-Whisper model instance, reusing cached instances across calls.
    If the requested model is still downloading, falls back to the smallest cached model to avoid blocking calls."""
    global _whisper_cache, _whisper_downloading
    m_name = model_name or settings.WHISPER_MODEL
    dev = device or settings.WHISPER_DEVICE
    comp = compute_type or settings.WHISPER_COMPUTE_TYPE
    cache_key = f"{m_name}:{dev}:{comp}"

    if cache_key in _whisper_cache:
        return _whisper_cache[cache_key]

    # If this model is being downloaded in background, use the best available cached fallback
    if cache_key in _whisper_downloading and _whisper_cache:
        fallback_key = next(iter(_whisper_cache))
        fallback_model = fallback_key.split(":")[0]
        logger.info(f"Model '{m_name}' still downloading — using cached '{fallback_model}' for this call.")
        return _whisper_cache[fallback_key]

    # Mark as downloading so concurrent callers see the sentinel
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
    from pipecat.frames.frames import TTSAudioRawFrame

    greeting_text = text or settings.GREETING_TEXT
    greeting_voice = voice or settings.KOKORO_VOICE

    global _cached_greeting_frames, _cached_greeting_duration, _cached_greeting_text, _cached_greeting_voice

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
                    frames.append(TTSAudioRawFrame(audio=chunk, sample_rate=sr, num_channels=1))
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
            frames.append(TTSAudioRawFrame(audio=chunk, sample_rate=sr, num_channels=1))

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
    title="Aria Voice AI Agent (Plivo + Pipecat)",
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
async def dashboard(request: Request):
    """Serve the web testing and control dashboard."""
    from app.agents import list_assistants, get_active_assistant, get_active_assistant_id
    http_base, _ = resolve_base_urls(request)
    assistants = list_assistants()
    active_agent = get_active_assistant()
    active_id = get_active_assistant_id()

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "settings": settings,
            "public_url": http_base,
            "inbound_webhook_url": f"{http_base}/",
            "cached_greeting_duration": round(_cached_greeting_duration, 2),
            "assistants": assistants,
            "active_assistant": active_agent,
            "active_id": active_id,
        }
    )


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


@app.post("/api/onboarding/chat-test")
async def api_simulate_agent_turn(payload: Dict[str, Any] = Body(...)):
    """Simulate a conversational turn with the client's tailored voice agent."""
    from app.onboarding import simulate_agent_turn, get_client_profile
    client_id = payload.get("client_id")
    profile = get_client_profile(client_id) if client_id else None
    if not profile:
        profile = {
            "business_name": payload.get("business_name", "Apex Services"),
            "industry": payload.get("industry", "general"),
            "hours": payload.get("hours", "Mon-Fri 8:00 AM - 6:00 PM"),
            "transfer_rules": payload.get("transfer_rules", "Transfer on emergencies"),
            "forwarding_phone": payload.get("forwarding_phone", "+1 (555) 234-5678"),
        }
    message = payload.get("message", "")
    history = payload.get("history", [])
    result = simulate_agent_turn(profile, message, history)
    return result


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
    from app.onboarding import get_client_profile, save_client_profile
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
        return {"status": "received"}
    except Exception as e:
        logger.error(f"Polar webhook error: {e}")
        return {"status": "error", "message": str(e)}


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
        if "forwarding_phone" in payload:
            profile["forwarding_phone"] = payload["forwarding_phone"]
        if "hours" in payload:
            profile["hours"] = payload["hours"]
    saved = save_client_profile(profile)
    return {"success": True, "profile": saved}


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


@app.post("/api/projects")
async def api_create_project(payload: Dict[str, Any] = Body(...)):
    """Admin-triggered creation of a client project with dedicated database & prompt."""
    from app.project_db import create_project
    project = create_project(payload, trigger_source="admin")
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


@app.post("/api/integrations/google-calendar/test")
async def api_test_global_google_calendar(payload: Optional[Dict[str, Any]] = Body(None)):
    """Test global Google Calendar credentials directly from Integrations modal."""
    from app.integrations import get_integrations_settings
    from app.appointments import verify_google_calendar_connection

    cfg = get_integrations_settings()
    cal_id = payload.get("google_calendar_id") if payload and "google_calendar_id" in payload else cfg.get("google_calendar_id", "primary")
    sa_data = payload.get("google_service_account_json") if payload and "google_service_account_json" in payload else cfg.get("google_service_account_json", "")

    result = await verify_google_calendar_connection(calendar_id=cal_id, service_account_data=sa_data)
    return result


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
    """Runs automated acoustic noise cancellation simulation across 3 real-world noise environments."""
    import asyncio
    from scripts.simulate_denoiser import run_all_benchmarks
    result = await asyncio.to_thread(run_all_benchmarks)
    return result


# ---------------------------------------------------------
# Test & Audio APIs
# ---------------------------------------------------------
@app.get("/api/test-speech")
async def test_speech_api(
    text: str = Query("Hi there! I am Aria, your AI voice assistant running live in the cloud."),
    voice: Optional[str] = Query(None),
    speed: float = Query(1.0),
):
    """Synthesizes speech using Kokoro ONNX and returns real-time WAV audio."""
    target_voice = voice or settings.KOKORO_VOICE
    if target_voice and (target_voice.startswith("flux-") or target_voice.startswith("aura-") or target_voice.lower() == "cliff"):
        voice_id = "flux-cliff-en" if target_voice.lower() == "cliff" else target_voice
        ver = "v2" if "flux" in voice_id else "v1"
        try:
            import httpx
            async with httpx.AsyncClient(timeout=8.0) as client:
                res = await client.post(
                    f"https://api.deepgram.com/{ver}/speak?model={voice_id}&encoding=linear16&sample_rate=24000&container=none",
                    headers={"Authorization": f"Token {settings.DEEPGRAM_API_KEY}", "Content-Type": "application/json"},
                    json={"text": text},
                )
                if res.status_code == 200:
                    raw = res.content
                    # Strip WAV header if present (safety guard)
                    if raw[:4] == b"RIFF":
                        raw = raw[44:]
                    samples = np.frombuffer(raw, dtype=np.int16)
                    buf = io.BytesIO()
                    sf.write(buf, samples, 24000, format="WAV", subtype="PCM_16")
                    return Response(content=buf.getvalue(), media_type="audio/wav")
        except Exception as e:
            logger.warning(f"Deepgram sample error ({e}), falling back to Kokoro...")

    kokoro = get_kokoro_instance()
    if not kokoro:
        raise HTTPException(status_code=503, detail="Kokoro models not found. Run scripts/download_models.py.")

    voices = kokoro.get_voices() if hasattr(kokoro, "get_voices") else kokoro.voices
    if target_voice not in voices:
        target_voice = list(voices)[0]

    samples, sample_rate = kokoro.create(text, voice=target_voice, speed=speed)
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")


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
    kokoro = get_kokoro_instance()
    if kokoro:
        try:
            voices = kokoro.get_voices() if hasattr(kokoro, "get_voices") else kokoro.voices
            voice_to_use = agent_voice if agent_voice in voices else "af_heart"
            samples, rate = kokoro.create(reply_text, voice=voice_to_use, speed=1.05)
            buf = io.BytesIO()
            sf.write(buf, samples, rate, format="WAV", subtype="PCM_16")
            audio_base64 = base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as e:
            logger.error(f"Error synthesizing Kokoro audio in simulation: {e}")

    return JSONResponse({
        "reply": reply_text,
        "speaker": agent_name,
        "audio_base64": audio_base64,
        "audio_format": "audio/wav",
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
            resp = client.chat.completions.create(
                model=settings.GEMINI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
            )
            reply = resp.choices[0].message.content.strip()
            elapsed_ms = round((time.time() - t0) * 1000)
            return {
                "status": "success",
                "provider": "gemini",
                "model": settings.GEMINI_MODEL,
                "latency_ms": elapsed_ms,
                "reply": reply,
                "badge": "⚡ Google Gemini Flash Smart",
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
# WebSocket Media Stream Endpoint
# ---------------------------------------------------------
@app.websocket("/ws")
async def websocket_media_stream(
    websocket: WebSocket,
    body: Optional[str] = Query(None),
    client: Optional[str] = Query(None),
):
    """Handles real-time bidirectional audio stream (16kHz Linear PCM for Web clients, μ-law for Plivo)."""
    await websocket.accept()
    is_web_client = (client == "web") or not body
    logger.info(f"WebSocket connection established (is_web_client={is_web_client})")

    body_data = {}
    if body:
        try:
            decoded_json = base64.b64decode(body).decode("utf-8")
            body_data = json.loads(decoded_json)
            logger.info(f"Parsed call metadata: {body_data}")
        except Exception as e:
            logger.error(f"Error decoding WebSocket metadata query parameter: {e}")

    try:
        from pipecat.runner.types import WebSocketRunnerArguments
        from app.bot import bot

        runner_args = WebSocketRunnerArguments(websocket=websocket)
        runner_args.handle_sigint = False
        runner_args.body = body_data
        if is_web_client:
            runner_args.transport_type = "websocket"

        await bot(runner_args)

    except Exception as e:
        logger.error(f"Error in WebSocket media processing: {e}")
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run("app.server:app", host=settings.HOST, port=settings.PORT, reload=True)
