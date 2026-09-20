#!/usr/bin/env python3
"""
Comprehensive Speed & Latency Benchmark for Aria Voice AI (Web & Cloud).
Benchmarks:
1. HTTP / Web API Latency (Cloudflare Tunnel vs Localhost)
2. Speech Synthesis (Kokoro ONNX TTS) over Cloud Web API
3. Speech-to-Text (Faster-Whisper STT) Processing Latency
4. Cloud WebSocket Telephony Streaming Handshake & TTFA
5. End-to-End Voice Turnaround (Mouth-to-Ear Latency)
"""

import asyncio
import audioop
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import io
import httpx
import numpy as np
import soundfile as sf
import websockets
from loguru import logger

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings

# G.711 μ-law decoding lookup table
MULAW_DECODE_TABLE = np.zeros(256, dtype=np.int16)
for i in range(256):
    byte = ~i
    sign = byte & 0x80
    exponent = (byte >> 4) & 0x07
    mantissa = byte & 0x0F
    sample = (mantissa << 3) + 132
    sample <<= exponent
    sample -= 132
    MULAW_DECODE_TABLE[i] = -sample if sign != 0 else sample


def decode_mulaw(raw_bytes: bytes) -> np.ndarray:
    indices = np.frombuffer(raw_bytes, dtype=np.uint8)
    return MULAW_DECODE_TABLE[indices]


# =========================================================
# 1. HTTP Web Endpoint Latency Benchmark
# =========================================================
async def benchmark_http_endpoints(base_url: str, runs: int = 5) -> Dict[str, Any]:
    logger.info(f"--- Benchmarking HTTP Endpoints on: {base_url} ({runs} runs) ---")
    results = {}
    async with httpx.AsyncClient(timeout=15.0) as client:
        # Warmup
        try:
            await client.get(f"{base_url}/health")
        except Exception:
            pass

        for endpoint in ["/health", "/dashboard"]:
            latencies = []
            status_codes = []
            for _ in range(runs):
                t0 = time.perf_counter()
                r = await client.get(f"{base_url}{endpoint}")
                elapsed = (time.perf_counter() - t0) * 1000.0
                latencies.append(elapsed)
                status_codes.append(r.status_code)
                await asyncio.sleep(0.05)

            results[endpoint] = {
                "min_ms": round(min(latencies), 2),
                "avg_ms": round(sum(latencies) / len(latencies), 2),
                "max_ms": round(max(latencies), 2),
                "status": status_codes[-1],
            }
            logger.info(
                f"  {endpoint:<12} -> Avg: {results[endpoint]['avg_ms']}ms "
                f"(Min: {results[endpoint]['min_ms']}ms, Max: {results[endpoint]['max_ms']}ms)"
            )
    return results


# =========================================================
# 2. Cloud TTS Speed Benchmark (/api/test-speech)
# =========================================================
async def benchmark_cloud_tts(base_url: str) -> List[Dict[str, Any]]:
    logger.info(f"--- Benchmarking Kokoro TTS over Web API: {base_url}/api/test-speech ---")
    test_cases = [
        ("Short (7 words)", "Hello! How can I help you today?"),
        (
            "Medium (14 words)",
            "I would be happy to book your appointment for this Thursday at two o'clock.",
        ),
        (
            "Long (30 words)",
            "Thank you for calling customer support. I have retrieved your account and everything looks good. I can guide you through resetting your password or updating your preferences right now.",
        ),
    ]

    results = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Warmup
        try:
            await client.get(f"{base_url}/api/test-speech", params={"text": "Warmup test."})
        except Exception:
            pass

        for label, text in test_cases:
            t0 = time.perf_counter()
            r = await client.get(f"{base_url}/api/test-speech", params={"text": text})
            total_elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if r.status_code == 200:
                audio_bytes = r.content
                wav_io = io.BytesIO(audio_bytes)
                audio_arr, sample_rate = sf.read(wav_io)
                duration_sec = len(audio_arr) / sample_rate
                # Real-Time Factor (RTF) = Generation Time / Audio Duration
                # An RTF < 1.0 means generation is faster than real-time speech!
                rtf = (total_elapsed_ms / 1000.0) / duration_sec
                speedup = 1.0 / rtf if rtf > 0 else 0

                item = {
                    "label": label,
                    "word_count": len(text.split()),
                    "latency_ms": round(total_elapsed_ms, 2),
                    "audio_duration_sec": round(duration_sec, 2),
                    "audio_size_bytes": len(audio_bytes),
                    "rtf": round(rtf, 3),
                    "speed_multiplier": round(speedup, 1),
                }
                results.append(item)
                logger.info(
                    f"  {label:<18} -> Latency: {item['latency_ms']}ms | "
                    f"Audio: {item['audio_duration_sec']}s | RTF: {item['rtf']} ({item['speed_multiplier']}x Real-time)"
                )
            else:
                logger.error(f"  {label} failed with status {r.status_code}")
    return results


# =========================================================
# 3. Faster-Whisper STT Benchmark
# =========================================================
def benchmark_whisper_stt() -> Dict[str, Any]:
    logger.info("--- Benchmarking Faster-Whisper STT Engine ---")
    from faster_whisper import WhisperModel

    model_name = settings.WHISPER_MODEL
    device = settings.WHISPER_DEVICE
    compute_type = settings.WHISPER_COMPUTE_TYPE

    t0 = time.perf_counter()
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    load_time_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(f"  Whisper ({model_name}, {compute_type}) model ready in {load_time_ms:.1f}ms")

    # Generate a synthetic 3-second 16kHz sine / speech wave for precise benchmark
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float32)
    synthetic_audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Warmup run
    list(model.transcribe(synthetic_audio, language="en")[0])

    # Benchmark run
    runs = 3
    times = []
    for _ in range(runs):
        t_start = time.perf_counter()
        segments, _ = model.transcribe(synthetic_audio, language="en")
        _ = list(segments)
        times.append((time.perf_counter() - t_start) * 1000.0)

    avg_transcribe_ms = sum(times) / len(times)
    stt_rtf = (avg_transcribe_ms / 1000.0) / duration
    speedup = 1.0 / stt_rtf if stt_rtf > 0 else 0

    res = {
        "model": model_name,
        "compute_type": compute_type,
        "sample_duration_sec": duration,
        "transcription_ms": round(avg_transcribe_ms, 2),
        "rtf": round(stt_rtf, 3),
        "speed_multiplier": round(speedup, 1),
    }
    logger.info(
        f"  STT on {duration}s audio -> Transcribed in {res['transcription_ms']}ms | "
        f"RTF: {res['rtf']} ({res['speed_multiplier']}x Real-time)"
    )
    return res


# =========================================================
# 4. Cloud WebSocket Handshake & Greeting TTFA Benchmark
# =========================================================
async def benchmark_websocket_stream(ws_url: str) -> Dict[str, Any]:
    logger.info(f"--- Benchmarking WebSocket Telephony Pipeline: {ws_url} ---")
    call_uuid = f"bench-call-{int(time.time())}"
    stream_id = f"bench-stream-{int(time.time())}"

    meta = {
        "call_uuid": call_uuid,
        "from": "+14155550199",
        "to": "+14155550100",
        "direction": "inbound",
    }
    encoded_meta = base64.b64encode(json.dumps(meta).encode("utf-8")).decode("utf-8")
    if "body=" not in ws_url:
        delimiter = "&" if "?" in ws_url else "?"
        ws_url = f"{ws_url}{delimiter}body={encoded_meta}"

    silence_chunk = base64.b64encode(b"\xff" * 160).decode("utf-8")

    t_connect_start = time.perf_counter()
    async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as ws:
        handshake_ms = (time.perf_counter() - t_connect_start) * 1000.0
        logger.info(f"  WSS Handshake established in {handshake_ms:.2f}ms")

        # 1. Send Plivo stream start handshake
        t_start_event = time.perf_counter()
        start_event = {
            "event": "start",
            "start": {
                "streamId": stream_id,
                "callId": call_uuid,
                "mediaFormat": {
                    "encoding": "audio/x-mulaw",
                    "sampleRate": 8000,
                    "channels": 1,
                },
            },
        }
        await ws.send(json.dumps(start_event))
        # Initial media packet to resolve Pipecat handshake
        await ws.send(json.dumps({"event": "media", "media": {"payload": silence_chunk}}))

        # Send background silence loop emulating phone line
        keep_sending = True

        async def silence_loop():
            while keep_sending:
                try:
                    await ws.send(json.dumps({"event": "media", "media": {"payload": silence_chunk}}))
                    await asyncio.sleep(0.02)
                except Exception:
                    break

        silence_task = asyncio.create_task(silence_loop())

        # Measure Time-To-First-Audio (TTFA)
        first_audio_ms = None
        packets_received = 0
        total_audio_bytes = 0

        t_listen_start = time.perf_counter()
        while time.perf_counter() - t_listen_start < 12.0:
            try:
                msg_raw = await asyncio.wait_for(ws.recv(), timeout=2.5)
                msg = json.loads(msg_raw)
                event = msg.get("event")
                if event in ("playAudio", "media"):
                    payload = msg.get("media", {}).get("payload", "")
                    if payload:
                        raw = base64.b64decode(payload)
                        packets_received += 1
                        total_audio_bytes += len(raw)
                        if first_audio_ms is None:
                            first_audio_ms = (time.perf_counter() - t_start_event) * 1000.0
                            logger.info(f"  First Greeting Audio Packet received in {first_audio_ms:.2f}ms!")
            except asyncio.TimeoutError:
                if packets_received > 0:
                    break

        keep_sending = False
        silence_task.cancel()

        total_audio_sec = total_audio_bytes / 8000.0
        res = {
            "wss_handshake_ms": round(handshake_ms, 2),
            "ttfa_greeting_ms": round(first_audio_ms, 2) if first_audio_ms else None,
            "packets_received": packets_received,
            "greeting_duration_sec": round(total_audio_sec, 2),
            "streaming_bps": round((total_audio_bytes * 8) / max(total_audio_sec, 0.001), 0),
        }
        logger.info(
            f"  Greeting Streaming -> TTFA: {res['ttfa_greeting_ms']}ms | "
            f"Received: {packets_received} frames ({res['greeting_duration_sec']}s audio)"
        )
        return res


# =========================================================
# 5. Full End-to-End Voice Turnaround (Mouth-to-Ear)
# =========================================================
async def benchmark_end_to_end_turn(ws_url: str) -> Dict[str, Any]:
    logger.info(f"--- Benchmarking Full End-to-End Voice Turnaround (Mouth-to-Ear) ---")
    from kokoro_onnx import Kokoro

    kokoro_dir = Path(settings.KOKORO_CACHE_DIR)
    kokoro = Kokoro(str(kokoro_dir / "kokoro-v1.0.onnx"), str(kokoro_dir / "voices-v1.0.bin"))

    call_uuid = f"turn-call-{int(time.time())}"
    stream_id = f"turn-stream-{int(time.time())}"

    meta = {
        "call_uuid": call_uuid,
        "from": "+14155550199",
        "to": "+14155550100",
        "direction": "inbound",
    }
    encoded_meta = base64.b64encode(json.dumps(meta).encode("utf-8")).decode("utf-8")
    if "body=" not in ws_url:
        delimiter = "&" if "?" in ws_url else "?"
        ws_url = f"{ws_url}{delimiter}body={encoded_meta}"

    silence_chunk = base64.b64encode(b"\xff" * 160).decode("utf-8")

    async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as ws:
        # Handshake
        start_event = {
            "event": "start",
            "start": {
                "streamId": stream_id,
                "callId": call_uuid,
                "mediaFormat": {
                    "encoding": "audio/x-mulaw",
                    "sampleRate": 8000,
                    "channels": 1,
                },
            },
        }
        await ws.send(json.dumps(start_event))
        await ws.send(json.dumps({"event": "media", "media": {"payload": silence_chunk}}))

        keep_sending = True

        async def silence_loop():
            while keep_sending:
                try:
                    await ws.send(json.dumps({"event": "media", "media": {"payload": silence_chunk}}))
                    await asyncio.sleep(0.02)
                except Exception:
                    break

        silence_task = asyncio.create_task(silence_loop())

        # Pre-synthesize caller question speech before session: "What services do you offer?"
        question = "What services do you offer?"
        samples, rate = kokoro.create(question, voice="am_adam", speed=1.1)
        pcm_24k = (samples * 32767).astype(np.int16).tobytes()
        pcm_8k, _ = audioop.ratecv(pcm_24k, 2, 1, rate, 8000, None)
        caller_bytes = audioop.lin2ulaw(pcm_8k, 2)
        caller_dur = len(caller_bytes) / 8000.0

        # Drain initial greeting completely
        greeting_pkts = 0
        t_wait = time.perf_counter()
        while time.perf_counter() - t_wait < 15.0:
            try:
                msg_raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                msg = json.loads(msg_raw)
                if msg.get("event") in ("playAudio", "media"):
                    if msg.get("media", {}).get("payload"):
                        greeting_pkts += 1
            except asyncio.TimeoutError:
                if greeting_pkts > 0:
                    logger.info(f"  Initial greeting finished ({greeting_pkts} packets). Ready for caller question.")
                    break
                continue

        keep_sending = False
        silence_task.cancel()

        logger.info(f"  Streaming caller audio: \"{question}\" ({caller_dur:.2f}s audio)...")
        chunk_size = 160
        for i in range(0, len(caller_bytes), chunk_size):
            chunk = caller_bytes[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk += b"\xff" * (chunk_size - len(chunk))
            pld = base64.b64encode(chunk).decode("utf-8")
            await ws.send(json.dumps({"event": "media", "media": {"payload": pld}}))
            await asyncio.sleep(0.02)

        # Mark turn handoff moment (user stopped speaking)
        t_caller_finished = time.perf_counter()
        logger.info("  Caller finished speaking. Streaming silence for VAD turn detection...")

        keep_sending = True
        silence_task = asyncio.create_task(silence_loop())

        ttfa_turn_ms = None
        turn_packets = 0
        total_resp_bytes = 0

        while time.perf_counter() - t_caller_finished < 15.0:
            try:
                msg_raw = await asyncio.wait_for(ws.recv(), timeout=2.5)
                msg = json.loads(msg_raw)
                event = msg.get("event")
                if event in ("playAudio", "media"):
                    payload = msg.get("media", {}).get("payload", "")
                    if payload:
                        raw = base64.b64decode(payload)
                        turn_packets += 1
                        total_resp_bytes += len(raw)
                        if ttfa_turn_ms is None:
                            ttfa_turn_ms = (time.perf_counter() - t_caller_finished) * 1000.0
                            logger.success(
                                f"  Turnaround Complete! First response audio packet received in {ttfa_turn_ms:.1f}ms!"
                            )
            except asyncio.TimeoutError:
                if turn_packets > 0:
                    break

        keep_sending = False
        silence_task.cancel()

        resp_dur_sec = total_resp_bytes / 8000.0
        res = {
            "question": question,
            "caller_audio_duration_sec": round(caller_dur, 2),
            "turnaround_latency_ms": round(ttfa_turn_ms, 2) if ttfa_turn_ms else None,
            "response_audio_duration_sec": round(resp_dur_sec, 2),
            "response_packets": turn_packets,
        }
        logger.info(
            f"  End-to-End Turnaround -> Latency: {res['turnaround_latency_ms']}ms | "
            f"Answer: {res['response_audio_duration_sec']}s audio"
        )
        return res


# =========================================================
# Main Execution Runner
# =========================================================
async def main():
    cloud_url = sys.argv[1] if len(sys.argv) > 1 else "https://seeks-downloadable-voluntary-missed.trycloudflare.com"
    local_url = "http://localhost:7860"

    cloud_url = cloud_url.rstrip("/")
    cloud_ws_url = cloud_url.replace("https://", "wss://").replace("http://", "ws://") + "/ws"

    logger.info(f"================================================================")
    logger.info(f"ARIA VOICE AI: COMPREHENSIVE SPEED & LATENCY BENCHMARK")
    logger.info(f"Cloud Target URL: {cloud_url}")
    logger.info(f"Local Target URL: {local_url}")
    logger.info(f"================================================================")

    benchmark_report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "cloud_url": cloud_url,
    }

    # 1. HTTP Endpoint Benchmarks
    logger.info("\n[BENCHMARK 1/5] Measuring HTTP Web API Latencies...")
    benchmark_report["http_localhost"] = await benchmark_http_endpoints(local_url, runs=5)
    benchmark_report["http_cloud"] = await benchmark_http_endpoints(cloud_url, runs=5)

    # 2. Cloud TTS Speed Benchmark
    logger.info("\n[BENCHMARK 2/5] Measuring Cloud TTS Synthesis Speed (/api/test-speech)...")
    benchmark_report["cloud_tts"] = await benchmark_cloud_tts(cloud_url)

    # 3. Whisper STT Benchmark
    logger.info("\n[BENCHMARK 3/5] Measuring Faster-Whisper STT Engine Latency...")
    benchmark_report["whisper_stt"] = benchmark_whisper_stt()

    # 4. WebSocket Streaming Benchmark
    logger.info("\n[BENCHMARK 4/5] Measuring Cloud WebSocket & Telephony Protocol Latency...")
    benchmark_report["websocket_stream"] = await benchmark_websocket_stream(cloud_ws_url)

    # 5. Full End-to-End Voice Turnaround (Mouth-to-Ear)
    logger.info("\n[BENCHMARK 5/5] Measuring Full End-to-End Voice Turnaround Latency...")
    benchmark_report["end_to_end_turn"] = await benchmark_end_to_end_turn(cloud_ws_url)

    # Save to JSON
    output_path = PROJECT_ROOT / "benchmark_results.json"
    output_path.write_text(json.dumps(benchmark_report, indent=2))
    logger.success(f"\nAll benchmarks complete! Detailed results saved to {output_path.name}")


if __name__ == "__main__":
    asyncio.run(main())
