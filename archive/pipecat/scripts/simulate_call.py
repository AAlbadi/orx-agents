#!/usr/bin/env python3
"""
Interactive Call Simulator for Aria Voice Agent.
Connects to the agent over WebSocket (local or cloud tunnel),
interacts with the agent, receives real-time audio, saves it as WAV,
and plays it back through your speakers.
"""

import asyncio
import audioop
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import websockets
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings

# G.711 μ-law decoding lookup table for rapid audio reconstruction
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
    """Decode 8-bit G.711 μ-law bytes into 16-bit linear PCM array."""
    indices = np.frombuffer(raw_bytes, dtype=np.uint8)
    return MULAW_DECODE_TABLE[indices]


def generate_caller_speech(text: str) -> bytes:
    """Synthesize caller question into 8kHz μ-law audio bytes."""
    from kokoro_onnx import Kokoro
    kokoro_dir = Path(settings.KOKORO_CACHE_DIR)
    kokoro = Kokoro(str(kokoro_dir / "kokoro-v1.0.onnx"), str(kokoro_dir / "voices-v1.0.bin"))
    samples, rate = kokoro.create(text, voice="am_adam", speed=1.0)
    pcm_24k = (samples * 32767).astype(np.int16).tobytes()
    pcm_8k, _ = audioop.ratecv(pcm_24k, 2, 1, rate, 8000, None)
    return audioop.lin2ulaw(pcm_8k, 2)


async def simulate_call(ws_url: str, question: str = "What time is it?"):
    """Simulates a live telephony session with the Aria agent."""
    logger.info(f"Connecting to Aria Voice Agent at: {ws_url}...")
    
    call_uuid = f"sim-call-{int(time.time())}"
    stream_id = f"sim-stream-{int(time.time())}"

    meta = {
        "call_uuid": call_uuid,
        "from": "+14155550199",
        "to": "+14155550100",
        "direction": "inbound"
    }
    encoded_meta = base64.b64encode(json.dumps(meta).encode("utf-8")).decode("utf-8")
    
    # Append body param if not already present
    if "body=" not in ws_url:
        delimiter = "&" if "?" in ws_url else "?"
        ws_url = f"{ws_url}{delimiter}body={encoded_meta}"

    greeting_audio = []
    response_audio = []

    silence_chunk = base64.b64encode(b"\xff" * 160).decode("utf-8")

    try:
        async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as ws:
            logger.success("WebSocket connected! Sending Plivo stream start handshake...")

            # 1. Send Plivo start event
            start_event = {
                "event": "start",
                "start": {
                    "streamId": stream_id,
                    "callId": call_uuid,
                    "mediaFormat": {
                        "encoding": "audio/x-mulaw",
                        "sampleRate": 8000,
                        "channels": 1
                    }
                }
            }
            await ws.send(json.dumps(start_event))

            # Send initial media packet so Pipecat handshake resolves immediately
            await ws.send(json.dumps({"event": "media", "media": {"payload": silence_chunk}}))
            logger.info("Sent handshake. Listening for Aria's initial greeting...")

            # Background task to send continuous silence frames (50 pkts/sec) emulating telephone mic
            keep_sending = True
            async def send_silence_loop():
                while keep_sending:
                    try:
                        await ws.send(json.dumps({"event": "media", "media": {"payload": silence_chunk}}))
                        await asyncio.sleep(0.02)
                    except Exception:
                        break

            silence_task = asyncio.create_task(send_silence_loop())

            # ==========================================
            # PHASE 1: Capture Greeting
            # ==========================================
            t_start = time.time()
            first_frame = False

            while time.time() - t_start < 20.0:
                try:
                    msg_raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    msg = json.loads(msg_raw)
                    event = msg.get("event")

                    if event in ("playAudio", "media"):
                        payload_b64 = msg.get("media", {}).get("payload", "")
                        if payload_b64:
                            raw_audio = base64.b64decode(payload_b64)
                            pcm_data = decode_mulaw(raw_audio)
                            greeting_audio.append(pcm_data)

                            if not first_frame:
                                first_frame = True
                                dur = (time.time() - t_start) * 1000
                                logger.success(f"Received Aria's greeting audio stream (TTFT: {dur:.1f}ms)!")

                except asyncio.TimeoutError:
                    if len(greeting_audio) > 0:
                        logger.info("Aria finished speaking greeting.")
                        break
                    continue

            # Play greeting
            if greeting_audio:
                full_greeting = np.concatenate(greeting_audio)
                sf.write("aria_greeting.wav", full_greeting, 8000, subtype="PCM_16")
                logger.success(f"Saved greeting ({len(full_greeting)/8000:.2f}s) to aria_greeting.wav")
                if sys.platform == "darwin":
                    logger.info("🔊 Playing Aria's greeting through speakers...")
                    subprocess.run(["afplay", "aria_greeting.wav"])

            # Pause silence loop while caller speaks
            keep_sending = False
            silence_task.cancel()

            # ==========================================
            # PHASE 2: Caller Speaks a Question
            # ==========================================
            logger.info(f"Synthesizing caller question: \"{question}\"...")
            caller_bytes = generate_caller_speech(question)
            logger.info(f"Streaming caller question ({len(caller_bytes)/8000:.2f}s) to agent...")

            chunk_size = 160  # 20ms
            for i in range(0, len(caller_bytes), chunk_size):
                chunk = caller_bytes[i:i+chunk_size]
                if len(chunk) < chunk_size:
                    chunk += b"\xff" * (chunk_size - len(chunk))
                pld = base64.b64encode(chunk).decode("utf-8")
                await ws.send(json.dumps({"event": "media", "media": {"payload": pld}}))
                await asyncio.sleep(0.02)

            logger.info("Finished speaking question. Streaming ambient pause for VAD...")
            
            # Stream silence so VAD detects turn end
            keep_sending = True
            silence_task = asyncio.create_task(send_silence_loop())

            # ==========================================
            # PHASE 3: Capture Aria's Response
            # ==========================================
            t_q = time.time()
            first_ans_frame = False

            while time.time() - t_q < 20.0:
                try:
                    msg_raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    msg = json.loads(msg_raw)
                    event = msg.get("event")

                    if event in ("playAudio", "media"):
                        payload_b64 = msg.get("media", {}).get("payload", "")
                        if payload_b64:
                            raw_audio = base64.b64decode(payload_b64)
                            pcm_data = decode_mulaw(raw_audio)
                            response_audio.append(pcm_data)

                            if not first_ans_frame:
                                first_ans_frame = True
                                dur = (time.time() - t_q) * 1000
                                logger.success(f"Aria responded to question in {dur:.1f}ms!")

                except asyncio.TimeoutError:
                    if len(response_audio) > 0:
                        logger.info("Aria finished answering.")
                        break
                    continue

            keep_sending = False
            silence_task.cancel()

    except Exception as e:
        logger.error(f"WebSocket session error: {e}")
        return False

    if response_audio:
        full_resp = np.concatenate(response_audio)
        sf.write("aria_response.wav", full_resp, 8000, subtype="PCM_16")
        logger.success(f"Saved response ({len(full_resp)/8000:.2f}s) to aria_response.wav")
        if sys.platform == "darwin":
            logger.info("🔊 Playing Aria's answer through speakers...")
            subprocess.run(["afplay", "aria_response.wav"])

    logger.success("=== End of Live Call Simulation ===")
    return True


if __name__ == "__main__":
    target_url = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:7860/ws"
    q = sys.argv[2] if len(sys.argv) > 2 else "What time is it?"
    asyncio.run(simulate_call(target_url, q))
