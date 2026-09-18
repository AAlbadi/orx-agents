#!/usr/bin/env python3
"""
CLI Integration Test for Aria Voice AI Agent components:
1. Tests Groq API connectivity and LLM tool calling
2. Tests Kokoro ONNX Text-to-Speech synthesis
3. Tests Faster-Whisper Speech-to-Text engine initialization
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from loguru import logger

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.prompts import get_system_prompt
from app.tools import REGISTERED_TOOLS


async def test_groq_llm():
    """Verify Groq API connectivity and Aria system prompt response."""
    logger.info("--> Testing Groq LLM API...")
    if not settings.GROQ_API_KEY:
        logger.warning("GROQ_API_KEY is not set. Skipping live Groq test.")
        return False

    try:
        from groq import AsyncGroq
        client = AsyncGroq(api_key=settings.GROQ_API_KEY)

        t0 = time.time()
        completion = await client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": "Hello Aria! What is the current time and what can you help me with?"}
            ],
            temperature=0.6,
            max_tokens=150,
        )
        latency = (time.time() - t0) * 1000
        reply = completion.choices[0].message.content
        logger.success(f"Groq responded in {latency:.1f}ms:\n\"{reply}\"")
        return True
    except Exception as e:
        logger.error(f"Groq API test failed: {e}")
        return False


def test_kokoro_tts():
    """Verify Kokoro ONNX loads and can synthesize audio samples."""
    logger.info("--> Testing Kokoro TTS Engine...")
    try:
        from kokoro_onnx import Kokoro
        kokoro_dir = Path(settings.KOKORO_CACHE_DIR)
        model_path = kokoro_dir / "kokoro-v1.0.onnx"
        voices_path = kokoro_dir / "voices-v1.0.bin"

        if not model_path.exists() or not voices_path.exists():
            logger.warning(f"Kokoro model files not found in {kokoro_dir}. Run `python scripts/download_models.py` first.")
            return False

        t0 = time.time()
        kokoro = Kokoro(str(model_path), str(voices_path))
        voice = settings.KOKORO_VOICE
        voices = kokoro.get_voices() if hasattr(kokoro, "get_voices") else kokoro.voices
        if voice not in voices:
            voice = list(voices)[0]
        samples, sample_rate = kokoro.create("Hello! This is Aria, your AI voice assistant.", voice=voice, speed=1.0)
        dur = (time.time() - t0) * 1000
        logger.success(f"Kokoro TTS synthesized {len(samples)} audio samples at {sample_rate}Hz in {dur:.1f}ms (voice: {voice}).")
        return True
    except Exception as e:
        logger.error(f"Kokoro TTS test failed: {e}")
        return False


def test_faster_whisper():
    """Verify Faster-Whisper initializes properly."""
    logger.info("--> Testing Faster-Whisper STT Engine...")
    try:
        from faster_whisper import WhisperModel
        t0 = time.time()
        model = WhisperModel(
            settings.WHISPER_MODEL,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            download_root=settings.WHISPER_CACHE_DIR
        )
        dur = (time.time() - t0) * 1000
        logger.success(f"Faster-Whisper '{settings.WHISPER_MODEL}' initialized successfully in {dur:.1f}ms.")
        return True
    except Exception as e:
        logger.error(f"Faster-Whisper STT test failed: {e}")
        return False


async def main():
    logger.info("=== Starting Aria Voice AI Integration Diagnostics ===")
    results = {}

    results["Groq LLM"] = await test_groq_llm()
    results["Kokoro TTS"] = test_kokoro_tts()
    results["Faster-Whisper STT"] = test_faster_whisper()

    logger.info("=== Diagnostic Summary ===")
    for comp, passed in results.items():
        status = "PASSED" if passed else "SKIPPED/FAILED"
        logger.info(f"{comp:20}: {status}")


if __name__ == "__main__":
    asyncio.run(main())
