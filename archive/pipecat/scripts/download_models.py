#!/usr/bin/env python3
"""
Pre-downloads Kokoro ONNX model weights and Faster-Whisper STT weights.
Used during Docker build or instance provisioning to ensure zero cold-start latency.
"""

import os
import sys
from pathlib import Path
import requests
from loguru import logger

# Model URLs for Kokoro ONNX
KOKORO_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
KOKORO_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"


def download_file(url: str, dest: Path):
    """Download a file with streaming and progress feedback."""
    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f"File already cached: {dest} ({dest.stat().st_size} bytes)")
        return

    logger.info(f"Downloading {url} -> {dest}...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=600)
    resp.raise_for_status()

    total_bytes = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_bytes:
                    done = int(50 * downloaded / total_bytes)
                    sys.stdout.write(f"\r[{'=' * done}{' ' * (50 - done)}] {downloaded / (1024*1024):.1f} MB / {total_bytes / (1024*1024):.1f} MB")
                    sys.stdout.flush()

    sys.stdout.write("\n")
    logger.info(f"Successfully saved {dest}")


def prefetch_kokoro(cache_dir: Path):
    """Prefetch Kokoro ONNX model and voices file."""
    logger.info(f"Checking Kokoro ONNX assets in {cache_dir}...")
    model_path = cache_dir / "kokoro-v1.0.onnx"
    voices_path = cache_dir / "voices-v1.0.bin"

    download_file(KOKORO_MODEL_URL, model_path)
    download_file(KOKORO_VOICES_URL, voices_path)


def prefetch_whisper(model_name: str = "small", cache_dir: Path = None):
    """Prefetch Faster-Whisper model weights via faster_whisper download_model."""
    logger.info(f"Checking Faster-Whisper weights for model '{model_name}'...")
    try:
        from faster_whisper import download_model
        download_model(model_name, output_dir=str(cache_dir) if cache_dir else None)
        logger.info(f"Faster-Whisper '{model_name}' model ready.")
    except Exception as e:
        logger.warning(f"Could not download Faster-Whisper model via API: {e}")
        logger.warning("Whisper will download on first run if not cached.")


def main():
    kokoro_dir = Path(os.environ.get("KOKORO_CACHE_DIR", Path.home() / ".cache" / "pipecat" / "kokoro-onnx"))
    whisper_model = os.environ.get("WHISPER_MODEL", "small")
    whisper_dir = Path(os.environ.get("WHISPER_CACHE_DIR", Path.home() / ".cache" / "whisper"))

    logger.info("=== Starting Model Prefetching ===")
    prefetch_kokoro(kokoro_dir)
    prefetch_whisper(whisper_model, whisper_dir)
    logger.info("=== All Models Prefetched Successfully! ===")


if __name__ == "__main__":
    main()
