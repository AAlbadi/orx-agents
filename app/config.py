"""Application configuration and environment settings."""

import os
from pathlib import Path
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore"
        )

        # Environment & Server
        ENV: str = "local"
        HOST: str = "0.0.0.0"
        PORT: int = 7860
        PUBLIC_URL: str = ""

        # Groq LLM Settings
        GROQ_API_KEY: str = ""
        GROQ_MODEL: str = "qwen/qwen3.8-27b"  # llama-3.3-70b-versatile removed from this account
        GROQ_TEMPERATURE: float = 0.6

        # Google Gemini LLM Settings
        GEMINI_API_KEY: str = ""
        GEMINI_MODEL: str = "gemini-3.1-flash-lite"
        GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

        # OpenRouter LLM Settings
        OPENROUTER_API_KEY: str = ""
        OPENROUTER_MODEL: str = "google/gemma-4-31b-it:free"

        # LLM Provider selection ('auto', 'groq', 'gemini', 'openrouter')
        LLM_PROVIDER: str = "auto"

        # Deepgram Settings (Nova-3 Streaming STT & Flux Cliff TTS)
        DEEPGRAM_API_KEY: str = "f5c0803425b85a228f4ff16eb0d56b20be13d05c"

        # Plivo Telephony Settings
        PLIVO_AUTH_ID: str = ""
        PLIVO_AUTH_TOKEN: str = ""
        PLIVO_PHONE_NUMBER: str = ""

        # STT Settings (Faster-Whisper) — hot-swappable at runtime
        WHISPER_MODEL: str = "base.en"
        WHISPER_DEVICE: str = "cpu"
        WHISPER_COMPUTE_TYPE: str = "int8"
        WHISPER_CACHE_DIR: str = str(Path.home() / ".cache" / "whisper")

        # TTS Settings (Kokoro ONNX) — hot-swappable at runtime
        KOKORO_VOICE: str = "af_heart"
        KOKORO_CACHE_DIR: str = str(Path.home() / ".cache" / "pipecat" / "kokoro-onnx")

        # Greeting — pre-rendered at boot for instant TTFA (Vapi-style)
        GREETING_TEXT: str = (
            "Hi there! I'm Aria, your AI voice assistant. "
            "How can I help you today?"
        )

        # Active model preset label for UI display
        ACTIVE_PRESET: str = "balanced"

        # Audio & Turn Management Latency Tuning
        SAMPLE_RATE: int = 16000
        VAD_STOP_SECS: float = 0.80
        VAD_CONFIDENCE: float = 0.70

        # Capabilities
        ENABLE_TOOLS: bool = True

        # ORX Brand & Polar Checkout Settings
        ORX_BRAND_NAME: str = "ORX Agents"
        ORX_DOMAIN: str = "agents.orxlabs"
        POLAR_ACCESS_TOKEN: str = ""
        POLAR_ORGANIZATION_ID: str = ""
        POLAR_PRODUCT_ID_STARTER: str = "polar_prod_starter"
        POLAR_PRODUCT_ID_GROWTH: str = "polar_prod_growth"
        POLAR_CHECKOUT_URL: str = "https://polar.sh/checkout"

        # Google Maps / Places API Configuration
        GOOGLE_MAPS_API_KEY: str = ""

except ImportError:
    from dataclasses import dataclass

    @dataclass
    class Settings:  # type: ignore
        ENV: str = os.getenv("ENV", "local")
        HOST: str = os.getenv("HOST", "0.0.0.0")
        PORT: int = int(os.getenv("PORT", "7860"))
        PUBLIC_URL: str = os.getenv("PUBLIC_URL", "")

        GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
        GROQ_MODEL: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0.6"))

        GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
        GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        GEMINI_BASE_URL: str = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

        OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
        OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")

        LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "auto")
        DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "f5c0803425b85a228f4ff16eb0d56b20be13d05c")

        PLIVO_AUTH_ID: str = os.getenv("PLIVO_AUTH_ID", "")
        PLIVO_AUTH_TOKEN: str = os.getenv("PLIVO_AUTH_TOKEN", "")
        PLIVO_PHONE_NUMBER: str = os.getenv("PLIVO_PHONE_NUMBER", "")

        WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")
        WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cpu")
        WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        WHISPER_CACHE_DIR: str = os.getenv("WHISPER_CACHE_DIR", str(Path.home() / ".cache" / "whisper"))

        KOKORO_VOICE: str = os.getenv("KOKORO_VOICE", "af_heart")
        KOKORO_CACHE_DIR: str = os.getenv("KOKORO_CACHE_DIR", str(Path.home() / ".cache" / "pipecat" / "kokoro-onnx"))

        GREETING_TEXT: str = os.getenv(
            "GREETING_TEXT",
            "Hi there! I'm Aria, your AI voice assistant. How can I help you today?"
        )
        ACTIVE_PRESET: str = os.getenv("ACTIVE_PRESET", "balanced")

        SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "16000"))
        VAD_STOP_SECS: float = float(os.getenv("VAD_STOP_SECS", "0.80"))
        VAD_CONFIDENCE: float = float(os.getenv("VAD_CONFIDENCE", "0.70"))

        ENABLE_TOOLS: bool = os.getenv("ENABLE_TOOLS", "true").lower() in ("1", "true", "yes")

        ORX_BRAND_NAME: str = os.getenv("ORX_BRAND_NAME", "ORX Agents")
        ORX_DOMAIN: str = os.getenv("ORX_DOMAIN", "agents.orxlabs")
        POLAR_ACCESS_TOKEN: str = os.getenv("POLAR_ACCESS_TOKEN", "")
        POLAR_ORGANIZATION_ID: str = os.getenv("POLAR_ORGANIZATION_ID", "")
        POLAR_PRODUCT_ID_STARTER: str = os.getenv("POLAR_PRODUCT_ID_STARTER", "polar_prod_starter")
        POLAR_PRODUCT_ID_GROWTH: str = os.getenv("POLAR_PRODUCT_ID_GROWTH", "polar_prod_growth")
        POLAR_CHECKOUT_URL: str = os.getenv("POLAR_CHECKOUT_URL", "https://polar.sh/checkout")

        GOOGLE_MAPS_API_KEY: str = os.getenv("GOOGLE_MAPS_API_KEY", "")


settings = Settings()


def get_active_llm_info() -> dict:
    """Returns the currently active LLM provider, model name, and intelligence status.
    Reads from the active assistant profile if available, falling back to global settings."""
    try:
        from app.agents import get_active_assistant
        asst = get_active_assistant()
        agent_llm = asst.get("llm_model", settings.GROQ_MODEL) if asst else settings.GROQ_MODEL
    except Exception:
        agent_llm = settings.GROQ_MODEL

    _DEPRECATED = {"llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "llama-3-70b"}
    if agent_llm in _DEPRECATED:
        agent_llm = "gemini-3.1-flash-lite"

    if "gemini" in agent_llm.lower() and bool(settings.GEMINI_API_KEY):
        return {
            "provider": "gemini",
            "model": agent_llm,
            "display_name": f"Google Gemini ({agent_llm})",
            "badge": "⚡ Gemini Flash (Smart)",
            "is_smart_llm": True,
            "configured": True,
        }
    elif bool(settings.GROQ_API_KEY) and agent_llm not in _DEPRECATED:
        return {
            "provider": "groq",
            "model": agent_llm,
            "display_name": f"Groq ({agent_llm})",
            "badge": "⚡ Groq LPU (Fast)",
            "is_smart_llm": True,
            "configured": bool(settings.GROQ_API_KEY),
        }
    elif bool(settings.GEMINI_API_KEY):
        return {
            "provider": "gemini",
            "model": settings.GEMINI_MODEL,
            "display_name": f"Google Gemini ({settings.GEMINI_MODEL})",
            "badge": "⚡ Gemini Flash (Smart)",
            "is_smart_llm": True,
            "configured": True,
        }
    elif bool(settings.OPENROUTER_API_KEY):
        return {
            "provider": "openrouter",
            "model": settings.OPENROUTER_MODEL,
            "display_name": f"OpenRouter ({settings.OPENROUTER_MODEL})",
            "badge": "⚡ OpenRouter (Free)",
            "is_smart_llm": True,
            "configured": bool(settings.OPENROUTER_API_KEY),
        }
    else:
        return {
            "provider": "demo",
            "model": "rule-based",
            "display_name": "Demo Fallback (No Key)",
            "badge": "⚠️ Demo Mode (Add Key)",
            "is_smart_llm": False,
            "configured": False,
        }

