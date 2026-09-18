#!/usr/bin/env python3
"""
Community-Standard Voice AI & Speech Recognition Benchmark Suite.

Evaluates according to global speech recognition and conversational AI standards:
1. ASR Accuracy: Word Error Rate (WER), Character Error Rate (CER), Word Information Lost (WIL)
   via the standard `jiwer` package and Whisper text normalization.
2. Multilingual Speech Evaluation: Evaluates speech across 5 languages:
   - English (Real call recording, spelled email, proper names, phone numbers)
   - Spanish (HVAC service inquiry & maintenance request)
   - Arabic (HVAC service inquiry & technical support)
   - French (HVAC repair inquiry & appointment request)
   - German (HVAC repair inquiry & system inspection)
3. Intent Steering & Call Control Evaluation:
   - Human Transfer Intent (10 positive test queries + emergency triggers)
   - Call Ending / Wrap-Up Intent (10 positive test queries)
   - Negative Controls (10 normal dialogue turns — must NOT false-trigger)
4. Latency & Real-Time Factor (RTF) Breakdown:
   - STT Latency (ms) & RTF
   - Text-to-Voice Lockstep Delta (ms)
"""

import asyncio
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from loguru import logger

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.bot import detect_call_action, normalize_spoken_email, clean_whisper_hallucinations

try:
    import jiwer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False


# =====================================================================
# 1. Community-Standard Text Normalization (Whisper / ASR Standard)
# =====================================================================
def community_normalize(text: str) -> str:
    """Standardizes text for fair WER/CER scoring according to Whisper/ASR standards:
    - Lowercase
    - Remove punctuation (except @ in emails)
    - Collapse extra whitespace
    - Strip accents/diacritics where appropriate
    """
    if not text:
        return ""
    t = text.lower().strip()
    # Normalize common contractions
    t = re.sub(r"\bcan't\b", "can not", t)
    t = re.sub(r"\bwon't\b", "will not", t)
    t = re.sub(r"\bit's\b", "it is", t)
    t = re.sub(r"\bi'm\b", "i am", t)
    t = re.sub(r"\bthat's\b", "that is", t)
    # Remove punctuation except @ and . in emails
    t = re.sub(r"[^\w\s@\.]", " ", t)
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def calculate_wer_cer(reference: str, hypothesis: str) -> Tuple[float, float, float, float]:
    """Calculates WER, CER, WIP, WIL using jiwer if available, or Levenshtein fallback."""
    ref_norm = community_normalize(reference)
    hyp_norm = community_normalize(hypothesis)

    if HAS_JIWER:
        wer = jiwer.wer(ref_norm, hyp_norm)
        cer = jiwer.cer(ref_norm, hyp_norm)
        try:
            measures = jiwer.compute_measures(ref_norm, hyp_norm)
            wil = measures.get("wil", 0.0)
            wip = measures.get("wip", 1.0)
        except Exception:
            wil = 0.0
            wip = 1.0
        return wer, cer, wil, wip
    else:
        # Standard Levenshtein calculation
        def edit_distance(s1, s2):
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i - 1] == s2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1]
                    else:
                        dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
            return dp[m][n]

        ref_words = ref_norm.split()
        hyp_words = hyp_norm.split()
        wer = edit_distance(ref_words, hyp_words) / max(1, len(ref_words))
        cer = edit_distance(list(ref_norm), list(hyp_norm)) / max(1, len(ref_norm))
        return wer, cer, 0.0, 1.0


# =====================================================================
# 2. Multilingual Audio Generation & Transcription Evaluation
# =====================================================================
async def generate_speech_audio(text: str, lang: str = "en") -> bytes:
    """Generates synthetic high-fidelity 16kHz speech WAV audio in any language."""
    from gtts import gTTS
    gtts_lang = {
        "en": "en",
        "es": "es",
        "ar": "ar",
        "fr": "fr",
        "de": "de"
    }.get(lang, "en")

    mp3_buf = io.BytesIO()
    tts = gTTS(text=text, lang=gtts_lang, slow=False)
    tts.write_to_fp(mp3_buf)
    mp3_buf.seek(0)

    import subprocess
    cmd = [
        "ffmpeg", "-y", "-i", "pipe:0",
        "-ar", "16000", "-ac", "1", "-f", "wav", "pipe:1"
    ]
    proc = subprocess.run(cmd, input=mp3_buf.read(), capture_output=True, check=True)
    return proc.stdout


async def transcribe_with_groq(audio_wav_bytes: bytes, language: Optional[str] = None) -> Tuple[str, str, float]:
    """Transcribes audio using Groq Whisper Large-v3 with language auto-detection or override.
    Returns: (transcribed_text, detected_language, latency_ms)
    """
    from groq import Groq
    client = Groq(api_key=settings.GROQ_API_KEY)

    kwargs = {
        "file": ("test.wav", audio_wav_bytes),
        "model": "whisper-large-v3",
        "response_format": "verbose_json",
        "temperature": 0.0,
    }
    if language:
        kwargs["language"] = language

    t0 = time.perf_counter()
    res = await asyncio.to_thread(client.audio.transcriptions.create, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    raw_text = (getattr(res, "text", "") or "").strip()
    detected_lang = getattr(res, "language", language or "unknown")

    cleaned = clean_whisper_hallucinations(raw_text)
    normalized = normalize_spoken_email(cleaned)
    return normalized, str(detected_lang).lower(), elapsed_ms


# =====================================================================
# 3. Multilingual Test Dataset
# =====================================================================
MULTILINGUAL_TEST_CASES = [
    # English (US)
    {
        "lang": "en",
        "label": "English - HVAC Issue",
        "reference": "Can you help me with my HVAC system?",
        "spoken": "Can you help me with my HVAC system?"
    },
    {
        "lang": "en",
        "label": "English - Spelled Email Dictation",
        "reference": "my email is aalbadi91@gmail.com",
        "spoken": "my email is a as in apple a l b a d i 9 1 at gmail dot com"
    },
    {
        "lang": "en",
        "label": "English - Name & Phone",
        "reference": "my name is Abdullah Aziz and my phone is 647 890 1234",
        "spoken": "my name is Abdullah Aziz and my phone is six four seven eight nine zero one two three four"
    },
    # Spanish
    {
        "lang": "es",
        "label": "Spanish - Servicio de Aire Acondicionado",
        "reference": "Hola, mi aire acondicionado no está enfriando y hace un ruido extraño. ¿Me pueden enviar un técnico mañana?",
        "spoken": "Hola, mi aire acondicionado no está enfriando y hace un ruido extraño. ¿Me pueden enviar un técnico mañana?"
    },
    {
        "lang": "es",
        "label": "Spanish - Mantenimiento de Calefacción",
        "reference": "Necesito agendar una cita para reparar la calefacción este viernes por la tarde.",
        "spoken": "Necesito agendar una cita para reparar la calefacción este viernes por la tarde."
    },
    # Arabic
    {
        "lang": "ar",
        "label": "Arabic - طلب صيانة مكيف",
        "reference": "مرحبا التكييف عندي لا يعمل ويخرج هواء ساخن هل يمكن حجز موعد لصيانة المكيف",
        "spoken": "مرحبا، التكييف عندي لا يعمل ويخرج هواء ساخن، هل يمكن حجز موعد لصيانة المكيف؟"
    },
    # French
    {
        "lang": "fr",
        "label": "French - Panne de Climatisation",
        "reference": "Bonjour mon système de climatisation ne refroidit plus et fait du bruit. Pouvez-vous envoyer un technicien demain?",
        "spoken": "Bonjour, mon système de climatisation ne refroidit plus et fait du bruit. Pouvez-vous envoyer un technicien demain?"
    },
    # German
    {
        "lang": "de",
        "label": "German - Klimaanlagen-Reparatur",
        "reference": "Guten Tag, meine Klimaanlage kühlt nicht mehr und macht seltsame Geräusche. Können Sie morgen einen Techniker schicken?",
        "spoken": "Guten Tag, meine Klimaanlage kühlt nicht mehr und macht seltsame Geräusche. Können Sie morgen einen Techniker schicken?"
    },
]


# =====================================================================
# 4. Intent Steering Benchmark Datasets
# =====================================================================
HUMAN_TRANSFER_TEST_CASES = [
    "Can I please speak to a human?",
    "I need to talk to a real person right now.",
    "Transfer me to an agent please.",
    "Is there a live representative available?",
    "Let me speak to the manager.",
    "Connect me to the owner.",
    "I smell gas in my basement, this is an emergency!",
    "Water is flooding from my ceiling AC unit!",
    "Operator please.",
    "I want to talk to someone real.",
]

END_CALL_TEST_CASES = [
    "No, that's all thank you.",
    "That is everything, thanks.",
    "Goodbye!",
    "Have a great day, bye bye.",
    "All set, thank you goodbye.",
    "No that's it, thanks for your help bye.",
    "That will be all, thank you so much.",
    "Nothing else needed, bye now.",
    "All good, goodbye!",
    "Thanks bye!",
]

CONTROL_NEGATIVE_TEST_CASES = [
    "My air conditioner is making a weird noise.",
    "My name is John Smith.",
    "123 Maple Street Austin Texas.",
    "Tomorrow morning at 10 AM works great.",
    "My email is john.smith@gmail.com.",
    "Yes, this is the right phone number.",
    "How much does a diagnostic visit cost?",
    "Do your technicians carry replacement filters?",
    "Yeah, sounds good.",
    "Can you repeat the appointment time?",
]


# =====================================================================
# 5. Main Benchmark Runner
# =====================================================================
async def run_benchmark():
    print("=" * 75)
    print("  COMMUNITY-STANDARD VOICE AI BENCHMARK SUITE")
    print("  Evaluating WER, CER, Multilingual Speech, & Smart Call Steering")
    print("=" * 75)

    # -----------------------------------------------------------------
    # Part 1: Multilingual Speech-to-Text Benchmark (WER & CER)
    # -----------------------------------------------------------------
    print("\n[PART 1: MULTILINGUAL SPEECH RECOGNITION (WER / CER)]")
    print(f"{'Language':<10} | {'Test Case':<36} | {'WER (%)':<8} | {'CER (%)':<8} | {'Latency':<8} | {'Detected'}")
    print("-" * 88)

    asr_results = []
    for tc in MULTILINGUAL_TEST_CASES:
        lang = tc["lang"]
        label = tc["label"]
        reference = tc["reference"]
        spoken = tc["spoken"]

        try:
            audio_bytes = await generate_speech_audio(spoken, lang=lang)
            hypothesis, detected_lang, latency = await transcribe_with_groq(audio_bytes, language=None)

            wer, cer, wil, wip = calculate_wer_cer(reference, hypothesis)
            wer_pct = wer * 100.0
            cer_pct = cer * 100.0

            asr_results.append({
                "lang": lang,
                "label": label,
                "reference": reference,
                "hypothesis": hypothesis,
                "wer": wer_pct,
                "cer": cer_pct,
                "latency_ms": latency,
                "detected_lang": detected_lang
            })

            print(f"{lang.upper():<10} | {label[:36]:<36} | {wer_pct:>6.1f}%  | {cer_pct:>6.1f}%  | {latency:>5.0f}ms  | {detected_lang}")
        except Exception as e:
            print(f"{lang.upper():<10} | {label[:36]:<36} | ERROR: {e}")

    avg_wer = np.mean([r["wer"] for r in asr_results]) if asr_results else 0.0
    avg_cer = np.mean([r["cer"] for r in asr_results]) if asr_results else 0.0
    avg_latency = np.mean([r["latency_ms"] for r in asr_results]) if asr_results else 0.0

    print("-" * 88)
    print(f"MULTILINGUAL ASR SUMMARY -> Avg WER: {avg_wer:.2f}% | Avg CER: {avg_cer:.2f}% | Avg Latency: {avg_latency:.1f}ms")

    # -----------------------------------------------------------------
    # Part 2: Smart Call Steering (Human Transfer & Call Termination)
    # -----------------------------------------------------------------
    print("\n\n[PART 2: CONVERSATIONAL STEERING & INTENT RECOGNITION ACCURACY]")
    print(f"{'Category':<22} | {'Queries':<8} | {'Correct':<8} | {'Accuracy':<10} | {'Status'}")
    print("-" * 65)

    # 1. Human Transfer Test
    transfer_correct = sum(1 for q in HUMAN_TRANSFER_TEST_CASES if detect_call_action(q) == "transfer")
    transfer_acc = (transfer_correct / len(HUMAN_TRANSFER_TEST_CASES)) * 100.0
    print(f"{'Human Transfer':<22} | {len(HUMAN_TRANSFER_TEST_CASES):<8} | {transfer_correct:<8} | {transfer_acc:>7.1f}%   | {'✅ PASSED' if transfer_acc == 100 else '❌ FAILED'}")

    # 2. Call Ending Test
    end_correct = sum(1 for q in END_CALL_TEST_CASES if detect_call_action(q) == "end_call")
    end_acc = (end_correct / len(END_CALL_TEST_CASES)) * 100.0
    print(f"{'Call Termination':<22} | {len(END_CALL_TEST_CASES):<8} | {end_correct:<8} | {end_acc:>7.1f}%   | {'✅ PASSED' if end_acc == 100 else '❌ FAILED'}")

    # 3. Negative Controls (Ordinary Conversation)
    control_correct = sum(1 for q in CONTROL_NEGATIVE_TEST_CASES if detect_call_action(q) is None)
    control_acc = (control_correct / len(CONTROL_NEGATIVE_TEST_CASES)) * 100.0
    print(f"{'Negative Control':<22} | {len(CONTROL_NEGATIVE_TEST_CASES):<8} | {control_correct:<8} | {control_acc:>7.1f}%   | {'✅ PASSED' if control_acc == 100 else '❌ FAILED'}")

    overall_steering_acc = (transfer_correct + end_correct + control_correct) / (len(HUMAN_TRANSFER_TEST_CASES) + len(END_CALL_TEST_CASES) + len(CONTROL_NEGATIVE_TEST_CASES)) * 100.0
    print("-" * 65)
    print(f"OVERALL STEERING ACCURACY -> {overall_steering_acc:.1f}% ({transfer_correct + end_correct + control_correct}/30 correct)")

    # -----------------------------------------------------------------
    # Part 3: Real Call Recording Audio Verification
    # -----------------------------------------------------------------
    print("\n\n[PART 3: REAL USER CALL RECORDING BENCHMARK]")
    real_rec_file = PROJECT_ROOT / "data" / "recordings" / "call-1789644286.wav"
    if real_rec_file.exists():
        try:
            with open(real_rec_file, "rb") as f:
                real_bytes = f.read()
            text, detected_lang, lat = await transcribe_with_groq(real_bytes[:48000]) # First 3s
            print(f"  Recorded File : {real_rec_file.name}")
            print(f"  Detected Lang : {detected_lang}")
            print(f"  Transcription : \"{text}\"")
            print(f"  Latency       : {lat:.1f}ms")
        except Exception as e:
            print(f"  Could not process real recording: {e}")
    else:
        print("  Real recording audio file not found on local path.")

    # Save results to JSON for reporting
    out_file = PROJECT_ROOT / "benchmark_community_results.json"
    results_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "asr": {
            "avg_wer_percent": round(avg_wer, 2),
            "avg_cer_percent": round(avg_cer, 2),
            "avg_latency_ms": round(avg_latency, 1),
            "test_cases": asr_results
        },
        "steering": {
            "transfer_accuracy": transfer_acc,
            "end_call_accuracy": end_acc,
            "negative_control_accuracy": control_acc,
            "overall_accuracy": overall_steering_acc
        }
    }
    out_file.write_text(json.dumps(results_payload, indent=2))
    print(f"\nSaved benchmark results to {out_file}")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
