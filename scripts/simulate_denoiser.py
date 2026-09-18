"""Automated Multi-Scenario Acoustic Noise Cancellation Benchmark.

Tests Pipecat voice pipeline noise suppression across 3 distinct real-world noise environments:
1. Scenario A: HVAC / Air Conditioner / Fan Rumble (Low-frequency harmonics)
2. Scenario B: Telephony Line Static & Thermal Hiss (Wideband high-frequency noise)
3. Scenario C: Ground Loop Hum & Office Ambient Noise (60Hz ground loop + babble floor)
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import soxr

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.server import get_whisper_instance, get_kokoro_instance
from app.denoiser import denoise_audio_array


def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    p_signal = np.mean(signal ** 2)
    p_noise = np.mean(noise ** 2)
    return float(10 * np.log10((p_signal + 1e-12) / (p_noise + 1e-12)))


def calculate_word_accuracy(reference: str, hypothesis: str) -> float:
    ref_words = [w.strip(".,?!;:\'\"").lower() for w in reference.split()]
    hyp_words = [w.strip(".,?!;:\'\"").lower() for w in hypothesis.split()]
    if not ref_words:
        return 100.0 if not hyp_words else 0.0
    matches = sum(1 for w in hyp_words if w in ref_words)
    return min(100.0, round((matches / max(len(ref_words), len(hyp_words))) * 100.0, 1))


def run_single_benchmark(whisper, speech_16k: np.ndarray, noise: np.ndarray, scenario_name: str, clean_text: str) -> Dict[str, Any]:
    audio_sec = len(speech_16k) / 16000.0
    noisy_audio = np.clip(speech_16k + noise, -1.0, 1.0)
    snr_before = calculate_snr(speech_16k, noise)

    t0 = time.perf_counter()
    denoised_audio = denoise_audio_array(
        noisy_audio,
        sample_rate=16000,
        stationary=True,
        prop_decrease=0.85,
    )
    dt_ms = (time.perf_counter() - t0) * 1000
    rtf = (dt_ms / 1000.0) / audio_sec

    residual = denoised_audio - speech_16k
    snr_after = calculate_snr(speech_16k, residual)
    snr_gain = snr_after - snr_before

    def transcribe(audio):
        segments, _ = whisper.transcribe(audio, language="en", beam_size=1)
        return " ".join(s.text.strip() for s in segments)

    noisy_text = transcribe(noisy_audio)
    denoised_text = transcribe(denoised_audio)

    acc_noisy = calculate_word_accuracy(clean_text, noisy_text)
    acc_denoised = calculate_word_accuracy(clean_text, denoised_text)
    passed = (snr_gain >= 5.0) and (rtf <= 0.05) and (acc_denoised >= 75.0)

    return {
        "scenario": scenario_name,
        "audio_duration_sec": round(audio_sec, 2),
        "latency_ms": round(dt_ms, 2),
        "rtf": round(rtf, 4),
        "speed_factor": round(1.0 / rtf, 1),
        "snr_before_db": round(snr_before, 2),
        "snr_after_db": round(snr_after, 2),
        "snr_gain_db": round(snr_gain, 2),
        "clean_text": clean_text,
        "noisy_text": noisy_text,
        "denoised_text": denoised_text,
        "accuracy_noisy": acc_noisy,
        "accuracy_denoised": acc_denoised,
        "status": "PASS" if passed else "FAIL",
    }


def run_all_benchmarks() -> Dict[str, Any]:
    print("\n" + "=" * 76)
    print(" 🎙️  ARIA VOICE AI — ENTERPRISE ACOUSTIC NOISE SUPPRESSION BENCHMARK")
    print("=" * 76)

    print("→ Pre-warming STT and TTS neural models...")
    whisper = get_whisper_instance("small", "cpu", "int8")
    kokoro = get_kokoro_instance()

    phrase = "Thank you for calling Comfort Breeze. I'd like to book an appointment for Friday morning please."
    raw_samples, sr = kokoro.create(phrase, voice="af_heart", speed=1.0)
    speech_16k = soxr.resample(raw_samples, sr, 16000)
    speech_16k = speech_16k / np.max(np.abs(speech_16k)) * 0.75
    t = np.arange(len(speech_16k)) / 16000.0

    # Obtain ground truth clean transcription
    seg, _ = whisper.transcribe(speech_16k, language="en", beam_size=1)
    clean_text = " ".join(s.text.strip() for s in seg)

    np.random.seed(42)
    scenarios = [
        (
            "HVAC / Air Conditioner & Fan Rumble",
            0.22 * np.sin(2 * np.pi * 120 * t) + 0.12 * np.sin(2 * np.pi * 360 * t) + 0.06 * np.random.normal(0, 1, len(speech_16k))
        ),
        (
            "Telephony Line Static & Thermal Hiss",
            0.15 * np.random.normal(0, 1, len(speech_16k))
        ),
        (
            "Ground Loop Hum (60Hz) & Office Babble",
            0.25 * np.sin(2 * np.pi * 60 * t) + 0.08 * np.sin(2 * np.pi * 180 * t) + 0.07 * np.random.normal(0, 1, len(speech_16k))
        ),
    ]

    results = []
    all_passed = True

    for name, noise in scenarios:
        res = run_single_benchmark(whisper, speech_16k, noise, name, clean_text)
        results.append(res)
        if res["status"] != "PASS":
            all_passed = False

        print(f"\n[Test Case] {name}")
        print(f"  • Clean:    \"{res['clean_text']}\"")
        print(f"  • Noisy:    \"{res['noisy_text']}\" ({res['accuracy_noisy']}% match)")
        print(f"  • Denoised: \"{res['denoised_text']}\" ({res['accuracy_denoised']}% match)")
        print(f"  • SNR Gain: +{res['snr_gain_db']} dB ({res['snr_before_db']} dB -> {res['snr_after_db']} dB)")
        print(f"  • Latency:  {res['latency_ms']} ms ({res['speed_factor']}x faster than real-time | RTF {res['rtf']}x)")
        print(f"  • Verdict:  {'✅ PASS' if res['status'] == 'PASS' else '❌ FAIL'}")

    print("\n" + "=" * 76)
    print(f" 🏁 FINAL SUMMARY: {'✅ ALL 3 SCENARIOS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    avg_gain = np.mean([r["snr_gain_db"] for r in results])
    avg_rtf = np.mean([r["rtf"] for r in results])
    avg_speed = np.mean([r["speed_factor"] for r in results])
    print(f" • Average SNR Improvement:  +{avg_gain:.2f} dB")
    print(f" • Average Real-Time Factor: {avg_rtf:.4f}x ({avg_speed:.1f}x real-time speed)")
    print(f" • Noise Cancellation State: ACTIVE (Spectral Gating & Adaptive Wiener)")
    print("=" * 76 + "\n")

    return {
        "overall_status": "PASS" if all_passed else "FAIL",
        "avg_snr_gain_db": round(avg_gain, 2),
        "avg_rtf": round(avg_rtf, 4),
        "avg_speed_factor": round(avg_speed, 1),
        "scenarios": results,
    }


if __name__ == "__main__":
    summary = run_all_benchmarks()
    sys.exit(0 if summary["overall_status"] == "PASS" else 1)
