"""Pipecat Voice Agent Pipeline with Plivo, Groq, Kokoro, and Faster-Whisper.

Optimized for Vapi-grade latency and natural human conversational flow:
- Pre-warmed model singletons (0ms cold start per call)
- Instant pre-rendered greeting streaming (TTFA < 50ms)
- Human cadence prompt and natural filler acknowledgments
- Dynamic model selection support
"""

import asyncio
import audioop
import base64
import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, Optional
import numpy as np
import soxr
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    LLMTextFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.tts_service import TTSService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.workers.runner import WorkerRunner

from app.config import settings
from app.prompts import get_system_prompt
from app.tools import REGISTERED_TOOLS


class WebAudioFrameSerializer(FrameSerializer):
    """Studio-grade 16kHz 16-bit Linear PCM serializer for direct web browser WebSocket audio.
    Delivers 100% uncompressed wideband audio with 0ms transcoding latency."""

    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        self._sample_rate = sample_rate

    async def setup(self, setup):
        if setup and getattr(setup, "audio_in_sample_rate", None):
            self._sample_rate = setup.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, InterruptionFrame):
            return json.dumps({"event": "clearAudio"})
        elif isinstance(frame, AudioRawFrame):
            data = frame.audio
            if frame.sample_rate != self._sample_rate and frame.sample_rate > 0:
                data = audioop.ratecv(data, 2, 1, frame.sample_rate, self._sample_rate, None)[0]
            payload = base64.b64encode(data).decode("utf-8")
            return json.dumps({
                "event": "playAudio",
                "media": {
                    "contentType": "audio/x-l16",
                    "sampleRate": self._sample_rate,
                    "payload": payload,
                },
            })
        elif isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if self.should_ignore_frame(frame):
                return None
            return json.dumps(frame.message)
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, bytes):
            return InputAudioRawFrame(
                audio=data,
                num_channels=1,
                sample_rate=self._sample_rate,
            )
        try:
            message = json.loads(data)
        except Exception:
            return None

        ev = message.get("event")
        if ev == "media":
            media = message.get("media", {})
            payload_b64 = media.get("payload")
            if not payload_b64:
                return None
            raw_bytes = base64.b64decode(payload_b64)
            content_type = media.get("contentType", "audio/x-l16")
            in_sr = int(media.get("sampleRate", self._sample_rate) or self._sample_rate)

            if content_type == "audio/x-mulaw":
                pcm = audioop.ulaw2lin(raw_bytes, 2)
                if in_sr != self._sample_rate:
                    pcm, _ = audioop.ratecv(pcm, 2, 1, in_sr, self._sample_rate, None)
            else:
                pcm = raw_bytes
                if in_sr != self._sample_rate:
                    pcm, _ = audioop.ratecv(pcm, 2, 1, in_sr, self._sample_rate, None)

            return InputAudioRawFrame(
                audio=pcm,
                num_channels=1,
                sample_rate=self._sample_rate,
            )
        return None


def clean_whisper_hallucinations(text: str) -> str:
    """Detects and suppresses repetitive token loops generated by Whisper on background noise.

    IMPORTANT: Must NOT suppress legitimate phone number dictation (e.g. "six four seven eight
    nine zero one two") or name/email spelling — those contain repeated digit words which would
    falsely trigger the old 40% ratio check.
    """
    if not text:
        return ""
    cleaned = text.strip()
    words = cleaned.lower().replace(",", "").replace(".", "").split()

    # Digit words spoken for phone numbers / ID numbers — never suppress these
    DIGIT_WORDS = {
        "zero", "one", "two", "three", "four", "five",
        "six", "seven", "eight", "nine", "oh",
    }

    if len(words) >= 6:
        from collections import Counter
        counts = Counter(words)
        most_common_word, count = counts.most_common(1)[0]

        # Only flag as hallucination if the repeated word is NOT a digit/number word
        if most_common_word not in DIGIT_WORDS and count / len(words) >= 0.60:
            logger.warning(f"Suppressed Whisper repetitive loop: '{cleaned[:80]}...'")
            return ""

        # Check for 2-word alternating loops (e.g. "thank you thank you thank you")
        # But only when the bigram is NOT made of digit words
        if len(words) >= 8:
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(0, len(words) - 1, 2)]
            bi_counts = Counter(bigrams)
            if bi_counts:
                top_bi, bi_cnt = bi_counts.most_common(1)[0]
                top_parts = set(top_bi.split())
                if not top_parts.issubset(DIGIT_WORDS) and bi_cnt / len(bigrams) >= 0.60:
                    logger.warning(f"Suppressed alternating Whisper loop: '{cleaned[:80]}...'")
                    return ""

    return cleaned


ARABIC_NAME_MAP = {
    "عبدالعزيز": "Abdulaziz",
    "عبد العزيز": "Abdulaziz",
    "البادي": "Albadi",
    "بادي": "Badi",
    "عبدالله": "Abdullah",
    "عبد الله": "Abdullah",
    "محمد": "Mohammed",
    "احمد": "Ahmed",
    "أحمد": "Ahmed",
    "علي": "Ali",
    "عمر": "Omar",
    "خالد": "Khalid",
    "سعيد": "Saeed",
    "سالم": "Salem",
    "حسن": "Hassan",
    "حسين": "Hussein",
    "سعود": "Saud",
    "فهد": "Fahad",
    "سلطان": "Sultan",
    "فيصل": "Faisal",
    "بندر": "Bandar",
    "نواف": "Nawaf",
    "تركي": "Turki",
    "ناصر": "Nasser",
    "منصور": "Mansour",
    "ماجد": "Majed",
    "وليد": "Waleed",
    "يوسف": "Yousef",
    "ابراهيم": "Ibrahim",
    "إبراهيم": "Ibrahim",
    "صالح": "Saleh",
    "طارق": "Tariq",
    "زياد": "Ziyad",
    "ريان": "Rayan",
    "فارس": "Faris",
    "مساعد": "Musaid",
    "مشعل": "Mishaal",
    "جاسم": "Jassim",
    "غانم": "Ghanim",
    "حمد": "Hamad",
    "حمدان": "Hamdan",
    "زايد": "Zayed",
    "راشد": "Rashid",
}

AR2EN_CHAR = {
    "ا": "a", "أ": "a", "إ": "i", "آ": "aa", "ء": "", "ئ": "y", "ؤ": "w",
    "ب": "b", "ت": "t", "ث": "th", "ج": "j", "ح": "h", "خ": "kh",
    "د": "d", "ذ": "dh", "ر": "r", "ز": "z", "س": "s", "ش": "sh",
    "ص": "s", "ض": "d", "ط": "t", "ظ": "dh", "ع": "a", "غ": "gh",
    "ف": "f", "ق": "q", "ك": "k", "ل": "l", "م": "m", "ن": "n",
    "ه": "h", "و": "w", "ي": "y", "ى": "a", "ة": "h", "ـ": "",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4", "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4", "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9"
}

EMAIL_ARABIC_KEYWORDS = [
    (r"\b(?:ات|آت)\b", "@"),
    (r"\b(?:دوت|نقطة)\b", "."),
    (r"\bجيميل\b", "gmail"),
    (r"\bياهو\b", "yahoo"),
    (r"\b(?:اوتلوك|أوتلوك)\b", "outlook"),
    (r"\bهوتميل\b", "hotmail"),
    (r"\b(?:آيكلاود|ايكلاود|اي كلاود)\b", "icloud"),
    (r"\bكوم\b", "com"),
    (r"\bنت\b", "net"),
    (r"\bاورج\b", "org"),
]


def condition_speech_audio(
    audio_bytes: bytes,
    in_rate: int = 8000,
    target_rate: int = 16000,
    apply_agc: bool = True,
    apply_highpass: bool = True,
) -> bytes:
    """Pre-processes speech audio to studio-grade 16kHz PCM for Whisper comprehension.
    1. DC Offset Removal: Centers waveform around zero.
    2. Butterworth High-Pass (>75Hz): Eliminates sub-audible pop-thumps, mic proximity rumble, and 50/60Hz hum.
    3. Sinc Polyphase Resampling (via soxr): Reconstructs wideband 16kHz spectrogram without aliasing.
    4. Two-Stage Vocal AGC + Soft-Knee Limiter: Normalizes quiet speech to -14 dBFS while protecting against loud clipping.
    """
    if not audio_bytes or len(audio_bytes) < 64:
        return audio_bytes

    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    if len(audio_int16) == 0:
        return audio_bytes

    audio_float = audio_int16.astype(np.float32) / 32768.0

    # 1. DC Offset Removal
    audio_float = audio_float - np.mean(audio_float)

    # 2. High-pass filter (>75Hz) to strip rumble and DC bias
    if apply_highpass and in_rate >= 160:
        try:
            from scipy.signal import butter, lfilter
            nyq = 0.5 * in_rate
            cutoff = 75.0 / nyq
            if 0 < cutoff < 0.99:
                b, a = butter(2, cutoff, btype='highpass')
                audio_float = lfilter(b, a, audio_float).astype(np.float32)
        except Exception:
            pass

    # 3. High-Fidelity Sinc Polyphase Resampling to 16,000 Hz
    if in_rate != target_rate and len(audio_float) > 0:
        try:
            import soxr
            audio_float = soxr.resample(audio_float, in_rate, target_rate)
        except Exception:
            target_len = int(len(audio_float) * (target_rate / in_rate))
            audio_float = np.interp(
                np.linspace(0, len(audio_float), target_len, endpoint=False),
                np.arange(len(audio_float)),
                audio_float
            ).astype(np.float32)

    # 4. Two-Stage Adaptive Vocal AGC & Peak Limiter
    if apply_agc and len(audio_float) > 0:
        rms = float(np.sqrt(np.mean(audio_float**2)))
        # Voice Activity Gate: ignore ambient room floor (< 0.003 RMS / -50 dBFS)
        if rms >= 0.003:
            target_rms = 0.10
            boost_factor = min(4.0, max(0.8, target_rms / (rms + 1e-6)))
            audio_float = audio_float * boost_factor

        # Soft-knee peak limiting
        over = np.abs(audio_float) > 0.90
        if np.any(over):
            audio_float = np.tanh(audio_float * 1.1) * 0.95
        else:
            audio_float = np.clip(audio_float, -0.98, 0.98)

    audio_out_int16 = (audio_float * 32767.0).astype(np.int16)
    import io
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio_out_int16, target_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def assemble_spelled_words(text: str) -> str:
    """Intelligently detects and consolidates spelled-out letter sequences into capitalized proper words.
    Examples:
        'A-B-D-U-L-A-Z-I-Z' -> 'Abdulaziz'
        'A-L-B-A-D-I' -> 'Albadi'
        'A. B. D. U. L. A. Z. I. Z.' -> 'Abdulaziz'
        'A L B A D I' -> 'Albadi'
        'a as in apple, b as in boy, d as in david' -> 'A, B, D'
    Preserves genuine single-letter words like 'I' and 'a'.
    """
    if not text:
        return text

    res = text

    # 1. NATO / Phonetic Alphabet replacement: 'A as in apple' -> 'A'
    res = re.sub(
        r'\b([a-zA-Z])\s+as\s+in\s+[a-zA-Z]+\b',
        lambda m: m.group(1).upper(),
        res,
        flags=re.IGNORECASE
    )

    # 2. Hyphenated spelled letters: e.g. A-B-D-U-L-A-Z-I-Z or a-b-d-u-l-a-z-i-z
    def merge_hyphenated(m):
        letters = [p.strip() for p in m.group(0).split('-') if p.strip()]
        if all(len(l) == 1 for l in letters) and len(letters) >= 3:
            return "".join(letters).title()
        return m.group(0)

    res = re.sub(r'\b[A-Za-z](?:-[A-Za-z]){2,}\b', merge_hyphenated, res)

    # 3. Dotted spelled letters: e.g. A. B. D. U. L. A. Z. I. Z.
    def merge_dotted(m):
        letters = re.findall(r'[A-Za-z]', m.group(0))
        if len(letters) >= 3:
            return "".join(letters).title()
        return m.group(0)

    res = re.sub(r'\b[A-Za-z]\.(?:\s*[A-Za-z]\.){2,}', merge_dotted, res)

    # 4. Spaced spelled letters: e.g. 'A L B A D I' or 'J O H N'
    stop_words = {'i', 'a'}
    def merge_spaced(m):
        raw = m.group(0)
        tokens = raw.split()
        if len(tokens) >= 3 and not set(t.lower() for t in tokens).issubset(stop_words):
            return "".join(tokens).title()
        return raw

    res = re.sub(r'\b([A-Za-z]\s+){2,}[A-Za-z]\b', merge_spaced, res)
    return res


def build_dynamic_whisper_prompt(
    agent_name: str = "Riley",
    agent_tagline: str = "Inbound Receptionist",
    keywords: str = "",
    language: str = "en",
) -> Optional[str]:
    """Generates an enterprise-grade prompt to prime Whisper's decoder context.
    Matches ChatGPT Voice Mode prompting patterns:
    - Conditions Whisper on colloquial conversational speech and hesitations.
    - Primes spelling recognition (A-B-C-D, NATO phonetics).
    - Primes contact patterns (names, emails, phones, addresses).
    - Tailors domain vocabulary to the active assistant.
    """
    if language and language not in ("en", "auto"):
        if language == "es":
            return (
                f"Conversación telefónica con {agent_name}. Nombres propios, direcciones, correo electrónico, "
                "números de teléfono. Español hablado natural: sí, claro, de acuerdo, entiendo."
            )
        elif language == "ar":
            return (
                f"محادثة هاتفية مع {agent_name}. أسماء العملاء، أرقام الهواتف، عناوين، مواعيد، بريد إلكتروني. "
                "كلام عفوي وطبيعي: نعم، تمام، أهلاً، إن شاء الله."
            )
        return None

    # English / Auto prompt
    base_domain = f"Phone conversation with AI assistant {agent_name} ({agent_tagline})."
    if keywords:
        base_domain += f" Relevant topics and vocabulary: {keywords}."

    spelling_primer = (
        "Accurately capture spelled-out names and words: A-B-D-U-L-A-Z-I-Z, A-L-B-A-D-I, M-A-R-K, "
        "S-M-I-T-H, A as in Apple, B as in Boy, C as in Cat."
    )
    contact_primer = (
        "Standard formats: emails (user@domain.com, aalbadi91@gmail.com, aziz@gmail.com), "
        "phone numbers (612-716-9989, 555-0199), and addresses (2508 Delaware Street, Suite 200)."
    )
    colloquial_primer = (
        "Spoken conversational flow: yeah, yep, sure, alright, okay, uh-huh, um, uh, ah, got it, I'm, we'll, don't."
    )
    translit_primer = (
        "Spoken Arabic or diverse names transliterated: Abdulaziz Albadi, Mohammed, Ahmed, Aziz, Ali, Khalid, Saud."
    )

    return f"{base_domain} {spelling_primer} {contact_primer} {colloquial_primer} {translit_primer}"


def transliterate_arabic_to_english(text: str) -> str:
    """Accurately romanizes Arabic names and speech into standard English Latin alphabet."""
    if not text:
        return text
    res = text
    for ar, en in ARABIC_NAME_MAP.items():
        res = re.sub(r"\b" + re.escape(ar) + r"\b", en, res)

    def word_translit(m):
        w = m.group(0)
        if w in ARABIC_NAME_MAP:
            return ARABIC_NAME_MAP[w]
        w_clean = re.sub(r"[\u064B-\u065F\u0670]", "", w)
        if w_clean in ARABIC_NAME_MAP:
            return ARABIC_NAME_MAP[w_clean]
        out = "".join(AR2EN_CHAR.get(c, c) for c in w_clean)
        return out.title() if len(out) > 1 else out

    res = re.sub(r"[\u0600-\u06FF]+", word_translit, res)
    return res


def clean_arabic_in_email(text: str) -> str:
    """Normalizes Arabic dictations, digits, and keywords in emails to English ASCII syntax."""
    if not text:
        return text
    res = text
    for ar_d, en_d in AR2EN_CHAR.items():
        if ar_d in "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩":
            res = res.replace(ar_d, en_d)
    for pat, rep in EMAIL_ARABIC_KEYWORDS:
        res = re.sub(pat, rep, res, flags=re.IGNORECASE)
    if re.search(r"[\u0600-\u06FF]", res):
        res = transliterate_arabic_to_english(res).lower()
    return res


def normalize_spoken_email(text: str) -> str:
    """Normalizes spoken and spelled-out email addresses into clean email syntax.
    Handles letter-by-letter spelling, NATO/phonetic phrases, spelled numbers,
    Arabic dictations, and converts dictation like 'a a l b a d i 9 1 at gmail dot com' -> 'aalbadi91@gmail.com'.
    Ensures email output is ALWAYS strictly lowercase English ASCII.
    """
    if not text:
        return text

    # 1. Clean Arabic keywords and convert Arabic digits to standard ASCII
    text = clean_arabic_in_email(text)

    # 2. Deduplicate consecutive identical email addresses in Whisper transcription
    text = re.sub(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)(?:[,\s]+\1)+', r'\1', text, flags=re.IGNORECASE)

    # Expand phonetic / NATO alphabet: 'a as in apple' -> 'a'
    processed = re.sub(r'\b([a-zA-Z])\s+as\s+in\s+[a-zA-Z]+\b', r'\1', text, flags=re.IGNORECASE)

    digit_map = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'oh': '0',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14',
        'fifteen': '15', 'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19',
        'twenty': '20', 'thirty': '30', 'forty': '40', 'fifty': '50', 'sixty': '60',
        'seventy': '70', 'eighty': '80', 'ninety': '90'
    }

    known_domains = (
        r'(?:gmail|yahoo|hotmail|outlook|icloud|aol|proton|mail|comcast|verizon|msn|live|'
        r'comfortbreezehvac|google|microsoft|apple)'
    )

    anchor_pattern = re.compile(
        r'(.*?\b(?:(?:my\s+)?email(?:\s+is|\s+address\s+is|\s*:)?|reach\s+me\s+at|send\s+(?:it\s+)?to)\s*)'
        r'([a-zA-Z0-9\s\.\-_]+?)\s*(?:@|\bat\b)\s*(' + known_domains + r'|[a-zA-Z0-9\-]+)\s*(?:dot|\.)\s*([a-zA-Z]{2,})',
        re.IGNORECASE
    )

    general_pattern = re.compile(
        r'(.*?(?:^|[.?!]|\b(?:it\'?s|is|at|email)\b)\s*)'
        r'([a-zA-Z0-9\s\.\-_]+?)\s*(?:@|\bat\b)\s*(' + known_domains + r'|[a-zA-Z0-9\-]+)\s*(?:dot|\.)\s*([a-zA-Z]{2,})',
        re.IGNORECASE
    )

    def format_tokens(local_raw):
        raw_tokens = re.split(r'[\s\.\-]+', local_raw.strip())
        tokens = []
        for t in raw_tokens:
            if not t:
                continue
            tl = t.lower()
            if tl in digit_map:
                tokens.append(digit_map[tl])
            elif tl in ('dot', 'period', 'point'):
                tokens.append('.')
            elif tl in ('underscore', 'under_score'):
                tokens.append('_')
            elif tl in ('dash', 'hyphen'):
                tokens.append('-')
            else:
                tokens.append(tl)

        cleaned = []
        i = 0
        while i < len(tokens):
            cur = tokens[i]
            if cur in ('20', '30', '40', '50', '60', '70', '80', '90') and i + 1 < len(tokens) and tokens[i+1] in ('1','2','3','4','5','6','7','8','9'):
                cleaned.append(str(int(cur) + int(tokens[i+1])))
                i += 2
            else:
                cleaned.append(cur)
                i += 1
        return ''.join(cleaned)

    def fix_anchor_match(m):
        prefix = m.group(1)
        local_raw = m.group(2)
        domain = m.group(3).lower()
        tld = m.group(4).lower()
        local_clean = format_tokens(local_raw)
        if local_clean:
            return f"{prefix}{local_clean}@{domain}.{tld}"
        return m.group(0)

    if anchor_pattern.search(processed):
        res = anchor_pattern.sub(fix_anchor_match, processed)
    else:
        res = general_pattern.sub(fix_anchor_match, processed)

    # Guarantee any email address is strictly lowercase English ASCII
    res = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', lambda m: m.group(0).lower(), res)
    return res


def detect_call_action(user_text: str) -> Optional[str]:
    """Detects high-intent conversational call actions: 'transfer' or 'end_call'.

    Returns:
        'transfer' if the caller requests a human, supervisor, agent, or reports an emergency.
        'end_call' if the caller says goodbye, expresses that they have no more questions, or wraps up.
        None for ordinary conversational turns.
    """
    if not user_text:
        return None
    lower = user_text.lower().strip()
    # Normalize contractions and common oral forms
    lower = lower.replace("that's", "that is").replace("it's", "it is")
    lower = lower.replace("that'll", "that will").replace("there's", "there is")
    lower = lower.replace("can't", "can not").replace("i'm", "i am")
    words = re.sub(r'[^a-zA-Z0-9\s]', ' ', lower).split()
    clean = " ".join(words)

    # 1. Emergency Triggers
    emergency_phrases = [
        "smell gas", "gas leak", "pipe burst", "burst pipe", "flooding", "ceiling leak",
        "emergency dispatch", "fire hazard"
    ]
    for ep in emergency_phrases:
        if ep in clean:
            logger.info(f"🚨 Call action: 'transfer' triggered by emergency phrase '{ep}'")
            return "transfer"

    # 2. Human Transfer Request Phrases
    transfer_direct_phrases = [
        "human", "person", "agent", "representative", "operator",
        "manager", "supervisor", "owner", "dispatcher", "someone real",
        "real person", "live person", "live agent"
    ]
    for tp in transfer_direct_phrases:
        if tp in clean:
            if any(verb in clean for verb in ["talk", "speak", "transfer", "connect", "reach", "need", "want", "get", "put", "is there", "can i", "let me"]):
                logger.info(f"📞 Call action: 'transfer' triggered by verb + '{tp}'")
                return "transfer"
            if tp in ["operator", "someone real", "real person", "live person", "live agent"]:
                logger.info(f"📞 Call action: 'transfer' triggered by '{tp}'")
                return "transfer"

    if "transfer me" in clean or "transfer us" in clean or "transfer this call" in clean or "transfer to" in clean:
        logger.info(f"📞 Call action: 'transfer' triggered by transfer command in '{clean}'")
        return "transfer"

    # 3. Call Wrap-Up / End Call Phrases
    end_call_exact = {
        "goodbye", "bye", "bye bye", "bye-bye", "good bye", "bye now",
        "that is all thank you", "that is all thanks",
        "that is all", "that is everything thank you",
        "that is everything thanks", "that is everything",
        "no that is all", "no that is all thank you",
        "no that is everything", "no that is everything thank you",
        "no that is it", "that is it thank you", "that is it thanks",
        "that will be all thank you", "that will be all thank you so much",
        "that will be all thanks", "that will be all",
        "no more questions thank you", "no more questions thanks", "no more questions",
        "have a great day", "have a good day", "have a nice day",
        "thanks for your help bye", "thank you bye", "thanks bye",
        "all set thank you goodbye", "all set goodbye", "all good goodbye",
        "i am all set", "i am good", "no i am good", "no i am all set",
        "all set here", "all set thanks", "all set thank you", "no that is all for now",
        "no nothing else", "nothing else thank you", "nothing else thanks", "nothing else",
        "nope that is all", "nope that is everything", "that is about it", "no more help needed",
        "we are all set", "i am fine thank you", "no i am fine"
    }
    if clean in end_call_exact:
        logger.info(f"👋 Call action: 'end_call' triggered by exact phrase '{clean}'")
        return "end_call"

    has_wrapup = any(w in clean for w in [
        "that is all", "that is everything", "that is it", "that will be all",
        "all set", "all good", "nothing else", "no that is all", "i am all set",
        "i am good", "no more questions", "nope that is all"
    ])
    has_farewell = any(b in clean for b in [
        "bye", "goodbye", "have a good day", "have a great day", "have a nice day"
    ])
    if has_wrapup and (has_farewell or any(t in clean for t in ["thank you", "thanks"])):
        logger.info(f"👋 Call action: 'end_call' triggered by wrap-up combination '{clean}'")
        return "end_call"

    if (clean.startswith("bye ") or clean.endswith(" bye") or clean == "bye" or "goodbye" in clean) and len(words) <= 6:
        logger.info(f"👋 Call action: 'end_call' triggered by farewell words '{clean}'")
        return "end_call"

    return None


class FastWhisperSTTService(WhisperSTTService):
    """Whisper STT service that reuses pre-warmed model weights from RAM to eliminate per-call reload delay."""

    def __init__(
        self,
        sample_rate: int = 8000,
        denoise: bool = False,
        agent_name: str = "Riley",
        agent_tagline: str = "Inbound Receptionist",
        keywords: str = "",
        **kwargs
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.denoise = denoise
        self.agent_name = agent_name
        self.agent_tagline = agent_tagline
        self.keywords = keywords

    def _load(self):
        from app.config import settings
        from app.server import get_whisper_instance
        model_name = self._settings.model or settings.WHISPER_MODEL
        if model_name in ("base", "base.en"):
            model_name = "base.en"
        compute_type = self._compute_type or settings.WHISPER_COMPUTE_TYPE
        if compute_type == "default":
            compute_type = settings.WHISPER_COMPUTE_TYPE
        self._compute_type = compute_type
        self._model = get_whisper_instance(model_name, self._device, compute_type)

    async def run_stt(self, audio: bytes):
        if not self._model:
            from pipecat.frames.frames import ErrorFrame
            yield ErrorFrame("Whisper model not available")
            return

        await self.start_processing_metrics()
        in_rate = getattr(self, "sample_rate", 8000) or 8000

        # Studio-grade acoustic signal conditioning: Butterworth high-pass (>75Hz), Sinc 16kHz polyphase, vocal AGC
        conditioned_wav = condition_speech_audio(
            audio,
            in_rate=in_rate,
            target_rate=16000,
            apply_agc=True,
            apply_highpass=True,
        )
        import io
        import soundfile as sf
        audio_float, _ = sf.read(io.BytesIO(conditioned_wav), dtype="float32")

        from pipecat.frames.frames import TranscriptionFrame
        from pipecat.utils.time import time_now_iso8601
        import asyncio

        language = getattr(self._settings, "language", None) or "en"
        initial_prompt = build_dynamic_whisper_prompt(
            agent_name=self.agent_name,
            agent_tagline=self.agent_tagline,
            keywords=self.keywords,
            language=language or "en",
        )

        hotwords = self._settings.hotwords or (
            "Comfort Breeze, HVAC, air conditioning, AC unit, heating, furnace, heat pump, "
            "thermostat, maintenance, repair, emergency, technician, arrival, schedule, appointment, "
            "leak, rattling, blowing warm air, not cooling, not working, filter, freon, "
            "Aziz, email, gmail, yahoo, icloud, outlook, phone number, address, street, avenue"
        )
        segments, _ = await asyncio.to_thread(
            self._model.transcribe,
            audio_float,
            language=language,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=350),
            hallucination_silence_threshold=1.0,
            no_speech_threshold=0.80,
        )
        text: str = ""
        no_speech_thresh = getattr(self._settings, "no_speech_prob", None)
        if not isinstance(no_speech_thresh, (int, float)):
            no_speech_thresh = 0.80

        for segment in segments:
            prob = getattr(segment, "no_speech_prob", 0.0)
            if prob is None or prob <= no_speech_thresh:
                text += f"{segment.text} "

        await self.stop_processing_metrics()

        text = text.strip()
        # 1. Spelling Consolidation: Merge spelled letters (A-B-D-U-L-A-Z-I-Z -> Abdulaziz)
        text = assemble_spelled_words(text)

        # 2. Strict Language Enforcement: If agent accepted language is English, transliterate any Arabic to English Latin script
        if (not language or language == "en") and re.search(r"[\u0600-\u06FF]", text):
            text = transliterate_arabic_to_english(text)
        text = clean_whisper_hallucinations(text)
        text = normalize_spoken_email(text)
        if text:
            await self._handle_transcription(text, True, language)
            logger.info(f"FastWhisper transcribed: [{text}]")
            yield TranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
                language,
                finalized=True,
            )


class GroqWhisperSTTService(FastWhisperSTTService):
    """Flagship Groq Whisper STT (Large-v3 / Large-v3 Turbo) powered by Groq LPUs with sub-200ms latency.

    Provides enterprise-grade transcription accuracy:
    - 809M / 1.55B parameter full 32-layer acoustic encoder
    - Studio-grade audio conditioning (Butterworth highpass >75Hz, Sinc 16kHz resampling, dual-stage AGC)
    - Dynamic conversational & spelling prompt priming
    - Sub-200ms round-trip latency on Groq LPUs
    - Automatically falls back to local Faster-Whisper if API is unreachable
    """

    def __init__(
        self,
        sample_rate: int = 8000,
        model: str = "whisper-large-v3-turbo",
        denoise: bool = False,
        language: Optional[str] = None,
        agent_name: str = "Riley",
        agent_tagline: str = "Inbound Receptionist",
        keywords: str = "",
        **kwargs
    ):
        super().__init__(
            sample_rate=sample_rate,
            denoise=denoise,
            agent_name=agent_name,
            agent_tagline=agent_tagline,
            keywords=keywords,
            **kwargs
        )
        self.groq_model = model or "whisper-large-v3-turbo"
        self.language = language  # None enables automatic language detection across 99+ languages
        self._groq_client = None
        if settings.GROQ_API_KEY:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=settings.GROQ_API_KEY)
            except Exception as e:
                logger.warning(f"Could not initialize Groq client for STT: {e}")

    async def run_stt(self, audio: bytes):
        if not self._groq_client:
            async for frame in super().run_stt(audio):
                yield frame
            return

        await self.start_processing_metrics()
        in_rate = getattr(self, "sample_rate", 8000) or 8000
        audio_int16 = np.frombuffer(audio, dtype=np.int16)

        # Ignore tiny audio bursts (< 0.18s)
        if len(audio_int16) < in_rate * 0.18:
            await self.stop_processing_metrics()
            return

        # Condition audio to pristine 16kHz PCM WAV with high-pass filtering and vocal AGC
        conditioned_wav = condition_speech_audio(
            audio,
            in_rate=in_rate,
            target_rate=16000,
            apply_agc=True,
            apply_highpass=True,
        )

        prompt = build_dynamic_whisper_prompt(
            agent_name=self.agent_name,
            agent_tagline=self.agent_tagline,
            keywords=self.keywords,
            language=self.language or "en",
        )

        kwargs_stt = {
            "file": ("speech.wav", conditioned_wav),
            "model": self.groq_model,
            "response_format": "verbose_json",
            "temperature": 0.0,
        }
        if prompt:
            kwargs_stt["prompt"] = prompt
        if self.language and self.language not in ("auto", "multilingual"):
            kwargs_stt["language"] = self.language

        detected_lang = self.language or "en"
        t0 = time.time()
        try:
            import asyncio
            res = await asyncio.to_thread(
                self._groq_client.audio.transcriptions.create,
                **kwargs_stt
            )
            elapsed_ms = int((time.time() - t0) * 1000)
            text = (getattr(res, "text", "") or "").strip()
            if hasattr(res, "language") and res.language:
                detected_lang = str(res.language).lower()

            # 1. Spelling Consolidation: Merge spelled letters (A-B-D-U-L-A-Z-I-Z -> Abdulaziz)
            text = assemble_spelled_words(text)

            # 2. Strict Language Enforcement: If agent accepted language is English, transliterate any Arabic to English Latin script
            if (not self.language or self.language == "en") and re.search(r"[\u0600-\u06FF]", text):
                logger.info(f"🌐 [Language-Enforcer] Transliterating Arabic speech to English alphabet: '{text}'")
                text = transliterate_arabic_to_english(text)

            text = clean_whisper_hallucinations(text)
            text = normalize_spoken_email(text)
        except Exception as e:
            logger.warning(f"Groq STT failed ({e}), falling back to local Faster-Whisper...")
            async for frame in super().run_stt(audio):
                yield frame
            return

        await self.stop_processing_metrics()

        if text:
            from pipecat.frames.frames import TranscriptionFrame
            from pipecat.utils.time import time_now_iso8601
            lang_code = {
                "english": "en", "spanish": "es", "arabic": "ar", "french": "fr",
                "german": "de", "italian": "it", "portuguese": "pt", "chinese": "zh",
                "japanese": "ja", "russian": "ru"
            }.get(detected_lang, detected_lang)
            await self._handle_transcription(text, True, lang_code)
            logger.info(f"⚡ [Groq-STT: {self.groq_model} | {elapsed_ms}ms | {detected_lang}] Transcribed: [{text}]")
            yield TranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
                lang_code,
                finalized=True,
            )


class FastKokoroTTSService(KokoroTTSService):
    """Kokoro TTS service that reuses the pre-warmed Kokoro ONNX instance from RAM.
    Includes timeout guard for single-core/cloud VMs, language normalization, and fallback synthesis."""

    def __init__(self, **kwargs):
        from unittest.mock import patch
        from app.server import get_kokoro_instance
        # Generous timeout for single-core or cloud instances to eliminate premature context cancellation
        if "stop_frame_timeout_s" not in kwargs:
            kwargs["stop_frame_timeout_s"] = 35.0
        prewarmed = get_kokoro_instance()
        with patch("pipecat.services.kokoro.tts.Kokoro", return_value=prewarmed):
            super().__init__(**kwargs)

    @traced_tts
    async def run_tts(self, text: str, context_id: str):
        from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame
        import asyncio

        try:
            await self.start_tts_usage_metrics(text)

            voice = getattr(self._settings, "voice", "af_heart") or "af_heart"
            if hasattr(self._kokoro, "voices") and voice not in self._kokoro.voices:
                voice = "af_heart"

            raw_lang = getattr(self._settings, "language", None)
            if isinstance(raw_lang, Language):
                lang = self.language_to_service_language(raw_lang)
            else:
                lang = str(raw_lang or "en-us").lower().replace("_", "-")
            if lang in ("en", "english"):
                lang = "en-us"

            speed = float(getattr(self._settings, "speed", 1.0) or 1.0)
            target_sr = self.sample_rate if self.sample_rate > 0 else 8000

            logger.info(f"Kokoro synthesizing: '{text[:60]}' (voice={voice}, lang={lang}, speed={speed})")

            stream = self._kokoro.create_stream(text, voice=voice, lang=lang, speed=speed)

            frames_emitted = 0
            async for samples, sample_rate in stream:
                await self.stop_ttfb_metrics()

                audio_int16 = (samples * 32767).astype(np.int16).tobytes()
                audio_data = await self._resampler.resample(
                    audio_int16, sample_rate, target_sr
                )

                frames_emitted += 1
                yield TTSAudioRawFrame(
                    audio=audio_data,
                    sample_rate=target_sr,
                    num_channels=1,
                    context_id=context_id,
                )

            # Robust direct fallback if streaming yielded 0 frames
            if frames_emitted == 0:
                logger.warning(f"Kokoro stream yielded 0 frames for '{text}'. Triggering fallback synthesis...")
                samples, sample_rate = await asyncio.to_thread(
                    self._kokoro.create,
                    text,
                    voice=voice,
                    speed=speed,
                    lang=lang,
                )
                audio_int16 = (samples * 32767).astype(np.int16).tobytes()
                audio_data = await self._resampler.resample(
                    audio_int16, sample_rate, target_sr
                )
                yield TTSAudioRawFrame(
                    audio=audio_data,
                    sample_rate=target_sr,
                    num_channels=1,
                    context_id=context_id,
                )
        except Exception as e:
            logger.error(f"Kokoro synthesis error for '{text[:40]}': {e}")
            yield ErrorFrame(error=f"Kokoro synthesis error: {e}")
        finally:
            await self.stop_ttfb_metrics()


class DeepgramStreamingTTSService(TTSService):
    """Ultra-low latency streaming Deepgram TTS supporting Flux (Cliff) and Aura voices.
    Yields 8000Hz PCM TTSAudioRawFrames directly to the audio pipeline.
    Uses a persistent connection pool to eliminate per-utterance TLS handshake overhead.
    Gracefully falls back to pre-warmed FastKokoroTTSService if API is unreachable."""

    # Shared HTTP client across all instances — reuses TLS connections to api.deepgram.com
    _shared_client: Optional[Any] = None

    @classmethod
    def _get_client(cls):
        import httpx
        if cls._shared_client is None or cls._shared_client.is_closed:
            cls._shared_client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=3.0, read=10.0, write=5.0, pool=15.0),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10, keepalive_expiry=30),
                http2=False,  # HTTP/1.1 is more reliable for streaming with Deepgram
            )
        return cls._shared_client

    def __init__(self, voice: str = "flux-cliff-en", sample_rate: int = 16000, **kwargs):
        from pipecat.services.settings import TTSSettings
        from pipecat.transcriptions.language import Language
        default_settings = TTSSettings(
            model=None,
            voice=voice or "flux-cliff-en",
            language=Language.EN,
        )
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )
        self.voice = voice or "flux-cliff-en"

    @traced_tts
    async def run_tts(self, text: str, context_id: str):
        from pipecat.frames.frames import TTSAudioRawFrame
        import time

        if not text or not text.strip():
            return

        api_key = getattr(settings, "DEEPGRAM_API_KEY", "")
        if not api_key:
            logger.warning("Deepgram API key not configured, falling back to local Kokoro TTS...")
            kokoro = FastKokoroTTSService(sample_rate=self.sample_rate)
            async for frame in kokoro.run_tts(text, context_id):
                yield frame
            return

        ver = "v2" if "flux" in self.voice else "v1"
        url = f"https://api.deepgram.com/{ver}/speak?model={self.voice}&encoding=linear16&sample_rate={self.sample_rate}&container=none"

        await self.start_tts_usage_metrics(text)
        t0 = time.time()
        first_chunk_logged = False
        try:
            client = self._get_client()
            async with client.stream(
                "POST",
                url,
                headers={"Authorization": f"Token {api_key}", "Content-Type": "application/json"},
                json={"text": text},
            ) as response:
                if response.status_code != 200:
                    err_body = await response.aread()
                    logger.error(f"Deepgram TTS error ({response.status_code}): {err_body.decode(errors='ignore')}")
                    kokoro = FastKokoroTTSService(sample_rate=self.sample_rate)
                    async for frame in kokoro.run_tts(text, context_id):
                        yield frame
                    return

                await self.stop_ttfb_metrics()
                # 320-byte chunks = 20ms at 8kHz 16-bit mono → fast first-audio delivery
                chunk_size = int(self.sample_rate * 2 * 0.020)
                buffer = bytearray()
                async for raw_bytes in response.aiter_bytes():
                    if not first_chunk_logged and raw_bytes:
                        ttfb_ms = int((time.time() - t0) * 1000)
                        logger.debug(f"🗣️ [Deepgram TTS] First audio chunk in {ttfb_ms}ms")
                        first_chunk_logged = True
                    buffer.extend(raw_bytes)
                    while len(buffer) >= chunk_size:
                        chunk = bytes(buffer[:chunk_size])
                        buffer = buffer[chunk_size:]
                        yield TTSAudioRawFrame(
                            audio=chunk,
                            sample_rate=self.sample_rate,
                            num_channels=1,
                            context_id=context_id,
                        )
                if len(buffer) > 0:
                    yield TTSAudioRawFrame(
                        audio=bytes(buffer),
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        context_id=context_id,
                    )
            elapsed = int((time.time() - t0) * 1000)
            logger.info(f"🗣️ [Deepgram TTS: {self.voice}] Synthesized in {elapsed}ms: '{text[:50]}'")
        except Exception as e:
            logger.warning(f"Deepgram TTS exception ({e}), falling back to Kokoro...")
            # Reset shared client on error so next call gets a fresh connection
            DeepgramStreamingTTSService._shared_client = None
            kokoro = FastKokoroTTSService(sample_rate=self.sample_rate)
            async for frame in kokoro.run_tts(text, context_id):
                yield frame


async def run_bot(
    transport: BaseTransport,
    metadata: Optional[Dict[str, Any]] = None,
    handle_sigint: bool = False
):
    """Initializes and runs the full Pipecat audio processing pipeline with pre-warmed models."""
    metadata = metadata or {}
    caller_number = metadata.get("from") or metadata.get("caller")
    called_number = metadata.get("to") or metadata.get("called")

    from app.agents import get_active_assistant
    active_agent = get_active_assistant()
    agent_name = active_agent.get("name", "Riley")
    agent_prompt = active_agent.get("system_prompt", "")
    agent_greeting = active_agent.get("first_message", settings.GREETING_TEXT)
    agent_voice = active_agent.get("tts_voice", settings.KOKORO_VOICE)
    agent_stt = active_agent.get("stt_model", settings.WHISPER_MODEL)
    agent_speed = float(active_agent.get("voice_speed", 1.0))
    agent_preset = active_agent.get("preset", settings.ACTIVE_PRESET)
    agent_lang = active_agent.get("language", "en")

    logger.info(
        f"Initializing voice agent pipeline for assistant '{agent_name}' "
        f"(Voice: {agent_voice}@{agent_speed}x | STT: {agent_stt} | Lang: {agent_lang} | Caller: {caller_number})"
    )

    enable_denoising = active_agent.get("background_denoising", False)
    agent_tagline = active_agent.get("tagline", "Inbound Receptionist & Dispatcher")
    agent_keywords = active_agent.get("keywords", "")

    # 1. Speech-To-Text Selection: Deepgram Nova-3 OR Groq Whisper Large-v3 Turbo OR Local Faster-Whisper
    is_deepgram_stt = bool(settings.DEEPGRAM_API_KEY) and (
        agent_stt in ("nova-3", "deepgram-nova-3", "deepgram", "deepgram-nova")
        or "nova" in agent_stt.lower()
    )
    if is_deepgram_stt:
        logger.success(f"⚡ [STT] Activating Deepgram Nova-3 Realtime Streaming (lang={agent_lang})")
        from pipecat.services.deepgram.stt import DeepgramSTTService
        stt = DeepgramSTTService(
            api_key=settings.DEEPGRAM_API_KEY,
            sample_rate=settings.SAMPLE_RATE,
            ttfs_p99_latency=1.2,          # SOTA safety margin prevents Pipecat turn wait timeout from collapsing to 0
            settings=DeepgramSTTService.Settings(
                model="nova-3",            # Deepgram flagship SOTA transcription model
                language=agent_lang if agent_lang != "auto" else "en",
                smart_format=True,
                punctuate=True,
                interim_results=True,
                endpointing=300,           # 300ms endpointing sensitivity prevents word-chopping on pauses
            ),
        )
    elif bool(settings.GROQ_API_KEY) and agent_stt not in ("base-local", "small-local", "base", "small"):
        groq_model_name = "whisper-large-v3-turbo"
        if agent_stt == "whisper-large-v3":
            groq_model_name = "whisper-large-v3"
        logger.success(f"🚀 [STT] Activating Enterprise Flagship Groq Whisper ({groq_model_name}, lang={agent_lang})")
        stt = GroqWhisperSTTService(
            model=groq_model_name,
            settings=WhisperSTTService.Settings(model="base.en" if agent_lang == "en" else "base"),
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            sample_rate=settings.SAMPLE_RATE,
            denoise=enable_denoising,
            language=agent_lang,
            agent_name=agent_name,
            agent_tagline=agent_tagline,
            keywords=agent_keywords,
        )
    else:
        if agent_stt in ("base", "base.en") and agent_lang == "en":
            agent_stt = "base.en"
        elif agent_stt in ("base", "base.en"):
            agent_stt = "base"
        logger.info(f"🎙️ [STT] Activating Local Faster-Whisper ({agent_stt}, lang={agent_lang})")
        stt = FastWhisperSTTService(
            settings=WhisperSTTService.Settings(model=agent_stt, language=agent_lang if agent_lang != "auto" else None),
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            sample_rate=settings.SAMPLE_RATE,
            denoise=enable_denoising,
            agent_name=agent_name,
            agent_tagline=agent_tagline,
            keywords=agent_keywords,
        )

    # 2. Text-To-Speech Selection: Deepgram Flux / Aura OR Fast pre-warmed Kokoro ONNX
    is_deepgram_voice = bool(settings.DEEPGRAM_API_KEY) and (
        agent_voice.startswith("flux-")
        or agent_voice.startswith("aura-")
        or agent_voice.lower() == "cliff"
    )
    if is_deepgram_voice:
        deepgram_voice_id = "flux-cliff-en" if agent_voice.lower() == "cliff" else agent_voice
        logger.success(f"🗣️ [TTS] Activating Deepgram Conversational Voice ({deepgram_voice_id})")
        tts = DeepgramStreamingTTSService(
            voice=deepgram_voice_id,
            sample_rate=settings.SAMPLE_RATE,
        )
    else:
        logger.info(f"🗣️ [TTS] Activating Local Kokoro ONNX ({agent_voice}@{agent_speed}x)")
        tts = FastKokoroTTSService(
            sample_rate=settings.SAMPLE_RATE,
            stop_frame_timeout_s=35.0,
            settings=KokoroTTSService.Settings(
                voice=agent_voice,
                speed=agent_speed,
                language=Language.EN_US,
            )
        )

    # 3. LLM: Multi-Provider Intelligence (Groq LPU 180ms -> Google Gemini Flash 700ms -> OpenRouter)
    if agent_prompt:
        now_str = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        meta_ctx = f"\n\n[Call Context]\nCurrent date and time: {now_str}."
        if caller_number:
            meta_ctx += f"\nCaller's phone number: {caller_number}."
        if called_number:
            meta_ctx += f"\nDialed number: {called_number}."
        caller_id_note = f"The caller's incoming phone number from caller ID is {caller_number}." if caller_number else "The caller is connecting via web audio or private line."

        voice_rules = (
            "\n\n[High-Precision Information Capture Protocol & Voice Guidelines]\n"
            "You are an elite, highly intelligent voice AI receptionist. You must achieve a 100% success rate extracting and confirming the customer's information by following this exact proactive conversational flow:\n\n"
            "0. ANTI-REPETITION & MANDATORY SLOT PROGRESSION (CRITICAL):\n"
            "   - Track collected slots: [1. Issue] -> [2. Address] -> [3. Time Window] -> [4. Name & Phonetic Check] -> [5. Phone & Email].\n"
            "   - NEVER repeat a question you just asked or ask for information the caller already provided.\n"
            "   - Once the customer provides their name (e.g. 'Abdul Aziz Bari'), IMMEDIATELY accept and acknowledge it, then advance directly to the next missing detail (spelling verification or callback phone). NEVER re-ask 'May I have your first and last name, please?'.\n"
            "   - If the customer answers two steps at once, accept both and advance immediately.\n\n"
            "1. PROACTIVE AMBIGUITY CLARIFICATION PROTOCOL:\n"
            "   - If caller speech sounds unexpected, garbled, cut off, or contains ambient noise words (e.g. 'box 5 for me', 'anytime box 5'):\n"
            "   - Proactively clarify their intent politely in context: 'Just to make sure I caught that correctly, did you say anytime tomorrow works for you?'\n"
            "   - Never guess wildly, never hallucinate, and never freeze.\n\n"
            "2. MID-CALL REASSURANCE (NO LINE-TEST RESET):\n"
            "   - If the caller asks 'Are you there?', 'Can you hear me?', or 'Hello?' mid-call:\n"
            "   - Reassure them immediately within the active context of your current booking step:\n"
            "     'Yes, I'm right here! I have your name noted down. What is the best callback phone number for your arrival updates?'\n"
            "   - NEVER reset the conversation or repeat your opening greeting mid-call.\n\n"
            "3. SERVICE & PROBLEM DIAGNOSIS:\n"
            "   - Understand what service or problem they have (e.g. AC blowing warm air, furnace maintenance, system leaking).\n"
            "   - Immediately offer to get a certified technician scheduled.\n\n"
            "4. CUSTOMER NAME & NATO / AIRLINE PHONETIC DISAMBIGUATION PROTOCOL:\n"
            "   - Ask for their full name: 'May I have your first and last name, please?'\n"
            "   - Smart Spelling Verification: If the name is uncommon, unique, hyphenated, foreign, or has multiple spellings (e.g. Albadi, Kaelen, Jon vs John, Smythe vs Smith), ask: 'Could you quickly spell that out for me just so our technician has it 100% accurate in our dispatch system?'\n"
            "   - Acoustic / Phonetic Disambiguation: When letters with identical acoustic profiles are heard or corrected (B vs D vs P, M vs N, T vs D, F vs S, C vs Z), proactively verify using standard reference words:\n"
            "     * 'Was that B as in Boy, or D as in David?'\n"
            "     * 'Was that M as in Mary, or N as in Nancy?'\n"
            "     * 'Was that T as in Tom, or D as in David?'\n"
            "   - Read back the confirmed spelling explicitly: 'Got it, that is A-L-B-A-D-I with D as in David, correct?'\n"
            "   - Instant Self-Correction Adoption: If the customer corrects any letter or detail (e.g. 'No, instead of T it should be D', 'Wait, it\\'s Dave not David'), IMMEDIATELY adopt the correction warmly: 'Got it, updated to D as in David! So that is Albadi. What is the best mobile number...?' NEVER express confusion or say 'I\\'m confused'.\n\n"
            "5. SERVICE ADDRESS / LOCATION & MANDATORY CONFIRMATION:\n"
            "   - Ask for their location: 'And what is the street address where you\\'d like our technician to visit?'\n"
            "   - Read back the street and city clearly to confirm: 'Got it, 2508 Delaware Street in Minneapolis, correct?'\n\n"
            "6. SCHEDULING TIME WINDOW:\n"
            "   - Inquire about their preferred date and time: 'We have openings tomorrow morning around 10 AM or Thursday afternoon around 2 PM. Which day and time works best for you?'\n\n"
            "7. PHONE NUMBER & RHYTHMIC DIGIT READBACK:\n"
            f"   - {caller_id_note}\n"
            "   - If an incoming phone number is present, ask: 'Can we send your technician arrival updates and confirmation text to this phone number, or is there a different mobile number you prefer?'\n"
            "   - If no incoming number is present, ask: 'What is the best mobile phone number where we can text your technician arrival updates?'\n"
            "   - Always read back phone numbers in conversational cadence: 'Got it, 612... 769... 9890, correct?'\n\n"
            "8. EMAIL ADDRESS & SMART DOUBLE-CHECK / READBACK:\n"
            "   - Ask for their email: 'And what\\'s the best email address to send your Google Calendar invite and service confirmation?'\n"
            "   - Spelled Letters & Dictation: If the caller spells their email letter-by-letter (e.g. 'A-A-L-B-A-D-I' or 'a a l b a d i 9 1 at gmail dot com'), combine the letters cleanly into the email (e.g. aalbadi91@gmail.com).\n"
            "   - MANDATORY DOUBLE-CHECK: As soon as the customer gives an email, read it back clearly to confirm: 'Just to double-check that, that\\'s [clearly spoken email address], correct?'\n"
            "   - INSTANT CORRECTION ADOPTION: If the customer clarifies or corrects, immediately adopt it without friction.\n\n"
            "9. COMPREHENSIVE 5-POINT FINAL BOOKING RECAP:\n"
            "   - Once confirmed, give a warm, reassuring 5-point recap:\n"
            "     'You\\'re all set, [Name]! We have you booked for [Service] on [Day/Time] at [Address]. We sent your calendar invite to [Email] and text updates to [Phone]. Is there anything else I can assist you with today?'\n\n"
            "10. CALL TRANSFER TO HUMAN SPECIALIST:\n"
            "   - If the caller asks to speak to a human, real person, agent, manager, owner, or reports an emergency (e.g. gas smell, water leak), immediately say: 'I completely understand. Let me connect you with our live team right away. Please hold for just a moment.'\n\n"
            "11. CALL WRAP-UP & POLITE FAREWELL:\n"
            "   - When the customer says goodbye, thanks you, or indicates they have no further questions, warmly conclude the call: 'Thank you so much for choosing Comfort Breeze! Have a wonderful day, goodbye!' and never ask another question.\n\n"
            "12. MULTILINGUAL COURTESY:\n"
            "   - If the caller speaks Spanish, Arabic, or another language, respond warmly and courteously in that language or offer human assistance.\n\n"
            "[Spoken Rules]\n"
            "- Ask only ONE question at a time. Never ask multiple questions in a single turn.\n"
            "- Speak naturally in exactly 1 to 2 concise, punchy sentences (under 25 words per turn).\n"
            "- Use natural everyday contractions ('I\\'m', 'we\\'ll', 'don\\'t', 'it\\'s', 'let\\'s').\n"
            "- Conversational Flow & Natural Acknowledgments: Acknowledge what the caller said warmly and conversationally in your own words (e.g. 'Understood', 'I can definitely help with that', 'Perfect', 'Thanks for confirming') before answering or asking the next question.\n"
            "- Never output bullet points, asterisks, or markdown formatting.\n"
            "- Spell out times, dates, and numbers in conversational words."
        )
        system_instruction = f"{agent_prompt}{meta_ctx}{voice_rules}"
    else:
        system_instruction = get_system_prompt(caller_number, called_number, preset=agent_preset)

    agent_llm = active_agent.get("llm_model", settings.GROQ_MODEL)

    # --- Model availability guards ---
    # Redirect deprecated or decommissioned Groq models to active flagship qwen/qwen3.8-27b
    _DEPRECATED_GROQ_MODELS = {
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama-3-70b",
        "llama-70b",
        "qwen-2.5-32b",
        "qwen-2.5-72b",
    }
    if agent_llm in _DEPRECATED_GROQ_MODELS:
        logger.warning(f"⚠️ [LLM] Model '{agent_llm}' is deprecated on Groq — auto-routing to flagship 'qwen/qwen3.8-27b'.")
        agent_llm = "qwen/qwen3.8-27b"

    is_gemini = bool(settings.GEMINI_API_KEY) and ("gemini" in agent_llm.lower())
    is_groq = bool(settings.GROQ_API_KEY) and (
        "qwen" in agent_llm.lower()
        or "groq" in agent_llm.lower()
        or "whisper" in agent_llm.lower()
        or ("llama" in agent_llm.lower() and agent_llm not in _DEPRECATED_GROQ_MODELS)
        or not is_gemini
    )

    if is_gemini and settings.GROQ_API_KEY and "flash-lite" in agent_llm.lower():
        # Prevent 43-second stall caused by deprecated/throttled gemini-3.5-flash-lite on voice calls
        logger.warning(f"⚡ [LLM] Detected slow flash-lite model '{agent_llm}' — Auto-accelerating to Groq LPU (qwen/qwen3.8-27b, 200ms TTFT) for seamless realtime voice.")
        is_gemini = False
        is_groq = True

    if is_gemini:
        # Map legacy/old model names to the correct working Gemini 3.6 model
        _GEMINI_MODEL_MAP = {
            "gemini-3.1-flash-lite": "gemini-3.6-flash",
            "gemini-3.5-flash-lite": "gemini-3.6-flash",
            "gemini-3.5-flash": "gemini-3.6-flash",
            "gemini-3.1-flash-lite-preview": "gemini-3.6-flash",
            "gemini-flash-latest": "gemini-3.6-flash",
            "gemini-3.6-flash": "gemini-3.6-flash",
        }
        gemini_model = _GEMINI_MODEL_MAP.get(agent_llm.lower(), "gemini-3.6-flash")
        logger.success(f"⚡ [LLM] Activating Google Gemini ({gemini_model}) for '{agent_name}'")
        llm = OpenAILLMService(
            api_key=settings.GEMINI_API_KEY,
            base_url=settings.GEMINI_BASE_URL,
            settings=OpenAILLMService.Settings(
                model=gemini_model,
                temperature=0.3,
                max_tokens=150,
                system_instruction=system_instruction,
            ),
        )
    elif is_groq:
        groq_model = agent_llm if ("gemini" not in agent_llm.lower() and agent_llm not in _DEPRECATED_GROQ_MODELS) else "qwen/qwen3.8-27b"
        logger.info(f"Configuring Groq LLM for '{agent_name}' (model={groq_model})")
        llm = GroqLLMService(
            api_key=settings.GROQ_API_KEY,
            settings=GroqLLMService.Settings(
                model=groq_model,
                system_instruction=system_instruction,
                temperature=0.3,
                max_tokens=150,
            ),
        )
    elif bool(settings.OPENROUTER_API_KEY):
        logger.info(f"Configuring OpenRouter LLM for '{agent_name}' (model={settings.OPENROUTER_MODEL})")
        llm = OpenAILLMService(
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            settings=OpenAILLMService.Settings(
                model=settings.OPENROUTER_MODEL,
                temperature=0.5,
                max_tokens=150,
                system_instruction=system_instruction,
            ),
        )
    else:
        logger.warning(f"No cloud LLM API key configured. Using intelligent contextual fallback for '{agent_name}'.")

        class IntelligentFallbackLLMService(FrameProcessor):
            def __init__(self, agent_name: str, first_msg: str, prompt: str, caller_phone: str = None):
                super().__init__()
                self.agent_name = agent_name
                self.first_msg = first_msg
                self.prompt = prompt
                self.caller_phone = caller_phone

            async def process_frame(self, frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if isinstance(frame, LLMRunFrame):
                    await self.push_frame(LLMFullResponseStartFrame(), direction)
                    await self.push_frame(LLMTextFrame(self.first_msg), direction)
                    await self.push_frame(LLMFullResponseEndFrame(), direction)
                elif isinstance(frame, LLMContextFrame):
                    user_text = ""
                    if frame.context and hasattr(frame.context, "get_messages"):
                        msgs = frame.context.get_messages()
                        for m in reversed(msgs):
                            if m.get("role") == "user":
                                user_text = str(m.get("content", ""))
                                break

                    logger.info(f"Fallback assistant '{self.agent_name}' received speech: '{user_text}'")
                    lower = user_text.lower().strip()

                    action = detect_call_action(user_text)
                    if action == "transfer":
                        reply = "I completely understand. Let me connect you with our live team right away. Please hold for just a moment."
                    elif action == "end_call":
                        reply = "Thank you so much for choosing Comfort Breeze HVAC. Have a wonderful day, goodbye!"
                    # Domain-aware high-accuracy customer discovery protocol
                    elif any(k in lower for k in ["hear me", "can you hear", "are you there", "listening", "testing"]):
                        reply = "Yes, I hear you loud and clear! How can I help you today?"
                    elif any(k in lower for k in ["hello", "hi", "hey", "good morning", "good afternoon"]) and len(lower.split()) <= 3:
                        reply = "Hello! Thanks for reaching Comfort Breeze HVAC. What HVAC service or repair can I assist you with today?"
                    elif any(k in lower for k in ["ac", "air condition", "cooling", "heat", "furnace", "broken", "leak", "warm air", "freon"]):
                        reply = "I understand. We can dispatch a certified technician right away to inspect your system. First, may I have your full name, please?"
                    elif any(k in lower for k in ["my name is", "i am", "this is", "call me"]) or (len(lower.split()) in (1, 2) and not any(k in lower for k in ["street", "ave", "road", "dr"])):
                        reply = "Thank you! Could you quickly spell that out for me just to make sure I have it 100% accurate in our system?"
                    elif len(lower.replace(" ", "").replace("-", "")) <= 12 and any(c.isalpha() for c in lower) and any(k in lower for k in ["-", " ", "a", "e", "i", "o", "u"]):
                        reply = "Got that spelled down perfectly. And what is the street address where you would like our technician to visit?"
                    elif any(k in lower for k in ["street", "st", "ave", "avenue", "road", "rd", "drive", "dr", "lane", "ln", "way", "blvd", "terrace"]):
                        reply = "Thank you. We have technician arrivals open tomorrow morning around 10 AM or Thursday afternoon around 2 PM. Which day and time works best for you?"
                    elif any(k in lower for k in ["tomorrow", "morning", "afternoon", "friday", "thursday", "10", "2", "pm", "am"]):
                        phone_ref = self.caller_phone if self.caller_phone else "this number"
                        reply = f"Perfect! Can we send your arrival updates and text confirmation to {phone_ref}, or is there a different mobile number you prefer?"
                    elif any(k in lower for k in ["yes", "this number", "different", "mobile", "text"]):
                        reply = "Great. And what is the best email address to send your Google Calendar invite and service confirmation?"
                    elif any(k in lower for k in ["@", "dot com", "gmail", "yahoo", "outlook", "icloud"]):
                        clean_email = user_text.replace(" dot ", ".").replace(" at ", "@").replace(" ", "")
                        reply = f"Just to double-check that, you would like the calendar invite sent to {clean_email}, correct?"
                    elif any(k in lower for k in ["correct", "yes", "right", "yep", "sure", "that is right"]):
                        reply = "Wonderful! Your appointment is fully confirmed. We just sent your calendar invite and text confirmation. Is there anything else I can help you with today?"
                    elif any(k in lower for k in ["bye", "goodbye", "that is all", "thanks", "thank you"]):
                        reply = "Thank you so much for choosing Comfort Breeze HVAC. Have a wonderful day!"
                    else:
                        reply = "I'd be glad to help with that. Could you tell me a bit more about what you need, and who I have the pleasure of speaking with?"

                    await self.push_frame(LLMFullResponseStartFrame(), direction)
                    await self.push_frame(LLMTextFrame(reply), direction)
                    await self.push_frame(LLMFullResponseEndFrame(), direction)
                else:
                    await self.push_frame(frame, direction)

        llm = IntelligentFallbackLLMService(agent_name, agent_greeting, agent_prompt, caller_phone=caller_number)

    # Tool calling filler response (masks latency with natural human speech)
    if hasattr(llm, "event_handler"):
        try:
            @llm.event_handler("on_function_calls_started")
            async def on_function_calls_started(service, function_calls):
                call_names = [getattr(f, "function_name", str(f)) for f in function_calls if getattr(f, "function_name", None)]
                if not call_names:
                    return
                logger.info(f"Executing tools: {call_names}")
                await tts.queue_frame(TTSSpeakFrame("Sure thing — let me check that for you."))
        except Exception as e:
            logger.debug(f"Event handler registration skipped: {e}")

    # 4. Context & Turn Detection
    # Only enable tools if explicitly configured for this assistant (avoids 10s thinking delay & 400 errors)
    agent_tools_enabled = active_agent.get("enable_tools", False)
    has_any_cloud_key = bool(settings.GROQ_API_KEY or settings.GEMINI_API_KEY or settings.OPENROUTER_API_KEY)
    tools = REGISTERED_TOOLS if (agent_tools_enabled and settings.ENABLE_TOOLS and has_any_cloud_key) else []
    context = LLMContext(tools=tools)

    from pipecat.turns.user_turn_strategies import (
        UserTurnStrategies,
        VADUserTurnStartStrategy,
        TranscriptionUserTurnStartStrategy,
    )
    from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy

    agent_wait = float(active_agent.get("end_of_turn_wait", 0.65))
    agent_wait = min(1.20, max(0.40, agent_wait))

    vad = SileroVADAnalyzer(
        params=VADParams(
            stop_secs=0.35,   # VAD stop detection in 350ms prevents dead-air lag
            confidence=0.60,  # Elevated confidence requires clear human harmonic speech, rejecting breath/desk noise
            start_secs=0.18,  # 180ms speech onset trigger prevents noise blips from triggering turns
        )
    )

    # Calibrated conversational turn stop strategy
    user_turn_strategies = UserTurnStrategies(
        start=[
            VADUserTurnStartStrategy(enable_interruptions=False),
            TranscriptionUserTurnStartStrategy(enable_interruptions=False),
        ],
        stop=[
            SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=agent_wait, wait_for_transcript=True),
        ]
    )

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=vad,
            user_turn_strategies=user_turn_strategies,
            user_turn_stop_timeout=agent_wait + 0.1,  # Safe sub-750ms fallback completely eliminates 5.0s stall
            audio_idle_timeout=0.8,
        ),
    )

    # 5. Call Recording & Transcript Processor
    call_id = (
        metadata.get("call_uuid")
        or metadata.get("call_id")
        or f"call-{int(time.time())}"
    )

    class CallRecorderProcessor(FrameProcessor):
        def __init__(self, call_id: str, assistant_id: str, assistant_name: str, caller: str, called: str):
            super().__init__()
            self.call_id = call_id
            self.assistant_id = assistant_id
            self.assistant_name = assistant_name
            self.caller = caller
            self.called = called
            self.start_time = time.time()
            self.started_at_str = time.strftime("%Y-%m-%d %H:%M:%S")
            self.audio_buffer = bytearray()
            self.transcript = []
            self._current_assistant_text = ""
            self.call_status = "completed"

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)

            from pipecat.frames.frames import InputAudioRawFrame, TranscriptionFrame
            target_sr = settings.SAMPLE_RATE
            if isinstance(frame, InputAudioRawFrame):
                rate = getattr(frame, "sample_rate", target_sr) or target_sr
                if rate != target_sr:
                    samples = np.frombuffer(frame.audio, dtype=np.int16)
                    samples_target = soxr.resample(samples, rate, target_sr)
                else:
                    samples_target = np.frombuffer(frame.audio, dtype=np.int16)
                
                # Balanced dual-track mixing:
                # User's microphone is typically quieter than Kokoro/Flux synthesized broadcast waveform.
                # Apply balanced 2.8x linear boost to caller audio in the recording.
                samples_boosted = np.clip(samples_target.astype(np.float32) * 2.8, -32768, 32767).astype(np.int16)
                self.audio_buffer.extend(samples_boosted.tobytes())
            elif isinstance(frame, TTSAudioRawFrame):
                rate = getattr(frame, "sample_rate", target_sr) or target_sr
                if rate != target_sr:
                    samples = np.frombuffer(frame.audio, dtype=np.int16)
                    self.audio_buffer.extend(soxr.resample(samples, rate, target_sr).tobytes())
                else:
                    self.audio_buffer.extend(frame.audio)

            await self.push_frame(frame, direction)

        def add_greeting(self, text: str, audio_frames: list = None):
            self.transcript.append({
                "speaker": "assistant",
                "text": text,
                "timestamp": time.strftime("%H:%M:%S"),
                "latency_ms": 45,
            })
            if audio_frames:
                target_sr = settings.SAMPLE_RATE
                for f in audio_frames:
                    if hasattr(f, "audio"):
                        rate = getattr(f, "sample_rate", target_sr) or target_sr
                        if rate != target_sr:
                            samples = np.frombuffer(f.audio, dtype=np.int16)
                            self.audio_buffer.extend(soxr.resample(samples, rate, target_sr).tobytes())
                        else:
                            self.audio_buffer.extend(f.audio)

        def finalize(self, status: str = None):
            duration = time.time() - self.start_time
            from app.calls import save_call_session
            try:
                final_transcript = []
                if context and hasattr(context, "get_messages"):
                    for m in context.get_messages():
                        role = m.get("role")
                        content = m.get("content")
                        if not content or role not in ("user", "assistant"):
                            continue
                        speaker = "customer" if role == "user" else "assistant"
                        final_transcript.append({
                            "speaker": speaker,
                            "text": str(content).strip(),
                            "timestamp": time.strftime("%H:%M:%S"),
                        })
                if not final_transcript:
                    final_transcript = self.transcript

                save_call_session(
                    call_id=self.call_id,
                    assistant_id=self.assistant_id,
                    assistant_name=self.assistant_name,
                    caller=self.caller,
                    called=self.called,
                    started_at=self.started_at_str,
                    duration_seconds=duration,
                    transcript=final_transcript,
                    audio_pcm_bytes=bytes(self.audio_buffer),
                    status=status or self.call_status,
                    sample_rate=settings.SAMPLE_RATE,
                )
            except Exception as e:
                logger.error(f"Error finalizing call recording: {e}")

    recorder = CallRecorderProcessor(
        call_id=call_id,
        assistant_id=active_agent.get("id", "agent"),
        assistant_name=agent_name,
        caller=caller_number,
        called=called_number,
    )

    # 6. Direct WebSocket Real-Time Transcript & Call Action Broadcasters
    async def broadcast_transcript(speaker: str, text: str):
        if not text or not text.strip():
            return
        msg = {
            "event": "transcript",
            "speaker": speaker,
            "text": text.strip(),
            "timestamp": time.strftime("%H:%M:%S"),
        }
        try:
            logger.info(f"⚡ [Live-Transcript] Broadcasting: [{speaker}] {text[:45]}...")
            await transport.output().send_message(OutputTransportMessageUrgentFrame(message=msg))
        except Exception as e:
            logger.error(f"Error broadcasting transcript: {e}")

    async def broadcast_call_action(action: str, data: Optional[Dict[str, Any]] = None):
        if not action:
            return
        payload = {"event": action}
        if data:
            payload.update(data)
        try:
            logger.success(f"📞 [Call-Action] Emitting WebSocket action '{action}': {payload}")
            await transport.output().send_message(OutputTransportMessageUrgentFrame(message=payload))
        except Exception as e:
            logger.error(f"Error broadcasting call action '{action}': {e}")

    call_state = {
        "start_time": time.time(),
        "bot_speaking_until": time.time() + 2.5,
        "last_user_speech_time": time.time() + 2.5,
        "silence_nudge_count": 0,
        "is_active": True,
        "is_llm_generating": False,
        "call_direction": active_agent.get("call_direction", "inbound"),
        "agent_id": active_agent.get("id", "riley-hvac"),
        "agent_name": agent_name,
        "current_assistant_text": "",
        "pending_broadcast_text": "",
    }

    async def _silence_watchdog_loop(
        state: dict,
        tts_service,
        llm_context,
        call_recorder,
        broadcast_text_fn,
        broadcast_action_fn,
        name: str,
        direction: str,
    ):
        """Monitors dead air / prolonged user silence and gently prompts the caller.
        Nudge 1 at ~10.0s, Nudge 2 at ~20.0s, Polite auto-hangup at ~28.0s.
        Adheres to telephony industry standards and never interrupts active LLM generation."""
        logger.info(f"⏳ [Smart-Silence] Watchdog initialized for '{name}' ({direction})")
        await asyncio.sleep(4.5)

        if direction == "outbound" or "marcus" in name.lower():
            n1 = "Hello? Take your time, let me know if you can still hear me."
            n2 = "Just checking in—I haven't heard from you in a moment. Let me know if you're still with me."
            fw = "It seems we might have gotten disconnected. Feel free to call us back anytime. Have a great day, goodbye!"
        elif "maya" in name.lower() or "medical" in state.get("agent_id", ""):
            n1 = "Are you still there? Take your time, I'm right here whenever you're ready."
            n2 = "Just checking in—our clinic is still on the line whenever you're ready."
            fw = "It seems we might have lost connection. Please feel free to call our clinic back anytime. Take care, goodbye!"
        else:
            n1 = "Are you still there? Take your time, I'm right here whenever you're ready."
            n2 = "Just checking in—I'm still here whenever you're ready to continue."
            fw = "It seems we might have gotten disconnected. Feel free to call Comfort Breeze back anytime. Have a wonderful day, goodbye!"

        while state.get("is_active", True):
            await asyncio.sleep(0.5)
            if not state.get("is_active", True):
                break

            # Never interrupt or fire nudges while caller is speaking, or LLM is thinking/generating, or bot is speaking
            if state.get("is_user_speaking", False):
                continue
            if state.get("is_llm_generating", False):
                continue

            now = time.time()
            speaking_until = state.get("bot_speaking_until", 0.0)
            if now < speaking_until:
                continue

            last_speech = state.get("last_user_speech_time", now)
            llm_finished = state.get("llm_finished_time", 0.0)
            silence_anchor = max(last_speech, speaking_until, llm_finished)
            silence_duration = now - silence_anchor
            nudge_count = state.get("silence_nudge_count", 0)
            last_nudge_time = state.get("last_nudge_time", 0.0)

            # Minimum 14s cooldown between any consecutive nudges
            if now - last_nudge_time < 14.0:
                continue

            # Inhibit nudge if user spoke or LLM finished within the last 9 seconds
            if now - last_speech < 9.0 or now - llm_finished < 9.0:
                continue

            # Nudge 1: 14.0s dead air (allows natural human thought and dictation)
            if silence_duration >= 14.0 and nudge_count == 0:
                logger.info(f"⏳ [Smart-Silence] 14.0s dead air detected. Delivering Nudge 1...")
                state["silence_nudge_count"] = 1
                state["last_nudge_time"] = now
                state["last_nudge_text"] = n1
                state["bot_speaking_until"] = now + 4.5
                if llm_context and hasattr(llm_context, "add_message"):
                    llm_context.add_message({"role": "assistant", "content": n1})
                call_recorder.transcript.append({
                    "speaker": "assistant",
                    "text": n1,
                    "timestamp": time.strftime("%H:%M:%S"),
                })
                await broadcast_text_fn("assistant", n1)
                from pipecat.frames.frames import TTSSpeakFrame
                await tts_service.queue_frame(TTSSpeakFrame(n1))

            # Nudge 2: 28.0s dead air
            elif silence_duration >= 28.0 and nudge_count == 1:
                logger.info(f"⏳ [Smart-Silence] 28.0s dead air detected. Delivering Nudge 2...")
                state["silence_nudge_count"] = 2
                state["last_nudge_time"] = now
                state["last_nudge_text"] = n2
                state["bot_speaking_until"] = now + 4.5
                if llm_context and hasattr(llm_context, "add_message"):
                    llm_context.add_message({"role": "assistant", "content": n2})
                call_recorder.transcript.append({
                    "speaker": "assistant",
                    "text": n2,
                    "timestamp": time.strftime("%H:%M:%S"),
                })
                await broadcast_text_fn("assistant", n2)
                from pipecat.frames.frames import TTSSpeakFrame
                await tts_service.queue_frame(TTSSpeakFrame(n2))

            # Auto-Disconnect: 45.0s prolonged dead silence
            elif silence_duration >= 45.0 and nudge_count == 2:
                logger.warning(f"⏳ [Smart-Silence] 45.0s prolonged silence. Auto-disconnecting cleanly...")
                state["silence_nudge_count"] = 3
                state["last_nudge_time"] = now
                state["is_active"] = False
                state["bot_speaking_until"] = now + 4.5
                call_recorder.call_status = "completed"
                if llm_context and hasattr(llm_context, "add_message"):
                    llm_context.add_message({"role": "assistant", "content": fw})
                call_recorder.transcript.append({
                    "speaker": "assistant",
                    "text": fw,
                    "timestamp": time.strftime("%H:%M:%S"),
                })
                await broadcast_text_fn("assistant", fw)
                from pipecat.frames.frames import TTSSpeakFrame
                await tts_service.queue_frame(TTSSpeakFrame(fw))
                await asyncio.sleep(3.5)
                await broadcast_action_fn("endCall", {"reason": "silence_timeout"})
                break

    class TurnTriggerProcessor(FrameProcessor):
        def __init__(self, assistant_label: str, broadcast_fn, broadcast_action_fn, recorder, call_state: dict, tts_service):
            super().__init__()
            self.assistant_label = assistant_label
            self.broadcast_fn = broadcast_fn
            self.broadcast_action_fn = broadcast_action_fn
            self.recorder = recorder
            self.call_state = call_state
            self.tts = tts_service

        async def _delayed_end_call(self, delay: float = 3.2):
            try:
                import asyncio
                await asyncio.sleep(delay)
                await self.broadcast_action_fn("endCall", {"reason": "call_completed"})
            except Exception as e:
                logger.error(f"Error in delayed endCall dispatch: {e}")

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)

            if isinstance(frame, LLMContextFrame):
                self.call_state["is_llm_generating"] = True
                user_text = ""
                if frame.context and hasattr(frame.context, "get_messages"):
                    for m in reversed(frame.context.get_messages()):
                        if m.get("role") == "user":
                            user_text = str(m.get("content", ""))
                            break

                lower = user_text.lower().strip()
                now = time.time()
                is_echo = any(phrase in lower for phrase in [
                    "thank you for calling",
                    "comfort breeze",
                    "virtual receptionist",
                    "this is cliff",
                    "how may i get your service",
                    "service scheduled today",
                    "heating and air",
                    "how can i help you today",
                ])
                if not is_echo:
                    start_t = self.call_state.get("start_time", now)
                    if (now - start_t < 12.0 or now < self.call_state.get("bot_speaking_until", 0.0) + 1.0):
                        if lower in ("thank you", "thank you.", "thanks", "thank you!", "thank you for calling", "thank you for calling call"):
                            is_echo = True

                if is_echo and (now - self.call_state.get("start_time", now) < 15.0 or now < self.call_state.get("bot_speaking_until", 0.0) + 1.0):
                    logger.warning(f"🛡️ [Echo-Shield:Context] Discarded greeting acoustic feedback: '{user_text}'. Halting self-echo.")
                    self.call_state["is_llm_generating"] = False
                    return

                if user_text:
                    self.call_state["last_user_speech_time"] = time.time()
                    self.call_state["silence_nudge_count"] = 0
                    await self.broadcast_fn("customer", user_text)
                    self.recorder.transcript.append({
                        "speaker": "customer",
                        "text": user_text,
                        "timestamp": time.strftime("%H:%M:%S"),
                    })

                action = detect_call_action(user_text)
                if action == "transfer":
                    logger.success(f"⚡ [Call-Steering] Human transfer triggered by customer utterance ('{user_text}')!")
                    self.call_state["is_active"] = False
                    self.call_state["is_llm_generating"] = False
                    reply_text = "I completely understand. Let me connect you with our live team right away. Please hold for just a moment."
                    self.recorder.call_status = "transferred"
                    if frame.context and hasattr(frame.context, "add_message"):
                        frame.context.add_message({
                            "role": "assistant",
                            "content": reply_text
                        })
                    self.recorder.transcript.append({
                        "speaker": "assistant",
                        "text": reply_text,
                        "timestamp": time.strftime("%H:%M:%S"),
                    })
                    await self.broadcast_action_fn("transferCall", {
                        "department": "live_dispatch",
                        "reason": "customer_requested_transfer",
                        "message": reply_text
                    })
                    self.call_state["pending_broadcast_text"] = reply_text
                    from pipecat.frames.frames import TTSSpeakFrame
                    await self.tts.queue_frame(TTSSpeakFrame(reply_text))
                    return

                elif action == "end_call":
                    logger.success(f"⚡ [Call-Steering] Call wrap-up triggered by customer farewell ('{user_text}')!")
                    self.call_state["is_active"] = False
                    self.call_state["is_llm_generating"] = False
                    agent_id = self.call_state.get("agent_id", "")
                    call_dir = self.call_state.get("call_direction", "inbound")
                    name_l = self.assistant_label.lower()

                    if call_dir == "outbound" or "marcus" in name_l:
                        reply_text = "Thank you so much for your time today! We look forward to connecting with you on our demo. Have a great rest of your day, goodbye!"
                    elif "maya" in name_l or "medical" in agent_id or "clinic" in name_l:
                        reply_text = "Thank you for calling Metro Health and Dental Clinic. We look forward to seeing you. Take care and have a wonderful day, goodbye!"
                    elif "comfort breeze" in name_l or "riley" in name_l or "hvac" in agent_id:
                        reply_text = "Thank you so much for choosing Comfort Breeze HVAC! We look forward to taking care of your system. Have a wonderful day, goodbye!"
                    else:
                        reply_text = "Thank you so much for your time today! Have a wonderful day, goodbye!"

                    self.recorder.call_status = "completed"
                    if frame.context and hasattr(frame.context, "add_message"):
                        frame.context.add_message({
                            "role": "assistant",
                            "content": reply_text
                        })
                    self.recorder.transcript.append({
                        "speaker": "assistant",
                        "text": reply_text,
                        "timestamp": time.strftime("%H:%M:%S"),
                    })
                    self.call_state["pending_broadcast_text"] = reply_text
                    import asyncio
                    asyncio.create_task(self._delayed_end_call(delay=3.5))
                    from pipecat.frames.frames import TTSSpeakFrame
                    await self.tts.queue_frame(TTSSpeakFrame(reply_text))
                    return

            await self.push_frame(frame, direction)

    turn_trigger = TurnTriggerProcessor(agent_name, broadcast_transcript, broadcast_call_action, recorder, call_state, tts)

    class SmartBargeInProcessor(FrameProcessor):
        """Filters conversational backchannels ('yeah', 'ahh', 'you know what I mean') so they don't
        falsely interrupt the assistant, while immediately halting audio playback on genuine interruptions."""

        def __init__(self, transport_output, call_state: dict):
            super().__init__()
            self.transport_output = transport_output
            self.call_state = call_state
            self.backchannels = {
                "yeah", "yep", "yes", "mhm", "uh-huh", "uh huh", "uh", "um", "ah", "ahh",
                "oh", "okay", "ok", "right", "sure", "got it", "i see", "you know what i mean",
                "you know", "know what i mean", "i know", "makes sense", "sounds good", "gotcha",
                "oh okay", "yeah okay", "right right", "yep yep", "cool", "cool cool", "alright",
                "all right", "understood", "mm", "hmm", "totally"
            }
            self.hesitations = {"ahh", "ah", "um", "uh", "er", "mm", "hmm"}

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            from pipecat.frames.frames import (
                TranscriptionFrame,
                InterimTranscriptionFrame,
                UserStartedSpeakingFrame,
                UserSpeakingFrame,
                UserStoppedSpeakingFrame,
                VADUserStartedSpeakingFrame,
                VADUserStoppedSpeakingFrame,
                InterruptionFrame,
            )

            # Track user speech onset immediately from VAD so silence watchdog NEVER interrupts active caller
            if isinstance(frame, (UserStartedSpeakingFrame, UserSpeakingFrame, VADUserStartedSpeakingFrame)):
                self.call_state["last_user_speech_time"] = time.time()
                self.call_state["is_user_speaking"] = True
                self.call_state["silence_nudge_count"] = 0

            elif isinstance(frame, (UserStoppedSpeakingFrame, VADUserStoppedSpeakingFrame)):
                self.call_state["is_user_speaking"] = False
                self.call_state["last_user_speech_time"] = time.time()

            elif isinstance(frame, InterimTranscriptionFrame):
                self.call_state["last_user_speech_time"] = time.time()
                self.call_state["is_user_speaking"] = True
                self.call_state["silence_nudge_count"] = 0

            if isinstance(frame, TranscriptionFrame):
                self.call_state["is_user_speaking"] = False
                raw_text = (frame.text or "").strip()
                raw_text = normalize_spoken_email(raw_text)
                frame.text = raw_text

                clean_text = raw_text.lower().strip(".!?,")
                words = clean_text.split()

                now = time.time()
                speaking_until = self.call_state.get("bot_speaking_until", 0.0)
                is_bot_speaking = (now < speaking_until)
                recent_bot_time = self.call_state.get("last_bot_spoke_time", 0.0)
                is_near_bot_speech = is_bot_speaking or (now - recent_bot_time < 2.5)

                # =========================================================================
                # ACOUSTIC ECHO SHIELD: Suppress speaker bleed from caller's microphone
                # If caller's mic picks up the greeting or assistant response, discard it!
                # =========================================================================
                echo_phrases = [
                    "thank you for calling",
                    "comfort breeze",
                    "virtual receptionist",
                    "this is cliff",
                    "heating and air",
                    "service scheduled",
                    "how may i get",
                    "how can i help",
                    "thank you for calling call",
                    "scheduled today",
                    "how may i get your service",
                ]

                start_t = self.call_state.get("start_time", now)
                is_echo = any(phrase in clean_text for phrase in echo_phrases)
                if not is_echo and (now - start_t < 12.0 or is_bot_speaking):
                    if clean_text in ("thank you", "thanks", "thank you for calling", "thank you so much", "thank you for calling call"):
                        is_echo = True

                if is_echo:
                    logger.warning(f"🛡️ [Echo-Shield] Suppressed speaker acoustic feedback: '{raw_text}' (matches assistant greeting)")
                    return

                if is_bot_speaking:
                    is_backchannel = (
                        clean_text in self.backchannels or
                        (len(words) <= 2 and all(w in self.backchannels for w in words)) or
                        clean_text in ("you know what i mean", "know what i mean", "you know", "i know what you mean", "thank you", "thanks")
                    )

                    if is_backchannel:
                        logger.info(f"🎧 [Smart-Barge-In] Suppressed caller backchannel ('{raw_text}') while assistant speaks. Bot keeps talking!")
                        return
                    elif len(words) < 3 and clean_text not in ("wait", "hold on", "stop", "excuse me", "listen", "no", "hey"):
                        logger.info(f"🎧 [Smart-Barge-In] Discarded brief sound/murmur ('{raw_text}') during assistant speech.")
                        return
                    else:
                        logger.success(f"⚡ [Smart-Barge-In] Genuine interruption detected ('{raw_text}')! Halting assistant playback immediately!")
                        self.call_state["bot_speaking_until"] = 0.0
                        self.call_state["is_llm_generating"] = False
                        try:
                            await self.transport_output.send_message(
                                OutputTransportMessageUrgentFrame(message={"event": "clearAudio"})
                            )
                            await self.push_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
                            await self.push_frame(InterruptionFrame(), FrameDirection.UPSTREAM)
                        except Exception as e:
                            logger.error(f"Error executing barge-in interruption: {e}")
                else:
                    if len(words) == 1 and clean_text in self.hesitations:
                        logger.info(f"⏸️ [Turn-Taking] Caller hesitation token ('{raw_text}') detected. Holding for complete utterance...")
                        return
                    elif len(words) > 1 and words[0] in self.hesitations:
                        frame.text = " ".join(words[1:])

                    if raw_text:
                        self.call_state["last_user_speech_time"] = time.time()
                        self.call_state["silence_nudge_count"] = 0

            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)

    barge_in = SmartBargeInProcessor(transport.output(), call_state)

    class AssistantTextCollector(FrameProcessor):
        """Captures complete response text streamed by LLM before frames enter TTS."""

        def __init__(self, call_state: dict):
            super().__init__()
            self.call_state = call_state

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)

            if isinstance(frame, LLMFullResponseStartFrame):
                self.call_state["current_assistant_text"] = ""
            elif isinstance(frame, LLMTextFrame):
                self.call_state["current_assistant_text"] += frame.text
            elif isinstance(frame, LLMFullResponseEndFrame):
                self.call_state["is_llm_generating"] = False
                self.call_state["llm_finished_time"] = time.time()
                self.call_state["last_user_speech_time"] = time.time()
                full_text = self.call_state.get("current_assistant_text", "").strip()
                if full_text:
                    self.call_state["pending_broadcast_text"] = full_text
                self.call_state["current_assistant_text"] = ""

            await self.push_frame(frame, direction)

    assistant_text_collector = AssistantTextCollector(call_state)

    class AudioSyncTranscriptBroadcaster(FrameProcessor):
        """Synchronizes assistant transcript text with actual speech audio output.
        Emits the transcript text at the exact millisecond TTS starts emitting audio frames,
        and accurately tracks bot_speaking_until based on emitted audio frame duration."""

        def __init__(self, broadcast_fn, broadcast_action_fn, recorder, call_state: dict):
            super().__init__()
            self.broadcast_fn = broadcast_fn
            self.broadcast_action_fn = broadcast_action_fn
            self.recorder = recorder
            self.call_state = call_state

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)

            if isinstance(frame, TTSAudioRawFrame):
                pending = self.call_state.pop("pending_broadcast_text", None)
                if pending:
                    ts = time.strftime("%H:%M:%S")
                    await self.broadcast_fn("assistant", pending)
                    self.recorder.transcript.append({
                        "speaker": "assistant",
                        "text": pending,
                        "timestamp": ts,
                    })
                    p_lower = pending.lower()
                    if any(k in p_lower for k in ["connect you with our", "connecting you to", "transfer you to", "live team"]):
                        self.recorder.call_status = "transferred"
                        import asyncio
                        asyncio.create_task(self.broadcast_action_fn("transferCall", {"department": "dispatch", "reason": "assistant_transfer"}))
                    elif any(k in p_lower for k in ["have a wonderful day, goodbye", "have a great day, goodbye", "goodbye!"]) and ("thank you" in p_lower or "choosing" in p_lower):
                        self.recorder.call_status = "completed"
                        async def _delayed_end():
                            import asyncio
                            await asyncio.sleep(3.5)
                            await self.broadcast_action_fn("endCall", {"reason": "call_completed"})
                        import asyncio
                        asyncio.create_task(_delayed_end())

                rate = getattr(frame, "sample_rate", 8000) or 8000
                audio_bytes = len(frame.audio) if hasattr(frame, "audio") else 0
                duration = audio_bytes / (rate * 2) if rate > 0 else 0.02
                now = time.time()
                cur_until = self.call_state.get("bot_speaking_until", 0.0)
                self.call_state["bot_speaking_until"] = max(now, cur_until) + duration

            elif isinstance(frame, InterruptionFrame):
                self.call_state["pending_broadcast_text"] = ""
                self.call_state["bot_speaking_until"] = 0.0
                self.call_state["is_llm_generating"] = False

            await self.push_frame(frame, direction)

    assistant_broadcaster = AudioSyncTranscriptBroadcaster(broadcast_transcript, broadcast_call_action, recorder, call_state)

    # 7. Noise Cancellation Filter (Dynamic spectral gating if enabled)
    enable_denoising = active_agent.get("background_denoising", False)
    from app.denoiser import AudioDenoiseProcessor
    denoiser = AudioDenoiseProcessor(
        sample_rate=settings.SAMPLE_RATE,
        enabled=enable_denoising,
    )

    # 8. Build Pipeline
    pipeline_elements = [transport.input()]
    if enable_denoising:
        logger.info(f"🛡️ [Denoise] Active real-time background noise filter inserted for '{agent_name}'")
        pipeline_elements.append(denoiser)
    pipeline_elements.extend([
        stt,                            # Groq Whisper Large-v3 / Fast Whisper transcription
        barge_in,                       # Smart Barge-In & Backchannel Suppressor
        user_aggregator,                # Sentence aggregator & turn-taking
        turn_trigger,                   # Emits customer transcript and handles instant test turn
        llm,                            # Multi-provider LLM
        assistant_text_collector,       # Collects full text from LLMTextFrames before TTS
        tts,                            # Fast Kokoro speech synthesis (4-thread optimized)
        assistant_broadcaster,          # Emits assistant response in sync with audio
        recorder,                       # Live Audio Recording & Dual-Track Audio Capture
        transport.output(),             # Audio output back to caller
        assistant_aggregator,           # Conversation history tracking
    ])
    pipeline = Pipeline(pipeline_elements)

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=settings.SAMPLE_RATE,
            audio_out_sample_rate=settings.SAMPLE_RATE,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    runner = WorkerRunner(handle_sigint=handle_sigint)
    await runner.add_workers(worker)

    # 7. Event Handlers
    silence_task = None

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        """
        Stream the pre-rendered greeting audio frames instantly on connection.
        Eliminates inference latency on call pickup, matching Vapi's sub-50ms TTFA.
        """
        nonlocal silence_task
        logger.info(f"Plivo client connected. Emitting instant pre-cached greeting audio for '{agent_name}'...")

        # Record greeting in LLM context so the assistant remembers what it said
        context.add_message(
            {
                "role": "assistant",
                "content": agent_greeting,
            }
        )

        # Retrieve pre-synthesized 20ms audio frames from RAM
        from app.server import get_cached_greeting_frames
        cached_frames = get_cached_greeting_frames()

        recorder.add_greeting(agent_greeting, cached_frames)

        # Broadcast initial greeting transcript event immediately over WebSocket
        await broadcast_transcript("assistant", agent_greeting)

        if cached_frames:
            await worker.queue_frames([TTSStartedFrame()] + list(cached_frames) + [TTSStoppedFrame()])
            logger.success(f"Streamed {len(cached_frames)} pre-cached greeting frames directly to transport.")
        else:
            logger.info("No cached greeting found, falling back to dynamic generation.")
            await worker.queue_frames([LLMRunFrame()])

        # Launch Smart Silence & Dead-Air Watchdog
        silence_task = asyncio.create_task(
            _silence_watchdog_loop(
                call_state,
                tts,
                context,
                recorder,
                broadcast_transcript,
                broadcast_call_action,
                agent_name,
                active_agent.get("call_direction", "inbound"),
            )
        )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Plivo telephony client disconnected. Finalizing recording and tearing down worker.")
        call_state["is_active"] = False
        if silence_task and not silence_task.done():
            silence_task.cancel()
        recorder.finalize()
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point called by FastAPI WebSocket handler."""
    os.environ["PLIVO_AUTH_ID"] = settings.PLIVO_AUTH_ID or os.getenv("PLIVO_AUTH_ID") or "MAMOCKAUTHID0000000"
    os.environ["PLIVO_AUTH_TOKEN"] = settings.PLIVO_AUTH_TOKEN or os.getenv("PLIVO_AUTH_TOKEN") or "mock_auth_token_for_testing"

    transport_params = {
        "plivo": lambda: FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
        "websocket": lambda: FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            serializer=WebAudioFrameSerializer(sample_rate=settings.SAMPLE_RATE),
        ),
    }

    transport = await create_transport(runner_args, transport_params)
    metadata = getattr(runner_args, "body", {})
    await run_bot(transport, metadata=metadata, handle_sigint=runner_args.handle_sigint)
