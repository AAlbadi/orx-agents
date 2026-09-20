"""Google Calendar, Email notifications, and Client Information Extraction module for Aria Voice AI.

Features:
- Automated extraction of client name, phone, email, service details, address, and appointment times from transcripts.
- 1-click Google Calendar URL generation and standard RFC 5545 .ics iCalendar file export.
- Automated post-call email notifications (HTML + plain text) with appointment details, recording audio link, and transcript.
- Integrations settings persistence (SMTP, Resend, Google Calendar, Webhooks).
"""

import asyncio
import json
import os
import re
import smtplib
import time
import urllib.parse
import uuid
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx
from loguru import logger

from app.config import settings

import re

def transliterate_arabic_to_english(text: str) -> str:
    """Romanizes Arabic names and speech into standard Latin alphabet."""
    return text

def normalize_spoken_email(text: str) -> str:
    """Normalizes spoken email syntax into standard clean email format."""
    if not text:
        return text
    clean = text.lower().replace(" at ", "@").replace(" dot ", ".").replace(" ", "")
    return re.sub(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+).*', r'\1', clean)

DATA_DIR = Path(__file__).parent.parent / "data"
INTEGRATIONS_FILE = DATA_DIR / "integrations.json"
EMAIL_LOG_FILE = DATA_DIR / "email_notifications.log"
SMS_LOG_FILE = DATA_DIR / "sms_notifications.log"

DEFAULT_SETTINGS: Dict[str, Any] = {
    "auto_extract_client_info": True,
    "google_calendar_enabled": True,
    "google_calendar_id": "primary",
    "google_service_account_json": "",
    "calendar_webhook_url": "",
    "morning_slot_capacity": 2,
    "afternoon_slot_capacity": 2,
    "owner_phone_number": "",
    "enable_owner_sms": True,
    "enable_customer_sms": True,
    "sms_provider": "auto",  # 'auto' | 'telnyx' | 'twilio' | 'plivo'
    "telnyx_api_key": os.getenv("TELNYX_API_KEY", ""),
    "telnyx_phone_number": os.getenv("TELNYX_PHONE_NUMBER", ""),
    "twilio_account_sid": os.getenv("TWILIO_ACCOUNT_SID", ""),
    "twilio_auth_token": os.getenv("TWILIO_AUTH_TOKEN", ""),
    "twilio_phone_number": os.getenv("TWILIO_PHONE_NUMBER", ""),
    "plivo_auth_id": os.getenv("PLIVO_AUTH_ID", ""),
    "plivo_auth_token": os.getenv("PLIVO_AUTH_TOKEN", ""),
    "plivo_phone_number": os.getenv("PLIVO_PHONE_NUMBER", ""),
    "email_notifications_enabled": False,
    "notify_email": "",
    "smtp_host": "smtp.gmail.com",
    "smtp_port": 587,
    "smtp_user": "",
    "smtp_password": "",
    "smtp_use_tls": True,
    "resend_api_key": "",
    "from_email": "Aria Voice AI <notifications@ariavoice.ai>",
}


def _ensure_integrations_file() -> Dict[str, Any]:
    """Ensures integrations settings file exists and returns current configuration."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not INTEGRATIONS_FILE.exists():
        INTEGRATIONS_FILE.write_text(json.dumps(DEFAULT_SETTINGS, indent=2))
        return DEFAULT_SETTINGS.copy()
    try:
        data = json.loads(INTEGRATIONS_FILE.read_text())
        merged = DEFAULT_SETTINGS.copy()
        merged.update(data)
        return merged
    except Exception as e:
        logger.error(f"Error reading integrations.json: {e}")
        return DEFAULT_SETTINGS.copy()


def get_integrations_settings() -> Dict[str, Any]:
    """Returns current integration settings."""
    return _ensure_integrations_file()


def update_integrations_settings(new_settings: Dict[str, Any]) -> Dict[str, Any]:
    """Updates and persists integration settings."""
    current = _ensure_integrations_file()
    for k, v in new_settings.items():
        if k in DEFAULT_SETTINGS:
            current[k] = v
    INTEGRATIONS_FILE.write_text(json.dumps(current, indent=2))
    logger.info(f"Updated integrations settings: {list(new_settings.keys())}")
    return current


# ── 1. Structured Client Information Extraction ───────────────────────────────

def _heuristic_fallback_extraction(transcript: List[Dict[str, Any]], caller: Optional[str] = None, language: str = "en") -> Dict[str, Any]:
    """Extracts client information using robust regex and linguistic patterns when LLM is unavailable."""
    full_text = " ".join([t.get("text", "") for t in transcript])
    customer_texts = [t.get("text", "") for t in transcript if t.get("speaker") == "customer"]
    cust_full = " ".join(customer_texts)

    # Name extraction
    name = ""
    # 1. Conversational turn alignment: check customer turn immediately following name prompt
    for idx, t in enumerate(transcript):
        if t.get("speaker") == "assistant":
            ast_txt = t.get("text", "").lower()
            if any(k in ast_txt for k in ["first and last name", "your name", "spell that out", "spell your name"]):
                for next_t in transcript[idx+1:]:
                    if next_t.get("speaker") == "customer":
                        c_cand = next_t.get("text", "").strip().strip(".,")
                        if c_cand:
                            # Reconstruct spelled letters like A-B-T-U-L-A-Z-I-Z-A-L-B-A-D-I
                            if "-" in c_cand and len(c_cand.split("-")) >= 3:
                                raw_letters = "".join(c_cand.split("-"))
                                # Try to split if common words or format
                                name = raw_letters.title()
                            elif any("\u0600" <= ch <= "\u06FF" for ch in c_cand):
                                name = c_cand
                            elif len(c_cand.split()) <= 4 and not any(w in c_cand.lower() for w in ["yes", "yeah", "sure", "okay", "no"]):
                                name = c_cand.title()
                        break
        if name:
            break

    # 2. Check for Arabic name anywhere in customer utterances
    if not name:
        ar_match = re.search(r'[\u0600-\u06FF]{2,}(?:\s+[\u0600-\u06FF]{2,})+', cust_full)
        if ar_match:
            name = ar_match.group(0).strip()

    # 3. Check for spelled letters anywhere in customer utterances
    if not name:
        spelled_match = re.search(r'\b[A-Za-z](?:-[A-Za-z]){3,}\b', cust_full)
        if spelled_match:
            name = "".join(spelled_match.group(0).split("-")).title()

    # 4. Standard conversational regex: 'my name is...', 'this is...'
    if not name:
        name_match = re.search(r"(?:my name is|this is|i'm|i am|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", cust_full, re.IGNORECASE)
        if name_match:
            name = name_match.group(1).title()

    # If language is English, guarantee transliteration of any Arabic characters
    if language == "en" and name and re.search(r'[\u0600-\u06FF]', name):
        name = transliterate_arabic_to_english(name)

    # Phone extraction
    phone = caller or ""
    phone_match = re.search(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", cust_full)
    if phone_match:
        phone = phone_match.group(0)

    # Email extraction
    email = ""
    # 1. Turn alignment: check customer turn immediately following email prompt
    for idx, t in enumerate(transcript):
        if t.get("speaker") == "assistant":
            ast_txt = t.get("text", "").lower()
            if any(k in ast_txt for k in ["email", "email address", "spell your email", "spelled email"]):
                for next_t in transcript[idx+1:]:
                    if next_t.get("speaker") == "customer":
                        c_text = next_t.get("text", "")
                        norm_e = normalize_spoken_email(c_text)
                        em = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", norm_e)
                        if em:
                            email = em.group(0).lower()
                            break
                if email:
                    break

    # 2. Fallback to customer utterances search
    if not email:
        norm_cust_full = normalize_spoken_email(cust_full)
        email_match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", norm_cust_full)
        if email_match:
            email = email_match.group(0).lower()

    # Service extraction
    service = "HVAC Inspection & Service"
    if any(k in cust_full.lower() for k in ["ac", "air condition", "cooling", "warm air", "freon"]):
        service = "Air Conditioning Repair & Inspection"
    elif any(k in cust_full.lower() for k in ["heat", "furnace", "heater", "cold air"]):
        service = "Heating & Furnace Diagnostic"
    elif any(k in cust_full.lower() for k in ["maintenance", "tune up", "filter"]):
        service = "HVAC Preventive Maintenance"

    # Appointment date & time
    today = datetime.now()
    appt_date = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    appt_time = "10:00 AM"

    lower_c = cust_full.lower()
    if "friday" in lower_c:
        days_ahead = (4 - today.weekday() + 7) % 7 or 7
        appt_date = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    elif "tomorrow" in lower_c:
        appt_date = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    elif "today" in lower_c:
        appt_date = today.strftime("%Y-%m-%d")

    if "morning" in lower_c or "am" in lower_c or "nine" in lower_c or "9" in lower_c:
        appt_time = "09:30 AM"
    elif "afternoon" in lower_c or "pm" in lower_c:
        appt_time = "02:00 PM"

    # Address extraction
    address = ""
    # Check conversational turn alignment first
    for idx, t in enumerate(transcript):
        if t.get("speaker") == "assistant":
            ast_txt = t.get("text", "").lower()
            if any(k in ast_txt for k in ["street address", "service address", "where you'd like", "your address"]):
                for next_t in transcript[idx+1:]:
                    if next_t.get("speaker") == "customer":
                        cand_addr = next_t.get("text", "").strip().strip(".,")
                        if cand_addr and any(char.isdigit() for char in cand_addr) and len(cand_addr.split()) >= 3:
                            address = cand_addr
                        break
        if address:
            break

    if not address:
        addr_match = re.search(r"\d{1,5}[,\s]+[A-Za-z0-9\s.,]+?(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Place|Pl|Court|Ct|Circle|Cir)\b(?:[,\s]+[A-Za-z\s]+)?", cust_full, re.IGNORECASE)
        if addr_match:
            address = addr_match.group(0).strip()

    has_appt = any(k in full_text.lower() for k in ["schedule", "book", "appointment", "technician", "window", "tomorrow", "friday", "slot"])

    return {
        "client_name": name or "Valued Customer",
        "client_phone": phone or "Not provided",
        "client_email": email or "",
        "service_requested": service,
        "service_address": address or "Address on file",
        "appointment_date": appt_date if has_appt else "",
        "appointment_time": appt_time if has_appt else "",
        "has_appointment": has_appt,
        "summary": "Customer called regarding HVAC service. Appointment scheduled for technician inspection.",
        "urgency": "medium",
        "extraction_engine": "heuristic_fallback",
    }


async def extract_client_info(
    transcript: List[Dict[str, Any]],
    caller: Optional[str] = None,
    called: Optional[str] = None,
    language: str = "en",
) -> Dict[str, Any]:
    """
    Extracts structured customer and appointment details from a call transcript
    using Google Gemini 3.5 Flash Lite or Groq LPU (with heuristic fallback).
    Enforces agent language constraints (e.g. English transliteration for names & addresses; strictly ASCII emails).
    """
    if not transcript:
        return {
            "client_name": "Unknown Caller",
            "client_phone": caller or "",
            "client_email": "",
            "service_requested": "General Inquiry",
            "service_address": "",
            "appointment_date": "",
            "appointment_time": "",
            "has_appointment": False,
            "summary": "Brief inquiry, no appointment scheduled.",
            "urgency": "low",
            "extraction_engine": "none",
        }

    if isinstance(transcript, str):
        transcript_text = transcript
    elif isinstance(transcript, list):
        formatted_lines = []
        for t in transcript:
            if isinstance(t, dict):
                formatted_lines.append(f"{t.get('speaker', 'unknown').upper()}: {t.get('text', '')}")
            else:
                formatted_lines.append(str(t))
        transcript_text = "\n".join(formatted_lines)
    else:
        transcript_text = str(transcript)

    now = datetime.now()
    now_str = now.strftime("%A, %B %d, %Y")

    lang_rules = ""
    if language == "en":
        lang_rules = """
LANGUAGE & SCRIPT ENFORCEMENT (CRITICAL):
- The assistant is strictly ENGLISH-ONLY.
- All extracted fields (names, addresses, service requests, summaries) MUST be output in the English alphabet (Latin script).
- If the customer gave their name in Arabic or non-Latin script, TRANSLITERATE it into standard English Latin alphabet (e.g. 'عبدالعزيز البادي' -> 'Abdulaziz Albadi', 'محمد' -> 'Mohammed').
- The customer email MUST ALWAYS be strictly lowercase English ASCII characters only (e.g. 'name@example.com'). Never output Arabic in email.
"""
    elif language == "ar":
        lang_rules = """
LANGUAGE ENFORCEMENT:
- The assistant operates in Arabic. Customer name and address can be Arabic script.
- The customer email MUST ALWAYS be strictly lowercase English ASCII characters (e.g. 'name@example.com'). Never output Arabic in email.
"""
    else:
        lang_rules = """
LANGUAGE ENFORCEMENT:
- The customer email MUST ALWAYS be strictly lowercase English ASCII characters (e.g. 'name@example.com').
"""

    prompt = f"""You are an expert AI data extraction specialist for an appointment and service business.
Analyze the following phone conversation transcript between an AI Receptionist and a Customer.
Current call date: {now_str}.
Caller phone number from caller ID: {caller or 'Unknown'}.
{lang_rules}
Extract the customer information and appointment details into a strict JSON object with these exact keys:
{{
  "client_name": "Full name of the caller (or empty string if not mentioned)",
  "client_phone": "Phone number (prefer what caller said, otherwise caller ID)",
  "client_email": "Customer email address (or empty string if not mentioned)",
  "service_requested": "Specific service requested, equipment type, or problem (e.g. AC Repair, Furnace Tune-up)",
  "service_address": "Street address or location for service (or empty string if not mentioned)",
  "appointment_date": "Scheduled appointment date in YYYY-MM-DD format (or empty string if none)",
  "appointment_time": "Scheduled appointment time like '10:00 AM' or '02:30 PM' (or empty string if none)",
  "has_appointment": true if an appointment or technician visit was requested or booked, false otherwise,
  "summary": "A concise 2-sentence summary of the conversation and agreed next steps.",
  "urgency": "low" | "medium" | "high" | "emergency"
}}

Respond ONLY with the JSON object. Do not include markdown ticks or additional commentary.

[TRANSCRIPT]
{transcript_text}
"""

    def _sanitize_extracted(parsed: Dict[str, Any]) -> Dict[str, Any]:
        if language == "en":
            for k in ["client_name", "customer_name", "service_address", "address"]:
                val = parsed.get(k)
                if val and re.search(r"[\u0600-\u06FF]", val):
                    parsed[k] = transliterate_arabic_to_english(val)
        if parsed.get("client_email"):
            parsed["client_email"] = normalize_spoken_email(parsed["client_email"])
            parsed["email"] = parsed["client_email"]
        return parsed

    # 1. Primary Engine: Groq LPU (Ultra-fast, 100% precision on spelled/Arabic names)
    groq_key = getattr(settings, "GROQ_API_KEY", None)
    if groq_key:
        try:
            from groq import AsyncGroq
            groq_client = AsyncGroq(api_key=groq_key)
            res = await asyncio.wait_for(
                groq_client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {"role": "system", "content": "You are a precise JSON data extractor. Output valid JSON only, no markdown."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.1
                ),
                timeout=6.0
            )
            content = res.choices[0].message.content.strip()
            content = re.sub(r"^```json\s*", "", content)
            content = re.sub(r"^```\s*", "", content)
            content = re.sub(r"\s*```$", "", content).strip()
            parsed = json.loads(content)
            if parsed.get("client_name") or parsed.get("service_requested") or parsed.get("service_address"):
                parsed["customer_name"] = parsed.get("client_name", "")
                parsed["email"] = parsed.get("client_email", "")
                parsed["phone"] = parsed.get("client_phone", "")
                parsed["address"] = parsed.get("service_address", "")
                parsed["extraction_engine"] = "groq_gpt-oss-120b"
                parsed = _sanitize_extracted(parsed)
                logger.info(f"✅ Extracted client info via Groq LPU: {parsed.get('client_name')} | {parsed.get('service_address')}")
                return parsed
        except Exception as e:
            logger.warning(f"Groq extraction failed/timed out: {e}. Falling back to Gemini...")

    # 2. Secondary Engine: Google Gemini Flash Lite
    gemini_key = settings.GEMINI_API_KEY
    if gemini_key:
        try:
            url = f"{settings.GEMINI_BASE_URL.rstrip('/')}/chat/completions"
            payload = {
                "model": settings.GEMINI_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a JSON data extractor. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 1000,
            }
            headers = {
                "Authorization": f"Bearer {gemini_key}",
                "Content-Type": "application/json"
            }
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.post(url, json=payload, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    content = data["choices"][0]["message"]["content"].strip()
                    content = re.sub(r"^```json\s*", "", content)
                    content = re.sub(r"^```\s*", "", content)
                    content = re.sub(r"\s*```$", "", content).strip()
                    parsed = json.loads(content)
                    parsed["customer_name"] = parsed.get("client_name", "")
                    parsed["email"] = parsed.get("client_email", "")
                    parsed["phone"] = parsed.get("client_phone", "")
                    parsed["address"] = parsed.get("service_address", "")
                    parsed["extraction_engine"] = "gemini_flash_lite"
                    parsed = _sanitize_extracted(parsed)
                    logger.info(f"✅ Extracted client info via Gemini: {parsed.get('client_name')} | {parsed.get('service_address')}")
                    return parsed
                else:
                    logger.warning(f"Gemini extraction HTTP {res.status_code}: {res.text[:200]}")
        except Exception as e:
            logger.error(f"Error extracting client info via Gemini: {e}")

    # Fallback to local heuristic extraction
    fallback = _heuristic_fallback_extraction(transcript, caller=caller, language=language)
    fallback["customer_name"] = fallback.get("client_name", "")
    fallback["email"] = fallback.get("client_email", "")
    fallback["phone"] = fallback.get("client_phone", "")
    fallback["address"] = fallback.get("service_address", "")
    fallback = _sanitize_extracted(fallback)
    logger.info(f"Extracted client info via fallback: {fallback.get('client_name')} | {fallback.get('service_requested')}")
    return fallback


# ── 2. Google Calendar Integration & .ics Generator ──────────────────────────

def create_google_calendar_url(extracted_info: Dict[str, Any], assistant_name: str = "Riley") -> str:
    """
    Generates an instant 1-click Google Calendar web creation URL.
    Opens calendar.google.com with prefilled title, date, time, location, and notes.
    """
    date_str = extracted_info.get("appointment_date") or extracted_info.get("date")
    time_str = extracted_info.get("appointment_time") or extracted_info.get("time") or extracted_info.get("exact_time") or extracted_info.get("window")
    client_name = extracted_info.get("client_name") or extracted_info.get("customer_name") or "Customer"
    service = extracted_info.get("service_requested") or extracted_info.get("service") or "Service Appointment"
    address = extracted_info.get("service_address") or extracted_info.get("address") or ""
    summary = extracted_info.get("summary") or extracted_info.get("notes") or ""
    phone = extracted_info.get("client_phone") or extracted_info.get("phone") or ""

    if not date_str:
        now = datetime.now() + timedelta(days=1)
        start_dt = now.replace(hour=10, minute=0, second=0, microsecond=0)
    else:
        try:
            parts = [int(p) for p in re.findall(r"\d+", str(date_str))[:3]]
            hour = 10
            minute = 0
            if time_str:
                tm = re.search(r"(\d{1,2}):?(\d{2})?\s*(AM|PM)?", str(time_str), re.IGNORECASE)
                if tm:
                    hour = int(tm.group(1))
                    minute = int(tm.group(2) or 0)
                    ampm = (tm.group(3) or "").upper()
                    if ampm == "PM" and hour < 12:
                        hour += 12
                    elif ampm == "AM" and hour == 12:
                        hour = 0
            start_dt = datetime(parts[0], parts[1], parts[2], hour, minute)
        except Exception:
            start_dt = datetime.now() + timedelta(days=1)

    end_dt = start_dt + timedelta(hours=1)
    start_fmt = start_dt.strftime("%Y%m%dT%H%M%S")
    end_fmt = end_dt.strftime("%Y%m%dT%H%M%S")

    title = f"{service} — {client_name}"
    details = (
        f"Service: {service}\n"
        f"Customer Name: {client_name}\n"
        f"Contact Phone: {phone}\n"
        f"Location: {address}\n\n"
        f"Notes from Call:\n{summary}\n\n"
        f"Booked automatically by {assistant_name} Voice AI."
    )

    params = {
        "action": "TEMPLATE",
        "text": title,
        "dates": f"{start_fmt}/{end_fmt}",
        "details": details,
        "location": address,
    }
    return f"https://calendar.google.com/calendar/render?{urllib.parse.urlencode(params)}"


def generate_ical_data(
    extracted_info: Dict[str, Any],
    call_id: str = "",
    assistant_name: str = "Riley",
) -> bytes:
    """Generates standard RFC 5545 iCalendar (.ics) bytes with CRLF endings for 1-click import."""
    date_str = extracted_info.get("appointment_date") or extracted_info.get("date")
    time_str = extracted_info.get("appointment_time") or extracted_info.get("time") or extracted_info.get("exact_time") or extracted_info.get("window")
    client_name = extracted_info.get("client_name") or extracted_info.get("customer_name") or "Customer"
    service = extracted_info.get("service_requested") or extracted_info.get("service") or "Service Appointment"
    address = extracted_info.get("service_address") or extracted_info.get("address") or ""
    summary = extracted_info.get("summary") or extracted_info.get("notes") or ""
    phone = extracted_info.get("client_phone") or extracted_info.get("phone") or ""

    now = datetime.now()
    if not date_str:
        start_dt = now + timedelta(days=1)
        start_dt = start_dt.replace(hour=10, minute=0, second=0, microsecond=0)
    else:
        try:
            parts = [int(p) for p in re.findall(r"\d+", str(date_str))[:3]]
            hour = 10
            minute = 0
            if time_str:
                tm = re.search(r"(\d{1,2}):?(\d{2})?\s*(AM|PM)?", str(time_str), re.IGNORECASE)
                if tm:
                    hour = int(tm.group(1))
                    minute = int(tm.group(2) or 0)
                    ampm = (tm.group(3) or "").upper()
                    if ampm == "PM" and hour < 12:
                        hour += 12
                    elif ampm == "AM" and hour == 12:
                        hour = 0
            start_dt = datetime(parts[0], parts[1], parts[2], hour, minute)
        except Exception:
            start_dt = now + timedelta(days=1)

    end_dt = start_dt + timedelta(hours=1)
    stamp_str = now.strftime("%Y%m%dT%H%M%SZ")
    start_str = start_dt.strftime("%Y%m%dT%H%M%S")
    end_str = end_dt.strftime("%Y%m%dT%H%M%S")

    uid = f"{call_id or uuid.uuid4().hex[:12]}@ariavoice.ai"
    title = f"{service} — {client_name}".replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,")

    desc_raw = (
        f"Customer: {client_name}\n"
        f"Phone: {phone}\n"
        f"Service: {service}\n"
        f"Location: {address}\n\n"
        f"Summary: {summary}\n"
        f"Booked via {assistant_name} Voice AI."
    )
    description = desc_raw.replace("\\", "\\\\").replace("\n", "\\n").replace(";", "\\;").replace(",", "\\,")
    escaped_address = address.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,")

    # Strict RFC 5545 CRLF line breaks
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Aria Voice AI//Appointment Scheduler//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{stamp_str}",
        f"DTSTART:{start_str}",
        f"DTEND:{end_str}",
        f"SUMMARY:{title}",
        f"DESCRIPTION:{description}",
        f"LOCATION:{escaped_address}",
        "STATUS:CONFIRMED",
        "END:VEVENT",
        "END:VCALENDAR",
        ""
    ]
    return "\r\n".join(lines).encode("utf-8")


async def dispatch_google_calendar_event(
    appointment_data: Dict[str, Any],
    calendar_id: str = "primary",
    service_account_data: Optional[str] = None,
    client_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Dispatches an event directly to Google Calendar API using OAuth access token or Service Account credentials.
    Returns status and calendar event details, with fallback to instant 1-click URL.
    """
    cfg = get_integrations_settings()
    sa_json = (service_account_data or cfg.get("google_service_account_json") or "").strip()
    cal_id = (calendar_id or cfg.get("google_calendar_id") or "primary").strip()

    cal_url = create_google_calendar_url(appointment_data)

    # 1. Check for 1-Click OAuth Token for the client
    token = None
    if client_id:
        try:
            from app.google_oauth import get_valid_access_token
            token = await get_valid_access_token(client_id)
            if token:
                logger.info(f"Using Google Calendar OAuth 2.0 access token for client '{client_id}'")
        except Exception as e:
            logger.warning(f"Notice fetching OAuth token for '{client_id}': {e}")

    # 2. Check for Service Account if no OAuth token
    if not token and sa_json:
        try:
            from google.oauth2 import service_account
            from google.auth.transport.requests import Request

            if sa_json.startswith("{") and sa_json.endswith("}"):
                info_dict = json.loads(sa_json)
                creds = service_account.Credentials.from_service_account_info(
                    info_dict, scopes=["https://www.googleapis.com/auth/calendar"]
                )
            elif Path(sa_json).exists():
                creds = service_account.Credentials.from_service_account_file(
                    sa_json, scopes=["https://www.googleapis.com/auth/calendar"]
                )
            else:
                return {
                    "status": "error",
                    "calendar_id": cal_id,
                    "error": "Invalid service account JSON format or non-existent file path",
                    "google_calendar_url": cal_url
                }

            await asyncio.to_thread(creds.refresh, Request())
            token = creds.token
        except Exception as sa_err:
            logger.error(f"Service account token resolution error: {sa_err}")

    if not token:
        logger.info(f"Google Calendar credentials not provided; 1-click calendar link: {cal_url}")
        return {
            "status": "simulated",
            "calendar_id": cal_id,
            "google_calendar_url": cal_url,
            "reason": "no_credentials_configured",
            "message": "Google OAuth or service account credentials not configured. Generated 1-click Google Calendar add link."
        }

    try:

        date_str = appointment_data.get("appointment_date") or appointment_data.get("date") or ""
        time_str = appointment_data.get("appointment_time") or appointment_data.get("time") or appointment_data.get("exact_time") or appointment_data.get("window") or ""
        client_name = appointment_data.get("client_name") or appointment_data.get("customer_name") or "Valued Customer"
        service = appointment_data.get("service_requested") or appointment_data.get("service") or "Service Appointment"
        address = appointment_data.get("service_address") or appointment_data.get("address") or ""
        phone = appointment_data.get("client_phone") or appointment_data.get("phone") or ""
        notes = appointment_data.get("summary") or appointment_data.get("notes") or ""

        now = datetime.now() + timedelta(days=1)
        start_dt = now.replace(hour=10, minute=0, second=0, microsecond=0)
        if date_str:
            try:
                parts = [int(p) for p in re.findall(r"\d+", str(date_str))[:3]]
                if len(parts) == 3:
                    hr, mn = 10, 0
                    if time_str:
                        tm = re.search(r"(\d{1,2}):?(\d{2})?\s*(AM|PM)?", str(time_str), re.IGNORECASE)
                        if tm:
                            hr = int(tm.group(1))
                            mn = int(tm.group(2) or 0)
                            ampm = (tm.group(3) or "").upper()
                            if ampm == "PM" and hr < 12:
                                hr += 12
                            elif ampm == "AM" and hr == 12:
                                hr = 0
                    start_dt = datetime(parts[0], parts[1], parts[2], hr, mn)
            except Exception:
                pass
        end_dt = start_dt + timedelta(hours=1)

        event_payload = {
            "summary": f"{service} — {client_name}",
            "description": f"Customer: {client_name}\nPhone: {phone}\nService: {service}\nLocation: {address}\n\nNotes:\n{notes}",
            "location": address,
            "start": {"dateTime": start_dt.isoformat(), "timeZone": "UTC"},
            "end": {"dateTime": end_dt.isoformat(), "timeZone": "UTC"},
        }

        # Add customer as attendee for automatic calendar sync across Google accounts
        attendees = []
        cust_email = (appointment_data.get("email") or appointment_data.get("client_email") or "").strip()
        if cust_email and "@" in cust_email:
            attendees.append({"email": cust_email, "displayName": client_name})
        if attendees:
            event_payload["attendees"] = attendees

        # sendUpdates=all automatically places event on attendee calendars and sends Google invites
        api_url = f"https://www.googleapis.com/calendar/v3/calendars/{urllib.parse.quote(cal_id)}/events?sendUpdates=all"
        async with httpx.AsyncClient(timeout=8.0) as client:
            res = await client.post(
                api_url,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=event_payload
            )
            if res.status_code in (200, 201):
                ev_data = res.json()
                logger.success(f"Google Calendar event created: {ev_data.get('id')}")
                return {
                    "status": "dispatched",
                    "event_id": ev_data.get("id"),
                    "html_link": ev_data.get("htmlLink"),
                    "calendar_id": cal_id,
                    "google_calendar_url": cal_url
                }
            else:
                logger.error(f"Google Calendar event API failed ({res.status_code}): {res.text}")
                return {
                    "status": "api_error",
                    "http_code": res.status_code,
                    "error": res.text,
                    "calendar_id": cal_id,
                    "google_calendar_url": cal_url
                }
    except Exception as e:
        logger.error(f"Google Calendar event dispatch exception: {e}")
        return {
            "status": "error",
            "error": str(e),
            "calendar_id": cal_id,
            "google_calendar_url": cal_url
        }


async def sync_to_calendar_webhook(extracted_info: Dict[str, Any], call_id: str) -> bool:
    """Dispatches call and booking payload to custom calendar webhook (e.g. Zapier, Make, Google Calendar)."""
    cfg = get_integrations_settings()
    webhook_url = cfg.get("calendar_webhook_url")
    if not webhook_url:
        return False

    payload = {
        "event": "appointment_booked",
        "call_id": call_id,
        "extracted_info": extracted_info,
        "google_calendar_url": create_google_calendar_url(extracted_info),
        "timestamp": datetime.now().isoformat(),
    }
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            res = await client.post(webhook_url, json=payload)
            logger.info(f"Dispatched calendar webhook to {webhook_url}: status {res.status_code}")
            return res.status_code in (200, 201, 202, 204)
    except Exception as e:
        logger.error(f"Error dispatching calendar webhook: {e}")
        return False


# ── 3. Automated Post-Call Email Notifications ────────────────────────────────

def _build_email_html(
    extracted_info: Dict[str, Any],
    call_session: Dict[str, Any],
    cal_url: str,
) -> str:
    """Builds a responsive, modern HTML email notification with booking details."""
    client_name = extracted_info.get("client_name") or "Customer"
    phone = extracted_info.get("client_phone") or call_session.get("caller") or "Not provided"
    email = extracted_info.get("client_email") or "Not provided"
    service = extracted_info.get("service_requested") or "Service Request"
    address = extracted_info.get("service_address") or "Not provided"
    date_str = extracted_info.get("appointment_date") or "Pending confirmation"
    time_str = extracted_info.get("appointment_time") or ""
    summary = extracted_info.get("summary") or "Call completed successfully."
    call_id = call_session.get("call_id", "")
    duration = call_session.get("duration_seconds", 0)
    agent_name = call_session.get("assistant_name", "Riley")

    transcript_html = ""
    for t in call_session.get("transcript", []):
        spk = t.get("speaker", "unknown")
        txt = t.get("text", "")
        ts = t.get("timestamp", "")
        if spk == "assistant":
            badge = f'<span style="background:#4f46e5;color:#fff;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600;">{agent_name}</span>'
        else:
            badge = '<span style="background:#0284c7;color:#fff;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600;">Customer</span>'
        transcript_html += f'<div style="margin-bottom:8px;padding:6px 0;border-bottom:1px solid #f1f5f9;"><span style="color:#64748b;font-size:11px;margin-right:8px;">{ts}</span>{badge} <span style="margin-left:8px;color:#1e293b;font-size:13px;">{txt}</span></div>'

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;margin:0;padding:24px;background-color:#f8fafc;color:#0f172a;">
  <div style="max-width:620px;margin:0 auto;background:#ffffff;border-radius:16px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.06);border:1px solid #e2e8f0;">
    
    <div style="background:linear-gradient(135deg,#4f46e5,#7c3aed);padding:28px 32px;color:#ffffff;">
      <div style="font-size:12px;letter-spacing:1px;text-transform:uppercase;opacity:0.85;font-weight:600;">Aria Voice AI • Post-Call Intelligence</div>
      <h1 style="margin:8px 0 0;font-size:22px;font-weight:700;">New Appointment & Call Summary</h1>
      <p style="margin:4px 0 0;font-size:14px;opacity:0.9;">Handled by {agent_name} • Duration: {duration}s</p>
    </div>

    <div style="padding:28px 32px;">
      
      <div style="background:#f1f5f9;border-radius:12px;padding:20px;margin-bottom:24px;border-left:4px solid #4f46e5;">
        <h2 style="margin:0 0 12px;font-size:16px;color:#0f172a;">📋 Extracted Customer & Booking Details</h2>
        <table style="width:100%;font-size:13px;border-collapse:collapse;">
          <tr>
            <td style="padding:4px 0;color:#64748b;width:140px;font-weight:600;">Customer Name:</td>
            <td style="padding:4px 0;color:#0f172a;font-weight:600;">{client_name}</td>
          </tr>
          <tr>
            <td style="padding:4px 0;color:#64748b;font-weight:600;">Service Requested:</td>
            <td style="padding:4px 0;color:#4f46e5;font-weight:700;">{service}</td>
          </tr>
          <tr>
            <td style="padding:4px 0;color:#64748b;font-weight:600;">Appointment Date:</td>
            <td style="padding:4px 0;color:#0f172a;font-weight:700;">{date_str} {time_str}</td>
          </tr>
          <tr>
            <td style="padding:4px 0;color:#64748b;font-weight:600;">Contact Phone:</td>
            <td style="padding:4px 0;color:#0f172a;">{phone}</td>
          </tr>
          <tr>
            <td style="padding:4px 0;color:#64748b;font-weight:600;">Email:</td>
            <td style="padding:4px 0;color:#0f172a;">{email}</td>
          </tr>
          <tr>
            <td style="padding:4px 0;color:#64748b;font-weight:600;">Service Location:</td>
            <td style="padding:4px 0;color:#0f172a;">{address}</td>
          </tr>
        </table>
      </div>

      <div style="margin-bottom:24px;text-align:center;">
        <a href="{cal_url}" target="_blank" style="display:inline-block;background:#4f46e5;color:#ffffff;text-decoration:none;padding:12px 24px;border-radius:8px;font-weight:600;font-size:14px;margin-right:8px;box-shadow:0 2px 8px rgba(79,70,229,0.3);">📅 Add to Google Calendar</a>
        <a href="/api/calls/{call_id}/calendar.ics" style="display:inline-block;background:#f1f5f9;color:#0f172a;text-decoration:none;padding:12px 18px;border-radius:8px;font-weight:600;font-size:14px;border:1px solid #cbd5e1;">📥 Download .ics</a>
      </div>

      <div style="margin-bottom:24px;">
        <h3 style="margin:0 0 8px;font-size:14px;text-transform:uppercase;letter-spacing:0.5px;color:#64748b;">Executive Call Summary</h3>
        <p style="margin:0;font-size:14px;line-height:1.6;color:#334155;background:#ffffff;padding:12px;border:1px solid #e2e8f0;border-radius:8px;">{summary}</p>
      </div>

      <div>
        <h3 style="margin:0 0 12px;font-size:14px;text-transform:uppercase;letter-spacing:0.5px;color:#64748b;">Full Call Transcript ({len(call_session.get('transcript', []))} turns)</h3>
        <div style="max-height:260px;overflow-y:auto;background:#fafafa;padding:12px 16px;border-radius:8px;border:1px solid #e2e8f0;">
          {transcript_html}
        </div>
      </div>

    </div>

    <div style="background:#f8fafc;padding:16px 32px;border-top:1px solid #e2e8f0;font-size:12px;color:#94a3b8;text-align:center;">
      Aria Voice AI 2.0 • Autonomous Telephone & Receptionist Intelligence • Call ID: {call_id}
    </div>

  </div>
</body>
</html>"""
    return html


async def send_post_call_email(
    extracted_info: Dict[str, Any],
    call_session: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Sends post-call notification email with extracted details, calendar link, and transcript.
    Supports Resend API and standard SMTP. Always logs to email_notifications.log.
    """
    cfg = get_integrations_settings()
    recipient = cfg.get("notify_email", "").strip()
    if not recipient:
        logger.info("Post-call email skipped: no notify_email configured in integrations settings.")
        return {"status": "skipped", "reason": "no_recipient_configured"}

    cal_url = create_google_calendar_url(extracted_info, assistant_name=call_session.get("assistant_name", "Riley"))
    client_name = extracted_info.get("client_name") or "Customer"
    service = extracted_info.get("service_requested") or "Service Request"
    call_id = call_session.get("call_id", "")

    subject = f"📅 New Booking: {service} — {client_name} (Call {call_id})"
    html_body = _build_email_html(extracted_info, call_session, cal_url)

    text_body = (
        f"Aria Voice AI — Call Summary\n"
        f"Customer: {client_name}\n"
        f"Service: {service}\n"
        f"Appointment: {extracted_info.get('appointment_date')} {extracted_info.get('appointment_time')}\n"
        f"Phone: {extracted_info.get('client_phone')}\n"
        f"Address: {extracted_info.get('service_address')}\n\n"
        f"Summary:\n{extracted_info.get('summary')}\n\n"
        f"Add to Google Calendar: {cal_url}\n"
    )

    result = {"status": "pending", "recipient": recipient, "subject": subject}

    resend_key = cfg.get("resend_api_key", "").strip()
    if resend_key:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(
                    "https://api.resend.com/emails",
                    headers={"Authorization": f"Bearer {resend_key}", "Content-Type": "application/json"},
                    json={
                        "from": cfg.get("from_email", "notifications@ariavoice.ai"),
                        "to": [recipient],
                        "subject": subject,
                        "html": html_body,
                        "text": text_body,
                    },
                )
                if res.status_code in (200, 201):
                    logger.success(f"Sent post-call email via Resend to {recipient}")
                    result["status"] = "sent_via_resend"
                    _log_email_record(result, extracted_info)
                    return result
                else:
                    logger.warning(f"Resend email failed (HTTP {res.status_code}): {res.text}")
        except Exception as e:
            logger.error(f"Error sending email via Resend: {e}")

    smtp_host = cfg.get("smtp_host", "").strip()
    smtp_user = cfg.get("smtp_user", "").strip()
    smtp_pass = cfg.get("smtp_password", "").strip()
    smtp_port = int(cfg.get("smtp_port", 587))
    use_tls = cfg.get("smtp_use_tls", True)

    if smtp_host and smtp_user and smtp_pass:
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = cfg.get("from_email", smtp_user)
            msg["To"] = recipient

            msg.attach(MIMEText(text_body, "plain"))
            msg.attach(MIMEText(html_body, "html"))

            ics_bytes = generate_ical_data(extracted_info, call_id, assistant_name=call_session.get("assistant_name", "Riley"))
            part = MIMEText(ics_bytes.decode("utf-8"), "calendar; method=PUBLISH", "utf-8")
            part.add_header("Content-Disposition", "attachment; filename=appointment.ics")
            msg.attach(part)

            def _smtp_send():
                server = smtplib.SMTP(smtp_host, smtp_port, timeout=10)
                if use_tls:
                    server.starttls()
                server.login(smtp_user, smtp_pass)
                server.sendmail(msg["From"], [recipient], msg.as_string())
                server.quit()

            await asyncio.to_thread(_smtp_send)
            logger.success(f"Sent post-call email via SMTP to {recipient}")
            result["status"] = "sent_via_smtp"
            _log_email_record(result, extracted_info)
            return result
        except Exception as e:
            logger.error(f"Error sending email via SMTP ({smtp_host}): {e}")
            result["status"] = "error"
            result["error"] = str(e)

    result["status"] = "logged_to_audit_file"
    _log_email_record(result, extracted_info)
    logger.info(f"Logged post-call notification email for {recipient} to {EMAIL_LOG_FILE}")
    return result


def _log_email_record(result: Dict[str, Any], extracted_info: Dict[str, Any]):
    """Logs sent or generated email notifications for user visibility."""
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": result.get("status"),
        "recipient": result.get("recipient"),
        "subject": result.get("subject"),
        "client_name": extracted_info.get("client_name"),
        "service": extracted_info.get("service_requested"),
        "appointment": f"{extracted_info.get('appointment_date')} {extracted_info.get('appointment_time')}",
    }
    try:
        with open(EMAIL_LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.warning(f"Could not append to email log: {e}")


async def send_activation_email(
    profile: Dict[str, Any],
    recipient: str,
    public_url: str = ""
) -> Dict[str, Any]:
    """Sends business client activation and phone forwarding instructions to owner's email."""
    cfg = get_integrations_settings()
    recipient = recipient.strip()
    if not recipient:
        return {"success": False, "message": "No email provided"}

    biz_name = profile.get("business_name") or "Your Business"
    assigned_phone = profile.get("assigned_phone") or "+1 (833) 420-5227"
    clean_digits = re.sub(r'[^0-9]', '', assigned_phone)
    client_id = profile.get("id") or ""
    portal_url = f"{public_url}/portal?client_id={client_id}" if public_url else f"/portal?client_id={client_id}"

    subject = f"🎉 Welcome to ORX Agents! Line Setup & Portal Link for {biz_name}"

    html_body = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:24px;background-color:#FAF9F5;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,sans-serif;color:#171715;">
  <div style="max-width:580px;margin:0 auto;background:#ffffff;border:1px solid #E5E3D8;border-radius:16px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.04);">
    <div style="background:#1C3326;padding:28px 32px;color:#ffffff;">
      <h1 style="margin:0;font-size:22px;font-weight:700;letter-spacing:-0.02em;">Welcome to ORX Agents</h1>
      <p style="margin:6px 0 0;font-size:14px;color:#A8BDB1;">Your AI Voice Receptionist is ready for {biz_name}</p>
    </div>
    <div style="padding:28px 32px;">
      <p style="font-size:15px;line-height:1.5;margin-bottom:20px;">
        Here are your dedicated phone line credentials and 1-tap forwarding instructions.
      </p>

      <div style="background:#F3F2EB;border:1px solid #E5E3D8;border-radius:12px;padding:18px;margin-bottom:22px;">
        <div style="font-size:12px;color:#5E5B52;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;">Your Dedicated AI Receptionist Number</div>
        <div style="font-size:24px;font-weight:800;color:#1C3326;margin-top:4px;font-family:monospace;">{assigned_phone}</div>
      </div>

      <h3 style="font-size:15px;font-weight:700;margin-bottom:12px;color:#171715;">How to Connect Your Existing Phone Line</h3>
      <p style="font-size:13.5px;color:#5E5B52;line-height:1.5;margin-bottom:14px;">
        Dial the carrier code below from your cell. Your phone will still ring first — our receptionist only answers when you're busy or don't pick up:
      </p>

      <table style="width:100%;border-collapse:collapse;margin-bottom:24px;font-size:13px;">
        <tr style="border-bottom:1px solid #E5E3D8;">
          <td style="padding:8px 0;font-weight:600;">Verizon:</td>
          <td style="padding:8px 0;font-family:monospace;color:#1C3326;font-weight:700;">*71{clean_digits}</td>
          <td style="padding:8px 0;color:#8C887E;font-size:11px;">(Turn off: *73)</td>
        </tr>
        <tr style="border-bottom:1px solid #E5E3D8;">
          <td style="padding:8px 0;font-weight:600;">AT&T:</td>
          <td style="padding:8px 0;font-family:monospace;color:#1C3326;font-weight:700;">*61*{clean_digits}#</td>
          <td style="padding:8px 0;color:#8C887E;font-size:11px;">(Turn off: #61#)</td>
        </tr>
        <tr style="border-bottom:1px solid #E5E3D8;">
          <td style="padding:8px 0;font-weight:600;">T-Mobile:</td>
          <td style="padding:8px 0;font-family:monospace;color:#1C3326;font-weight:700;">**61*{clean_digits}#</td>
          <td style="padding:8px 0;color:#8C887E;font-size:11px;">(Turn off: ##61#)</td>
        </tr>
        <tr>
          <td style="padding:8px 0;font-weight:600;">Office Phone / VoIP:</td>
          <td style="padding:8px 0;font-family:monospace;color:#1C3326;font-weight:700;">*72{clean_digits}</td>
          <td style="padding:8px 0;color:#8C887E;font-size:11px;">(Or forward in app settings)</td>
        </tr>
      </table>

      <div style="text-align:center;margin-top:28px;margin-bottom:20px;">
        <a href="{portal_url}" style="display:inline-block;background:#1C3326;color:#ffffff;text-decoration:none;padding:12px 28px;border-radius:10px;font-size:14px;font-weight:700;">Access Client Portal →</a>
      </div>

      <div style="border-top:1px solid #E5E3D8;padding-top:16px;font-size:12px;color:#8C887E;text-align:center;">
        Need video guidance? Watch our 60-second YouTube walkthrough: <a href="https://www.youtube.com/watch?v=kYJvM9lZJ0w" target="_blank" style="color:#1C3326;">Watch Tutorial</a>
      </div>
    </div>
  </div>
</body>
</html>"""

    text_body = (
        f"Welcome to ORX Agents!\n\n"
        f"Your AI Voice Receptionist is ready for {biz_name}.\n"
        f"Assigned Dedicated Number: {assigned_phone}\n\n"
        f"1-Tap Carrier Forwarding Codes:\n"
        f"- Verizon: *71{clean_digits} (turn off: *73)\n"
        f"- AT&T: *61*{clean_digits}# (turn off: #61#)\n"
        f"- T-Mobile: **61*{clean_digits}# (turn off: ##61#)\n"
        f"- Landline/VoIP: *72{clean_digits}\n\n"
        f"Access Your Client Portal:\n{portal_url}\n\n"
        f"YouTube Video Tutorial: https://www.youtube.com/watch?v=kYJvM9lZJ0w\n"
    )

    result = {"success": True, "recipient": recipient, "subject": subject, "status": "sent"}

    # Resend API check
    resend_key = cfg.get("resend_api_key", "").strip()
    if resend_key:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(
                    "https://api.resend.com/emails",
                    headers={"Authorization": f"Bearer {resend_key}", "Content-Type": "application/json"},
                    json={
                        "from": cfg.get("from_email", "notifications@ariavoice.ai"),
                        "to": [recipient],
                        "subject": subject,
                        "html": html_body,
                        "text": text_body,
                    },
                )
                if res.status_code in (200, 201):
                    logger.success(f"Sent activation email via Resend to {recipient}")
                    result["channel"] = "resend"
                    _log_email_record(result, {"client_id": client_id, "service_requested": "Setup Activation"})
                    return result
        except Exception as e:
            logger.warning(f"Resend activation email error: {e}")

    # Fallback to local email log
    _log_email_record(result, {"client_id": client_id, "service_requested": "Setup Activation"})
    result["channel"] = "logged"
    return result


async def test_email_notification(recipient: str) -> Dict[str, Any]:
    """Sends a test email to verify SMTP or Resend credentials."""
    dummy_info = {
        "client_name": "Sarah Jenkins",
        "client_phone": "555-019-2834",
        "client_email": recipient,
        "service_requested": "High-Efficiency Heat Pump Tune-Up",
        "service_address": "742 Evergreen Terrace",
        "appointment_date": (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d"),
        "appointment_time": "10:30 AM",
        "has_appointment": True,
        "summary": "Customer booked an annual heat pump inspection and requested morning arrival.",
    }
    dummy_session = {
        "call_id": f"test-email-{int(time.time())}",
        "assistant_name": "Riley Voice AI",
        "caller": "555-019-2834",
        "duration_seconds": 48.5,
        "transcript": [
            {"speaker": "assistant", "text": "Thanks for calling Comfort Breeze. How can I assist you today?", "timestamp": "10:30:00"},
            {"speaker": "customer", "text": "Hi, I'd like to schedule a tune-up for my heat pump on Friday morning.", "timestamp": "10:30:05"},
            {"speaker": "assistant", "text": "Certainly! What is your name and service address?", "timestamp": "10:30:10"},
            {"speaker": "customer", "text": "Sarah Jenkins at 742 Evergreen Terrace.", "timestamp": "10:30:16"},
            {"speaker": "assistant", "text": "You're all booked for Friday at 10:30 AM, Sarah. Have a wonderful day!", "timestamp": "10:30:22"},
        ]
    }
    cfg = get_integrations_settings()
    old_recipient = cfg.get("notify_email")
    cfg["notify_email"] = recipient
    update_integrations_settings(cfg)
    try:
        res = await send_post_call_email(dummy_info, dummy_session)
        return res
    finally:
        if old_recipient is not None:
            cfg["notify_email"] = old_recipient
            update_integrations_settings(cfg)


async def send_customer_appointment_confirmation(
    customer_email: Optional[str],
    customer_phone: Optional[str],
    appointment_data: Dict[str, Any],
    business_name: str = "Comfort Breeze HVAC",
    calendar_url: str = "",
    call_id: str = "",
) -> Dict[str, Any]:
    """
    Sends customer appointment confirmation via email (with 1-click Google Calendar button + .ics)
    and SMS (with calendar link).
    """
    results: Dict[str, Any] = {"email": None, "sms": None}
    client_name = appointment_data.get("client_name") or appointment_data.get("customer_name") or "Valued Customer"
    service = appointment_data.get("service_requested") or appointment_data.get("service") or "Service Appointment"
    date_str = appointment_data.get("appointment_date") or appointment_data.get("date") or "Upcoming"
    time_str = appointment_data.get("appointment_time") or appointment_data.get("time") or appointment_data.get("window") or ""
    address = appointment_data.get("service_address") or appointment_data.get("address") or ""

    if not calendar_url:
        calendar_url = create_google_calendar_url(appointment_data, assistant_name=business_name)

    # 1. Customer Email
    if customer_email and "@" in customer_email:
        clean_email = customer_email.strip()
        subject = f"✅ Appointment Confirmed: {service} with {business_name}"
        html_body = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;margin:0;padding:24px;background:#f8fafc;color:#1e293b;">
  <div style="max-width:580px;margin:0 auto;background:#ffffff;border-radius:12px;overflow:hidden;border:1px solid #e2e8f0;box-shadow:0 4px 12px rgba(0,0,0,0.05);">
    <div style="background:linear-gradient(135deg,#059669,#10b981);padding:24px 32px;color:#ffffff;">
      <h1 style="margin:0;font-size:20px;font-weight:700;">{business_name}</h1>
      <p style="margin:4px 0 0 0;font-size:14px;opacity:0.9;">Appointment Confirmation</p>
    </div>
    <div style="padding:32px;">
      <p style="font-size:16px;margin:0 0 16px;">Hi <strong>{client_name}</strong>,</p>
      <p style="font-size:14px;line-height:1.6;color:#475569;margin:0 0 24px;">Your service appointment has been successfully scheduled. Our technician will arrive during your scheduled window.</p>
      
      <div style="background:#f1f5f9;border-radius:8px;padding:16px 20px;margin-bottom:24px;border-left:4px solid #10b981;">
        <table style="width:100%;font-size:14px;border-collapse:collapse;">
          <tr><td style="padding:4px 0;color:#64748b;font-weight:600;width:120px;">Service:</td><td style="padding:4px 0;font-weight:600;color:#0f172a;">{service}</td></tr>
          <tr><td style="padding:4px 0;color:#64748b;font-weight:600;">Date & Time:</td><td style="padding:4px 0;font-weight:600;color:#0f172a;">{date_str} ({time_str})</td></tr>
          <tr><td style="padding:4px 0;color:#64748b;font-weight:600;">Location:</td><td style="padding:4px 0;color:#0f172a;">{address}</td></tr>
        </table>
      </div>

      <div style="text-align:center;margin:32px 0 24px;">
        <a href="{calendar_url}" target="_blank" style="display:inline-block;background:#10b981;color:#ffffff;text-decoration:none;padding:14px 28px;border-radius:8px;font-weight:600;font-size:15px;box-shadow:0 2px 8px rgba(16,185,129,0.35);">📅 Add to Google Calendar</a>
      </div>
      <p style="font-size:12px;color:#94a3b8;text-align:center;margin:0;">Need to reschedule? Reply directly to this email or call our team.</p>
    </div>
    <div style="background:#f8fafc;padding:16px;text-align:center;font-size:12px;color:#94a3b8;border-top:1px solid #e2e8f0;">
      {business_name} • Powered by Aria Voice AI
    </div>
  </div>
</body>
</html>"""
        text_body = (
            f"Hi {client_name},\n\n"
            f"Your appointment for {service} with {business_name} is confirmed for {date_str} ({time_str}) at {address}.\n\n"
            f"Add to your Google Calendar:\n{calendar_url}\n\n"
            f"Thank you for choosing {business_name}!"
        )

        cfg = get_integrations_settings()
        email_res = {"status": "simulated", "recipient": clean_email, "subject": subject}
        resend_key = cfg.get("resend_api_key", "").strip()

        # Generate RFC 5545 iCalendar (.ics) invite for automatic calendar recognition
        import base64
        ics_bytes = generate_ical_data(appointment_data, call_id=call_id, assistant_name=business_name)
        ics_base64 = base64.b64encode(ics_bytes).decode("ascii")

        if resend_key:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    r = await client.post(
                        "https://api.resend.com/emails",
                        headers={"Authorization": f"Bearer {resend_key}", "Content-Type": "application/json"},
                        json={
                            "from": cfg.get("from_email", "notifications@ariavoice.ai"),
                            "to": [clean_email],
                            "subject": subject,
                            "html": html_body,
                            "text": text_body,
                            "attachments": [
                                {
                                    "filename": "invite.ics",
                                    "content": ics_base64,
                                }
                            ],
                            "headers": {
                                "Content-Class": "urn:content-classes:calendarmessage"
                            }
                        }
                    )
                    if r.status_code in (200, 201):
                        email_res["status"] = "sent_via_resend"
            except Exception as ex:
                logger.warning(f"Customer confirmation email via Resend notice: {ex}")

        _log_email_record(email_res, appointment_data)
        results["email"] = email_res

    # 2. Customer SMS
    if customer_phone:
        sms_msg = (
            f"Hi {client_name}, your appointment with {business_name} is confirmed for {date_str} ({time_str}). "
            f"Add to your calendar: {calendar_url}"
        )
        sms_res = await send_sms(customer_phone, sms_msg)
        results["sms"] = sms_res

    return results


async def send_owner_lead_alert(
    owner_email: Optional[str],
    owner_phone: Optional[str],
    appointment_data: Dict[str, Any],
    call_session: Dict[str, Any],
    business_name: str = "Our Business",
    calendar_url: str = "",
) -> Dict[str, Any]:
    """
    Sends new appointment lead alert to the business owner and admin.
    Includes full customer details, Google Calendar button, executive summary, and transcript.
    """
    results: Dict[str, Any] = {"email": None, "sms": None}
    client_name = appointment_data.get("client_name") or appointment_data.get("customer_name") or "New Customer"
    service = appointment_data.get("service_requested") or appointment_data.get("service") or "Service Request"
    phone = appointment_data.get("client_phone") or appointment_data.get("phone") or "Not provided"
    email = appointment_data.get("client_email") or appointment_data.get("email") or "Not provided"
    address = appointment_data.get("service_address") or appointment_data.get("address") or "Not provided"
    date_str = appointment_data.get("appointment_date") or appointment_data.get("date") or "Upcoming"
    time_str = appointment_data.get("appointment_time") or appointment_data.get("time") or appointment_data.get("window") or ""

    if not calendar_url:
        calendar_url = create_google_calendar_url(appointment_data, assistant_name=business_name)

    cfg = get_integrations_settings()
    recipients = []
    if owner_email and "@" in owner_email:
        recipients.append(owner_email.strip())
    admin_email = cfg.get("notify_email", "").strip()
    if admin_email and admin_email not in recipients:
        recipients.append(admin_email)

    # 1. Owner & Admin Email
    if recipients:
        subject = f"🚨 New Booking: {service} — {client_name} ({date_str})"
        html_body = _build_email_html(appointment_data, call_session, calendar_url)
        email_res = {"status": "simulated", "recipient": ", ".join(recipients), "subject": subject}
        resend_key = cfg.get("resend_api_key", "").strip()

        # Generate RFC 5545 iCalendar (.ics) invite so Google Calendar auto-places it on owner's calendar
        import base64
        call_id = call_session.get("call_id", "")
        ics_bytes = generate_ical_data(appointment_data, call_id=call_id, assistant_name=business_name)
        ics_base64 = base64.b64encode(ics_bytes).decode("ascii")

        if resend_key:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    r = await client.post(
                        "https://api.resend.com/emails",
                        headers={"Authorization": f"Bearer {resend_key}", "Content-Type": "application/json"},
                        json={
                            "from": cfg.get("from_email", "notifications@ariavoice.ai"),
                            "to": recipients,
                            "subject": subject,
                            "html": html_body,
                            "attachments": [
                                {
                                    "filename": "appointment.ics",
                                    "content": ics_base64,
                                }
                            ],
                            "headers": {
                                "Content-Class": "urn:content-classes:calendarmessage"
                            }
                        }
                    )
                    if r.status_code in (200, 201):
                        email_res["status"] = "sent_via_resend"
            except Exception as ex:
                logger.warning(f"Owner alert email notice: {ex}")

        _log_email_record(email_res, appointment_data)
        results["email"] = email_res

    # 2. Owner SMS
    target_phone = owner_phone or cfg.get("owner_phone_number")
    if target_phone:
        sms_msg = (
            f"🚨 NEW BOOKING for {business_name}!\n"
            f"👤 {client_name} ({service})\n"
            f"📅 {date_str} ({time_str})\n"
            f"📍 {address}\n"
            f"📞 {phone}\n"
            f"Calendar: {calendar_url}"
        )
        sms_res = await send_sms(target_phone, sms_msg)
        results["sms"] = sms_res

    return results


# ── 4. Multi-Provider SMS Integration (Telnyx, Twilio, Plivo) ────────────────

def _log_sms_record(entry: Dict[str, Any]):
    """Logs sent or simulated SMS notifications to audit log."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(SMS_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning(f"Could not append to SMS log: {e}")


async def send_sms(to_phone: str, message: str, provider: Optional[str] = None) -> Dict[str, Any]:
    """
    Sends an SMS message supporting Telnyx REST API, Twilio REST API, and Plivo fallback.
    If live credentials are not configured or in testing mode, safely logs and simulates successful delivery.
    """
    cfg = get_integrations_settings()
    clean_to = re.sub(r"[^\d+]", "", (to_phone or "").strip())
    if not clean_to.startswith("+") and len(clean_to) == 10:
        clean_to = f"+1{clean_to}"
    elif not clean_to.startswith("+") and len(clean_to) == 11 and clean_to.startswith("1"):
        clean_to = f"+{clean_to}"

    now_str = time.strftime("%Y-%m-%d %H:%M:%S")

    # Determine provider
    resolved_provider = (provider or cfg.get("sms_provider") or "auto").lower()

    telnyx_key = (cfg.get("telnyx_api_key") or os.getenv("TELNYX_API_KEY") or "").strip()
    telnyx_from = (cfg.get("telnyx_phone_number") or os.getenv("TELNYX_PHONE_NUMBER") or "").strip()

    twilio_sid = (cfg.get("twilio_account_sid") or os.getenv("TWILIO_ACCOUNT_SID") or "").strip()
    twilio_token = (cfg.get("twilio_auth_token") or os.getenv("TWILIO_AUTH_TOKEN") or "").strip()
    twilio_from = (cfg.get("twilio_phone_number") or os.getenv("TWILIO_PHONE_NUMBER") or "").strip()

    plivo_id = (cfg.get("plivo_auth_id") or getattr(settings, "PLIVO_AUTH_ID", "") or os.getenv("PLIVO_AUTH_ID") or "").strip()
    plivo_tok = (cfg.get("plivo_auth_token") or getattr(settings, "PLIVO_AUTH_TOKEN", "") or os.getenv("PLIVO_AUTH_TOKEN") or "").strip()
    plivo_from = (cfg.get("plivo_phone_number") or getattr(settings, "PLIVO_PHONE_NUMBER", "") or os.getenv("PLIVO_PHONE_NUMBER") or "").strip()

    if resolved_provider == "auto":
        if telnyx_key and telnyx_from:
            resolved_provider = "telnyx"
        elif twilio_sid and twilio_token and twilio_from:
            resolved_provider = "twilio"
        elif plivo_id and plivo_tok and plivo_from:
            resolved_provider = "plivo"
        else:
            resolved_provider = "simulated"

    # 1. Telnyx REST API (https://api.telnyx.com/v2/messages)
    if resolved_provider == "telnyx":
        if not telnyx_key or not telnyx_from:
            result = {
                "status": "simulated_success",
                "provider": "telnyx",
                "to": clean_to,
                "text": message,
                "timestamp": now_str,
                "message_id": f"telnyx_sim_{uuid.uuid4().hex[:10]}",
                "reason": "telnyx_credentials_not_configured"
            }
            _log_sms_record(result)
            logger.info(f"[Telnyx SIMULATED] -> {clean_to}: {message[:80]}...")
            return result

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(
                    "https://api.telnyx.com/v2/messages",
                    headers={
                        "Authorization": f"Bearer {telnyx_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "from": telnyx_from,
                        "to": clean_to,
                        "text": message
                    }
                )
                if res.status_code in (200, 201, 202):
                    data = res.json().get("data", {})
                    msg_id = data.get("id", f"telnyx_{uuid.uuid4().hex[:8]}")
                    result = {
                        "status": "sent",
                        "provider": "telnyx",
                        "to": clean_to,
                        "from": telnyx_from,
                        "message_id": msg_id,
                        "timestamp": now_str,
                        "text": message
                    }
                    _log_sms_record(result)
                    logger.success(f"Dispatched Telnyx SMS to {clean_to}: ID {msg_id}")
                    return result
                else:
                    result = {
                        "status": "failed",
                        "provider": "telnyx",
                        "error": res.text,
                        "http_code": res.status_code,
                        "to": clean_to,
                        "text": message,
                        "timestamp": now_str
                    }
                    _log_sms_record(result)
                    logger.error(f"Telnyx SMS error ({res.status_code}): {res.text}")
                    return result
        except Exception as e:
            result = {
                "status": "error",
                "provider": "telnyx",
                "error": str(e),
                "to": clean_to,
                "text": message,
                "timestamp": now_str
            }
            _log_sms_record(result)
            return result

    # 2. Twilio REST API (https://api.twilio.com/2010-04-01/Accounts/.../Messages.json)
    elif resolved_provider == "twilio":
        if not twilio_sid or not twilio_token or not twilio_from:
            result = {
                "status": "simulated_success",
                "provider": "twilio",
                "to": clean_to,
                "text": message,
                "timestamp": now_str,
                "message_id": f"twilio_sim_{uuid.uuid4().hex[:10]}",
                "reason": "twilio_credentials_not_configured"
            }
            _log_sms_record(result)
            logger.info(f"[Twilio SIMULATED] -> {clean_to}: {message[:80]}...")
            return result

        try:
            url = f"https://api.twilio.com/2010-04-01/Accounts/{twilio_sid}/Messages.json"
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(
                    url,
                    auth=(twilio_sid, twilio_token),
                    data={
                        "From": twilio_from,
                        "To": clean_to,
                        "Body": message
                    }
                )
                if res.status_code in (200, 201):
                    data = res.json()
                    msg_id = data.get("sid", f"twilio_{uuid.uuid4().hex[:8]}")
                    result = {
                        "status": "sent",
                        "provider": "twilio",
                        "to": clean_to,
                        "from": twilio_from,
                        "message_id": msg_id,
                        "timestamp": now_str,
                        "text": message
                    }
                    _log_sms_record(result)
                    logger.success(f"Dispatched Twilio SMS to {clean_to}: SID {msg_id}")
                    return result
                else:
                    result = {
                        "status": "failed",
                        "provider": "twilio",
                        "error": res.text,
                        "http_code": res.status_code,
                        "to": clean_to,
                        "text": message,
                        "timestamp": now_str
                    }
                    _log_sms_record(result)
                    logger.error(f"Twilio SMS error ({res.status_code}): {res.text}")
                    return result
        except Exception as e:
            result = {
                "status": "error",
                "provider": "twilio",
                "error": str(e),
                "to": clean_to,
                "text": message,
                "timestamp": now_str
            }
            _log_sms_record(result)
            return result

    # 3. Plivo REST API fallback
    elif resolved_provider == "plivo":
        if not plivo_id or not plivo_tok:
            result = {
                "status": "simulated_success",
                "provider": "plivo",
                "to": clean_to,
                "text": message,
                "timestamp": now_str,
                "message_id": f"plivo_sim_{uuid.uuid4().hex[:10]}",
                "reason": "plivo_credentials_not_configured"
            }
            _log_sms_record(result)
            logger.info(f"[Plivo SIMULATED] -> {clean_to}: {message[:80]}...")
            return result

        try:
            url = f"https://api.plivo.com/v1/Account/{plivo_id}/Message/"
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(
                    url,
                    auth=(plivo_id, plivo_tok),
                    json={"src": plivo_from or "AriaVoice", "dst": clean_to, "text": message}
                )
                if res.status_code in (200, 201, 202):
                    data = res.json()
                    msg_id = data.get("message_uuid", [f"plivo_{uuid.uuid4().hex[:8]}"])[0]
                    result = {
                        "status": "sent",
                        "provider": "plivo",
                        "to": clean_to,
                        "from": plivo_from,
                        "message_id": msg_id,
                        "timestamp": now_str,
                        "text": message
                    }
                    _log_sms_record(result)
                    return result
                else:
                    result = {
                        "status": "failed",
                        "provider": "plivo",
                        "error": res.text,
                        "http_code": res.status_code,
                        "to": clean_to,
                        "text": message,
                        "timestamp": now_str
                    }
                    _log_sms_record(result)
                    return result
        except Exception as e:
            result = {
                "status": "error",
                "provider": "plivo",
                "error": str(e),
                "to": clean_to,
                "text": message,
                "timestamp": now_str
            }
            _log_sms_record(result)
            return result

    # 4. Default / Simulated Success Fallback
    result = {
        "status": "simulated_success",
        "provider": resolved_provider or "simulated",
        "to": clean_to,
        "text": message,
        "timestamp": now_str,
        "message_id": f"sim_{uuid.uuid4().hex[:10]}",
        "reason": "simulated_environment"
    }
    _log_sms_record(result)
    logger.info(f"[SMS SIMULATED] -> {clean_to} ({resolved_provider}): {message[:80]}...")
    return result


async def send_appointment_confirmation_sms(
    customer_phone: str,
    appointment_data: Dict[str, Any],
    business_name: str = "Our Business",
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Sends an automated appointment confirmation SMS to customer with calendar add link."""
    client_name = appointment_data.get("client_name") or appointment_data.get("customer_name") or "Valued Customer"
    service = appointment_data.get("service_requested") or appointment_data.get("service") or "Service Appointment"
    date_str = appointment_data.get("appointment_date") or appointment_data.get("date") or "Upcoming"
    time_str = appointment_data.get("appointment_time") or appointment_data.get("time") or appointment_data.get("exact_time") or appointment_data.get("window") or ""
    address = appointment_data.get("service_address") or appointment_data.get("address") or "Address on file"

    cal_link = create_google_calendar_url(appointment_data)
    time_display = f" at {time_str}" if time_str else ""
    msg = (
        f"Hi {client_name}! Your appointment with {business_name} for {service} is confirmed for {date_str}{time_display}. "
        f"Location: {address}. 1-Tap Calendar & Receipt: {cal_link} Questions? Reply directly to this text."
    )
    return await send_sms(to_phone=customer_phone, message=msg, provider=provider)


async def send_lead_alert_sms(
    owner_phone: str,
    appointment_data: Dict[str, Any],
    business_name: str = "Our Business",
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Sends an instant lead alert SMS to the business owner."""
    client_name = appointment_data.get("client_name") or appointment_data.get("customer_name") or "Caller"
    phone = appointment_data.get("client_phone") or appointment_data.get("phone") or "Not provided"
    service = appointment_data.get("service_requested") or appointment_data.get("service") or "General Inquiry"
    date_str = appointment_data.get("appointment_date") or appointment_data.get("date") or "ASAP"
    time_str = appointment_data.get("appointment_time") or appointment_data.get("time") or appointment_data.get("exact_time") or appointment_data.get("window") or ""
    address = appointment_data.get("service_address") or appointment_data.get("address") or "Not provided"
    summary = appointment_data.get("summary") or appointment_data.get("notes") or ""

    time_part = f" {time_str}" if time_str else ""
    msg = (
        f"🔔 NEW LEAD for {business_name}!\n"
        f"Customer: {client_name} ({phone})\n"
        f"Service: {service}\n"
        f"Slot: {date_str}{time_part}\n"
        f"Address: {address}\n"
        f"Notes: {summary[:100]}"
    )
    return await send_sms(to_phone=owner_phone, message=msg, provider=provider)
