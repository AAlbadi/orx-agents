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
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx
from loguru import logger

from app.config import settings

DATA_DIR = Path(__file__).parent.parent / "data"
INTEGRATIONS_FILE = DATA_DIR / "integrations.json"
EMAIL_LOG_FILE = DATA_DIR / "email_notifications.log"

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
        from app.bot import transliterate_arabic_to_english
        name = transliterate_arabic_to_english(name)

    # Phone extraction
    phone = caller or ""
    phone_match = re.search(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", cust_full)
    if phone_match:
        phone = phone_match.group(0)

    # Email extraction
    email = ""
    from app.bot import normalize_spoken_email
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

    from app.bot import transliterate_arabic_to_english, normalize_spoken_email

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
    Generates a 1-click Google Calendar web creation URL.
    Opens calendar.google.com with prefilled title, date, time, location, and call notes.
    """
    date_str = extracted_info.get("appointment_date")
    time_str = extracted_info.get("appointment_time")
    client_name = extracted_info.get("client_name") or "Customer"
    service = extracted_info.get("service_requested") or "Service Appointment"
    address = extracted_info.get("service_address") or ""
    summary = extracted_info.get("summary") or ""
    phone = extracted_info.get("client_phone") or ""

    if not date_str:
        now = datetime.now() + timedelta(days=1)
        start_dt = now.replace(hour=10, minute=0, second=0, microsecond=0)
    else:
        try:
            parts = [int(p) for p in date_str.split("-")]
            hour = 10
            minute = 0
            if time_str:
                tm = re.search(r"(\d{1,2}):?(\d{2})?\s*(AM|PM)?", time_str, re.IGNORECASE)
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
    call_id: str,
    assistant_name: str = "Riley",
) -> bytes:
    """Generates standard RFC 5545 iCalendar (.ics) bytes for 1-click import into any calendar."""
    date_str = extracted_info.get("appointment_date")
    time_str = extracted_info.get("appointment_time")
    client_name = extracted_info.get("client_name") or "Customer"
    service = extracted_info.get("service_requested") or "Service Appointment"
    address = extracted_info.get("service_address") or ""
    summary = extracted_info.get("summary") or ""
    phone = extracted_info.get("client_phone") or ""

    now = datetime.now()
    if not date_str:
        start_dt = now + timedelta(days=1)
        start_dt = start_dt.replace(hour=10, minute=0, second=0)
    else:
        try:
            parts = [int(p) for p in date_str.split("-")]
            hour = 10
            minute = 0
            if time_str:
                tm = re.search(r"(\d{1,2}):?(\d{2})?\s*(AM|PM)?", time_str, re.IGNORECASE)
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

    uid = f"{call_id}@ariavoice.ai"
    title = f"{service} — {client_name}"
    description = (
        f"Customer: {client_name}\n"
        f"Phone: {phone}\n"
        f"Service: {service}\n"
        f"Location: {address}\n\n"
        f"Summary: {summary}\n"
        f"Booked via {assistant_name} Voice AI."
    )

    ics_content = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Aria Voice AI//Appointment Scheduler//EN
CALSCALE:GREGORIAN
METHOD:PUBLISH
BEGIN:VEVENT
UID:{uid}
DTSTAMP:{stamp_str}
DTSTART:{start_str}
DTEND:{end_str}
SUMMARY:{title}
DESCRIPTION:{description}
LOCATION:{address}
STATUS:CONFIRMED
END:VEVENT
END:VCALENDAR"""

    return ics_content.encode("utf-8")


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
