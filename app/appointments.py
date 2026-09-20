"""HVAC Scheduling, Google Calendar FreeBusy, and 1-Tap SMS Dispatch Module for Aria Voice AI.

Features:
- Standard HVAC Arrival Windows (Morning 8am-12pm, Afternoon 12pm-4pm, Emergency 4pm-7pm).
- Google Calendar FreeBusy real-time availability check via Service Account / OAuth token.
- Local window capacity tracking (default max 2 jobs per window) to prevent overbooking.
- Automated Plivo SMS Dispatch to Customer (instant receipt) & Owner (1-tap approval/reschedule).
- Inbound SMS webhook parser to handle Owner confirmations ("1") or time changes ("Fri 2pm").
"""

import asyncio
import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from loguru import logger

from app.config import settings

DATA_DIR = Path(__file__).parent.parent / "data"
APPOINTMENTS_FILE = DATA_DIR / "appointments.json"
SMS_LOG_FILE = DATA_DIR / "sms_notifications.log"

HVAC_WINDOWS = {
    "morning": {
        "key": "morning",
        "label": "Morning (8:00 AM – 12:00 PM)",
        "start_hour": 8,
        "end_hour": 12,
        "start_time_str": "08:00 AM",
    },
    "afternoon": {
        "key": "afternoon",
        "label": "Afternoon (12:00 PM – 4:00 PM)",
        "start_hour": 12,
        "end_hour": 16,
        "start_time_str": "12:00 PM",
    },
    "evening": {
        "key": "evening",
        "label": "Late Afternoon / Emergency (4:00 PM – 7:00 PM)",
        "start_hour": 16,
        "end_hour": 19,
        "start_time_str": "04:00 PM",
    },
}


def _ensure_appointments_file() -> Dict[str, Any]:
    """Ensures appointments storage file exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not APPOINTMENTS_FILE.exists():
        initial = {"appointments": []}
        APPOINTMENTS_FILE.write_text(json.dumps(initial, indent=2))
        return initial
    try:
        return json.loads(APPOINTMENTS_FILE.read_text())
    except Exception as e:
        logger.error(f"Error reading appointments.json: {e}")
        return {"appointments": []}


def get_all_appointments() -> List[Dict[str, Any]]:
    """Retrieves all stored appointments."""
    storage = _ensure_appointments_file()
    return storage.get("appointments", [])


def save_appointment(appointment: Dict[str, Any]) -> Dict[str, Any]:
    """Saves or updates an appointment in storage."""
    storage = _ensure_appointments_file()
    appts = storage.get("appointments", [])
    found = False
    for i, a in enumerate(appts):
        if a.get("id") == appointment.get("id"):
            appts[i] = appointment
            found = True
            break
    if not found:
        appts.append(appointment)
    storage["appointments"] = appts
    APPOINTMENTS_FILE.write_text(json.dumps(storage, indent=2))
    return appointment


def find_appointment_by_id(apt_id: str) -> Optional[Dict[str, Any]]:
    """Finds an appointment by its ID or reference number."""
    clean_id = apt_id.upper().strip().lstrip("#")
    for a in get_all_appointments():
        if a.get("id", "").upper() == clean_id:
            return a
    return None


def find_latest_pending_appointment_for_phone(phone: str) -> Optional[Dict[str, Any]]:
    """Finds the most recent appointment waiting for owner or customer reply."""
    clean_phone = re.sub(r"[^\d+]", "", phone)
    appts = get_all_appointments()
    # Reverse search for newest first
    for a in reversed(appts):
        if a.get("status") in ("pending_owner_approval", "rescheduled_pending_customer"):
            return a
    return appts[-1] if appts else None


# ── Date & Window Normalization ───────────────────────────────────────────────

def normalize_date_string(date_input: str) -> str:
    """Converts relative dates like 'tomorrow', 'friday', or '2026-09-20' to YYYY-MM-DD."""
    today = datetime.now()
    clean = (date_input or "").lower().strip()

    if not clean or clean in ("today", "asap", "now"):
        return today.strftime("%Y-%m-%d")
    if "tomorrow" in clean:
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")

    day_map = {
        "monday": 0, "mon": 0,
        "tuesday": 1, "tue": 1, "tues": 1,
        "wednesday": 2, "wed": 2,
        "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
        "friday": 4, "fri": 4,
        "saturday": 5, "sat": 5,
        "sunday": 6, "sun": 6,
    }
    for day_name, day_idx in day_map.items():
        if day_name in clean:
            days_ahead = (day_idx - today.weekday() + 7) % 7
            if days_ahead == 0:
                days_ahead = 7
            return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    # Match YYYY-MM-DD
    iso_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", clean)
    if iso_match:
        return iso_match.group(0)

    # Match MM/DD or MM-DD
    md_match = re.search(r"\b(\d{1,2})[/-](\d{1,2})\b", clean)
    if md_match:
        month = int(md_match.group(1))
        day = int(md_match.group(2))
        year = today.year
        if month < today.month or (month == today.month and day < today.day):
            year += 1
        return f"{year:04d}-{month:02d}-{day:02d}"

    # Default fallback: tomorrow
    return (today + timedelta(days=1)).strftime("%Y-%m-%d")


def normalize_window(window_or_time: str) -> str:
    """Categorizes input text or time into 'morning', 'afternoon', or 'evening'."""
    val = (window_or_time or "").lower().strip()
    # Strip any ref codes like #APT-12345 first so digits in IDs don't interfere
    clean_text = re.sub(r"#?apt-\d+", "", val)

    # Keyword checks
    if any(w in clean_text for w in ["evening", "night", "late"]):
        return "evening"
    if any(w in clean_text for w in ["afternoon", "after lunch", "midday"]):
        return "afternoon"
    if any(w in clean_text for w in ["morning", "early", "first thing"]):
        return "morning"

    # Match hour with PM/AM (e.g. '2:00 PM', '2pm', '10am', '14:00')
    hour_match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", clean_text)
    if hour_match:
        hour = int(hour_match.group(1))
        ampm = (hour_match.group(3) or "").lower()
        if ampm == "pm" and hour < 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        elif not ampm and 1 <= hour <= 6:
            # For HVAC service work, bare 1-6 almost always means afternoon (1pm - 6pm)
            hour += 12

        if 8 <= hour < 12:
            return "morning"
        elif 12 <= hour < 16:
            return "afternoon"
        elif 16 <= hour <= 20:
            return "evening"

    if re.search(r"\bpm\b", clean_text):
        return "afternoon"
    if re.search(r"\bam\b", clean_text):
        return "morning"

    return "morning"


def extract_time_and_window(text: str) -> Tuple[Optional[str], str, str]:
    """
    Extracts exact time (e.g. '2:30 PM') and HVAC arrival window ('morning'|'afternoon'|'evening').
    Returns (exact_time_str, window_key, window_label).
    """
    clean_text = (text or "").lower()
    clean_text = re.sub(r"#?apt-\d+", "", clean_text).strip()

    exact_time = None
    window_key = None

    # Check arrival window keywords first
    if any(w in clean_text for w in ["evening", "night", "late", "4-7", "4pm-7pm"]):
        window_key = "evening"
    elif any(w in clean_text for w in ["afternoon", "after lunch", "midday", "12-4", "12pm-4pm"]):
        window_key = "afternoon"
    elif any(w in clean_text for w in ["morning", "early", "first thing", "8-12", "8am-12pm"]):
        window_key = "morning"

    # Match exact time patterns: e.g. '2:30pm', '10:15am', '2pm', '14:30', '9 am', 'at 3'
    time_regex = r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b"
    for match in re.finditer(time_regex, clean_text):
        hr_str, mn_str, ampm_str = match.groups()
        hr = int(hr_str)
        mn = int(mn_str) if mn_str else 0
        ampm = (ampm_str or "").lower()

        start_pos = match.start()
        prefix = clean_text[max(0, start_pos - 10):start_pos]
        suffix = clean_text[match.end():min(len(clean_text), match.end() + 10)]

        # Skip relative numbers like "in 2 weeks", "in 3 days"
        if "in" in prefix and ("week" in suffix or "day" in suffix):
            continue

        # Skip calendar dates like "oct 15" without time context
        if not ampm and not mn_str:
            month_names = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
            if any(m in prefix for m in month_names):
                continue

        # Require explicit time context: am/pm, minutes, 'at', or 'around'
        is_explicit_time = bool(ampm or mn_str or "at" in prefix or "around" in prefix)
        if not is_explicit_time:
            continue

        if 1 <= hr <= 24 and 0 <= mn < 60:
            hour_24 = hr
            if ampm == "pm" and hr < 12:
                hour_24 += 12
            elif ampm == "am" and hr == 12:
                hour_24 = 0
            elif not ampm and 1 <= hr <= 6:
                hour_24 += 12  # HVAC jobs between 1 and 6 default to PM

            display_hr = hour_24 % 12
            if display_hr == 0:
                display_hr = 12
            display_ampm = "PM" if hour_24 >= 12 else "AM"
            exact_time = f"{display_hr}:{mn:02d} {display_ampm}"

            if not window_key:
                if 8 <= hour_24 < 12:
                    window_key = "morning"
                elif 12 <= hour_24 < 16:
                    window_key = "afternoon"
                else:
                    window_key = "evening"
            break

    if not window_key:
        window_key = "morning"

    win_label = HVAC_WINDOWS.get(window_key, HVAC_WINDOWS["morning"])["label"]
    return exact_time, window_key, win_label


def get_window_datetimes(date_str: str, window_key: str) -> Tuple[datetime, datetime]:
    """Returns (start_datetime, end_datetime) for a date and arrival window."""
    win = HVAC_WINDOWS.get(window_key, HVAC_WINDOWS["morning"])
    parts = [int(p) for p in date_str.split("-")]
    start_dt = datetime(parts[0], parts[1], parts[2], win["start_hour"], 0, 0)
    end_dt = datetime(parts[0], parts[1], parts[2], win["end_hour"], 0, 0)
    return start_dt, end_dt


def get_appointment_datetimes(
    date_str: str,
    window_key: str,
    exact_time_str: Optional[str] = None,
) -> Tuple[datetime, datetime]:
    """Returns (start_datetime, end_datetime) for a date, arrival window, and optional exact time."""
    parts = [int(p) for p in date_str.split("-")]
    if exact_time_str:
        try:
            t_obj = datetime.strptime(exact_time_str, "%I:%M %p")
            start_dt = datetime(parts[0], parts[1], parts[2], t_obj.hour, t_obj.minute, 0)
            end_dt = start_dt + timedelta(hours=2)
            return start_dt, end_dt
        except Exception:
            pass
    return get_window_datetimes(date_str, window_key)


# ── Google Calendar FreeBusy & Sync ───────────────────────────────────────────

async def get_google_auth_token(service_account_info_or_path: str) -> Optional[str]:
    """Generates an OAuth2 bearer token from a Google Service Account JSON string or file."""
    if not service_account_info_or_path:
        return None
    try:
        from google.oauth2 import service_account
        from google.auth.transport.requests import Request

        content = service_account_info_or_path.strip()
        if content.startswith("{") and content.endswith("}"):
            info_dict = json.loads(content)
            creds = service_account.Credentials.from_service_account_info(
                info_dict, scopes=["https://www.googleapis.com/auth/calendar"]
            )
        elif Path(content).exists():
            creds = service_account.Credentials.from_service_account_file(
                content, scopes=["https://www.googleapis.com/auth/calendar"]
            )
        else:
            return None

        await asyncio.to_thread(creds.refresh, Request())
        return creds.token
    except Exception as e:
        logger.error(f"Failed to generate Google auth token: {e}")
        return None


async def check_google_calendar_freebusy(
    start_dt: datetime,
    end_dt: datetime,
    calendar_id: str = "primary",
    service_account_data: str = "",
    client_id: Optional[str] = None,
) -> bool:
    """Queries Google Calendar FreeBusy API. Returns True if FREE, False if BUSY."""
    token = None
    if client_id:
        try:
            from app.google_oauth import get_valid_access_token
            token = await get_valid_access_token(client_id)
        except Exception:
            token = None
    if not token:
        token = await get_google_auth_token(service_account_data)
    if not token:
        return True  # Fallback to local capacity check if no Google token

    url = "https://www.googleapis.com/calendar/v3/freeBusy"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "timeMin": start_dt.isoformat() + "Z",
        "timeMax": end_dt.isoformat() + "Z",
        "items": [{"id": calendar_id}],
    }

    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            res = await client.post(url, json=payload, headers=headers)
            if res.status_code == 200:
                data = res.json()
                cal_data = data.get("calendars", {}).get(calendar_id, {})
                busy_periods = cal_data.get("busy", [])
                logger.info(f"Google FreeBusy query for {calendar_id}: {len(busy_periods)} busy slots.")
                return len(busy_periods) == 0
            else:
                logger.warning(f"Google FreeBusy HTTP {res.status_code}: {res.text[:200]}")
                return True
    except Exception as e:
        logger.warning(f"Google FreeBusy check error: {e}. Defaulting to open.")
        return True


async def create_google_calendar_event(
    appointment: Dict[str, Any],
    calendar_id: str = "primary",
    service_account_data: str = "",
    client_id: Optional[str] = None,
) -> Optional[str]:
    """Creates a confirmed event directly on the technician's Google Calendar."""
    token = None
    if client_id:
        try:
            from app.google_oauth import get_valid_access_token
            token = await get_valid_access_token(client_id)
        except Exception:
            token = None
    if not token:
        token = await get_google_auth_token(service_account_data)
    if not token:
        return None

    date_str = appointment.get("appointment_date", "")
    win_key = appointment.get("window", "morning")
    exact_time = appointment.get("exact_time")
    start_dt, end_dt = get_appointment_datetimes(date_str, win_key, exact_time)

    client_name = appointment.get("client_name") or "Valued Customer"
    service = appointment.get("service_requested") or "HVAC Service"
    address = appointment.get("service_address") or ""
    phone = appointment.get("client_phone") or ""
    notes = appointment.get("summary") or appointment.get("notes") or ""

    summary = f"🔧 {service} — {client_name}"
    desc = (
        f"Customer: {client_name}\n"
        f"Phone: {phone}\n"
        f"Service: {service}\n"
        f"Location: {address}\n\n"
        f"Notes:\n{notes}\n\n"
        f"Booked automatically by Aria Voice AI (ID: {appointment.get('id')})"
    )

    event_payload = {
        "summary": summary,
        "description": desc,
        "location": address,
        "start": {"dateTime": start_dt.isoformat(), "timeZone": "UTC"},
        "end": {"dateTime": end_dt.isoformat(), "timeZone": "UTC"},
    }

    url = f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            res = await client.post(url, json=event_payload, headers=headers)
            if res.status_code in (200, 201):
                event_data = res.json()
                event_id = event_data.get("id")
                logger.success(f"Inserted event on Google Calendar: {event_id}")
                return event_id
            else:
                logger.error(f"Failed to create Google Calendar event: {res.text[:250]}")
                return None
    except Exception as e:
        logger.error(f"Error creating Google Calendar event: {e}")
        return None


async def verify_google_calendar_connection(
    calendar_id: str = "primary",
    service_account_data: str = "",
) -> Dict[str, Any]:
    """
    Tests and verifies Google Calendar connection.
    Validates service account credentials, fetches calendar metadata, and performs a FreeBusy test.
    """
    sa_clean = (service_account_data or "").strip()
    cal_clean = (calendar_id or "primary").strip()

    if not sa_clean:
        return {
            "connected": False,
            "calendar_id": cal_clean,
            "status": "missing_credentials",
            "message": "Google Service Account JSON is empty. Paste service account credentials or set file path."
        }

    token = await get_google_auth_token(sa_clean)
    if not token:
        return {
            "connected": False,
            "calendar_id": cal_clean,
            "status": "auth_failed",
            "message": "Failed to authenticate with Google. Ensure JSON credentials are valid and have not expired."
        }

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            cal_title = cal_clean
            cal_tz = "UTC"
            # 1. Query calendar metadata if possible
            try:
                cal_url = f"https://www.googleapis.com/calendar/v3/calendars/{urllib.parse.quote(cal_clean)}"
                res = await client.get(cal_url, headers=headers)
                if res.status_code == 200:
                    cal_info = res.json()
                    cal_title = cal_info.get("summary", cal_clean)
                    cal_tz = cal_info.get("timeZone", "UTC")
            except Exception:
                pass

            # 2. Test FreeBusy API for 1 hour test window
            now_dt = datetime.utcnow()
            fb_url = "https://www.googleapis.com/calendar/v3/freeBusy"
            fb_payload = {
                "timeMin": now_dt.isoformat() + "Z",
                "timeMax": (now_dt + timedelta(hours=1)).isoformat() + "Z",
                "items": [{"id": cal_clean}],
            }
            fb_res = await client.post(fb_url, json=fb_payload, headers=headers)
            if fb_res.status_code == 200:
                data = fb_res.json()
                cal_data = data.get("calendars", {}).get(cal_clean, {})
                errors = cal_data.get("errors", [])
                if errors:
                    err_msg = errors[0].get("reason", "unknown calendar error")
                    return {
                        "connected": False,
                        "calendar_id": cal_clean,
                        "status": "permission_denied",
                        "message": f"Google Calendar error for '{cal_clean}': {err_msg}. Ensure the calendar is shared with the service account client email."
                    }
                return {
                    "connected": True,
                    "calendar_id": cal_clean,
                    "calendar_title": cal_title,
                    "timezone": cal_tz,
                    "status": "connected",
                    "message": f"Successfully connected to Google Calendar '{cal_title}' ({cal_tz}). Live FreeBusy sync is operational.",
                    "verified_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            else:
                return {
                    "connected": False,
                    "calendar_id": cal_clean,
                    "status": "api_error",
                    "message": f"Google Calendar API responded with HTTP {fb_res.status_code}: {fb_res.text[:200]}."
                }
    except Exception as e:
        return {
            "connected": False,
            "calendar_id": cal_clean,
            "status": "error",
            "message": f"Network or connection error during Google verification: {str(e)}"
        }


# ── Live Availability Checker ─────────────────────────────────────────────────

async def check_technician_availability(
    date_input: str,
    window_input: str = "morning",
    client_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Checks if an HVAC technician is available on the requested date and arrival window.
    Combines Google Calendar FreeBusy check with local arrival window capacity counters.
    Supports client-specific configuration or global system default.
    """
    from app.integrations import get_integrations_settings

    global_cfg = get_integrations_settings()
    norm_date = normalize_date_string(date_input)
    norm_win = normalize_window(window_input)
    win_info = HVAC_WINDOWS.get(norm_win, HVAC_WINDOWS["morning"])

    # Resolve calendar config & limits (per-client or global)
    if client_id:
        from app.project_db import get_project_calendar_config
        cal_cfg = get_project_calendar_config(client_id)
        cap_morning = int(cal_cfg.get("morning_slot_capacity", 2))
        cap_afternoon = int(cal_cfg.get("afternoon_slot_capacity", 2))
        cal_id = cal_cfg.get("calendar_id", "primary")
        sa_data = cal_cfg.get("service_account_json", "")
    else:
        cap_morning = int(global_cfg.get("morning_slot_capacity", 2))
        cap_afternoon = int(global_cfg.get("afternoon_slot_capacity", 2))
        cal_id = global_cfg.get("google_calendar_id", "primary")
        sa_data = global_cfg.get("google_service_account_json", "")

    cap_limit = cap_morning if norm_win == "morning" else cap_afternoon

    # 1. Check local capacity limits (from project db if client_id specified, otherwise global)
    if client_id:
        from app.project_db import get_project_appointments
        existing_appts = get_project_appointments(client_id)
    else:
        existing_appts = get_all_appointments()

    booked_count = sum(
        1
        for a in existing_appts
        if a.get("appointment_date") == norm_date
        and a.get("window") == norm_win
        and a.get("status") in ("confirmed", "pending_owner_approval")
    )

    if booked_count >= cap_limit:
        alt_win = "afternoon" if norm_win == "morning" else "morning"
        alt_label = HVAC_WINDOWS[alt_win]["label"]
        return {
            "available": False,
            "date": norm_date,
            "window": norm_win,
            "window_label": win_info["label"],
            "alternative_window": alt_win,
            "alternative_label": alt_label,
            "message": f"We are fully booked for the {win_info['label']} on {norm_date}, but we have openings in our {alt_label} window.",
        }

    # 2. Check Google Calendar FreeBusy (if credentials supplied)
    start_dt, end_dt = get_window_datetimes(norm_date, norm_win)
    is_gcal_free = await check_google_calendar_freebusy(
        start_dt, end_dt, calendar_id=cal_id, service_account_data=sa_data, client_id=client_id
    )

    if not is_gcal_free:
        alt_win = "afternoon" if norm_win == "morning" else "morning"
        alt_label = HVAC_WINDOWS[alt_win]["label"]
        return {
            "available": False,
            "date": norm_date,
            "window": norm_win,
            "window_label": win_info["label"],
            "alternative_window": alt_win,
            "alternative_label": alt_label,
            "message": f"The technician has a conflicting job during the {win_info['label']} on {norm_date}, but our {alt_label} window is open.",
        }

    return {
        "available": True,
        "date": norm_date,
        "window": norm_win,
        "window_label": win_info["label"],
        "message": f"Great news! The technician is available for the {win_info['label']} window on {norm_date}.",
    }


# ── Plivo SMS Dispatcher ──────────────────────────────────────────────────────

async def send_plivo_sms(to_number: str, message_text: str) -> Dict[str, Any]:
    """
    Sends an SMS message using Plivo REST API.
    Gracefully logs and simulates if Plivo credentials are not configured.
    """
    auth_id = settings.PLIVO_AUTH_ID
    auth_token = settings.PLIVO_AUTH_TOKEN
    from_number = settings.PLIVO_PHONE_NUMBER or "AriaVoice"

    clean_to = re.sub(r"[^\d+]", "", to_number)
    if not clean_to.startswith("+") and len(clean_to) == 10:
        clean_to = f"+1{clean_to}"

    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "to": clean_to,
        "from": from_number,
        "text": message_text,
    }

    if not auth_id or not auth_token:
        log_entry["status"] = "simulated_success"
        log_entry["reason"] = "plivo_credentials_not_configured"
        logger.info(f"[SMS SIMULATED] -> {clean_to}: {message_text[:80]}...")
        _append_sms_log(log_entry)
        return log_entry

    url = f"https://api.plivo.com/v1/Account/{auth_id}/Message/"
    payload = {
        "src": from_number,
        "dst": clean_to,
        "text": message_text,
    }

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            res = await client.post(url, auth=(auth_id, auth_token), json=payload)
            if res.status_code in (200, 201, 202):
                data = res.json()
                log_entry["status"] = "sent"
                log_entry["message_uuid"] = data.get("message_uuid", [""])[0]
                logger.success(f"SMS dispatched to {clean_to}: UUID {log_entry['message_uuid']}")
            else:
                log_entry["status"] = "failed"
                log_entry["error"] = res.text
                logger.error(f"Plivo SMS failed ({res.status_code}): {res.text}")
    except Exception as e:
        log_entry["status"] = "error"
        log_entry["error"] = str(e)
        logger.error(f"Error sending SMS to {clean_to}: {e}")

    _append_sms_log(log_entry)
    return log_entry


def _append_sms_log(entry: Dict[str, Any]):
    """Appends SMS audit records to file."""
    try:
        with open(SMS_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning(f"Could not write to SMS log: {e}")


def generate_schedule_options(requested_date_str: str, requested_window_key: str) -> Dict[str, Dict[str, str]]:
    """
    Computes 3 explicit, logical options for the HVAC owner:
    - Option 1: The requested slot (e.g. Fri Sep 25, 8am - 12pm)
    - Option 2: Same-day alternate window (e.g. Fri Sep 25, 12pm - 4pm)
    - Option 3: Next business day window (e.g. Mon Sep 28, 8am - 12pm)
    """
    try:
        req_date = datetime.strptime(requested_date_str, "%Y-%m-%d")
    except Exception:
        req_date = datetime.now() + timedelta(days=1)
        requested_date_str = req_date.strftime("%Y-%m-%d")

    req_day_name = req_date.strftime("%a %b %d")

    # Option 1: Exact requested slot
    opt1_win = requested_window_key
    opt1_time_label = "8am-12pm" if opt1_win == "morning" else "12pm-4pm"
    opt1_label = f"{req_day_name} ({opt1_time_label})"

    # Option 2: Opposite window on the same date
    opt2_win = "afternoon" if opt1_win == "morning" else "morning"
    opt2_time_label = "8am-12pm" if opt2_win == "morning" else "12pm-4pm"
    opt2_label = f"{req_day_name} ({opt2_time_label})"

    # Option 3: Next business day (skip Saturday/Sunday)
    days_to_add = 1
    if req_date.weekday() == 4:  # Friday -> Monday (+3 days)
        days_to_add = 3
    elif req_date.weekday() == 5:  # Saturday -> Monday (+2 days)
        days_to_add = 2
    next_biz_date = req_date + timedelta(days=days_to_add)
    next_day_name = next_biz_date.strftime("%a %b %d")
    opt3_date_str = next_biz_date.strftime("%Y-%m-%d")
    opt3_win = "morning"
    opt3_label = f"{next_day_name} (8am-12pm)"

    return {
        "1": {
            "key": "1",
            "date": requested_date_str,
            "window": opt1_win,
            "label": opt1_label,
            "window_label": HVAC_WINDOWS.get(opt1_win, HVAC_WINDOWS["morning"])["label"],
        },
        "2": {
            "key": "2",
            "date": requested_date_str,
            "window": opt2_win,
            "label": opt2_label,
            "window_label": HVAC_WINDOWS.get(opt2_win, HVAC_WINDOWS["afternoon"])["label"],
        },
        "3": {
            "key": "3",
            "date": opt3_date_str,
            "window": opt3_win,
            "label": opt3_label,
            "window_label": HVAC_WINDOWS.get(opt3_win, HVAC_WINDOWS["morning"])["label"],
        },
    }


# ── 1-Tap HVAC Booking Dispatch ───────────────────────────────────────────────

async def dispatch_hvac_booking(
    extracted_info: Dict[str, Any],
    call_session: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Creates an appointment record and triggers 1-tap SMS flow:
    1. Pre-calculates 3 numbered options (Requested slot, Alternate slot, Next day slot).
    2. Sends pre-confirmation SMS to the Customer.
    3. Sends dispatch alert SMS to the HVAC Owner with numbered choices + 1-tap mobile link.
    """
    from app.integrations import get_integrations_settings

    cfg = get_integrations_settings()
    client_name = extracted_info.get("client_name") or "Valued Customer"
    client_phone = extracted_info.get("client_phone") or call_session.get("caller") or ""
    service = extracted_info.get("service_requested") or "HVAC Inspection & Repair"
    address = extracted_info.get("service_address") or "Address on file"
    date_str = normalize_date_string(extracted_info.get("appointment_date") or "")
    win_key = normalize_window(extracted_info.get("appointment_time") or "")
    win_label = HVAC_WINDOWS.get(win_key, HVAC_WINDOWS["morning"])["label"]
    call_id = call_session.get("call_id", f"call_{int(time.time())}")

    import random
    apt_code = f"APT-{random.randint(10000, 99999)}"
    options = generate_schedule_options(date_str, win_key)

    base_url = (settings.PUBLIC_URL or "http://localhost:7860").rstrip("/")
    action_url = f"{base_url}/a/{apt_code}"

    appointment = {
        "id": apt_code,
        "call_id": call_id,
        "client_name": client_name,
        "client_phone": client_phone,
        "service_requested": service,
        "service_address": address,
        "appointment_date": date_str,
        "window": win_key,
        "window_label": win_label,
        "status": "pending_owner_approval",
        "created_at": datetime.now().isoformat(),
        "summary": extracted_info.get("summary", ""),
        "options": options,
        "action_url": action_url,
        "google_event_id": None,
    }

    # Associate with dedicated client project if detected
    client_id = extracted_info.get("client_id") or call_session.get("client_id")
    if not client_id:
        try:
            from app.project_db import find_client_by_phone
            target_phone = call_session.get("called_number") or call_session.get("to") or cfg.get("owner_phone_number")
            matched_proj = find_client_by_phone(target_phone)
            if matched_proj:
                client_id = matched_proj.get("id")
        except Exception:
            pass

    appointment["client_id"] = client_id
    save_appointment(appointment)
    if client_id:
        try:
            from app.project_db import save_project_appointment
            save_project_appointment(client_id, appointment)
        except Exception as p_ex:
            logger.warning(f"Could not save appointment to dedicated project DB: {p_ex}")

    logger.info(f"Created pending HVAC appointment #{apt_code} for {client_name} ({date_str} {win_label})")

    results = {"appointment": appointment, "sms_customer": None, "sms_owner": None}

    # 1. Send SMS to Customer
    if cfg.get("enable_customer_sms", True) and client_phone:
        customer_msg = (
            f"Thanks for calling Comfort Breeze, {client_name}! We received your request for {service} on {date_str} ({win_label}). "
            f"Our technician is confirming their route and will text you momentary confirmation.\n\n"
            f"Need a different day or time? Text us your preferred slot:\n"
            f"Format: YYYY-MM-DD:HHMM\n"
            f"Example: 2026-10-15:1330\n"
            f"Ref: #{apt_code}"
        )
        results["sms_customer"] = await send_plivo_sms(client_phone, customer_msg)

    # 2. Send SMS to Owner / Dispatcher with Foolproof Numbered Options + Any Day/Time Instructions
    owner_phone = cfg.get("owner_phone_number", "").strip()
    if cfg.get("enable_owner_sms", True) and owner_phone:
        opt1_lbl = options["1"]["label"]
        opt2_lbl = options["2"]["label"]
        opt3_lbl = options["3"]["label"]

        owner_msg = (
            f"🚨 NEW HVAC JOB #{apt_code}\n"
            f"👤 {client_name} ({service})\n"
            f"📍 {address}\n"
            f"📞 {client_phone}\n\n"
            f"Reply with a number:\n"
            f"1️⃣ Confirm ({opt1_lbl})\n"
            f"2️⃣ Move to {opt2_lbl}\n"
            f"3️⃣ Move to {opt3_lbl}\n"
            f"4️⃣ Decline / Busy\n\n"
            f"📅 Or text ANY exact day & time or window:\n"
            f"Format: YYYY-MM-DD:HHMM\n"
            f"Example: 2026-10-15:1330\n\n"
            f"Or tap: {action_url}"
        )
        results["sms_owner"] = await send_plivo_sms(owner_phone, owner_msg)
    else:
        logger.info("Owner SMS skipped: no owner_phone_number configured in integrations.")

    return results


# ── Smart Flexible Schedule Intent Parser ──────────────────────────────────────

async def parse_flexible_schedule_intent(
    text: str,
    reference_date: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    """
    Parses flexible relative/exact date, time, and arrival window instructions.
    Handles phrases like:
    - "in 2 weeks Friday 2:30pm"
    - "in 3 days afternoon"
    - "next Tuesday at 2pm"
    - "October 15 10am"
    - "tomorrow morning"
    - "Friday 3pm"
    - "10/24 11am"
    Returns:
    {
        "date": "YYYY-MM-DD",
        "exact_time": "2:30 PM" (or None),
        "window": "morning" | "afternoon" | "evening",
        "window_label": "Afternoon (12:00 PM – 4:00 PM)",
        "display_label": "Fri Oct 02 at 2:30 PM (Afternoon window 12pm–4pm)",
    }
    """
    today = reference_date or datetime.now()
    clean = (text or "").lower().strip()
    clean = re.sub(r"#?apt-\d+", "", clean).strip()

    # Fast guard: require at least one date, weekday, month, or time indicator
    DATE_TIME_INDICATORS = [
        "week", "weeks", "day", "days", "tomorrow", "tmrw", "next", "later", "this",
        "monday", "mon", "tuesday", "tue", "wednesday", "wed", "thursday", "thu",
        "friday", "fri", "saturday", "sat", "sunday", "sun",
        "jan", "january", "feb", "february", "mar", "march", "apr", "april", "may",
        "jun", "june", "jul", "july", "aug", "august", "sep", "sept", "september",
        "oct", "october", "nov", "november", "dec", "december",
        "morning", "afternoon", "evening", "am", "pm", "clock", "noon", "at",
    ]
    has_date_term = any(re.search(r"\b" + re.escape(ind) + r"\b", clean) for ind in DATE_TIME_INDICATORS)
    has_iso_or_slash = bool(re.search(r"\b\d{1,4}[-/]\d{1,2}(?:[-/]\d{1,4})?\b", clean))

    if not (has_date_term or has_iso_or_slash):
        return None

    target_date = None
    exact_time, window_key, win_label = extract_time_and_window(clean)

    # 0. Check primary standard format: YYYY-MM-DD:HHMM (e.g. 2026-10-15:1330 or 2026-10-15:13:30)
    std_m = re.search(r"\b(\d{4}-\d{2}-\d{2})[:\s]([a-zA-Z0-9:]+)\b", clean)
    if std_m:
        d_part = std_m.group(1)
        t_part = std_m.group(2).lower()
        try:
            target_date = datetime.strptime(d_part, "%Y-%m-%d")
        except Exception:
            target_date = None

        if target_date:
            num_m = re.match(r"^(\d{1,2})(\d{2})$", t_part)
            col_m = re.match(r"^(\d{1,2}):(\d{2})$", t_part)
            ampm_m = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)$", t_part)

            if num_m or col_m:
                m_obj = num_m or col_m
                hr = int(m_obj.group(1))
                mn = int(m_obj.group(2))
                if 0 <= hr <= 24 and 0 <= mn < 60:
                    display_hr = hr % 12
                    if display_hr == 0:
                        display_hr = 12
                    display_ampm = "PM" if hr >= 12 else "AM"
                    exact_time = f"{display_hr}:{mn:02d} {display_ampm}"
                    if 8 <= hr < 12:
                        window_key = "morning"
                    elif 12 <= hr < 16:
                        window_key = "afternoon"
                    else:
                        window_key = "evening"
                    win_label = HVAC_WINDOWS.get(window_key, HVAC_WINDOWS["morning"])["label"]

            elif ampm_m:
                hr = int(ampm_m.group(1))
                mn = int(ampm_m.group(2)) if ampm_m.group(2) else 0
                ampm = ampm_m.group(3)
                hour_24 = hr
                if ampm == "pm" and hr < 12:
                    hour_24 += 12
                elif ampm == "am" and hr == 12:
                    hour_24 = 0
                display_hr = hour_24 % 12
                if display_hr == 0:
                    display_hr = 12
                display_ampm = "PM" if hour_24 >= 12 else "AM"
                exact_time = f"{display_hr}:{mn:02d} {display_ampm}"
                if 8 <= hour_24 < 12:
                    window_key = "morning"
                elif 12 <= hour_24 < 16:
                    window_key = "afternoon"
                else:
                    window_key = "evening"
                win_label = HVAC_WINDOWS.get(window_key, HVAC_WINDOWS["morning"])["label"]

            elif t_part in ("morning", "afternoon", "evening"):
                window_key = t_part
                win_label = HVAC_WINDOWS.get(window_key, HVAC_WINDOWS["morning"])["label"]

    # 1. Check "tomorrow" / "tmrw"
    if not target_date and ("tomorrow" in clean or "tmrw" in clean):
        target_date = today + timedelta(days=1)

    # 2. Check "in X weeks [day]" or "in X weeks"
    if not target_date:
        weeks_match = re.search(r"in\s+(\d+)\s+weeks?(?:\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun))?", clean)
        if weeks_match:
            num_weeks = int(weeks_match.group(1))
            spec_day = weeks_match.group(2)
            base_target = today + timedelta(weeks=num_weeks)
            if spec_day:
                day_map = {
                    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1,
                    "wednesday": 2, "wed": 2, "thursday": 3, "thu": 3,
                    "friday": 4, "fri": 4, "saturday": 5, "sat": 5,
                    "sunday": 6, "sun": 6,
                }
                target_dow = day_map.get(spec_day.lower(), 0)
                start_of_week = base_target - timedelta(days=base_target.weekday())
                target_date = start_of_week + timedelta(days=target_dow)
                if target_date < today:
                    target_date += timedelta(weeks=1)
            else:
                target_date = base_target

    # 3. Check "in X days"
    if not target_date:
        days_match = re.search(r"in\s+(\d+)\s+days?", clean)
        if days_match:
            num_days = int(days_match.group(1))
            target_date = today + timedelta(days=num_days)

    # 4. Check "next [weekday]" (e.g. "next tuesday", "next friday")
    if not target_date:
        next_day_match = re.search(r"next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)", clean)
        if next_day_match:
            spec_day = next_day_match.group(1)
            day_map = {
                "monday": 0, "mon": 0, "tuesday": 1, "tue": 1,
                "wednesday": 2, "wed": 2, "thursday": 3, "thu": 3,
                "friday": 4, "fri": 4, "saturday": 5, "sat": 5,
                "sunday": 6, "sun": 6,
            }
            target_dow = day_map.get(spec_day.lower(), 0)
            days_ahead = (target_dow - today.weekday() + 7) % 7
            if days_ahead == 0:
                days_ahead = 7
            target_date = today + timedelta(days=days_ahead)

    # 5. Check "(this)? [weekday]" (e.g. "this friday", "friday", "monday 2pm")
    if not target_date:
        this_day_match = re.search(r"\b(?:this\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)\b", clean)
        if this_day_match:
            spec_day = this_day_match.group(1)
            day_map = {
                "monday": 0, "mon": 0, "tuesday": 1, "tue": 1,
                "wednesday": 2, "wed": 2, "thursday": 3, "thu": 3,
                "friday": 4, "fri": 4, "saturday": 5, "sat": 5,
                "sunday": 6, "sun": 6,
            }
            target_dow = day_map.get(spec_day.lower(), 0)
            days_ahead = (target_dow - today.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            target_date = today + timedelta(days=days_ahead)

    # 6. Check explicit month + day (e.g. "oct 15", "october 15", "nov 2") or day + of + month
    if not target_date:
        month_map = {
            "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
            "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
            "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
            "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12
        }
        # e.g. "oct 15"
        month_pattern = r"\b(" + "|".join(month_map.keys()) + r")\s+(\d{1,2})(?:st|nd|rd|th)?\b"
        m_match = re.search(month_pattern, clean)
        if m_match:
            m_name = m_match.group(1)
            m_day = int(m_match.group(2))
            m_month = month_map[m_name]
            yr = today.year
            if m_month < today.month or (m_month == today.month and m_day < today.day):
                yr += 1
            target_date = datetime(yr, m_month, m_day)
        else:
            # e.g. "15th of oct"
            reverse_pattern = r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(" + "|".join(month_map.keys()) + r")\b"
            rev_m = re.search(reverse_pattern, clean)
            if rev_m:
                m_day = int(rev_m.group(1))
                m_name = rev_m.group(2)
                m_month = month_map[m_name]
                yr = today.year
                if m_month < today.month or (m_month == today.month and m_day < today.day):
                    yr += 1
                target_date = datetime(yr, m_month, m_day)

    # 7. Check slash or dash date MM/DD or MM/DD/YYYY
    if not target_date:
        slash_m = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", clean)
        if slash_m:
            try:
                m_val = int(slash_m.group(1))
                d_val = int(slash_m.group(2))
                y_val = int(slash_m.group(3)) if slash_m.group(3) else today.year
                if y_val < 100:
                    y_val += 2000
                if m_val < today.month or (m_val == today.month and d_val < today.day):
                    if not slash_m.group(3):
                        y_val += 1
                target_date = datetime(y_val, m_val, d_val)
            except Exception:
                pass

    # 8. Check ISO date YYYY-MM-DD
    if not target_date:
        iso_m = re.search(r"\b\d{4}-\d{2}-\d{2}\b", clean)
        if iso_m:
            try:
                target_date = datetime.strptime(iso_m.group(0), "%Y-%m-%d")
            except Exception:
                pass

    # 9. LLM Fast Zero-Shot Fallback
    if not target_date and (settings.GROQ_API_KEY or settings.GEMINI_API_KEY):
        try:
            today_str = today.strftime("%A, %B %d, %Y")
            prompt = (
                f"Today is {today_str}. The user wants to schedule an HVAC service appointment with this message: '{clean}'.\n"
                "Extract the target date, exact time if mentioned, and arrival window.\n"
                "Output ONLY a valid JSON object with keys:\n"
                "{\"date\": \"YYYY-MM-DD\", \"exact_time\": \"HH:MM AM/PM\" or null, \"window\": \"morning\" | \"afternoon\" | \"evening\"}"
            )
            llm_result = None
            if settings.GROQ_API_KEY:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    resp = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"},
                        json={
                            "model": settings.GROQ_MODEL,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.0,
                            "max_tokens": 100,
                        }
                    )
                    if resp.status_code == 200:
                        raw_c = resp.json()["choices"][0]["message"]["content"]
                        raw_c = re.sub(r"^```json\s*", "", raw_c)
                        raw_c = re.sub(r"^```\s*", "", raw_c)
                        raw_c = re.sub(r"\s*```$", "", raw_c).strip()
                        llm_result = json.loads(raw_c)
            if llm_result and llm_result.get("date"):
                target_date = datetime.strptime(llm_result["date"], "%Y-%m-%d")
                if llm_result.get("exact_time"):
                    exact_time = llm_result["exact_time"]
                window_key = llm_result.get("window", window_key)
                win_label = HVAC_WINDOWS.get(window_key, HVAC_WINDOWS["morning"])["label"]
        except Exception as e:
            logger.debug(f"LLM date parsing skipped/failed: {e}")

    if not target_date:
        return None

    date_str = target_date.strftime("%Y-%m-%d")
    day_name = target_date.strftime("%a %b %d")

    if exact_time:
        disp_lbl = f"{day_name} at {exact_time} ({win_label})"
    else:
        time_lbl = "8am-12pm" if window_key == "morning" else ("12pm-4pm" if window_key == "afternoon" else "4pm-7pm")
        disp_lbl = f"{day_name} ({time_lbl})"

    return {
        "date": date_str,
        "exact_time": exact_time,
        "window": window_key,
        "window_label": win_label,
        "display_label": disp_lbl,
    }


# ── Action Applicator (Shared by SMS & Web Link) ──────────────────────────────

async def apply_appointment_action(
    appointment: Dict[str, Any],
    action_input: Any,
    from_phone: str = "",
) -> Dict[str, Any]:
    """
    Executes an action on an appointment (from owner SMS or web action card).
    Supports:
    - 1 / Confirm (Option 1)
    - 2 / Shift to Option 2 (Afternoon / Alternate)
    - 3 / Shift to Option 3 (Next Business Day)
    - 4 / Decline
    - Custom payload: {"action": "custom", "date": "2026-10-02", "window": "morning"}
    - Natural Language Freeform: "in 2 weeks Friday morning", "next Tuesday 2pm", "in 3 days"
    - Safety-net help menu fallback
    """
    from app.integrations import get_integrations_settings

    cfg = get_integrations_settings()
    clean_from = re.sub(r"[^\d+]", "", from_phone)

    apt_id = appointment.get("id")
    client_phone = appointment.get("client_phone")
    client_name = appointment.get("client_name", "Customer")
    address = appointment.get("service_address", "Address on file")
    options = appointment.get("options") or generate_schedule_options(
        appointment.get("appointment_date", datetime.now().strftime("%Y-%m-%d")),
        appointment.get("window", "morning")
    )
    action_url = appointment.get("action_url") or f"{(settings.PUBLIC_URL or 'http://localhost:7860').rstrip('/')}/a/{apt_id}"

    # Handle structured custom dictionary payload from Web UI
    if isinstance(action_input, dict) and (action_input.get("action") == "custom" or action_input.get("date")):
        custom_date = normalize_date_string(action_input.get("date", ""))
        custom_win = normalize_window(action_input.get("window", "morning"))
        custom_time = action_input.get("time") or action_input.get("exact_time")
        win_info = HVAC_WINDOWS.get(custom_win, HVAC_WINDOWS["morning"])

        try:
            d_obj = datetime.strptime(custom_date, "%Y-%m-%d")
            day_str = d_obj.strftime("%a %b %d")
            if custom_time:
                disp_lbl = f"{day_str} at {custom_time} ({win_info['label']})"
            else:
                time_lbl = "8am-12pm" if custom_win == "morning" else ("12pm-4pm" if custom_win == "afternoon" else "4pm-7pm")
                disp_lbl = f"{day_str} ({time_lbl})"
        except Exception:
            disp_lbl = f"{custom_date} ({win_info['label']})"

        appointment["appointment_date"] = custom_date
        appointment["exact_time"] = custom_time
        appointment["appointment_time"] = custom_time or win_info["label"]
        appointment["window"] = custom_win
        appointment["window_label"] = win_info["label"]
        appointment["status"] = "rescheduled_pending_customer"
        save_appointment(appointment)

        if client_phone:
            cust_resched_msg = (
                f"Update from Comfort Breeze: Our technician can schedule your service for {disp_lbl}. "
                f"Does this day and time work for you? Reply YES to confirm. Ref: #{apt_id}"
            )
            await send_plivo_sms(client_phone, cust_resched_msg)

        owner_ack = f"⏱️ Custom proposal sent to {client_name} for {disp_lbl}. We'll alert you the moment they confirm."
        if clean_from:
            await send_plivo_sms(clean_from, owner_ack)

        return {
            "status": "rescheduled_by_owner",
            "appointment_id": apt_id,
            "new_date": custom_date,
            "new_window": disp_lbl,
            "exact_time": custom_time,
            "message": owner_ack,
        }

    raw = str(action_input or "").strip()
    norm = raw.lower().strip()

    # Strip appointment reference tag so text like '👍 #APT-12345' or '1 #APT-12345' matches cleanly
    clean_cmd = re.sub(r"#?apt-\d+", "", norm).strip()

    # Interactive selector query: "days", "schedule", "when", "options"
    if clean_cmd in ("days", "day", "options", "schedule", "when", "help", "slots"):
        opt1_lbl = options.get("1", {}).get("label", "Original Time")
        opt2_lbl = options.get("2", {}).get("label", "Alternate Window")
        opt3_lbl = options.get("3", {}).get("label", "Next Day")
        help_sms = (
            f"📅 Select day & time for #{apt_id}:\n"
            f"Format: YYYY-MM-DD:HHMM\n"
            f"Example: 2026-10-15:1330\n\n"
            f"Or reply with a quick option:\n"
            f"1️⃣ {opt1_lbl}\n"
            f"2️⃣ {opt2_lbl}\n"
            f"3️⃣ {opt3_lbl}\n"
            f"4️⃣ Decline / Busy\n\n"
            f"📲 Or tap: {action_url}"
        )
        if clean_from:
            await send_plivo_sms(clean_from, help_sms)
        return {
            "status": "schedule_options_sent",
            "appointment_id": apt_id,
            "message": help_sms,
        }

    # Normalize emojis, letters A-D, and common words to action keys
    action_key = None
    if clean_cmd in ("1", "1️⃣", "a", "opt1", "one", "yes", "y", "confirm", "ok", "okay", "approved", "sure", "sounds good", "👍", "great", "done") or "👍" in norm or "1️⃣" in norm:
        action_key = "1"
    elif clean_cmd in ("2", "2️⃣", "b", "opt2", "two", "afternoon", "pm", "later", "lunch", "second") or "2️⃣" in norm:
        action_key = "2"
    elif clean_cmd in ("3", "3️⃣", "c", "opt3", "three", "tomorrow", "next day", "mon", "monday", "next morning", "third") or "3️⃣" in norm:
        action_key = "3"
    elif clean_cmd in ("4", "4️⃣", "d", "opt4", "four", "decline", "no", "busy", "cant", "cancel", "pass", "full", "reject") or "4️⃣" in norm:
        action_key = "4"

    # Action 1: Confirm Option 1 (or Customer's requested slot if rescheduled)
    if action_key == "1":
        if appointment.get("status") == "rescheduled_by_customer":
            confirmed_date = appointment.get("appointment_date")
            confirmed_win = appointment.get("window", "morning")
            confirmed_label = appointment.get("display_label") or appointment.get("window_label")
        else:
            opt = options.get("1", {})
            confirmed_date = opt.get("date", appointment.get("appointment_date"))
            confirmed_win = opt.get("window", appointment.get("window"))
            confirmed_label = opt.get("label", appointment.get("appointment_date"))
            appointment["appointment_date"] = confirmed_date
            appointment["window"] = confirmed_win
            appointment["window_label"] = opt.get("window_label", appointment.get("window_label"))

        appointment["status"] = "confirmed"
        appointment["confirmed_at"] = datetime.now().isoformat()
        save_appointment(appointment)

        # Sync to Google Calendar (using dedicated project credentials if available)
        cid = appointment.get("client_id")
        if cid:
            try:
                from app.project_db import get_project_calendar_config
                p_cal = get_project_calendar_config(cid)
                sa_data = p_cal.get("service_account_json", "")
                cal_id = p_cal.get("calendar_id", "primary")
            except Exception:
                sa_data = cfg.get("google_service_account_json", "")
                cal_id = cfg.get("google_calendar_id", "primary")
        else:
            sa_data = cfg.get("google_service_account_json", "")
            cal_id = cfg.get("google_calendar_id", "primary")

        event_id = await create_google_calendar_event(
            appointment, calendar_id=cal_id, service_account_data=sa_data
        )
        if event_id:
            appointment["google_event_id"] = event_id
        save_appointment(appointment)
        if cid:
            try:
                from app.project_db import save_project_appointment
                save_project_appointment(cid, appointment)
            except Exception:
                pass

        time_desc = appointment.get("exact_time") or appointment.get("appointment_time") or appointment.get("window_label")
        disp_time = f"{appointment['appointment_date']} at {time_desc}" if appointment.get("exact_time") else f"{appointment['appointment_date']} during the {appointment.get('window_label', '')} window"

        # Text Customer confirmation
        if client_phone:
            cust_confirm_msg = (
                f"✅ Confirmed! Comfort Breeze will arrive at {address} on {disp_time}. "
                f"Our technician will text you 30 minutes before arrival. Ref: #{apt_id}"
            )
            await send_plivo_sms(client_phone, cust_confirm_msg)

        owner_ack = f"✅ Confirmed! Job #{apt_id} for {client_name} is locked in for {disp_time}. Customer has been notified."
        if clean_from:
            await send_plivo_sms(clean_from, owner_ack)

        return {
            "status": "owner_confirmed",
            "appointment_id": apt_id,
            "date": appointment["appointment_date"],
            "window": appointment["window_label"],
            "exact_time": appointment.get("exact_time"),
            "message": owner_ack,
        }

    # Action 2: Shift to Option 2
    elif action_key == "2":
        opt = options.get("2", {})
        appointment["appointment_date"] = opt.get("date", appointment.get("appointment_date"))
        appointment["window"] = opt.get("window", "afternoon")
        appointment["window_label"] = opt.get("window_label", HVAC_WINDOWS[appointment["window"]]["label"])
        appointment["status"] = "rescheduled_pending_customer"
        save_appointment(appointment)

        if client_phone:
            cust_resched_msg = (
                f"Update from Comfort Breeze: Our technician had a routing adjustment and can arrive on {opt.get('date')} during the {opt.get('label')} window instead. "
                f"Does this time work for you? Reply YES to confirm. Ref: #{apt_id}"
            )
            await send_plivo_sms(client_phone, cust_resched_msg)

        owner_ack = f"⏱️ Shifted! Proposed {opt.get('label')} to {client_name}. We'll alert you the moment they reply."
        if clean_from:
            await send_plivo_sms(clean_from, owner_ack)

        return {
            "status": "rescheduled_by_owner",
            "appointment_id": apt_id,
            "new_date": opt.get("date"),
            "new_window": opt.get("label"),
            "message": owner_ack,
        }

    # Action 3: Shift to Option 3
    elif action_key == "3":
        opt = options.get("3", {})
        appointment["appointment_date"] = opt.get("date", appointment.get("appointment_date"))
        appointment["window"] = opt.get("window", "morning")
        appointment["window_label"] = opt.get("window_label", HVAC_WINDOWS[appointment["window"]]["label"])
        appointment["status"] = "rescheduled_pending_customer"
        save_appointment(appointment)

        if client_phone:
            cust_resched_msg = (
                f"Update from Comfort Breeze: Our technician's schedule is fully committed today, but we have an opening on {opt.get('date')} during the {opt.get('label')} window. "
                f"Does this time work for you? Reply YES to confirm. Ref: #{apt_id}"
            )
            await send_plivo_sms(client_phone, cust_resched_msg)

        owner_ack = f"⏱️ Shifted! Proposed {opt.get('label')} to {client_name}. We'll alert you the moment they reply."
        if clean_from:
            await send_plivo_sms(clean_from, owner_ack)

        return {
            "status": "rescheduled_by_owner",
            "appointment_id": apt_id,
            "new_date": opt.get("date"),
            "new_window": opt.get("label"),
            "message": owner_ack,
        }

    # Action 4: Decline
    elif action_key == "4":
        appointment["status"] = "declined"
        save_appointment(appointment)

        if client_phone:
            cust_decline_msg = (
                f"Notice from Comfort Breeze: Our technicians are currently at peak capacity and unable to take on this service slot. "
                f"We apologize for the inconvenience! Please call us if you'd like to arrange another day."
            )
            await send_plivo_sms(client_phone, cust_decline_msg)

        owner_ack = f"❌ Job #{apt_id} marked as Declined. Customer has been notified."
        if clean_from:
            await send_plivo_sms(clean_from, owner_ack)

        return {
            "status": "declined",
            "appointment_id": apt_id,
            "message": owner_ack,
        }

    # Smart Flexible Parsing: "in 2 weeks Friday morning", "in 3 days afternoon", "next Tuesday 2pm", "Oct 15", etc.
    parsed_intent = await parse_flexible_schedule_intent(raw)
    if parsed_intent:
        custom_date = parsed_intent["date"]
        custom_win = parsed_intent["window"]
        exact_time = parsed_intent.get("exact_time")
        disp_lbl = parsed_intent["display_label"]

        appointment["appointment_date"] = custom_date
        appointment["exact_time"] = exact_time
        appointment["appointment_time"] = exact_time or parsed_intent["window_label"]
        appointment["window"] = custom_win
        appointment["window_label"] = parsed_intent["window_label"]
        appointment["display_label"] = disp_lbl
        appointment["status"] = "rescheduled_pending_customer"
        save_appointment(appointment)

        if client_phone:
            cust_resched_msg = (
                f"Update from Comfort Breeze: Our technician can schedule your service for {disp_lbl}. "
                f"Does this day and time work for you? Reply YES to confirm. Ref: #{apt_id}"
            )
            await send_plivo_sms(client_phone, cust_resched_msg)

        owner_ack = f"⏱️ Flexible proposal sent to {client_name} for {disp_lbl}. We'll alert you when they confirm."
        if clean_from:
            await send_plivo_sms(clean_from, owner_ack)

        return {
            "status": "rescheduled_by_owner",
            "appointment_id": apt_id,
            "new_date": custom_date,
            "new_window": disp_lbl,
            "exact_time": exact_time,
            "message": owner_ack,
        }

    # Safety-Net Guide: Unrecognized input
    opt1_lbl = options.get("1", {}).get("label", "Original Time")
    opt2_lbl = options.get("2", {}).get("label", "Alternate Window")
    opt3_lbl = options.get("3", {}).get("label", "Next Day")

    help_sms = (
        f"❓ Reply with a number for #{apt_id}:\n"
        f"1️⃣ Confirm ({opt1_lbl})\n"
        f"2️⃣ Move to {opt2_lbl}\n"
        f"3️⃣ Move to {opt3_lbl}\n"
        f"4️⃣ Decline / Busy\n\n"
        f"📅 Or text ANY exact day & time or window:\n"
        f"Format: YYYY-MM-DD:HHMM\n"
        f"Example: 2026-10-15:1330\n\n"
        f"Or tap: {action_url}"
    )
    if clean_from:
        await send_plivo_sms(clean_from, help_sms)

    return {
        "status": "safety_net_prompted",
        "appointment_id": apt_id,
        "message": "Sent safety net numbered choices to owner.",
    }


# ── Inbound SMS Webhook Processor ─────────────────────────────────────────────

async def process_inbound_sms(from_number: str, message_text: str) -> Dict[str, Any]:
    """
    Processes incoming SMS messages from Plivo.
    Handles:
    - Owner messages -> delegates to apply_appointment_action with numbered options, smart dates & safety net.
    - Customer replies 'yes' / 'confirm' -> finalizes appointment.
    - Customer replies with new date/time -> updates appointment and notifies owner.
    """
    from app.integrations import get_integrations_settings

    cfg = get_integrations_settings()
    owner_phone = cfg.get("owner_phone_number", "").strip()
    clean_from = re.sub(r"[^\d+]", "", from_number)
    clean_owner = re.sub(r"[^\d+]", "", owner_phone)
    text = (message_text or "").strip()

    logger.info(f"Incoming SMS from {clean_from}: '{text}'")

    # Check if text contains a specific #APT-XXXX reference code
    ref_match = re.search(r"#?(APT-\d{5})", text, re.IGNORECASE)
    if ref_match:
        appointment = find_appointment_by_id(ref_match.group(1))
    else:
        appointment = find_latest_pending_appointment_for_phone(clean_from)

    if not appointment:
        logger.warning(f"No active pending appointment found for SMS from {clean_from}")
        return {"status": "unmatched", "message": "No active appointment found for this message."}

    apt_id = appointment.get("id")
    client_name = appointment.get("client_name", "Customer")
    date_str = appointment.get("appointment_date", "")
    win_label = appointment.get("window_label", "")

    # Is the sender the Owner?
    is_owner = bool(clean_owner and (clean_from.endswith(clean_owner[-10:]) or clean_owner.endswith(clean_from[-10:])))

    if is_owner:
        return await apply_appointment_action(appointment, text, from_phone=clean_from)

    # If sender is the Customer
    else:
        if text.lower() in ("yes", "y", "confirm", "ok", "sure", "sounds good", "👍"):
            appointment["status"] = "confirmed"
            appointment["confirmed_at"] = datetime.now().isoformat()
            save_appointment(appointment)

            # Sync to Google Calendar (using dedicated project credentials if available)
            cid = appointment.get("client_id")
            if cid:
                try:
                    from app.project_db import get_project_calendar_config
                    p_cal = get_project_calendar_config(cid)
                    sa_data = p_cal.get("service_account_json", "")
                    cal_id = p_cal.get("calendar_id", "primary")
                except Exception:
                    sa_data = cfg.get("google_service_account_json", "")
                    cal_id = cfg.get("google_calendar_id", "primary")
            else:
                sa_data = cfg.get("google_service_account_json", "")
                cal_id = cfg.get("google_calendar_id", "primary")

            event_id = await create_google_calendar_event(
                appointment, calendar_id=cal_id, service_account_data=sa_data
            )
            if event_id:
                appointment["google_event_id"] = event_id
            save_appointment(appointment)
            if cid:
                try:
                    from app.project_db import save_project_appointment
                    save_project_appointment(cid, appointment)
                except Exception:
                    pass

            time_desc = appointment.get("exact_time") or appointment.get("appointment_time") or appointment.get("window_label")
            disp_time = f"{date_str} at {time_desc}" if appointment.get("exact_time") else f"{date_str} ({win_label})"

            # Text Customer
            await send_plivo_sms(clean_from, f"✅ You're all set! We have you confirmed for {disp_time}. See you then! - Comfort Breeze")

            # Alert Owner
            if owner_phone:
                await send_plivo_sms(owner_phone, f"🎉 Customer {client_name} confirmed adjusted slot: {disp_time} for #{apt_id}.")

            return {"status": "customer_confirmed", "appointment_id": apt_id}

        # Customer texts a different day or time! E.g. "Can we do in 2 weeks Friday at 2:30pm?"
        cust_intent = await parse_flexible_schedule_intent(text)
        if cust_intent:
            custom_date = cust_intent["date"]
            custom_win = cust_intent["window"]
            exact_time = cust_intent.get("exact_time")
            disp_lbl = cust_intent["display_label"]

            appointment["appointment_date"] = custom_date
            appointment["exact_time"] = exact_time
            appointment["appointment_time"] = exact_time or cust_intent["window_label"]
            appointment["window"] = custom_win
            appointment["window_label"] = cust_intent["window_label"]
            appointment["status"] = "rescheduled_by_customer"
            save_appointment(appointment)

            cust_ack = (
                f"Got it, {client_name}! We received your request for {disp_lbl}. "
                f"Our technician is reviewing the schedule and will confirm shortly. Ref: #{apt_id}"
            )
            await send_plivo_sms(clean_from, cust_ack)

            if owner_phone:
                action_url = appointment.get("action_url") or f"{(settings.PUBLIC_URL or 'http://localhost:7860').rstrip('/')}/a/{apt_id}"
                owner_alert = (
                    f"📩 Customer {client_name} requested new slot for #{apt_id}:\n"
                    f"📅 {disp_lbl}\n\n"
                    f"Reply '1' to confirm, or text another day/time to counter-propose!\n"
                    f"📲 Or tap: {action_url}"
                )
                await send_plivo_sms(owner_phone, owner_alert)

            return {
                "status": "customer_rescheduled",
                "appointment_id": apt_id,
                "new_date": custom_date,
                "new_time": disp_lbl,
                "exact_time": exact_time,
                "message": cust_ack,
            }

    return {"status": "unhandled_intent", "text": text}
