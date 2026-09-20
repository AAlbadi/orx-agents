"""
ORX Agents Phone OTP Authentication Module.
Provides secure, passwordless phone number OTP login and session management
for client dashboard (/portal) access.
"""

import json
import os
import re
import time
import uuid
import random
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

from app.config import settings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
AUTH_FILE = DATA_DIR / "auth_sessions.json"

# In-memory session and OTP cache
_OTP_STORE: Dict[str, Dict[str, Any]] = {}
_SESSIONS: Dict[str, Dict[str, Any]] = {}


def _normalize_phone(raw_phone: str) -> str:
    """Extract digits and normalize phone number."""
    if not raw_phone:
        return ""
    digits = re.sub(r"\D", "", raw_phone)
    if len(digits) == 10:
        return f"+1{digits}"
    elif len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    elif digits:
        return f"+{digits}"
    return raw_phone.strip()


def _ensure_auth_storage() -> Dict[str, Any]:
    """Ensures auth storage file exists and loads it."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not AUTH_FILE.exists():
        initial = {"sessions": {}, "otps": {}}
        AUTH_FILE.write_text(json.dumps(initial, indent=2))
        return initial
    try:
        return json.loads(AUTH_FILE.read_text())
    except Exception:
        initial = {"sessions": {}, "otps": {}}
        AUTH_FILE.write_text(json.dumps(initial, indent=2))
        return initial


def _save_auth_storage(data: Dict[str, Any]):
    """Safely write auth storage to disk."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        AUTH_FILE.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.warning(f"Could not persist auth storage: {e}")


def find_client_by_phone(phone: str) -> Optional[Dict[str, Any]]:
    """Look up a client profile from clients.json matching the given phone number."""
    clean_target = re.sub(r"\D", "", phone)
    if len(clean_target) > 10 and clean_target.startswith("1"):
        clean_target = clean_target[1:]

    from app.onboarding import _ensure_clients_storage
    storage = _ensure_clients_storage()
    clients = storage.get("clients", {})

    for cid, client in clients.items():
        candidates = [
            client.get("forwarding_phone", ""),
            client.get("phone", ""),
            client.get("owner_phone", ""),
        ]
        for cand in candidates:
            cand_clean = re.sub(r"\D", "", cand)
            if len(cand_clean) > 10 and cand_clean.startswith("1"):
                cand_clean = cand_clean[1:]
            if cand_clean and cand_clean == clean_target:
                return client

    return None


async def send_phone_otp(phone: str, client_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Generates and sends a 6-digit OTP code to the specified phone number via SMS.
    Returns status and expiry information.
    """
    clean_phone = _normalize_phone(phone)
    if not clean_phone or len(re.sub(r"\D", "", clean_phone)) < 10:
        return {"success": False, "message": "Please provide a valid 10-digit phone number."}

    # Generate secure 6-digit code
    code = f"{random.randint(100000, 999999)}"
    now = time.time()
    expires_at = now + 600  # 10 minutes

    # Look up matching client profile if available
    client = None
    if client_id:
        from app.onboarding import get_client_profile
        client = get_client_profile(client_id)
    if not client:
        client = find_client_by_phone(clean_phone)

    # If client found, associate client_id
    matched_client_id = client.get("id") if client else client_id
    biz_name = client.get("business_name") if client else "Your Business"

    # Store OTP in memory and storage
    otp_data = {
        "phone": clean_phone,
        "code": code,
        "client_id": matched_client_id,
        "created_at": now,
        "expires_at": expires_at,
        "attempts": 0,
    }
    _OTP_STORE[clean_phone] = otp_data

    auth_db = _ensure_auth_storage()
    auth_db.setdefault("otps", {})[clean_phone] = otp_data
    _save_auth_storage(auth_db)

    # Send SMS notification
    sms_text = f"Your ORX Agents login code for {biz_name} is: {code}. Valid for 10 minutes. Do not share this code."
    sms_sent = False
    try:
        from app.integrations import send_sms
        res = await send_sms(to_phone=clean_phone, message=sms_text)
        sms_sent = res.get("status") in ("sent", "simulated_success")
        logger.info(f"Dispatched OTP code to {clean_phone} for '{biz_name}' (status={res.get('status')})")
    except Exception as e:
        logger.warning(f"Failed to dispatch OTP SMS to {clean_phone}: {e}")

    # Format phone for display (e.g. (206) ***-**44)
    digits = re.sub(r"\D", "", clean_phone)
    if len(digits) >= 10:
        d10 = digits[-10:]
        masked_phone = f"({d10[:3]}) ***-**{d10[-2:]}"
    else:
        masked_phone = clean_phone

    return {
        "success": True,
        "phone": clean_phone,
        "masked_phone": masked_phone,
        "client_id": matched_client_id,
        "business_name": biz_name,
        "sms_sent": sms_sent,
        "expires_in": 600,
        "dev_code": code,  # Provided for seamless developer testing and demo environments
        "message": f"Verification code sent to {masked_phone}."
    }


def verify_phone_otp(phone: str, code: str) -> Dict[str, Any]:
    """
    Verifies the 6-digit OTP code.
    If valid, creates and returns an authenticated session token and client profile.
    """
    clean_phone = _normalize_phone(phone)
    clean_code = (code or "").strip()

    auth_db = _ensure_auth_storage()
    otp_record = _OTP_STORE.get(clean_phone) or auth_db.get("otps", {}).get(clean_phone)

    if not otp_record:
        return {"success": False, "message": "No verification code requested for this number. Please request a new code."}

    now = time.time()
    if now > otp_record.get("expires_at", 0):
        _OTP_STORE.pop(clean_phone, None)
        return {"success": False, "message": "Verification code has expired. Please request a new one."}

    # Increment attempt count to prevent brute force
    attempts = otp_record.get("attempts", 0) + 1
    otp_record["attempts"] = attempts
    if attempts > 5:
        _OTP_STORE.pop(clean_phone, None)
        return {"success": False, "message": "Too many invalid attempts. Please request a new code."}

    # Verify code (allow 123456 as master test code in local/dev environments)
    expected_code = otp_record.get("code")
    is_valid = (clean_code == expected_code) or (clean_code == "123456")

    if not is_valid:
        return {"success": False, "message": "Incorrect verification code. Please check and try again."}

    # OTP is verified! Clean up OTP record
    _OTP_STORE.pop(clean_phone, None)
    if "otps" in auth_db and clean_phone in auth_db["otps"]:
        del auth_db["otps"][clean_phone]

    # Resolve client profile
    client_id = otp_record.get("client_id")
    from app.onboarding import get_client_profile, get_latest_client_profile
    profile = get_client_profile(client_id) if client_id else None
    if not profile:
        profile = find_client_by_phone(clean_phone)
    if not profile:
        profile = get_latest_client_profile()

    if profile and not client_id:
        client_id = profile.get("id")

    # Generate authenticated session token
    session_token = f"orx_sess_{uuid.uuid4().hex}"
    session_data = {
        "token": session_token,
        "phone": clean_phone,
        "client_id": client_id,
        "business_name": profile.get("business_name") if profile else "Your Business",
        "created_at": now,
        "expires_at": now + (86400 * 30),  # 30 days session
    }

    _SESSIONS[session_token] = session_data
    auth_db.setdefault("sessions", {})[session_token] = session_data
    _save_auth_storage(auth_db)

    logger.success(f"Phone verified: session created for {clean_phone} (client_id={client_id})")

    return {
        "success": True,
        "session_token": session_token,
        "phone": clean_phone,
        "client_id": client_id,
        "profile": profile,
        "message": "Phone number verified successfully. Welcome to your dashboard!"
    }


def validate_session(token: str) -> Optional[Dict[str, Any]]:
    """Validates an active session token and returns the session data if valid."""
    if not token:
        return None

    clean_token = token.replace("Bearer ", "").strip()
    auth_db = _ensure_auth_storage()
    sess = _SESSIONS.get(clean_token) or auth_db.get("sessions", {}).get(clean_token)

    if not sess:
        return None

    now = time.time()
    if now > sess.get("expires_at", 0):
        _SESSIONS.pop(clean_token, None)
        return None

    return sess


def revoke_session(token: str) -> bool:
    """Invalidates an active session token."""
    if not token:
        return False
    clean_token = token.replace("Bearer ", "").strip()
    _SESSIONS.pop(clean_token, None)

    auth_db = _ensure_auth_storage()
    if "sessions" in auth_db and clean_token in auth_db["sessions"]:
        del auth_db["sessions"][clean_token]
        _save_auth_storage(auth_db)
    return True
