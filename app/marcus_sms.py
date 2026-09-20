"""
app.marcus_sms
--------------
Dedicated two-way SMS communication manager for Marcus / Ana Outbound Campaigns.
Handles:
- Storing outbound follow-up SMS (demo links, pricing, activation)
- Storing incoming prospect SMS replies & inquiries via Telnyx webhooks
- Fetching aggregated conversation threads by business / phone
- Dispatching manual replies via Telnyx REST API
- Linking SMS threads with voice call recordings and summaries
"""

import os
import re
import time
import uuid
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SMS_DB_FILE = DATA_DIR / "marcus_sms.db"


def _get_db_conn() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(SMS_DB_FILE), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_marcus_sms_db():
    """Ensure the SMS messages table and indexes exist."""
    conn = _get_db_conn()
    cur = conn.cursor()
    try:
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS marcus_sms_messages (
            id TEXT PRIMARY KEY,
            phone_number TEXT NOT NULL,
            business_name TEXT,
            direction TEXT NOT NULL, -- 'outbound' or 'inbound'
            text TEXT NOT NULL,
            status TEXT DEFAULT 'sent', -- 'sent', 'received', 'failed'
            provider TEXT DEFAULT 'telnyx',
            message_id TEXT,
            call_id TEXT,
            created_at TEXT NOT NULL,
            read_status INTEGER DEFAULT 0 -- 0=unread, 1=read
        );

        CREATE INDEX IF NOT EXISTS idx_marcus_sms_phone ON marcus_sms_messages(phone_number);
        CREATE INDEX IF NOT EXISTS idx_marcus_sms_created ON marcus_sms_messages(created_at);
        """)
        conn.commit()
    finally:
        conn.close()


def normalize_phone(phone: str) -> str:
    """Normalize phone number to standard E.164-like format."""
    clean = re.sub(r"[^\d+]", "", (phone or "").strip())
    if not clean.startswith("+") and len(clean) == 10:
        return f"+1{clean}"
    elif not clean.startswith("+") and len(clean) == 11 and clean.startswith("1"):
        return f"+{clean}"
    return clean


def record_sms_message(
    phone_number: str,
    text: str,
    direction: str,
    business_name: Optional[str] = None,
    call_id: Optional[str] = None,
    status: str = "sent",
    provider: str = "telnyx",
    message_id: Optional[str] = None,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Records an outbound or inbound SMS message into the SQLite database."""
    init_marcus_sms_db()
    norm_phone = normalize_phone(phone_number)
    msg_id = message_id or f"msg_{uuid.uuid4().hex[:12]}"
    now_str = created_at or time.strftime("%Y-%m-%d %H:%M:%S")

    # If business_name not passed, try resolving from past records
    resolved_biz = business_name
    if not resolved_biz:
        resolved_biz = find_business_name_for_phone(norm_phone) or "HVAC Contractor"

    read_val = 1 if direction == "outbound" else 0

    conn = _get_db_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO marcus_sms_messages (
                id, phone_number, business_name, direction, text,
                status, provider, message_id, call_id, created_at, read_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            msg_id, norm_phone, resolved_biz, direction, text,
            status, provider, msg_id, call_id, now_str, read_val
        ))
        conn.commit()
    except Exception as e:
        logger.error(f"[Marcus SMS] Error recording message: {e}")
    finally:
        conn.close()

    return {
        "id": msg_id,
        "phone_number": norm_phone,
        "business_name": resolved_biz,
        "direction": direction,
        "text": text,
        "status": status,
        "provider": provider,
        "call_id": call_id,
        "created_at": now_str,
        "read_status": read_val,
    }


def find_business_name_for_phone(phone_number: str) -> Optional[str]:
    """Look up business name for a phone number across SMS records and call logs."""
    norm = normalize_phone(phone_number)
    conn = _get_db_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT business_name FROM marcus_sms_messages
            WHERE phone_number = ? AND business_name IS NOT NULL AND business_name != ''
            ORDER BY created_at DESC LIMIT 1
        """, (norm,))
        row = cur.fetchone()
        if row and row["business_name"]:
            return row["business_name"]
    except Exception:
        pass
    finally:
        conn.close()

    # Fallback to calls.json
    try:
        from app.calls import list_calls
        calls = list_calls(limit=100)
        digits = re.sub(r"[^\d]", "", norm)[-10:]
        for c in calls:
            caller_clean = re.sub(r"[^\d]", "", c.get("caller", ""))[-10:]
            called_clean = re.sub(r"[^\d]", "", c.get("called", ""))[-10:]
            if digits and (digits == caller_clean or digits == called_clean):
                biz = c.get("business_name") or (c.get("extracted_info") or {}).get("business_name")
                if biz:
                    return biz
    except Exception:
        pass

    return None


def get_sms_conversations() -> List[Dict[str, Any]]:
    """Returns a list of all distinct conversations grouped by phone number."""
    init_marcus_sms_db()
    conn = _get_db_conn()
    cur = conn.cursor()
    results = []
    try:
        # Group by phone_number to find latest message and unread count
        cur.execute("""
            SELECT 
                phone_number,
                MAX(created_at) as last_time
            FROM marcus_sms_messages
            GROUP BY phone_number
            ORDER BY last_time DESC
        """)
        groups = cur.fetchall()

        for g in groups:
            phone = g["phone_number"]
            # Fetch latest message info
            cur.execute("""
                SELECT * FROM marcus_sms_messages
                WHERE phone_number = ?
                ORDER BY created_at DESC LIMIT 1
            """, (phone,))
            latest = cur.fetchone()

            # Count unread inbound messages
            cur.execute("""
                SELECT COUNT(*) as unread FROM marcus_sms_messages
                WHERE phone_number = ? AND direction = 'inbound' AND read_status = 0
            """, (phone,))
            unread_row = cur.fetchone()
            unread_cnt = unread_row["unread"] if unread_row else 0

            # Count total messages
            cur.execute("""
                SELECT COUNT(*) as total FROM marcus_sms_messages
                WHERE phone_number = ?
            """, (phone,))
            tot_row = cur.fetchone()
            total_cnt = tot_row["total"] if tot_row else 0

            results.append({
                "phone_number": phone,
                "business_name": latest["business_name"] if latest else "HVAC Business",
                "last_message": latest["text"] if latest else "",
                "last_timestamp": latest["created_at"] if latest else "",
                "last_direction": latest["direction"] if latest else "outbound",
                "last_status": latest["status"] if latest else "sent",
                "unread_count": unread_cnt,
                "total_messages": total_cnt,
                "call_id": latest["call_id"] if latest else None,
            })
    except Exception as e:
        logger.error(f"[Marcus SMS] Error listing conversations: {e}")
    finally:
        conn.close()

    return results


def get_conversation_thread(phone_number: str) -> Dict[str, Any]:
    """
    Returns full chronological message history for a phone number
    and marks inbound messages as read.
    """
    init_marcus_sms_db()
    norm_phone = normalize_phone(phone_number)
    conn = _get_db_conn()
    cur = conn.cursor()
    messages = []
    biz_name = "HVAC Business"
    call_id = None
    try:
        # Mark inbound as read
        cur.execute("""
            UPDATE marcus_sms_messages
            SET read_status = 1
            WHERE phone_number = ? AND direction = 'inbound' AND read_status = 0
        """, (norm_phone,))
        conn.commit()

        cur.execute("""
            SELECT * FROM marcus_sms_messages
            WHERE phone_number = ?
            ORDER BY created_at ASC
        """, (norm_phone,))
        rows = cur.fetchall()
        for r in rows:
            if r["business_name"] and r["business_name"] != "HVAC Contractor":
                biz_name = r["business_name"]
            if r["call_id"]:
                call_id = r["call_id"]
            messages.append({
                "id": r["id"],
                "direction": r["direction"],
                "text": r["text"],
                "status": r["status"],
                "provider": r["provider"],
                "created_at": r["created_at"],
                "call_id": r["call_id"],
            })
    except Exception as e:
        logger.error(f"[Marcus SMS] Error fetching thread for {norm_phone}: {e}")
    finally:
        conn.close()

    # Look up call metadata if available
    call_meta = None
    if call_id:
        try:
            from app.calls import get_call
            call_meta = get_call(call_id)
        except Exception:
            pass

    return {
        "phone_number": norm_phone,
        "business_name": biz_name,
        "call_id": call_id,
        "call_summary": (call_meta or {}).get("extracted_info", {}).get("summary", ""),
        "recording_file": (call_meta or {}).get("recording_file", ""),
        "messages": messages,
    }


async def send_manual_reply(phone_number: str, text: str, business_name: Optional[str] = None) -> Dict[str, Any]:
    """Dispatches a manual SMS reply to a prospect via Telnyx and records the message."""
    from app.integrations import send_sms
    norm_phone = normalize_phone(phone_number)
    clean_text = (text or "").strip()
    if not clean_text:
        raise ValueError("Message text cannot be empty")

    res = await send_sms(to_phone=norm_phone, message=clean_text, provider="telnyx")
    status = "sent" if res.get("status") in ("sent", "simulated_success") else "failed"

    record = record_sms_message(
        phone_number=norm_phone,
        text=clean_text,
        direction="outbound",
        business_name=business_name,
        status=status,
        provider="telnyx",
        message_id=res.get("message_id"),
    )
    return {
        "success": status == "sent",
        "result": res,
        "message": record,
    }


def handle_inbound_sms(from_phone: str, text: str, provider: str = "telnyx", message_id: Optional[str] = None) -> Dict[str, Any]:
    """Processes an incoming SMS from a contractor/prospect and records it."""
    norm_phone = normalize_phone(from_phone)
    clean_text = (text or "").strip()
    biz_name = find_business_name_for_phone(norm_phone) or "HVAC Contractor"

    logger.info(f"[Marcus Inbound SMS] Received from {norm_phone} ({biz_name}): {clean_text}")

    record = record_sms_message(
        phone_number=norm_phone,
        text=clean_text,
        direction="inbound",
        business_name=biz_name,
        status="received",
        provider=provider,
        message_id=message_id,
    )
    return record


# Initialize database schema upon import
init_marcus_sms_db()
