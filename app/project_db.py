"""Dedicated Client Project Database Engine for Multi-Tenant Aria Voice AI.

Every client project (HVAC, plumbing, electrical, legal, medical, etc.) gets:
1. An isolated directory: `data/projects/{client_id}/`
2. A dedicated SQLite database: `data/projects/{client_id}/client.db`
3. Synced human-readable JSON files (`project.json`, `prompt.json`, `calendar.json`)
4. Isolated custom prompts, Google Calendar configurations, appointments, and call logs.

Can be triggered either:
- Automatically when a client completes onboarding (`trigger_source="onboarding"`)
- Manually by an admin via the Dashboard / API (`trigger_source="admin"`)
"""

import json
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger

DATA_DIR = Path(__file__).parent.parent / "data"
PROJECTS_DIR = DATA_DIR / "projects"
PROJECTS_INDEX_FILE = PROJECTS_DIR / "index.json"
CLIENTS_FILE = DATA_DIR / "clients.json"


def get_projects_dir() -> Path:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    return PROJECTS_DIR


def get_project_dir(client_id: str) -> Path:
    clean_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', client_id)
    pdir = get_projects_dir() / clean_id
    pdir.mkdir(parents=True, exist_ok=True)
    return pdir


def get_db_path(client_id: str) -> Path:
    return get_project_dir(client_id) / "client.db"


def get_db_connection(client_id: str) -> sqlite3.Connection:
    db_path = get_db_path(client_id)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_project_db(client_id: str):
    """Initializes dedicated SQLite tables for a client project."""
    conn = get_db_connection(client_id)
    cur = conn.cursor()
    try:
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS project_meta (
            client_id TEXT PRIMARY KEY,
            project_name TEXT,
            business_name TEXT,
            industry TEXT,
            address TEXT,
            forwarding_phone TEXT,
            assigned_phone TEXT,
            owner_phone TEXT,
            owner_email TEXT,
            timezone TEXT DEFAULT 'America/New_York',
            status TEXT DEFAULT 'active',
            trigger_source TEXT DEFAULT 'admin',
            sms_notifications_enabled INTEGER DEFAULT 1,
            created_at TEXT,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS custom_prompt (
            client_id TEXT PRIMARY KEY,
            persona_name TEXT,
            system_prompt TEXT,
            livekit_prompt TEXT,
            first_message TEXT,
            tts_voice TEXT,
            voice_speed REAL DEFAULT 1.0,
            services TEXT,
            emergency_triggers TEXT,
            hours TEXT,
            pricing_policy TEXT,
            booking_action TEXT,
            custom_qa TEXT,
            updated_at TEXT,
            FOREIGN KEY(client_id) REFERENCES project_meta(client_id)
        );

        CREATE TABLE IF NOT EXISTS google_calendar_config (
            client_id TEXT PRIMARY KEY,
            calendar_id TEXT DEFAULT 'primary',
            service_account_json TEXT DEFAULT '',
            calendar_webhook_url TEXT DEFAULT '',
            sync_enabled INTEGER DEFAULT 1,
            morning_slot_capacity INTEGER DEFAULT 2,
            afternoon_slot_capacity INTEGER DEFAULT 2,
            evening_slot_capacity INTEGER DEFAULT 2,
            is_connected INTEGER DEFAULT 0,
            last_tested_at TEXT,
            last_status TEXT DEFAULT 'untested',
            last_error TEXT,
            FOREIGN KEY(client_id) REFERENCES project_meta(client_id)
        );

        CREATE TABLE IF NOT EXISTS appointments (
            id TEXT PRIMARY KEY,
            client_id TEXT,
            client_name TEXT,
            client_phone TEXT,
            service_requested TEXT,
            service_address TEXT,
            appointment_date TEXT,
            window TEXT,
            exact_time TEXT,
            status TEXT,
            google_event_id TEXT,
            summary TEXT,
            created_at TEXT,
            updated_at TEXT,
            raw_payload TEXT,
            FOREIGN KEY(client_id) REFERENCES project_meta(client_id)
        );

        CREATE TABLE IF NOT EXISTS call_logs (
            id TEXT PRIMARY KEY,
            client_id TEXT,
            caller_phone TEXT,
            call_duration REAL,
            recording_file TEXT,
            transcript TEXT,
            extracted_info TEXT,
            created_at TEXT,
            FOREIGN KEY(client_id) REFERENCES project_meta(client_id)
        );

        CREATE INDEX IF NOT EXISTS idx_appointments_date ON appointments(appointment_date);
        CREATE INDEX IF NOT EXISTS idx_appointments_status ON appointments(status);
        CREATE INDEX IF NOT EXISTS idx_call_logs_caller ON call_logs(caller_phone);
        """)

        # Migrations for existing project databases
        cur.execute("PRAGMA table_info(custom_prompt)")
        cp_cols = [c[1] for c in cur.fetchall()]
        if "livekit_prompt" not in cp_cols:
            try:
                cur.execute("ALTER TABLE custom_prompt ADD COLUMN livekit_prompt TEXT")
            except Exception:
                pass
        if "tone_preset" not in cp_cols:
            try:
                cur.execute("ALTER TABLE custom_prompt ADD COLUMN tone_preset TEXT DEFAULT 'warm_empathetic'")
            except Exception:
                pass

        cur.execute("PRAGMA table_info(project_meta)")
        pm_cols = [c[1] for c in cur.fetchall()]
        if "sms_notifications_enabled" not in pm_cols:
            try:
                cur.execute("ALTER TABLE project_meta ADD COLUMN sms_notifications_enabled INTEGER DEFAULT 1")
            except Exception:
                pass

        cur.execute("PRAGMA table_info(google_calendar_config)")
        gcc_cols = [c[1] for c in cur.fetchall()]
        for col_name, col_def in [
            ("oauth_access_token", "TEXT DEFAULT ''"),
            ("oauth_refresh_token", "TEXT DEFAULT ''"),
            ("oauth_token_expiry", "INTEGER DEFAULT 0"),
            ("oauth_user_email", "TEXT DEFAULT ''"),
            ("auth_type", "TEXT DEFAULT 'oauth'")
        ]:
            if col_name not in gcc_cols:
                try:
                    cur.execute(f"ALTER TABLE google_calendar_config ADD COLUMN {col_name} {col_def}")
                except Exception:
                    pass

        conn.commit()
    finally:
        conn.close()


def _write_json_safely(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(data, indent=2, default=str))
    temp_path.replace(path)


def compile_livekit_voice_prompt(data: Dict[str, Any]) -> str:
    """
    Compiles voice instructions tailored for LiveKit voice agents.
    Uses custom system prompt or livekit_prompt if explicitly provided (>400 chars),
    or generates a Marcus-grade master prompt dynamically from client onboarding data.
    """
    if data.get("livekit_prompt") and len(data["livekit_prompt"]) > 400:
        return data["livekit_prompt"].strip()
    if data.get("system_prompt") and len(data["system_prompt"]) > 400:
        return data["system_prompt"].strip()
    if data.get("compiled_prompt") and len(data["compiled_prompt"]) > 400:
        return data["compiled_prompt"].strip()
    
    from app.onboarding import compile_agent_prompt
    return compile_agent_prompt(data)


def sync_project_json_files(client_id: str):
    """Synchronizes SQLite data into human-readable JSON files in the project folder."""
    pdir = get_project_dir(client_id)
    project = get_project(client_id)
    if not project:
        return

    # 1. project.json
    _write_json_safely(pdir / "project.json", project.get("meta", {}))

    # 2. prompt.json
    prompt_dict = project.get("prompt", {}).copy()
    if "livekit_prompt" in project and "livekit_prompt" not in prompt_dict:
        prompt_dict["livekit_prompt"] = project["livekit_prompt"]
    prompt_dict["character_count"] = len(prompt_dict.get("livekit_prompt", ""))
    _write_json_safely(pdir / "prompt.json", prompt_dict)

    # 3. calendar.json
    _write_json_safely(pdir / "calendar.json", project.get("calendar", {}))


def create_project(data: Dict[str, Any], trigger_source: str = "admin") -> Dict[str, Any]:
    """
    Creates or registers a dedicated project for any client business type.
    Triggered either automatically post-onboarding or manually by admin.
    """
    client_id = data.get("id") or data.get("client_id")
    if not client_id:
        biz_slug = re.sub(r'[^a-zA-Z0-9]', '', data.get("business_name", "client")).lower()[:12]
        client_id = f"proj_{biz_slug}_{uuid.uuid4().hex[:6]}"

    clean_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', client_id)
    init_project_db(clean_id)

    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_connection(clean_id)
    cur = conn.cursor()

    try:
        # 1. Project Meta
        biz_name = data.get("business_name") or data.get("name") or "New Client Business"
        project_name = data.get("project_name") or f"{biz_name} Project"
        industry = data.get("industry") or "hvac"
        address = data.get("address") or ""
        forwarding = data.get("forwarding_phone") or ""
        assigned = data.get("assigned_phone") or data.get("phone") or "+1 (833) 420-5227"
        owner_phone = data.get("owner_phone") or forwarding
        owner_email = data.get("owner_email") or data.get("email") or ""
        timezone = data.get("timezone") or "America/New_York"
        status = data.get("status") or "active"
        sms_enabled = 1 if data.get("sms_notifications_enabled", True) else 0

        cur.execute("""
            INSERT INTO project_meta (
                client_id, project_name, business_name, industry, address,
                forwarding_phone, assigned_phone, owner_phone, owner_email,
                timezone, status, trigger_source, sms_notifications_enabled,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(client_id) DO UPDATE SET
                project_name=excluded.project_name,
                business_name=excluded.business_name,
                industry=excluded.industry,
                address=excluded.address,
                forwarding_phone=excluded.forwarding_phone,
                assigned_phone=excluded.assigned_phone,
                owner_phone=excluded.owner_phone,
                owner_email=excluded.owner_email,
                timezone=excluded.timezone,
                status=excluded.status,
                sms_notifications_enabled=excluded.sms_notifications_enabled,
                updated_at=excluded.updated_at
        """, (
            clean_id, project_name, biz_name, industry, address,
            forwarding, assigned, owner_phone, owner_email,
            timezone, status, trigger_source, sms_enabled, now_str, now_str
        ))

        # 2. Custom Prompt & Character-Optimized LiveKit Voice Instructions
        persona_name = data.get("persona_name") or "Riley"
        from app.onboarding import compile_agent_prompt
        system_prompt = data.get("compiled_prompt") or data.get("system_prompt") or ""
        if not system_prompt or len(system_prompt) < 400:
            system_prompt = compile_agent_prompt(data)

        livekit_prompt = data.get("livekit_prompt")
        if not livekit_prompt or len(livekit_prompt) < 400:
            livekit_prompt = system_prompt

        first_message = data.get("first_message") or (
            f"Thank you for calling {biz_name}. This is {persona_name}, your virtual receptionist. "
            f"How may I help get your service scheduled today?"
        )
        tts_voice = data.get("persona_voice") or data.get("tts_voice") or "af_heart"
        voice_speed = float(data.get("voice_speed", 1.0))
        services = data.get("services") or ""
        emergency_triggers = data.get("transfer_rules") or data.get("emergency_triggers") or ""
        hours = data.get("hours") or "Monday to Friday 8:00 AM to 6:00 PM. 24/7 on-call dispatch for emergencies."
        pricing_policy = data.get("pricing_policy") or data.get("pricing_preset") or ""
        booking_action = data.get("booking_action") or "schedule arrival window"
        custom_qa = data.get("custom_qa", [])
        qa_str = json.dumps(custom_qa) if isinstance(custom_qa, list) else str(custom_qa)
        tone_preset = data.get("tone_preset") or data.get("tone_id") or "warm_empathetic"

        cur.execute("""
            INSERT INTO custom_prompt (
                client_id, persona_name, system_prompt, livekit_prompt, first_message,
                tts_voice, voice_speed, services, emergency_triggers,
                hours, pricing_policy, booking_action, custom_qa, tone_preset, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(client_id) DO UPDATE SET
                persona_name=excluded.persona_name,
                system_prompt=excluded.system_prompt,
                livekit_prompt=excluded.livekit_prompt,
                first_message=excluded.first_message,
                tts_voice=excluded.tts_voice,
                voice_speed=excluded.voice_speed,
                services=excluded.services,
                emergency_triggers=excluded.emergency_triggers,
                hours=excluded.hours,
                pricing_policy=excluded.pricing_policy,
                booking_action=excluded.booking_action,
                custom_qa=excluded.custom_qa,
                tone_preset=excluded.tone_preset,
                updated_at=excluded.updated_at
        """, (
            clean_id, persona_name, system_prompt, livekit_prompt, first_message,
            tts_voice, voice_speed, services, emergency_triggers,
            hours, pricing_policy, booking_action, qa_str, tone_preset, now_str
        ))

        # 3. Google Calendar Config
        cal_id = data.get("google_calendar_id") or data.get("calendar_id") or "primary"
        sa_json = data.get("google_service_account_json") or data.get("service_account_json") or ""
        cal_webhook = data.get("calendar_webhook_url") or ""
        sync_enabled = 1 if data.get("calendar_sync_enabled", True) else 0
        m_cap = int(data.get("morning_slot_capacity", 2))
        a_cap = int(data.get("afternoon_slot_capacity", 2))
        e_cap = int(data.get("evening_slot_capacity", 2))

        oauth_email = data.get("oauth_user_email") or (cal_id if "@" in cal_id else "")
        oauth_access = data.get("oauth_access_token", "")
        oauth_refresh = data.get("oauth_refresh_token", "")
        is_conn = 1 if (data.get("is_connected") or oauth_email) else 0
        auth_type = data.get("auth_type", "oauth" if oauth_email else "manual")
        last_stat = 'connected' if is_conn else 'untested'

        cur.execute("""
            INSERT INTO google_calendar_config (
                client_id, calendar_id, service_account_json, calendar_webhook_url,
                sync_enabled, morning_slot_capacity, afternoon_slot_capacity,
                evening_slot_capacity, is_connected, last_tested_at, last_status,
                oauth_access_token, oauth_refresh_token, oauth_user_email, auth_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(client_id) DO UPDATE SET
                calendar_id=CASE WHEN excluded.calendar_id != 'primary' THEN excluded.calendar_id ELSE google_calendar_config.calendar_id END,
                service_account_json=CASE WHEN excluded.service_account_json != '' THEN excluded.service_account_json ELSE google_calendar_config.service_account_json END,
                calendar_webhook_url=excluded.calendar_webhook_url,
                sync_enabled=excluded.sync_enabled,
                morning_slot_capacity=excluded.morning_slot_capacity,
                afternoon_slot_capacity=excluded.afternoon_slot_capacity,
                evening_slot_capacity=excluded.evening_slot_capacity,
                is_connected=CASE WHEN excluded.is_connected = 1 THEN 1 ELSE google_calendar_config.is_connected END,
                oauth_access_token=CASE WHEN excluded.oauth_access_token != '' THEN excluded.oauth_access_token ELSE google_calendar_config.oauth_access_token END,
                oauth_refresh_token=CASE WHEN excluded.oauth_refresh_token != '' THEN excluded.oauth_refresh_token ELSE google_calendar_config.oauth_refresh_token END,
                oauth_user_email=CASE WHEN excluded.oauth_user_email != '' THEN excluded.oauth_user_email ELSE google_calendar_config.oauth_user_email END,
                auth_type=CASE WHEN excluded.auth_type != '' THEN excluded.auth_type ELSE google_calendar_config.auth_type END
        """, (
            clean_id, cal_id, sa_json, cal_webhook, sync_enabled, m_cap, a_cap, e_cap,
            is_conn, now_str if is_conn else None, last_stat,
            oauth_access, oauth_refresh, oauth_email, auth_type
        ))

        conn.commit()
    finally:
        conn.close()

    sync_project_json_files(clean_id)
    _update_projects_index()
    logger.success(f"✅ Created dedicated project & database for client '{clean_id}' ({biz_name}) via {trigger_source}")
    return get_project(clean_id) or {}


def trigger_new_client_project(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Invoked when a client buys our voice agents (e.g. from stripe webhook, polar checkout,
    subscribe page, or API).
    Provisions:
    1. Dedicated project directory `data/projects/{client_id}/`
    2. Dedicated SQLite database `data/projects/{client_id}/client.db`
    3. Human-readable `project.json`, `prompt.json`, `calendar.json`
    4. Automatically tailored character-optimized voice instructions for LiveKit (<600 chars)
    5. Connects assigned phone number, Google Calendar, and SMS notifications.
    """
    payload = data.copy()
    client_id = payload.get("id") or payload.get("client_id")
    if not client_id:
        biz_name_raw = payload.get("business_name") or payload.get("name") or "client"
        biz_slug = re.sub(r'[^a-zA-Z0-9]', '', biz_name_raw).lower()[:12]
        client_id = f"proj_{biz_slug}_{uuid.uuid4().hex[:6]}"
    payload["id"] = client_id
    payload["client_id"] = client_id

    # 1. Connect assigned phone number
    if not payload.get("assigned_phone"):
        payload["assigned_phone"] = os.getenv("PLIVO_PHONE_NUMBER") or "+1 (833) 420-5227"

    # 2. Connect forwarding and owner phone
    if not payload.get("forwarding_phone") and payload.get("phone"):
        payload["forwarding_phone"] = payload.get("phone")
    if not payload.get("owner_phone"):
        payload["owner_phone"] = payload.get("forwarding_phone") or payload.get("phone") or ""

    # 3. Connect SMS notifications
    if "sms_notifications_enabled" not in payload:
        payload["sms_notifications_enabled"] = 1

    # 4. Connect Google Calendar config
    if "calendar_sync_enabled" not in payload:
        payload["calendar_sync_enabled"] = True
    if not payload.get("google_calendar_id") and not payload.get("calendar_id"):
        from app.integrations import get_integrations_settings
        global_cfg = get_integrations_settings()
        payload["google_calendar_id"] = global_cfg.get("google_calendar_id", "primary")

    # 5. Compile character-optimized LiveKit prompt (<600 chars)
    livekit_prompt = compile_livekit_voice_prompt(payload)
    payload["livekit_prompt"] = livekit_prompt

    trigger_src = payload.get("trigger_source") or "onboarding_buy"
    project = create_project(payload, trigger_source=trigger_src)

    # 6. Automated welcome SMS notification to owner/forwarding phone if requested
    if payload.get("send_welcome_sms", False) and payload.get("owner_phone"):
        try:
            from app.integrations import send_sms
            biz = payload.get("business_name") or "your business"
            assigned = payload.get("assigned_phone")
            welcome_text = (
                f"🎉 Welcome to ORX Voice AI for {biz}! "
                f"Your dedicated line {assigned} is active. "
                f"Your LiveKit AI receptionist is online and ready to answer calls."
            )
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(send_sms(to_phone=payload["owner_phone"], message=welcome_text))
                else:
                    loop.run_until_complete(send_sms(to_phone=payload["owner_phone"], message=welcome_text))
            except Exception:
                asyncio.run(send_sms(to_phone=payload["owner_phone"], message=welcome_text))
        except Exception as sms_err:
            logger.warning(f"Could not dispatch welcome SMS for project '{client_id}': {sms_err}")

    logger.success(f"🚀 Successfully triggered new client project '{client_id}' with dedicated DB & LiveKit prompt ({len(livekit_prompt)} chars)")
    return project


def get_project(client_id: str) -> Optional[Dict[str, Any]]:
    """Fetches full project details including meta, prompt, and calendar config."""
    clean_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', client_id)
    db_path = get_db_path(clean_id)
    if not db_path.exists():
        return None

    conn = get_db_connection(clean_id)
    cur = conn.cursor()
    try:
        cur.execute("SELECT * FROM project_meta WHERE client_id = ?", (clean_id,))
        meta_row = cur.fetchone()
        if not meta_row:
            return None
        meta = dict(meta_row)

        cur.execute("SELECT * FROM custom_prompt WHERE client_id = ?", (clean_id,))
        prompt_row = cur.fetchone()
        prompt = dict(prompt_row) if prompt_row else {}
        if prompt.get("custom_qa"):
            try:
                prompt["custom_qa"] = json.loads(prompt["custom_qa"])
            except Exception:
                pass

        cur.execute("SELECT * FROM google_calendar_config WHERE client_id = ?", (clean_id,))
        cal_row = cur.fetchone()
        cal = dict(cal_row) if cal_row else {}

        # Counts
        cur.execute("SELECT COUNT(*) as count FROM appointments WHERE client_id = ?", (clean_id,))
        apt_count = cur.fetchone()["count"]
        cur.execute("SELECT COUNT(*) as count FROM call_logs WHERE client_id = ?", (clean_id,))
        call_count = cur.fetchone()["count"]

        livekit_prompt = prompt.get("livekit_prompt") or ""
        tone_preset = prompt.get("tone_preset") or "warm_empathetic"

        return {
            "id": clean_id,
            "meta": meta,
            "prompt": prompt,
            "tone_preset": tone_preset,
            "livekit_prompt": livekit_prompt,
            "character_count": len(livekit_prompt),
            "calendar": cal,
            "stats": {
                "appointments_count": apt_count,
                "calls_count": call_count
            }
        }
    finally:
        conn.close()


def list_projects() -> List[Dict[str, Any]]:
    """Lists all available client projects with overview stats and calendar status."""
    get_projects_dir()
    migrate_legacy_clients()  # Ensure all clients in clients.json are in projects

    results = []
    for item in PROJECTS_DIR.iterdir():
        if item.is_dir() and (item / "client.db").exists():
            proj = get_project(item.name)
            if proj:
                results.append(proj)

    results.sort(key=lambda p: p.get("meta", {}).get("created_at", ""), reverse=True)
    return results


def update_project(client_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Updates project metadata or prompt or calendar settings."""
    clean_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', client_id)
    proj = get_project(clean_id)
    if not proj:
        # Create if not exists
        updates["id"] = clean_id
        return create_project(updates, trigger_source="admin")

    init_project_db(clean_id)
    conn = get_db_connection(clean_id)
    cur = conn.cursor()
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        # Update Meta fields
        meta_fields = ["project_name", "business_name", "industry", "address",
                       "forwarding_phone", "assigned_phone", "owner_phone",
                       "owner_email", "timezone", "status", "sms_notifications_enabled"]
        meta_updates = {k: updates[k] for k in meta_fields if k in updates}
        if meta_updates:
            set_clause = ", ".join([f"{k} = ?" for k in meta_updates.keys()])
            params = list(meta_updates.values()) + [now_str, clean_id]
            cur.execute(f"UPDATE project_meta SET {set_clause}, updated_at = ? WHERE client_id = ?", params)

        # Update Prompt fields
        prompt_fields = ["persona_name", "system_prompt", "livekit_prompt", "first_message", "tts_voice",
                         "voice_speed", "services", "emergency_triggers", "hours",
                         "pricing_policy", "booking_action"]
        prompt_updates = {k: updates[k] for k in prompt_fields if k in updates}
        if "custom_qa" in updates:
            qa_val = updates["custom_qa"]
            prompt_updates["custom_qa"] = json.dumps(qa_val) if isinstance(qa_val, list) else str(qa_val)

        # If voice instructions or business settings changed and no explicit livekit_prompt given, recompile
        if "livekit_prompt" not in updates and any(k in updates for k in ["business_name", "persona_name", "hours", "services", "pricing_policy", "transfer_rules"]):
            combined_data = proj.get("meta", {}).copy()
            combined_data.update(proj.get("prompt", {}))
            combined_data.update(updates)
            prompt_updates["livekit_prompt"] = compile_livekit_voice_prompt(combined_data)

        if prompt_updates:
            set_clause = ", ".join([f"{k} = ?" for k in prompt_updates.keys()])
            params = list(prompt_updates.values()) + [now_str, clean_id]
            cur.execute(f"UPDATE custom_prompt SET {set_clause}, updated_at = ? WHERE client_id = ?", params)

        # Update Calendar fields
        cal_fields = ["calendar_id", "service_account_json", "calendar_webhook_url",
                      "sync_enabled", "morning_slot_capacity", "afternoon_slot_capacity",
                      "evening_slot_capacity", "is_connected", "last_tested_at", "last_status", "last_error"]
        cal_updates = {k: updates[k] for k in cal_fields if k in updates}
        if cal_updates:
            set_clause = ", ".join([f"{k} = ?" for k in cal_updates.keys()])
            params = list(cal_updates.values()) + [clean_id]
            cur.execute(f"UPDATE google_calendar_config SET {set_clause} WHERE client_id = ?", params)

        conn.commit()
    finally:
        conn.close()

    sync_project_json_files(clean_id)
    _update_projects_index()
    logger.info(f"Updated project '{clean_id}'")
    return get_project(clean_id) or {}


def delete_project(client_id: str) -> bool:
    """Deletes a client project directory and records."""
    clean_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', client_id)
    pdir = get_project_dir(clean_id)
    if pdir.exists():
        import shutil
        shutil.rmtree(pdir)
        _update_projects_index()
        logger.info(f"Deleted project '{clean_id}'")
        return True
    return False


def save_project_calendar_config(client_id: str, cal_data: Dict[str, Any]) -> Dict[str, Any]:
    """Updates client-specific Google Calendar configuration and slot limits."""
    clean_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', client_id)
    conn = get_db_connection(clean_id)
    cur = conn.cursor()
    try:
        cur.execute("""
            UPDATE google_calendar_config SET
                calendar_id = COALESCE(?, calendar_id),
                service_account_json = COALESCE(?, service_account_json),
                calendar_webhook_url = COALESCE(?, calendar_webhook_url),
                sync_enabled = COALESCE(?, sync_enabled),
                morning_slot_capacity = COALESCE(?, morning_slot_capacity),
                afternoon_slot_capacity = COALESCE(?, afternoon_slot_capacity),
                evening_slot_capacity = COALESCE(?, evening_slot_capacity),
                is_connected = COALESCE(?, is_connected),
                last_tested_at = COALESCE(?, last_tested_at),
                last_status = COALESCE(?, last_status),
                last_error = COALESCE(?, last_error)
            WHERE client_id = ?
        """, (
            cal_data.get("calendar_id"),
            cal_data.get("service_account_json"),
            cal_data.get("calendar_webhook_url"),
            cal_data.get("sync_enabled"),
            cal_data.get("morning_slot_capacity"),
            cal_data.get("afternoon_slot_capacity"),
            cal_data.get("evening_slot_capacity"),
            cal_data.get("is_connected"),
            cal_data.get("last_tested_at"),
            cal_data.get("last_status"),
            cal_data.get("last_error"),
            clean_id
        ))
        conn.commit()
    finally:
        conn.close()

    sync_project_json_files(clean_id)
    proj = get_project(clean_id)
    return proj.get("calendar", {}) if proj else {}


def get_project_calendar_config(client_id: str) -> Dict[str, Any]:
    """Retrieves client-specific calendar configuration with global fallback."""
    clean_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', client_id)
    from app.integrations import get_integrations_settings
    global_cfg = get_integrations_settings()

    # 1. Check dedicated calendar.json file
    cal_file = get_project_dir(clean_id) / "calendar.json"
    if cal_file.exists():
        try:
            cal_data = json.loads(cal_file.read_text())
            if cal_data:
                return cal_data
        except Exception:
            pass

    proj = get_project(clean_id)
    if not proj or not proj.get("calendar"):
        return {
            "calendar_id": global_cfg.get("google_calendar_id", "primary"),
            "service_account_json": global_cfg.get("google_service_account_json", ""),
            "morning_slot_capacity": global_cfg.get("morning_slot_capacity", 2),
            "afternoon_slot_capacity": global_cfg.get("afternoon_slot_capacity", 2),
            "evening_slot_capacity": 2,
            "is_connected": False,
            "is_inherited_global": True
        }

    cal = proj["calendar"]
    # Fallback to global service account if client left it blank
    sa_json = (cal.get("service_account_json") or "").strip()
    if not sa_json:
        sa_json = (global_cfg.get("google_service_account_json") or "").strip()
        cal["service_account_json"] = sa_json
        cal["is_inherited_global"] = bool(sa_json)
    else:
        cal["is_inherited_global"] = False

    return cal


def save_project_appointment(client_id: str, appointment_data: Dict[str, Any]) -> Dict[str, Any]:
    """Saves or updates an appointment in the client's dedicated database."""
    clean_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', client_id)
    init_project_db(clean_id)

    apt_id = appointment_data.get("id") or f"APT-{uuid.uuid4().hex[:5].upper()}"
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")

    conn = get_db_connection(clean_id)
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO appointments (
                id, client_id, client_name, client_phone, service_requested,
                service_address, appointment_date, window, exact_time,
                status, google_event_id, summary, created_at, updated_at, raw_payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                client_name=excluded.client_name,
                client_phone=excluded.client_phone,
                service_requested=excluded.service_requested,
                service_address=excluded.service_address,
                appointment_date=excluded.appointment_date,
                window=excluded.window,
                exact_time=excluded.exact_time,
                status=excluded.status,
                google_event_id=excluded.google_event_id,
                summary=excluded.summary,
                updated_at=excluded.updated_at,
                raw_payload=excluded.raw_payload
        """, (
            apt_id,
            clean_id,
            appointment_data.get("client_name") or "",
            appointment_data.get("client_phone") or "",
            appointment_data.get("service_requested") or "",
            appointment_data.get("service_address") or "",
            appointment_data.get("appointment_date") or "",
            appointment_data.get("window") or "morning",
            appointment_data.get("exact_time") or "",
            appointment_data.get("status") or "pending_owner_confirmation",
            appointment_data.get("google_event_id") or "",
            appointment_data.get("summary") or "",
            appointment_data.get("created_at") or now_str,
            now_str,
            json.dumps(appointment_data, default=str)
        ))
        conn.commit()
    finally:
        conn.close()

    logger.info(f"Saved appointment #{apt_id} in dedicated DB for '{clean_id}'")
    return appointment_data


def get_project_appointments(client_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves appointments for a specific client project."""
    clean_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', client_id)
    db_path = get_db_path(clean_id)
    if not db_path.exists():
        return []

    conn = get_db_connection(clean_id)
    cur = conn.cursor()
    try:
        cur.execute("SELECT * FROM appointments WHERE client_id = ? ORDER BY created_at DESC LIMIT ?", (clean_id, limit))
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def save_project_call(client_id: str, call_data: Dict[str, Any]) -> Dict[str, Any]:
    """Records a call into the client's dedicated database."""
    clean_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', client_id)
    init_project_db(clean_id)

    call_id = call_data.get("id") or f"call-{int(time.time())}"
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")

    conn = get_db_connection(clean_id)
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO call_logs (
                id, client_id, caller_phone, call_duration,
                recording_file, transcript, extracted_info, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                call_duration=excluded.call_duration,
                transcript=excluded.transcript,
                extracted_info=excluded.extracted_info
        """, (
            call_id,
            clean_id,
            call_data.get("caller") or call_data.get("caller_phone") or "",
            float(call_data.get("duration", 0.0)),
            call_data.get("recording_file") or "",
            json.dumps(call_data.get("transcript", [])) if isinstance(call_data.get("transcript"), list) else str(call_data.get("transcript", "")),
            json.dumps(call_data.get("extracted_info", {})) if isinstance(call_data.get("extracted_info"), dict) else str(call_data.get("extracted_info", "")),
            now_str
        ))
        conn.commit()
    finally:
        conn.close()

    logger.info(f"Recorded call #{call_id} in dedicated DB for '{clean_id}'")
    return call_data


def get_project_calls(client_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves call logs for a specific client project."""
    clean_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', client_id)
    db_path = get_db_path(clean_id)
    if not db_path.exists():
        return []

    conn = get_db_connection(clean_id)
    cur = conn.cursor()
    try:
        cur.execute("SELECT * FROM call_logs WHERE client_id = ? ORDER BY created_at DESC LIMIT ?", (clean_id, limit))
        rows = cur.fetchall()
        result = []
        for r in rows:
            item = dict(r)
            try:
                item["transcript"] = json.loads(item["transcript"])
            except Exception:
                pass
            try:
                item["extracted_info"] = json.loads(item["extracted_info"])
            except Exception:
                pass
            result.append(item)
        return result
    finally:
        conn.close()


def find_client_by_phone(phone: str) -> Optional[Dict[str, Any]]:
    """Looks up a client project by assigned or forwarding or owner phone."""
    if not phone:
        return None
    clean_target = re.sub(r'[^0-9]', '', phone)
    if not clean_target:
        return None

    for proj in list_projects():
        meta = proj.get("meta", {})
        phones = [
            meta.get("assigned_phone", ""),
            meta.get("forwarding_phone", ""),
            meta.get("owner_phone", "")
        ]
        for p in phones:
            p_clean = re.sub(r'[^0-9]', '', p)
            if p_clean and (p_clean == clean_target or p_clean.endswith(clean_target) or clean_target.endswith(p_clean)):
                return proj
    return None


def _update_projects_index():
    """Maintains a lightweight fast-lookup index of all projects."""
    get_projects_dir()
    index_data = []
    for item in PROJECTS_DIR.iterdir():
        if item.is_dir() and (item / "project.json").exists():
            try:
                meta = json.loads((item / "project.json").read_text())
                index_data.append({
                    "id": meta.get("client_id") or item.name,
                    "business_name": meta.get("business_name"),
                    "industry": meta.get("industry"),
                    "status": meta.get("status"),
                    "assigned_phone": meta.get("assigned_phone"),
                    "created_at": meta.get("created_at")
                })
            except Exception:
                pass
    _write_json_safely(PROJECTS_INDEX_FILE, index_data)


def migrate_legacy_clients():
    """Discovers legacy profiles from data/clients.json and ensures each has a dedicated database."""
    if not CLIENTS_FILE.exists():
        return
    try:
        data = json.loads(CLIENTS_FILE.read_text())
        clients_dict = data.get("clients", {})
        for cid, profile in clients_dict.items():
            clean_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', cid)
            db_path = get_db_path(clean_id)
            if not db_path.exists():
                logger.info(f"Auto-migrating legacy client '{cid}' to dedicated project DB...")
                create_project(profile, trigger_source="onboarding")
    except Exception as e:
        logger.error(f"Error during legacy client migration: {e}")
