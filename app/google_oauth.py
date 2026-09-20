"""Google Calendar OAuth 2.0 Integration for 1-Click Fast Business Owner Connection.
Supports:
1. Google OAuth 2.0 Authorization Flow (Offline refresh tokens, calendar.events scope)
2. Token refreshing and storage in dedicated client SQLite DB and calendar.json
3. Instant 1-click Dev / Sandbox connection for frictionless testing
4. Direct Google Calendar event insertion with sendUpdates=all
"""

import os
import json
import time
import base64
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import httpx
from loguru import logger

from app.config import settings

GOOGLE_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_ENDPOINT = "https://www.googleapis.com/oauth2/v3/userinfo"
CALENDAR_SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.readonly",
    "openid",
    "email",
    "profile",
]


def get_oauth_credentials() -> Tuple[str, str, str]:
    """Retrieves Google OAuth Client ID, Secret, and Redirect URI."""
    client_id = os.getenv("GOOGLE_OAUTH_CLIENT_ID", "").strip()
    client_secret = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET", "").strip()
    redirect_uri = os.getenv("GOOGLE_OAUTH_REDIRECT_URI", "").strip()

    if not redirect_uri:
        if settings.PUBLIC_URL:
            base = settings.PUBLIC_URL.strip().rstrip("/")
            redirect_uri = f"{base}/api/auth/google/callback"
        else:
            redirect_uri = f"http://localhost:{settings.PORT}/api/auth/google/callback"

    return client_id, client_secret, redirect_uri


def is_google_oauth_configured() -> bool:
    """Returns True if Google OAuth Client ID and Secret are configured."""
    client_id, client_secret, _ = get_oauth_credentials()
    return bool(client_id and client_secret)


def build_google_auth_url(
    client_id_project: str,
    redirect_uri: Optional[str] = None,
    email: Optional[str] = None
) -> str:
    """Builds the 1-Click Google OAuth authorization URL for the business owner.
    If Google OAuth credentials are not set, returns a sandbox fast-connect URL for instant testing."""
    oauth_id, _, default_redirect = get_oauth_credentials()
    final_redirect = redirect_uri or default_redirect

    if not oauth_id:
        # Fast Sandbox Connect URL (Instant 1-click testing)
        params_dict = {
            "client_id": client_id_project,
            "redirect_uri": final_redirect,
            "sandbox": "1"
        }
        if email:
            params_dict["email"] = email
        params = urllib.parse.urlencode(params_dict)
        return f"/api/auth/google/sandbox-connect?{params}"

    # Real Google OAuth 2.0 URL
    state_payload = json.dumps({"client_id": client_id_project, "ts": int(time.time()), "email": email or ""})
    state_b64 = base64.urlsafe_b64encode(state_payload.encode()).decode()

    params = {
        "client_id": oauth_id,
        "redirect_uri": final_redirect,
        "response_type": "code",
        "scope": " ".join(CALENDAR_SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "state": state_b64,
        "include_granted_scopes": "true",
    }
    if email:
        params["login_hint"] = email
    return f"{GOOGLE_AUTH_ENDPOINT}?{urllib.parse.urlencode(params)}"


async def exchange_google_code(
    code: str,
    redirect_uri: Optional[str] = None,
    client_id_project: Optional[str] = None,
) -> Dict[str, Any]:
    """Exchanges authorization code for access and refresh tokens, extracts user email,
    and persists them to client's dedicated SQLite DB and calendar.json."""
    oauth_id, oauth_secret, default_redirect = get_oauth_credentials()
    final_redirect = redirect_uri or default_redirect

    if not oauth_id or not oauth_secret:
        raise ValueError("Google OAuth credentials (GOOGLE_OAUTH_CLIENT_ID / GOOGLE_OAUTH_CLIENT_SECRET) not set.")

    async with httpx.AsyncClient(timeout=10.0) as http_client:
        token_res = await http_client.post(
            GOOGLE_TOKEN_ENDPOINT,
            data={
                "code": code,
                "client_id": oauth_id,
                "client_secret": oauth_secret,
                "redirect_uri": final_redirect,
                "grant_type": "authorization_code",
            },
        )
        if token_res.status_code != 200:
            logger.error(f"Google token exchange failed ({token_res.status_code}): {token_res.text}")
            raise ValueError(f"Google Token Exchange Error: {token_res.text}")

        token_data = token_res.json()
        access_token = token_data.get("access_token", "")
        refresh_token = token_data.get("refresh_token", "")
        expires_in = int(token_data.get("expires_in", 3600))

        # Fetch authenticated user profile / email
        user_email = ""
        user_name = ""
        try:
            userinfo_res = await http_client.get(
                GOOGLE_USERINFO_ENDPOINT,
                headers={"Authorization": f"Bearer {access_token}"}
            )
            if userinfo_res.status_code == 200:
                uinfo = userinfo_res.json()
                user_email = uinfo.get("email", "")
                user_name = uinfo.get("name", "")
        except Exception as ex:
            logger.warning(f"Failed to fetch userinfo from Google: {ex}")

        # Persist connection to client's project
        if client_id_project:
            save_oauth_connection(
                client_id=client_id_project,
                user_email=user_email,
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=expires_in,
                user_name=user_name,
            )

        return {
            "success": True,
            "email": user_email,
            "name": user_name,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_in": expires_in,
            "client_id": client_id_project,
        }


def save_oauth_connection(
    client_id: str,
    user_email: str,
    access_token: str,
    refresh_token: str,
    expires_in: int = 3600,
    user_name: str = "",
) -> Dict[str, Any]:
    """Saves Google OAuth connection details into client's dedicated SQLite database and calendar.json."""
    from app.project_db import get_db_connection, init_project_db, get_project_dir

    init_project_db(client_id)
    now_ts = int(time.time())
    expiry_ts = now_ts + expires_in
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")

    conn = get_db_connection(client_id)
    cur = conn.cursor()
    try:
        # Check if oauth columns exist, if not add them
        cur.execute("PRAGMA table_info(google_calendar_config)")
        cols = [c[1] for c in cur.fetchall()]

        if "oauth_access_token" not in cols:
            cur.execute("ALTER TABLE google_calendar_config ADD COLUMN oauth_access_token TEXT DEFAULT ''")
        if "oauth_refresh_token" not in cols:
            cur.execute("ALTER TABLE google_calendar_config ADD COLUMN oauth_refresh_token TEXT DEFAULT ''")
        if "oauth_token_expiry" not in cols:
            cur.execute("ALTER TABLE google_calendar_config ADD COLUMN oauth_token_expiry INTEGER DEFAULT 0")
        if "oauth_user_email" not in cols:
            cur.execute("ALTER TABLE google_calendar_config ADD COLUMN oauth_user_email TEXT DEFAULT ''")
        if "auth_type" not in cols:
            cur.execute("ALTER TABLE google_calendar_config ADD COLUMN auth_type TEXT DEFAULT 'oauth'")

        # Update or Insert
        cur.execute("""
            INSERT INTO google_calendar_config (
                client_id, calendar_id, sync_enabled, is_connected,
                last_tested_at, last_status, oauth_access_token,
                oauth_refresh_token, oauth_token_expiry, oauth_user_email, auth_type
            ) VALUES (?, 'primary', 1, 1, ?, 'connected', ?, ?, ?, ?, 'oauth')
            ON CONFLICT(client_id) DO UPDATE SET
                calendar_id='primary',
                sync_enabled=1,
                is_connected=1,
                last_tested_at=excluded.last_tested_at,
                last_status='connected',
                oauth_access_token=excluded.oauth_access_token,
                oauth_refresh_token=CASE WHEN excluded.oauth_refresh_token != '' THEN excluded.oauth_refresh_token ELSE google_calendar_config.oauth_refresh_token END,
                oauth_token_expiry=excluded.oauth_token_expiry,
                oauth_user_email=excluded.oauth_user_email,
                auth_type='oauth'
        """, (
            client_id, now_str, access_token, refresh_token, expiry_ts, user_email
        ))
        conn.commit()
    finally:
        conn.close()

    # Also update calendar.json
    pdir = get_project_dir(client_id)
    cal_file = pdir / "calendar.json"
    cal_data: Dict[str, Any] = {}
    if cal_file.exists():
        try:
            cal_data = json.loads(cal_file.read_text())
        except Exception:
            pass

    cal_data.update({
        "client_id": client_id,
        "calendar_id": "primary",
        "auth_type": "oauth",
        "is_connected": True,
        "oauth_user_email": user_email,
        "oauth_user_name": user_name,
        "oauth_access_token": access_token,
        "oauth_refresh_token": refresh_token or cal_data.get("oauth_refresh_token", ""),
        "oauth_token_expiry": expiry_ts,
        "last_connected_at": now_str,
    })
    cal_file.write_text(json.dumps(cal_data, indent=2))
    logger.success(f"✅ Google Calendar OAuth connected for '{client_id}' ({user_email})")
    return cal_data


async def get_valid_access_token(client_id: str) -> Optional[str]:
    """Retrieves a valid OAuth access token for the client, automatically refreshing if expired."""
    from app.project_db import get_project_dir

    pdir = get_project_dir(client_id)
    cal_file = pdir / "calendar.json"
    if not cal_file.exists():
        return None

    try:
        data = json.loads(cal_file.read_text())
    except Exception:
        return None

    if data.get("auth_type") != "oauth" or not data.get("is_connected"):
        return None

    access_token = data.get("oauth_access_token")
    refresh_token = data.get("oauth_refresh_token")
    expiry = data.get("oauth_token_expiry", 0)

    # Token still valid with >60s buffer
    if access_token and time.time() < (expiry - 60):
        return access_token

    # Token expired or missing, try refresh
    if refresh_token:
        oauth_id, oauth_secret, _ = get_oauth_credentials()
        if oauth_id and oauth_secret:
            try:
                async with httpx.AsyncClient(timeout=8.0) as http_client:
                    r = await http_client.post(
                        GOOGLE_TOKEN_ENDPOINT,
                        data={
                            "client_id": oauth_id,
                            "client_secret": oauth_secret,
                            "refresh_token": refresh_token,
                            "grant_type": "refresh_token",
                        }
                    )
                    if r.status_code == 200:
                        new_data = r.json()
                        new_access = new_data.get("access_token")
                        new_expiry = int(time.time()) + int(new_data.get("expires_in", 3600))
                        data["oauth_access_token"] = new_access
                        data["oauth_token_expiry"] = new_expiry
                        cal_file.write_text(json.dumps(data, indent=2))
                        return new_access
            except Exception as ex:
                logger.error(f"Error refreshing Google OAuth token for '{client_id}': {ex}")

    return access_token


def disconnect_google_calendar(client_id: str) -> bool:
    """Disconnects Google Calendar for a client."""
    from app.project_db import get_db_connection, get_project_dir

    conn = get_db_connection(client_id)
    cur = conn.cursor()
    try:
        cur.execute("""
            UPDATE google_calendar_config
            SET is_connected = 0,
                oauth_access_token = '',
                oauth_refresh_token = '',
                last_status = 'disconnected'
            WHERE client_id = ?
        """, (client_id,))
        conn.commit()
    finally:
        conn.close()

    pdir = get_project_dir(client_id)
    cal_file = pdir / "calendar.json"
    if cal_file.exists():
        try:
            data = json.loads(cal_file.read_text())
            data["is_connected"] = False
            data["oauth_access_token"] = ""
            data["oauth_refresh_token"] = ""
            cal_file.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    logger.info(f"Disconnected Google Calendar for '{client_id}'")
    return True
