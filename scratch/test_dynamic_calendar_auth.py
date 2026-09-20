"""Verification script for dynamic Google Calendar OAuth connection and switching."""

import asyncio
from fastapi.testclient import TestClient
from app.server import app
from app.project_db import get_project_calendar_config

client = TestClient(app)

def test_dynamic_calendar_flow():
    test_id = "test_dynamic_cal_client"
    email_1 = "randomowner@apexair.com"
    email_2 = "completely_different@coolbreeze.net"

    print(f"--- 1. Testing URL generation with dynamic email: {email_1} ---")
    res1 = client.get(f"/api/auth/google/url?client_id={test_id}&email={email_1}")
    assert res1.status_code == 200, f"URL fetch failed: {res1.text}"
    url_data = res1.json()
    print(f"Auth URL: {url_data.get('url')}")
    assert "randomowner" in url_data.get("url"), "Expected email in auth URL"

    print(f"--- 2. Testing Sandbox connect with {email_1} ---")
    res2 = client.get(f"/api/auth/google/sandbox-connect?client_id={test_id}&email={email_1}&redirect=0")
    assert res2.status_code == 200
    data2 = res2.json()
    print(f"Sandbox connect response: {data2}")
    assert data2.get("email") == email_1
    assert data2.get("connected") is True

    print(f"--- 3. Testing Status endpoint for {email_1} ---")
    res3 = client.get(f"/api/auth/google/status?client_id={test_id}")
    assert res3.status_code == 200
    st3 = res3.json()
    print(f"Status: {st3}")
    assert st3.get("is_connected") is True
    assert st3.get("user_email") == email_1

    print(f"--- 4. Testing Disconnect / Switch Account ---")
    res4 = client.post("/api/auth/google/disconnect", json={"client_id": test_id})
    assert res4.status_code == 200
    st4 = client.get(f"/api/auth/google/status?client_id={test_id}").json()
    print(f"Status after disconnect: {st4}")
    assert st4.get("is_connected") is False

    print(f"--- 5. Testing Re-connecting with DIFFERENT random email: {email_2} ---")
    res5 = client.get(f"/api/auth/google/sandbox-connect?client_id={test_id}&email={email_2}&redirect=0")
    assert res5.status_code == 200
    data5 = res5.json()
    print(f"Sandbox connect response for email_2: {data5}")
    assert data5.get("email") == email_2

    st5 = client.get(f"/api/auth/google/status?client_id={test_id}").json()
    print(f"Status for email_2: {st5}")
    assert st5.get("is_connected") is True
    assert st5.get("user_email") == email_2

    print(f"--- 6. Testing Project Creation with OAuth connected state ---")
    res6 = client.post("/api/projects/create", json={
        "business_name": "Dynamic Cool Air",
        "industry": "hvac",
        "owner_phone": "+16125551234",
        "owner_email": email_2,
        "google_calendar_id": email_2,
        "oauth_user_email": email_2,
        "is_connected": 1
    })
    assert res6.status_code == 200
    proj_data = res6.json()
    proj_id = proj_data["project"]["id"]
    print(f"Created project: {proj_id}")

    cal_cfg = get_project_calendar_config(proj_id)
    print(f"Project calendar config: {cal_cfg}")
    assert cal_cfg.get("is_connected") == 1
    assert cal_cfg.get("oauth_user_email") == email_2

    print("ALL DYNAMIC CALENDAR AUTH TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_dynamic_calendar_flow()
