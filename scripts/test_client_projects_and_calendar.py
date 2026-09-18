"""Comprehensive Automated Test Suite for Multi-Tenant Client Projects,
Dedicated SQLite Databases, Custom Prompts, and Google Calendar Connections.
"""

import asyncio
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from app.server import app
from app.project_db import (
    create_project,
    get_project,
    list_projects,
    update_project,
    delete_project,
    save_project_calendar_config,
    get_project_calendar_config,
    save_project_appointment,
    get_project_appointments,
    find_client_by_phone,
    get_project_dir,
    get_db_path,
)
from app.onboarding import save_client_profile
from app.appointments import (
    verify_google_calendar_connection,
    check_technician_availability,
    dispatch_hvac_booking,
    apply_appointment_action,
)

client = TestClient(app)


def test_1_dedicated_database_provisioning():
    print("\n--- Test 1: Dedicated SQLite Database & JSON Synchronization ---")
    test_id = "cli_unit_test_hvac_99"
    delete_project(test_id)

    payload = {
        "id": test_id,
        "business_name": "Polar Air & Heating",
        "industry": "hvac",
        "address": "123 Glacier Way, Denver, CO",
        "owner_phone": "+13035550199",
        "assigned_phone": "+18334205227",
        "google_calendar_id": "tech-dispatch@polarair.com",
        "morning_slot_capacity": 3,
        "afternoon_slot_capacity": 4,
        "custom_qa": [{"question": "Do you do duct cleanings?", "answer": "Yes, whole-home sanitization."}]
    }

    proj = create_project(payload, trigger_source="admin")
    assert proj is not None
    assert proj["id"] == test_id
    assert proj["meta"]["business_name"] == "Polar Air & Heating"
    assert proj["meta"]["industry"] == "hvac"
    assert proj["calendar"]["morning_slot_capacity"] == 3
    assert proj["calendar"]["afternoon_slot_capacity"] == 4

    # Verify dedicated SQLite file exists
    db_file = get_db_path(test_id)
    assert db_file.exists(), f"SQLite file {db_file} does not exist!"

    # Verify SQLite schema
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cur.fetchall()}
    assert "project_meta" in tables
    assert "custom_prompt" in tables
    assert "google_calendar_config" in tables
    assert "appointments" in tables
    assert "call_logs" in tables
    conn.close()

    # Verify human-readable JSON files exist
    pdir = get_project_dir(test_id)
    assert (pdir / "project.json").exists()
    assert (pdir / "prompt.json").exists()
    assert (pdir / "calendar.json").exists()

    cal_data = json.loads((pdir / "calendar.json").read_text())
    assert cal_data.get("calendar_id") == "tech-dispatch@polarair.com"

    prompt_data = json.loads((pdir / "prompt.json").read_text())
    assert "Polar Air & Heating" in prompt_data.get("system_prompt", "")

    print(f"✅ Verified dedicated SQLite db and JSON files created at: {pdir}")


def test_2_onboarding_triggered_provisioning():
    print("\n--- Test 2: Client Onboarding Trigger Flow ---")
    onboard_id = "cli_auto_onboard_plumb_88"
    delete_project(onboard_id)

    profile_data = {
        "id": onboard_id,
        "business_name": "Blue River Plumbing",
        "industry": "plumbing",
        "address": "88 Riverwalk Ave, San Antonio, TX",
        "forwarding_phone": "+12105557788",
        "hours": "Mon-Sat 7:00 AM to 7:00 PM",
        "services": "Drain clearing, water heater replacement, slab leak detection",
        "transfer_rules": "Active flooding or sewage backup",
        "assigned_phone": "+18334205227"
    }

    # Simulate client completing onboarding form
    saved_profile = save_client_profile(profile_data)
    assert saved_profile["id"] == onboard_id

    # Verify dedicated project was automatically created by onboarding trigger
    proj = get_project(onboard_id)
    assert proj is not None, "Project was not auto-provisioned upon onboarding!"
    assert proj["meta"]["trigger_source"] == "onboarding"
    assert proj["meta"]["business_name"] == "Blue River Plumbing"
    assert "Blue River Plumbing" in proj["prompt"]["system_prompt"]
    assert "Drain clearing" in proj["prompt"]["system_prompt"]

    print(f"✅ Verified onboarding automatically triggered project creation: #{onboard_id}")


def test_3_admin_triggered_api_creation():
    print("\n--- Test 3: Admin-Triggered API Creation (POST /api/projects) ---")
    admin_payload = {
        "business_name": "Metro Electrical Experts",
        "industry": "electrical",
        "address": "400 Power Line Rd, Phoenix, AZ",
        "owner_phone": "+16025553311",
        "assigned_phone": "+18334205227",
        "google_calendar_id": "service@metroelectrical.com",
        "system_prompt": "You are Marcus, master electrical assistant for Metro Electrical."
    }

    res = client.post("/api/projects", json=admin_payload)
    assert res.status_code == 200, f"Status: {res.status_code}, Body: {res.text}"
    data = res.json()
    assert data["success"] is True
    proj = data["project"]
    client_id = proj["id"]

    assert proj["meta"]["trigger_source"] == "admin"
    assert proj["meta"]["business_name"] == "Metro Electrical Experts"
    assert proj["meta"]["industry"] == "electrical"
    assert proj["prompt"]["system_prompt"] == "You are Marcus, master electrical assistant for Metro Electrical."

    # Verify retrieval via GET /api/projects/{id}
    get_res = client.get(f"/api/projects/{client_id}")
    assert get_res.status_code == 200
    assert get_res.json()["project"]["meta"]["business_name"] == "Metro Electrical Experts"

    # Verify in list
    list_res = client.get("/api/projects")
    assert list_res.status_code == 200
    projects = list_res.json()["projects"]
    assert any(p["id"] == client_id for p in projects)

    print(f"✅ Verified Admin API created dedicated project: #{client_id}")


def test_4_google_calendar_verification():
    print("\n--- Test 4: Google Calendar Verification & Health Check ---")
    # A. Test missing credentials
    res_missing = asyncio.run(verify_google_calendar_connection(calendar_id="primary", service_account_data=""))
    assert res_missing["connected"] is False
    assert res_missing["status"] == "missing_credentials"

    # B. Test invalid auth data
    res_bad = asyncio.run(verify_google_calendar_connection(calendar_id="primary", service_account_data="invalid-json-content"))
    assert res_bad["connected"] is False
    assert res_bad["status"] == "auth_failed"

    # C. Test project calendar test endpoint via API
    test_id = "cli_unit_test_hvac_99"
    cal_res = client.post(f"/api/projects/{test_id}/calendar/test", json={"calendar_id": "primary"})
    assert cal_res.status_code == 200
    cal_data = cal_res.json()
    assert "connected" in cal_data

    # D. Test global calendar test endpoint via API
    global_res = client.post("/api/integrations/google-calendar/test", json={"google_calendar_id": "primary"})
    assert global_res.status_code == 200
    assert "connected" in global_res.json()

    print("✅ Verified Google Calendar connection diagnostic tools & API endpoints.")


def test_5_project_calendar_config_persistence():
    print("\n--- Test 5: Client-Specific Calendar Configuration & Capacities ---")
    test_id = "cli_unit_test_hvac_99"

    update_payload = {
        "calendar_id": "hvac-dispatch-east@gmail.com",
        "morning_slot_capacity": 5,
        "afternoon_slot_capacity": 6,
        "evening_slot_capacity": 3
    }
    res = client.post(f"/api/projects/{test_id}/calendar", json=update_payload)
    assert res.status_code == 200
    cal = res.json()["calendar"]
    assert cal["calendar_id"] == "hvac-dispatch-east@gmail.com"
    assert cal["morning_slot_capacity"] == 5
    assert cal["afternoon_slot_capacity"] == 6

    # Verify retrieval
    saved_cfg = get_project_calendar_config(test_id)
    assert saved_cfg["calendar_id"] == "hvac-dispatch-east@gmail.com"
    assert saved_cfg["morning_slot_capacity"] == 5

    print(f"✅ Verified per-client calendar settings saved to dedicated DB for {test_id}.")


def test_6_dedicated_appointments_routing():
    print("\n--- Test 6: Dedicated Appointments Storage in Client SQLite Database ---")
    test_id = "cli_unit_test_hvac_99"

    sample_apt = {
        "id": "APT-TEST-001",
        "client_name": "Eleanor Vance",
        "client_phone": "+14155554321",
        "service_requested": "Emergency AC Repair",
        "service_address": "900 Summit Ave, Denver, CO",
        "appointment_date": "2026-10-20",
        "window": "morning",
        "exact_time": "10:00 AM",
        "status": "confirmed"
    }

    # Save to dedicated client project database
    save_project_appointment(test_id, sample_apt)

    # Retrieve from dedicated client database
    apts = get_project_appointments(test_id)
    assert len(apts) >= 1
    found = next((a for a in apts if a["id"] == "APT-TEST-001"), None)
    assert found is not None
    assert found["client_name"] == "Eleanor Vance"
    assert found["service_address"] == "900 Summit Ave, Denver, CO"
    assert found["status"] == "confirmed"

    # Verify via API GET /api/projects/{client_id}/appointments
    api_res = client.get(f"/api/projects/{test_id}/appointments")
    assert api_res.status_code == 200
    api_apts = api_res.json()["appointments"]
    assert any(a["id"] == "APT-TEST-001" for a in api_apts)

    print(f"✅ Verified appointment #{sample_apt['id']} saved and queried from dedicated SQLite database.")


def test_7_find_client_by_phone():
    print("\n--- Test 7: Lookup Client Project by Phone ---")
    test_id = "cli_unit_test_hvac_99"
    proj = find_client_by_phone("+13035550199")
    assert proj is not None
    assert proj["id"] == test_id
    assert proj["meta"]["business_name"] == "Polar Air & Heating"

    print(f"✅ Successfully resolved client project #{proj['id']} from phone +13035550199")


def test_8_activate_project_in_studio():
    print("\n--- Test 8: Activate Client Project in Studio ---")
    test_id = "cli_unit_test_hvac_99"
    act_res = client.post(f"/api/projects/{test_id}/activate")
    assert act_res.status_code == 200
    act_data = act_res.json()
    assert act_data["success"] is True
    assert act_data["project"]["business_name"] == "Polar Air & Heating"

    # Verify active assistant in agents.py
    from app.agents import get_active_assistant
    active = get_active_assistant()
    assert "Polar Air & Heating" in active["name"]
    assert "Polar Air & Heating" in active["system_prompt"]

    print(f"✅ Verified client project #{test_id} activated live in studio with custom prompt!")


if __name__ == "__main__":
    print("===================================================================")
    print("🚀 RUNNING MULTI-TENANT CLIENT PROJECTS & GCAL CONNECTOR TEST SUITE")
    print("===================================================================")

    test_1_dedicated_database_provisioning()
    test_2_onboarding_triggered_provisioning()
    test_3_admin_triggered_api_creation()
    test_4_google_calendar_verification()
    test_5_project_calendar_config_persistence()
    test_6_dedicated_appointments_routing()
    test_7_find_client_by_phone()
    test_8_activate_project_in_studio()

    print("\n===================================================================")
    print("🎉 ALL 8 MULTI-TENANT & GOOGLE CALENDAR TESTS PASSED!")
    print("===================================================================")
