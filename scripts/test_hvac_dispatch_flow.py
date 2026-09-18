"""Comprehensive test suite for HVAC Foolproof 1-Tap SMS & Mobile Action Card Flow."""

import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.appointments import (
    check_technician_availability,
    dispatch_hvac_booking,
    process_inbound_sms,
    find_appointment_by_id,
    apply_appointment_action,
)
from app.integrations import update_integrations_settings


async def main():
    print("===================================================================")
    print("🧪 Testing Foolproof HVAC Owner SMS & Mobile Action Flow")
    print("===================================================================")

    owner_phone = "+14155552671"
    customer_phone = "+14155559876"
    test_date = "2026-09-25"  # A Friday

    update_integrations_settings({
        "owner_phone_number": owner_phone,
        "enable_owner_sms": True,
        "enable_customer_sms": True,
        "morning_slot_capacity": 2,
        "afternoon_slot_capacity": 2,
    })

    # Test 1: Inbound Booking Creation & Numbered Options in SMS
    print("\n--- Test 1: Numbered Options Generation in Dispatch SMS ---")
    mock_extracted = {
        "client_name": "Sarah Jenkins",
        "client_phone": customer_phone,
        "service_requested": "High-Efficiency Heat Pump Tune-Up",
        "service_address": "742 Evergreen Terrace",
        "appointment_date": test_date,
        "appointment_time": "09:30 AM",
        "has_appointment": True,
        "summary": "Customer AC blowing warm air, wants Friday morning inspection.",
    }
    dispatch_res = await dispatch_hvac_booking(mock_extracted, {"call_id": "test_call_101", "caller": customer_phone})
    apt1 = dispatch_res["appointment"]
    apt1_id = apt1["id"]
    owner_sms_text = dispatch_res["sms_owner"]["text"]

    print(f"Generated Owner SMS:\n{owner_sms_text}\n")
    assert "1️⃣ Confirm (" in owner_sms_text, "SMS must contain Option 1!"
    assert "2️⃣ Move to" in owner_sms_text, "SMS must contain Option 2!"
    assert "3️⃣ Move to" in owner_sms_text, "SMS must contain Option 3!"
    assert "4️⃣ Decline" in owner_sms_text, "SMS must contain Option 4!"
    assert f"/a/{apt1_id}" in owner_sms_text, "SMS must contain mobile 1-tap web link!"
    print("✅ Verified Dispatch SMS has clear numbered options and 1-tap link!")

    # Test 2: Owner replies with single digit "1"
    print("\n--- Test 2: Single-Digit Reply '1' (Confirm) ---")
    res1 = await process_inbound_sms(owner_phone, "1")
    assert res1["status"] == "owner_confirmed"
    assert find_appointment_by_id(apt1_id)["status"] == "confirmed"
    print(f"✅ Verified single-digit '1' confirmed #{apt1_id}")

    # Test 3: Owner replies with Thumbs Up Emoji "👍"
    print("\n--- Test 3: Emoji Confirmation ('👍') ---")
    mock_extracted2 = {
        "client_name": "Mike Henderson",
        "client_phone": "+14155551111",
        "service_requested": "Emergency AC Leak",
        "service_address": "88 Ocean Avenue",
        "appointment_date": test_date,
        "appointment_time": "10:00 AM",
        "has_appointment": True,
    }
    d2 = await dispatch_hvac_booking(mock_extracted2, {"call_id": "call_102", "caller": "+14155551111"})
    apt2_id = d2["appointment"]["id"]

    res_emoji = await process_inbound_sms(owner_phone, f"👍 #{apt2_id}")
    assert res_emoji["status"] == "owner_confirmed"
    assert find_appointment_by_id(apt2_id)["status"] == "confirmed"
    print(f"✅ Verified emoji '👍' confirmed #{apt2_id}")

    # Test 4: Single digit "2" (Shift to afternoon)
    print("\n--- Test 4: Single-Digit Reply '2' (Shift to Afternoon) ---")
    mock_extracted3 = {
        "client_name": "Alice Wong",
        "client_phone": "+14155552222",
        "service_requested": "Duct Inspection",
        "service_address": "500 Market St",
        "appointment_date": test_date,
        "appointment_time": "09:00 AM",
        "has_appointment": True,
    }
    d3 = await dispatch_hvac_booking(mock_extracted3, {"call_id": "call_103", "caller": "+14155552222"})
    apt3_id = d3["appointment"]["id"]

    res_opt2 = await process_inbound_sms(owner_phone, f"2 #{apt3_id}")
    assert res_opt2["status"] == "rescheduled_by_owner"
    apt3 = find_appointment_by_id(apt3_id)
    assert apt3["window"] == "afternoon"
    print(f"✅ Verified single-digit '2' shifted #{apt3_id} to Afternoon window!")

    # Test 5: Single digit "3" (Shift to next business day)
    print("\n--- Test 5: Single-Digit Reply '3' (Shift to Next Business Day) ---")
    mock_extracted4 = {
        "client_name": "Carlos Gomez",
        "client_phone": "+14155554444",
        "service_requested": "Thermostat Installation",
        "service_address": "77 Mission St",
        "appointment_date": test_date,
        "appointment_time": "09:00 AM",
        "has_appointment": True,
    }
    d4 = await dispatch_hvac_booking(mock_extracted4, {"call_id": "call_104", "caller": "+14155554444"})
    apt4_id = d4["appointment"]["id"]

    res_opt3 = await process_inbound_sms(owner_phone, f"3 #{apt4_id}")
    assert res_opt3["status"] == "rescheduled_by_owner"
    apt4 = find_appointment_by_id(apt4_id)
    # Friday + 3 days -> Monday 2026-09-28
    assert apt4["appointment_date"] == "2026-09-28"
    print(f"✅ Verified single-digit '3' shifted #{apt4_id} to Monday ({apt4['appointment_date']})!")

    # Test 6: Single digit "4" (Decline)
    print("\n--- Test 6: Single-Digit Reply '4' (Decline) ---")
    mock_extracted5 = {
        "client_name": "Tom Brady",
        "client_phone": "+14155557777",
        "service_requested": "Commercial Rooftop AC",
        "service_address": "12 Stadium Way",
        "appointment_date": test_date,
        "appointment_time": "11:00 AM",
        "has_appointment": True,
    }
    d5 = await dispatch_hvac_booking(mock_extracted5, {"call_id": "call_105", "caller": "+14155557777"})
    apt5_id = d5["appointment"]["id"]

    res_opt4 = await process_inbound_sms(owner_phone, f"4 #{apt5_id}")
    assert res_opt4["status"] == "declined"
    assert find_appointment_by_id(apt5_id)["status"] == "declined"
    print(f"✅ Verified single-digit '4' declined #{apt5_id}!")

    # Test 7: Safety-Net Guide for Unrecognized Input
    print("\n--- Test 7: Safety-Net Auto-Guide on Unrecognized Input ---")
    mock_extracted6 = {
        "client_name": "Karen Smith",
        "client_phone": "+14155558888",
        "service_requested": "Filter Replacement",
        "service_address": "90 Pine St",
        "appointment_date": test_date,
        "appointment_time": "10:00 AM",
        "has_appointment": True,
    }
    d6 = await dispatch_hvac_booking(mock_extracted6, {"call_id": "call_106", "caller": "+14155558888"})
    apt6_id = d6["appointment"]["id"]

    # Owner types something vague like "where is this located again?"
    res_help = await process_inbound_sms(owner_phone, f"where is this located again? #{apt6_id}")
    assert res_help["status"] == "safety_net_prompted"
    print(f"✅ Verified unrecognized input triggered Safety-Net Guide with numbered choices for #{apt6_id}!")

    # Test 8: Mobile Web Action Page & Action API
    print("\n--- Test 8: Mobile 1-Tap Web Page & Action API ---")
    from fastapi.testclient import TestClient
    from app.server import app

    client = TestClient(app)
    # Check GET /a/{apt_id}
    web_resp = client.get(f"/a/{apt6_id}")
    assert web_resp.status_code == 200
    assert "HVAC Dispatch Alert" in web_resp.text
    assert "Google Maps" in web_resp.text
    assert "Call Customer" in web_resp.text
    assert "1️⃣" in web_resp.text
    print(f"GET /a/{apt6_id} returned 200 OK with mobile card HTML!")

    # Check POST /api/appointments/{apt_id}/action
    action_resp = client.post(f"/api/appointments/{apt6_id}/action", json={"action": "1"})
    assert action_resp.status_code == 200
    assert action_resp.json()["status"] == "owner_confirmed"
    assert find_appointment_by_id(apt6_id)["status"] == "confirmed"
    print(f"POST /api/appointments/{apt6_id}/action confirmed #{apt6_id} via 1-tap web!")

    print("\n===================================================================")
    print("🎉 ALL FOOLPROOF TESTS PASSED! 100% Friction-Free HVAC Dispatch!")
    print("===================================================================")


if __name__ == "__main__":
    asyncio.run(main())
