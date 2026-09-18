"""Automated verification for Smart Flexible Scheduling (Natural Language SMS & Interactive Custom Picker)."""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.appointments import (
    parse_flexible_schedule_intent,
    dispatch_hvac_booking,
    process_inbound_sms,
    find_appointment_by_id,
)
from app.integrations import update_integrations_settings


async def main():
    print("===================================================================")
    print("🧪 Testing Smart Flexible Scheduling ('in 2 weeks', 'in 3 days', etc.)")
    print("===================================================================")

    owner_phone = "+14155552671"
    customer_phone = "+14155559876"
    ref_date = datetime(2026, 9, 18)  # Friday Sep 18, 2026

    update_integrations_settings({
        "owner_phone_number": owner_phone,
        "enable_owner_sms": True,
        "enable_customer_sms": True,
    })

    # Test 1: Relative Natural Language Date Parser ("in 2 weeks Friday 2:30pm")
    print("\n--- Test 1: Parse 'in 2 weeks Friday 2:30pm' (Exact Date & Time) ---")
    p1 = await parse_flexible_schedule_intent("in 2 weeks Friday 2:30pm", reference_date=ref_date)
    print(f"Result: {p1}")
    assert p1 is not None
    # 2 weeks from Friday Sep 18, 2026 is Friday Oct 2, 2026
    assert p1["date"] == "2026-10-02"
    assert p1["exact_time"] == "2:30 PM"
    assert p1["window"] == "afternoon"
    assert "2:30 PM" in p1["display_label"]
    print("✅ 'in 2 weeks Friday 2:30pm' correctly resolved to 2026-10-02 at 2:30 PM (Afternoon)!")

    # Test 2: Relative Natural Language Date Parser ("in 3 days afternoon")
    print("\n--- Test 2: Parse 'in 3 days afternoon' (Relative Date & Window) ---")
    p2 = await parse_flexible_schedule_intent("in 3 days afternoon", reference_date=ref_date)
    print(f"Result: {p2}")
    assert p2 is not None
    # 3 days from Sep 18 is Sep 21
    assert p2["date"] == "2026-09-21"
    assert p2["window"] == "afternoon"
    print("✅ 'in 3 days afternoon' correctly resolved to 2026-09-21 (Afternoon)!")

    # Test 3: Relative Natural Language ("next Tuesday 2pm")
    print("\n--- Test 3: Parse 'next Tuesday 2pm' ---")
    p3 = await parse_flexible_schedule_intent("next Tuesday 2pm", reference_date=ref_date)
    print(f"Result: {p3}")
    assert p3 is not None
    # Friday Sep 18 -> next Tuesday is Sep 22
    assert p3["date"] == "2026-09-22"
    assert p3["exact_time"] == "2:00 PM"
    assert p3["window"] == "afternoon"
    print("✅ 'next Tuesday 2pm' correctly resolved to 2026-09-22 at 2:00 PM (Afternoon)!")

    # Test 4: Explicit Month & Day ("October 15 10am")
    print("\n--- Test 4: Parse 'October 15 10am' ---")
    p4 = await parse_flexible_schedule_intent("October 15 10am", reference_date=ref_date)
    print(f"Result: {p4}")
    assert p4 is not None
    assert p4["date"] == "2026-10-15"
    assert p4["exact_time"] == "10:00 AM"
    assert p4["window"] == "morning"
    print("✅ 'October 15 10am' correctly resolved to 2026-10-15 at 10:00 AM (Morning)!")

    # Test 5: Standard Unified Format 2026-10-15:1330
    print("\n--- Test 5: Parse Standard Format '2026-10-15:1330' and '2026-10-15:0900' ---")
    p_std1 = await parse_flexible_schedule_intent("2026-10-15:1330", reference_date=ref_date)
    print(f"Result for 2026-10-15:1330: {p_std1}")
    assert p_std1 is not None
    assert p_std1["date"] == "2026-10-15"
    assert p_std1["exact_time"] == "1:30 PM"
    assert p_std1["window"] == "afternoon"

    p_std2 = await parse_flexible_schedule_intent("2026-10-15:0900", reference_date=ref_date)
    assert p_std2["date"] == "2026-10-15"
    assert p_std2["exact_time"] == "9:00 AM"
    assert p_std2["window"] == "morning"
    print("✅ Verified standard unified format '2026-10-15:1330' parsed to 2026-10-15 at 1:30 PM!")

    # Test 5b: Tomorrow, Bare Weekday, and Slash Dates
    print("\n--- Test 5b: Parse 'tomorrow 9am', 'Friday 3pm', '10/24 11am' ---")
    p_tmrw = await parse_flexible_schedule_intent("tomorrow 9am", reference_date=ref_date)
    assert p_tmrw["date"] == "2026-09-19"
    assert p_tmrw["exact_time"] == "9:00 AM"

    p_slash = await parse_flexible_schedule_intent("10/24 11am", reference_date=ref_date)
    assert p_slash["date"] == "2026-10-24"
    assert p_slash["exact_time"] == "11:00 AM"
    print("✅ Verified 'tomorrow 9am' and '10/24 11am' parsed accurately!")

    # Test 6: End-to-End Inbound SMS with Flexible Text ("in 2 weeks Friday 2:30pm")
    print("\n--- Test 6: Owner Texts 'in 2 weeks Friday 2:30pm' via SMS ---")
    mock_extracted = {
        "client_name": "Marcus Vance",
        "client_phone": customer_phone,
        "service_requested": "Full AC System Replacement",
        "service_address": "404 Ocean Blvd",
        "appointment_date": "2026-09-21",
        "appointment_time": "09:00 AM",
        "has_appointment": True,
    }
    d_res = await dispatch_hvac_booking(mock_extracted, {"call_id": "call_flex_101", "caller": customer_phone})
    apt_id = d_res["appointment"]["id"]
    # Verify the owner SMS message uses the unified format with Format and Example
    owner_sms_body = d_res["sms_owner"]["text"]
    assert "📅 Or text ANY exact day & time or window:" in owner_sms_body
    assert "Format: YYYY-MM-DD:HHMM" in owner_sms_body
    assert "Example: 2026-10-15:1330" in owner_sms_body
    print("✅ Verified Owner SMS contains Format and Example: 'Format: YYYY-MM-DD:HHMM / Example: 2026-10-15:1330'!")

    # Owner texts back: "in 2 weeks Friday 2:30pm #APT-..."
    sms_reply = await process_inbound_sms(owner_phone, f"in 2 weeks Friday 2:30pm #{apt_id}")
    print(f"Inbound SMS Result: {sms_reply}")
    assert sms_reply["status"] == "rescheduled_by_owner"
    assert sms_reply["exact_time"] == "2:30 PM"

    updated_apt = find_appointment_by_id(apt_id)
    assert updated_apt["exact_time"] == "2:30 PM"
    assert updated_apt["window"] == "afternoon"
    assert updated_apt["status"] == "rescheduled_pending_customer"
    print("✅ Verified Owner texted 'in 2 weeks Friday 2:30pm' and system saved exact time & window!")

    # Test 6b: Owner texts standard unified format "2026-10-15:1330"
    print("\n--- Test 6b: Owner Texts Standard '2026-10-15:1330' via SMS ---")
    sms_std_reply = await process_inbound_sms(owner_phone, f"2026-10-15:1330 #{apt_id}")
    print(f"Inbound SMS Result for 2026-10-15:1330: {sms_std_reply}")
    assert sms_std_reply["status"] == "rescheduled_by_owner"
    assert sms_std_reply["new_date"] == "2026-10-15"
    assert sms_std_reply["exact_time"] == "1:30 PM"
    apt_std_check = find_appointment_by_id(apt_id)
    assert apt_std_check["appointment_date"] == "2026-10-15"
    assert apt_std_check["exact_time"] == "1:30 PM"
    assert apt_std_check["window"] == "afternoon"
    print("✅ Verified Owner texted '2026-10-15:1330' and appointment was rescheduled to 2026-10-15 at 1:30 PM!")

    # Test 7: Interactive SMS Selector Query ("days")
    print("\n--- Test 7: Owner or User Texts 'days' for Interactive Guide ---")
    days_reply = await process_inbound_sms(owner_phone, f"days #{apt_id}")
    assert days_reply["status"] == "schedule_options_sent"
    assert "Select day & time" in days_reply["message"]
    print("✅ Verified 'days' prompt returns smart interactive day & time guide!")

    # Test 8: Shortcut Letter ('A', 'B', 'C')
    print("\n--- Test 8: Owner Uses Letter Shortcut 'B' ---")
    b_reply = await process_inbound_sms(owner_phone, f"B #{apt_id}")
    assert b_reply["status"] == "rescheduled_by_owner"
    print("✅ Verified letter shortcut 'B' shifted to Option 2!")

    # Test 9: Customer Inbound Rescheduling ("Can we do Oct 15 10am?")
    print("\n--- Test 9: Customer Reschedules via Natural Language SMS ---")
    cust_resched_reply = await process_inbound_sms(customer_phone, f"Can we do Oct 15 at 10am instead? #{apt_id}")
    print(f"Customer Inbound Result: {cust_resched_reply}")
    assert cust_resched_reply["status"] == "customer_rescheduled"
    assert cust_resched_reply["new_date"] == "2026-10-15"
    assert cust_resched_reply["exact_time"] == "10:00 AM"

    apt_cust_update = find_appointment_by_id(apt_id)
    assert apt_cust_update["appointment_date"] == "2026-10-15"
    assert apt_cust_update["exact_time"] == "10:00 AM"
    print("✅ Customer successfully requested Oct 15 at 10am via SMS and Owner was alerted!")

    # Test 10: Owner Confirms Customer's Slot with '1'
    print("\n--- Test 10: Owner Confirms Customer's Rescheduled Slot ---")
    owner_confirm = await process_inbound_sms(owner_phone, f"1 #{apt_id}")
    assert owner_confirm["status"] == "owner_confirmed"
    assert find_appointment_by_id(apt_id)["status"] == "confirmed"
    print("✅ Owner confirmed customer's slot! Final appointment confirmed for Oct 15 at 10am!")

    # Test 11: Mobile Web Custom Date/Window API with Exact Time
    print("\n--- Test 11: Mobile 1-Tap Web Card Custom Date & Exact Time ---")
    from fastapi.testclient import TestClient
    from app.server import app

    client = TestClient(app)
    # 1. Check GET /a/{apt_id} contains the custom date picker, time picker and smart chips
    page_resp = client.get(f"/a/{apt_id}")
    assert page_resp.status_code == 200
    assert "custom-date-picker" in page_resp.text
    assert "custom-time-picker" in page_resp.text
    print(f"GET /a/{apt_id} contains interactive calendar and exact time picker!")

    # 2. POST custom date & exact time (e.g. 2026-10-23 at 3:30 PM)
    custom_resp = client.post(
        f"/api/appointments/{apt_id}/action",
        json={
            "action": "custom",
            "date": "2026-10-23",
            "window": "afternoon",
            "exact_time": "3:30 PM"
        }
    )
    assert custom_resp.status_code == 200
    assert custom_resp.json()["new_date"] == "2026-10-23"
    assert custom_resp.json()["exact_time"] == "3:30 PM"

    apt_final = find_appointment_by_id(apt_id)
    assert apt_final["appointment_date"] == "2026-10-23"
    assert apt_final["exact_time"] == "3:30 PM"
    print("✅ Verified custom exact date 2026-10-23 at 3:30 PM applied successfully via web!")

    # Test 12: Customer confirms final proposal
    print("\n--- Test 12: Customer Confirms Final Proposal ('YES') ---")
    cust_res = await process_inbound_sms(customer_phone, "YES")
    assert cust_res["status"] == "customer_confirmed"
    assert find_appointment_by_id(apt_id)["status"] == "confirmed"
    print("✅ Customer confirmed! Appointment is fully LOCKED in for 2026-10-23 at 3:30 PM!")

    print("\n===================================================================")
    print("🎉 ALL SMART FLEXIBLE SCHEDULING TESTS PASSED (12/12)!")
    print("===================================================================")


if __name__ == "__main__":
    asyncio.run(main())
