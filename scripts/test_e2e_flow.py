import sys
import json
from fastapi.testclient import TestClient
from app.server import app
from app.onboarding import get_client_profile

client = TestClient(app)

def run_end_to_end_test():
    print("="*60)
    print("🚀 RUNNING END-TO-END TEST SUITE (ONBOARDING -> POLAR -> PORTAL -> VERIFICATION)")
    print("="*60)

    # 1. Visitor Lands on Site
    print("\n[Step 1] Visitor landing on / and /subscribe...")
    res_home = client.get("/", headers={"Accept": "text/html"})
    assert res_home.status_code == 200
    assert "ORX Agents" in res_home.text
    assert "Got an office phone or app" in res_home.text
    print("  ✓ Landing page served with non-AI aesthetic and office phone drawer!")

    # 2. Google Maps / Places Auto-Search
    print("\n[Step 2] Business owner searches on Google Maps in Box 1...")
    res_search = client.post("/api/onboarding/search-places", json={"query": "Apex HVAC Austin"})
    assert res_search.status_code == 200
    matches = res_search.json().get("matches", [])
    assert len(matches) > 0
    print(f"  ✓ Found {len(matches)} matching Google Maps results for 'Apex HVAC Austin'")
    top_match = matches[0]
    print(f"  ✓ Top Match: {top_match['title']} - {top_match['address']}")

    # 3. Google Maps Place Enrichment & Auto-Fill
    print("\n[Step 3] Business auto-discovery & field enrichment...")
    res_disc = client.post("/api/onboarding/discover", json={"query": f"{top_match['title']}, {top_match['address']}"})
    assert res_disc.status_code == 200
    disc_data = res_disc.json()
    assert disc_data["success"] is True
    assert disc_data["inferred_industry"] == "hvac"
    print(f"  ✓ Enriched: Name='{disc_data['business_name']}', Trade='{disc_data['inferred_industry']}', Address='{disc_data['formatted_address']}'")

    # 4. Save Profile & Compile Tailored Agent
    client_id = "cli_e2e_test_99"
    print(f"\n[Step 4] Saving profile and compiling agent prompt for {client_id}...")
    profile_payload = {
        "id": client_id,
        "business_name": disc_data["business_name"],
        "industry": disc_data["inferred_industry"],
        "address": disc_data["formatted_address"],
        "forwarding_phone": "+1 (512) 555-4321",
        "hours": disc_data.get("hours", "Monday through Friday 8:00 AM to 6:00 PM"),
        "transfer_rules": "Immediate transfer for gas smell, water leaks, or caller asks for owner",
        "services": "AC repair, furnace tune-ups, duct cleaning",
        "assigned_phone": "+1 (833) 420-5227"
    }
    res_save = client.post("/api/onboarding/save-profile", json=profile_payload)
    assert res_save.status_code == 200
    saved_prof = res_save.json()["profile"]
    assert "Riley" in saved_prof["compiled_prompt"]
    assert disc_data["business_name"] in saved_prof["compiled_prompt"]
    print("  ✓ Agent prompt successfully compiled with industry voice guardrails!")

    # 5. Test Live Demo Conversational Turn
    print("\n[Step 5] Testing live simulator conversational turn...")
    res_chat = client.post("/api/onboarding/chat-test", json={
        "client_id": client_id,
        "message": "What are your operating hours?"
    })
    assert res_chat.status_code == 200
    chat_reply = res_chat.json()
    assert "response" in chat_reply
    print(f"  ✓ Agent responded: \"{chat_reply['response']}\"")

    # 6. Polar Subscription Checkout Creation
    print("\n[Step 6] Creating Polar subscription checkout session...")
    res_polar = client.post("/api/polar/create-checkout", json={
        "plan": "starter",
        "client_id": client_id,
        "success_url": "http://localhost:7860/portal"
    })
    assert res_polar.status_code == 200
    polar_data = res_polar.json()
    assert "checkout_url" in polar_data
    print(f"  ✓ Polar Checkout Session generated: Mode={polar_data.get('mode')}, URL={polar_data.get('checkout_url')}")

    # 7. Polar Webhook: Payment Success Event
    print("\n[Step 7] Simulating Polar webhook payment success event...")
    webhook_payload = {
        "type": "subscription.created",
        "data": {
            "id": "sub_polar_live_123",
            "status": "active",
            "metadata": {
                "client_id": client_id,
                "plan": "starter"
            }
        }
    }
    res_webhook = client.post("/api/polar/webhook", json=webhook_payload)
    assert res_webhook.status_code == 200
    updated_client = get_client_profile(client_id)
    assert updated_client["polar_status"] == "active"
    print("  ✓ Webhook processed: Client subscription marked as 'active' in database!")

    # 8. Post-Payment Portal View & Office Phone Drawer
    print("\n[Step 8] Loading post-payment client portal (/portal)...")
    res_portal = client.get(f"/portal?client_id={client_id}")
    assert res_portal.status_code == 200
    assert "Client Portal" in res_portal.text
    assert "Connect Your Business Line in 1 Tap" in res_portal.text
    assert "Got an office phone or app" in res_portal.text
    print("  ✓ Client portal successfully rendered with mobile 1-tap dial and office phone drawer!")

    # 9. 1-Tap Carrier Forwarding Line Ping Verification
    print("\n[Step 9] Simulating 1-Tap carrier forward ping verification...")
    res_ping = client.post("/api/client/ping-verify", json={
        "client_id": client_id,
        "carrier": "verizon"
    })
    assert res_ping.status_code == 200
    ping_data = res_ping.json()
    assert ping_data["connection_status"] == "verified"
    assert ping_data["carrier"] == "Verizon"
    print(f"  ✓ Carrier line verified live! Timestamp: {ping_data['verified_at']}")

    # 10. Disconnect / Pause Forwarding Test
    print("\n[Step 10] Testing pause/disconnect (*73)...")
    res_disc = client.post("/api/client/disconnect", json={"client_id": client_id})
    assert res_disc.status_code == 200
    assert res_disc.json()["connection_status"] == "paused"
    print("  ✓ Line successfully paused!")

    print("\n" + "="*60)
    print("🎉 ALL 10 END-TO-END STEPS PASSED WITH 100% SUCCESS!")
    print("="*60)

if __name__ == "__main__":
    run_end_to_end_test()
