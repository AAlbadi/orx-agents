import sys
from fastapi.testclient import TestClient
from app.server import app

client = TestClient(app)

def test_all():
    print("--- 1. Testing GET /subscribe ---")
    res = client.get("/subscribe")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert "ORX Agents" in res.text
    assert "Your 24/7 Phone Receptionist" in res.text
    assert "Activate Receptionist via Polar" in res.text
    print("✓ GET /subscribe passed!")

    print("--- 2. Testing GET /portal ---")
    res = client.get("/portal")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert "Client Portal" in res.text
    assert "Recent Inbound Calls" in res.text
    print("✓ GET /portal passed!")

    print("--- 3. Testing Root Route / for Browser ---")
    res = client.get("/", headers={"Accept": "text/html"})
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert "ORX Agents" in res.text
    print("✓ Root route browser visit passed!")

    print("--- 4. Testing Root Route / for Plivo Call ---")
    res = client.get("/?CallUUID=call_12345&From=%2B15551234567")
    assert res.status_code == 200
    assert "<Response>" in res.text
    print("✓ Root route Plivo webhook passed!")

    print("--- 5. Testing POST /api/onboarding/discover ---")
    res = client.post("/api/onboarding/discover", json={"query": "Apex Plumbing, 100 Congress Ave, Austin TX"})
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["inferred_industry"] == "plumbing"
    print(f"✓ Discovery passed! Discovered: {data['business_name']} ({data['inferred_industry']})")

    print("--- 6. Testing POST /api/onboarding/save-profile ---")
    profile_payload = {
        "id": "cli_test_1",
        "business_name": "Apex Plumbing Specialists",
        "industry": "plumbing",
        "address": "100 Congress Ave, Austin TX",
        "forwarding_phone": "+1 (512) 555-0199",
        "hours": "Mon-Sat 7:00 AM to 7:00 PM",
        "transfer_rules": "Transfer on burst pipes or water heater flooding",
        "services": "Drain clearing, water heaters, leak detection",
        "custom_qa": [{"question": "Do you offer emergency service?", "answer": "Yes, 24/7 emergency dispatch."}]
    }
    res = client.post("/api/onboarding/save-profile", json=profile_payload)
    assert res.status_code == 200
    saved = res.json()["profile"]
    assert saved["business_name"] == "Apex Plumbing Specialists"
    assert "compiled_prompt" in saved
    assert "Apex Plumbing Specialists" in saved["compiled_prompt"]
    print("✓ Profile saved & prompt compiled successfully!")

    print("--- 7. Testing POST /api/onboarding/chat-test ---")
    res = client.post("/api/onboarding/chat-test", json={
        "client_id": "cli_test_1",
        "message": "What are your operating hours?"
    })
    assert res.status_code == 200
    reply = res.json()
    assert "response" in reply
    print(f"✓ Chat response: {reply['response']}")

    # Test transfer trigger
    res_transfer = client.post("/api/onboarding/chat-test", json={
        "client_id": "cli_test_1",
        "message": "I have an emergency burst pipe in my basement!"
    })
    assert res_transfer.status_code == 200
    transfer_data = res_transfer.json()
    assert transfer_data["is_transfer"] is True
    print(f"✓ Emergency transfer trigger verified: {transfer_data['response']}")

    print("--- 8. Testing POST /api/polar/create-checkout ---")
    res = client.post("/api/polar/create-checkout", json={
        "plan": "starter",
        "client_id": "cli_test_1",
        "success_url": "http://localhost:7860/portal"
    })
    assert res.status_code == 200
    checkout_data = res.json()
    assert "checkout_url" in checkout_data
    print(f"✓ Polar checkout session created: mode={checkout_data.get('mode')}")

    print("--- 9. Testing Client Profile & Update ---")
    res = client.get("/api/client/profile?client_id=cli_test_1")
    assert res.status_code == 200
    assert res.json()["profile"]["id"] == "cli_test_1"

    res_up = client.post("/api/client/update", json={
        "client_id": "cli_test_1",
        "forwarding_phone": "+1 (512) 555-9999"
    })
    assert res_up.status_code == 200
    assert res_up.json()["profile"]["forwarding_phone"] == "+1 (512) 555-9999"
    print("✓ Client profile update verified!")

    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_all()
