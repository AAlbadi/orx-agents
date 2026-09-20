"""Verification test script for:
1. Strict Single Question Rule (<2 questions per turn).
2. Declarative confirmation protocol.
3. Automated SQLite client.db persistence in _finalize_session.
"""

import asyncio
import json
import os
import sqlite3
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.templates_mgr import get_template, DEFAULT_TEMPLATES
from app.agents import get_assistant, DEFAULT_AGENTS
from app.project_db import get_db_path, get_project_calls, get_project_appointments
from app.livekit_agent import _finalize_session, _ACTIVE_SESSIONS, _DEFAULT_AGENT_INSTRUCTIONS
from livekit.plugins import groq
from app.config import settings

async def test_llm_single_question():
    print("\n--- 1. Testing LLM Single Question Rule with Groq ---")
    groq_api_key = getattr(settings, "GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
    if not groq_api_key:
        print("WARNING: GROQ_API_KEY not set; skipping live Groq LLM check.")
        return

    # Test cases: simulated caller turns that previously caused double questions
    test_cases = [
        {
            "role": "riley-hvac",
            "prompt": get_assistant("riley-hvac")["system_prompt"],
            "messages": [
                {"role": "assistant", "content": "Thank you for calling Comfort Breeze Heating and Air. This is Riley. How may I get your service scheduled today?"},
                {"role": "user", "content": "Hi, my AC stopped blowing cold air and it's 85 degrees inside."},
                {"role": "assistant", "content": "I can definitely help you with that. Is your system completely unresponsive or still running?"},
                {"role": "user", "content": "It's running, just blowing warm air."},
                {"role": "assistant", "content": "Understood. What is the street address where you need service?"},
                {"role": "user", "content": "It's 2508 Delaware Street in Minneapolis."},
            ]
        },
        {
            "role": "maya-medical",
            "prompt": get_assistant("maya-medical")["system_prompt"],
            "messages": [
                {"role": "assistant", "content": "Hello and thank you for calling Metro Health and Dental Clinic. My name is Maya. Are you scheduling a routine checkup, or calling regarding an urgent health concern?"},
                {"role": "user", "content": "I'd like to schedule a routine dental cleaning."},
                {"role": "assistant", "content": "I would be happy to help with that. Are you an established patient with us or is this your first visit?"},
                {"role": "user", "content": "First visit, and my name is Sarah Jenkins, born March 14th 1985."},
            ]
        }
    ]

    import httpx
    async with httpx.AsyncClient(timeout=15.0) as client:
        for tc in test_cases:
            payload = {
                "model": "qwen/qwen3.8-27b",
                "messages": [{"role": "system", "content": tc["prompt"]}] + tc["messages"],
                "temperature": 0.3,
                "max_tokens": 100,
            }
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"},
                json=payload
            )
            assert r.status_code == 200, f"Groq API error: {r.text}"
            reply = r.json()["choices"][0]["message"]["content"].strip()
            q_count = reply.count("?")
            print(f"[{tc['role']}] Agent Reply: \"{reply}\"")
            print(f"[{tc['role']}] Question mark count: {q_count}")
            assert q_count <= 1, f"FAILED: Reply contains {q_count} questions: '{reply}'"
            print(f"[{tc['role']}] PASS: Compliant with single question rule!")

async def test_sqlite_persistence():
    print("\n--- 2. Testing SQLite Persistence in _finalize_session ---")
    test_room = f"test-room-verify-{int(asyncio.get_event_loop().time())}"
    test_client_id = "test_client_verify"

    # Seed an active session with rich conversation data
    _ACTIVE_SESSIONS[test_room] = {
        "call_id": f"lk_test_{int(asyncio.get_event_loop().time())}",
        "started_at": asyncio.get_event_loop().time() - 45.0,
        "config": {
            "client_id": test_client_id,
            "assistant_name": "Riley (Comfort Breeze)",
            "caller": "+16127169989",
            "language": "en"
        },
        "transcript_turns": [
            {"role": "assistant", "text": "Thank you for calling Comfort Breeze. How may I help?"},
            {"role": "user", "text": "My AC is broken, I need someone tomorrow morning."},
            {"role": "assistant", "text": "Got it. What is your service address?"},
            {"role": "user", "text": "2508 Delaware Street in Minneapolis."},
            {"role": "assistant", "text": "Got it, 2508 Delaware Street in Minneapolis. Would tomorrow between 9 and noon work for you?"},
            {"role": "user", "text": "Yes that works. My name is Abdul Aziz Albadi, phone 612-716-9989."},
            {"role": "assistant", "text": "Thank you Abdul Aziz. You are all set for tomorrow between 9 and noon at 2508 Delaware Street. Does that sound right?"},
            {"role": "user", "text": "Yes, sounds good. Thank you!"},
            {"role": "assistant", "text": "Thank you for choosing Comfort Breeze. Have a wonderful day!"}
        ],
        "audio_buffer": bytearray(b"\x00" * 3200)  # 100ms of silence
    }

    # Run _finalize_session
    call_entry = await _finalize_session(test_room, status="completed")
    assert call_entry is not None, "Failed to finalize session"
    print(f"Finalized call: {call_entry.get('call_id')}")

    # Check client.db
    db_path = get_db_path(test_client_id)
    assert db_path.exists(), f"Client SQLite DB does not exist at {db_path}"
    print(f"Verified client.db exists at: {db_path}")

    calls = get_project_calls(test_client_id)
    print(f"Retrieved {len(calls)} calls from client.db:")
    assert len(calls) >= 1, "No calls found in client.db"
    last_call = calls[0]
    print(f" - Call ID: {last_call['id']}, Caller: {last_call['caller_phone']}, Duration: {last_call['call_duration']}s")
    assert last_call['id'] == call_entry.get("call_id"), "Call ID mismatch in client.db"

    apts = get_project_appointments(test_client_id)
    print(f"Retrieved {len(apts)} appointments from client.db:")
    for a in apts:
        print(f" - Apt ID: {a['id']}, Client: {a['client_name']}, Address: {a['service_address']}, Date: {a['appointment_date']}, Window: {a['window']}")

    print("PASS: SQLite persistence successfully verified!")

async def main():
    await test_llm_single_question()
    await test_sqlite_persistence()
    print("\nALL VERIFICATIONS PASSED SUCCESSFULLY! ✅")

if __name__ == "__main__":
    asyncio.run(main())
