import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.agents import get_assistant
from app.config import settings
import httpx

async def main():
    groq_api_key = getattr(settings, "GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
    agent = get_assistant("riley-hvac")
    prompt = agent["system_prompt"]

    print("--- Testing Address Confirmation Flow with Live Groq LPU ---")

    test_conversations = [
        # Scenario 1: Caller provides address -> Agent must confirm address ONLY and STOP
        {
            "name": "Turn after address given",
            "messages": [
                {"role": "assistant", "content": "Thank you for calling Comfort Breeze Heating and Air. This is Riley, your virtual receptionist. How may I get your service scheduled today?"},
                {"role": "user", "content": "Hi, my AC is blowing warm air and it's 85 degrees inside."},
                {"role": "assistant", "content": "Understood. To get our technician out to you, what is the service address where we will be working?"},
                {"role": "user", "content": "2508 Delaware Street in Minneapolis."},
            ],
            "expected_check": lambda reply: "correct" in reply.lower() or "right" in reply.lower()
        },
        # Scenario 2: Caller confirms address -> Agent must propose appointment windows
        {
            "name": "Turn after address confirmed",
            "messages": [
                {"role": "assistant", "content": "Understood. To get our technician out to you, what is the service address where we will be working?"},
                {"role": "user", "content": "2508 Delaware Street in Minneapolis."},
                {"role": "assistant", "content": "Got it, 2508 Delaware Street in Minneapolis, is that correct?"},
                {"role": "user", "content": "Yes, that's right."},
            ],
            "expected_check": lambda reply: "tomorrow" in reply.lower() or "window" in reply.lower() or "morning" in reply.lower()
        },
        # Scenario 3: Caller corrects misheard address -> Agent must adopt correction immediately
        {
            "name": "Turn after address correction",
            "messages": [
                {"role": "assistant", "content": "Understood. To get our technician out to you, what is the service address where we will be working?"},
                {"role": "user", "content": "2508 Delaware Street in Minneapolis."},
                {"role": "assistant", "content": "Got it, 2508 The Lowest Street in Minneapolis, is that correct?"},
                {"role": "user", "content": "No, it's 2508 Delaware Street, not lowest!"},
            ],
            "expected_check": lambda reply: "delaware" in reply.lower()
        }
    ]

    async with httpx.AsyncClient(timeout=15.0) as client:
        for tc in test_conversations:
            payload = {
                "model": "qwen/qwen3.8-27b",
                "messages": [{"role": "system", "content": prompt}] + tc["messages"],
                "temperature": 0.3,
                "max_tokens": 100,
            }
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"},
                json=payload
            )
            reply = r.json()["choices"][0]["message"]["content"].strip()
            q_count = reply.count("?")
            print(f"\n[{tc['name']}]")
            print(f"Reply: \"{reply}\"")
            print(f"Question Count: {q_count}")
            assert q_count <= 1, f"Failed: Multiple questions detected ({q_count})"
            assert tc["expected_check"](reply), f"Failed expected check on: '{reply}'"
            print("PASS! ✅")

    print("\nALL ADDRESS CONFIRMATION SCENARIOS PASSED WITH EXACTLY 1 QUESTION! 🎉")

if __name__ == "__main__":
    asyncio.run(main())
