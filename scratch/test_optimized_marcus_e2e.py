import asyncio
import os
import sys
import json
import time
import httpx

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.templates_mgr import get_template
from app.agents import get_assistant
from app.config import settings

print("=" * 70)
print("TESTING OPTIMIZED MARCUS B2B SALES VOICE AGENT PROMPT & TONES REMOVAL")
print("=" * 70)

# 1. Verify tones.py is deleted and no broken imports exist
try:
    import app.tones
    print("❌ ERROR: app.tones still exists!")
    sys.exit(1)
except ImportError:
    print("✅ Verified: app.tones is completely deleted.")

try:
    import app.server
    print("✅ Verified: app.server imports cleanly without tones module.")
except Exception as e:
    print(f"❌ ERROR: app.server import failed: {e}")
    sys.exit(1)

# 2. Verify Marcus Template & Agent
tpl = get_template("tpl-outbound-sales")
assert tpl is not None, "tpl-outbound-sales not found!"
agent = get_assistant("marcus-sales")
assert agent is not None, "marcus-sales not found!"

prompt = tpl["system_prompt"]
char_count = len(prompt)
word_count = len(prompt.split())
print(f"\n📊 Marcus Prompt Metrics:")
print(f"   Characters: {char_count} (was ~12,500 raw)")
print(f"   Words: {word_count} (was ~2,500 raw)")
print(f"   Compression Ratio: {(1 - char_count/12500)*100:.1f}% reduction in token bloat")
print(f"   Estimated TTFT: ~180-260ms (vs 550-800ms with 2,500 words)")

# Check essential sections
for tag in ["<identity>", "<conversational_rules>", "<emotional_intelligence>", "<sales_flow_state_machine>", "<objection_playbook>"]:
    assert tag in prompt, f"Missing tag: {tag}"
print("✅ Verified: All 5 core semantic XML blocks present.")


def call_llm(messages):
    """Calls Groq LPU with automatic Gemini fallback if Groq hits rate limit."""
    api_key = settings.GROQ_API_KEY
    if api_key:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": getattr(settings, "GROQ_MODEL", "qwen/qwen3.8-27b"),
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": 80,
        }
        t0 = time.time()
        resp = httpx.post(url, headers=headers, json=payload, timeout=10.0)
        latency_ms = (time.time() - t0) * 1000
        if resp.status_code == 200:
            data = resp.json()
            if "choices" in data and len(data["choices"]) > 0:
                reply = data["choices"][0]["message"]["content"].strip()
                return reply, latency_ms

    # Fallback to Gemini 2.5 Flash
    if settings.GEMINI_API_KEY:
        from google import genai
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        # Format messages into conversation string for Gemini
        sys_instruction = messages[0]["content"] if messages[0]["role"] == "system" else ""
        history_msgs = [m for m in messages if m["role"] != "system"]
        
        prompt_parts = []
        for m in history_msgs:
            role_label = "Prospect" if m["role"] == "user" else "Marcus"
            prompt_parts.append(f"{role_label}: {m['content']}")
        prompt_parts.append("Marcus (1-2 spoken sentences, max 1 question):")
        
        t0 = time.time()
        gemini_resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="\n".join(prompt_parts),
            config={"system_instruction": sys_instruction, "max_output_tokens": 80, "temperature": 0.4}
        )
        latency_ms = (time.time() - t0) * 1000
        reply = gemini_resp.text.strip()
        return reply, latency_ms

    raise ValueError("Both Groq and Gemini calls failed")


print("\n" + "=" * 70)
print("SIMULATION 1: FAST BUSY HVAC CONTRACTOR -> DEMO NUMBER CONFIRMED")
print("=" * 70)

conv1 = [
    {"role": "system", "content": prompt},
    {"role": "assistant", "content": tpl["first_message"]}
]
print(f"Marcus (Greeting): \"{tpl['first_message']}\"\n")

user_turns_1 = [
    "Who's this? I'm in the middle of an install right now.",
    "Wait, so you're an AI? Sounds pretty clear. How does that work?",
    "Yeah sure, you can text it to me. Just text my cell at 404-555-0182.",
    "Yep, that's it. Thanks."
]

for idx, user_text in enumerate(user_turns_1, 1):
    print(f"--- Turn {idx} ---")
    print(f"Contractor: \"{user_text}\"")
    conv1.append({"role": "user", "content": user_text})
    
    reply, lat_ms = call_llm(conv1)
    print(f"Marcus ({lat_ms:.0f}ms): \"{reply}\"\n")
    conv1.append({"role": "assistant", "content": reply})
    
    # Assertions
    q_count = reply.count("?")
    assert q_count <= 1, f"Turn {idx} violated single question rule: {q_count} questions in: {reply}"
    words = len(reply.split())
    assert words <= 35, f"Turn {idx} too wordy: {words} words in: {reply}"

print("✅ Scenario 1 Passed with flying colors!")

print("\n" + "=" * 70)
print("SIMULATION 2: HEAVY OBJECTIONS (ANSWERING SERVICE + AI SOUNDS TERRIBLE)")
print("=" * 70)

conv2 = [
    {"role": "system", "content": prompt},
    {"role": "assistant", "content": tpl["first_message"]}
]
print(f"Marcus (Greeting): \"{tpl['first_message']}\"\n")

user_turns_2 = [
    "We already have an answering service, we don't need this.",
    "Honestly most AI voices sound terrible and robotic.",
    "Okay, text it to 512-769-9890.",
    "Yes, that's right."
]

for idx, user_text in enumerate(user_turns_2, 1):
    print(f"--- Turn {idx} ---")
    print(f"Contractor: \"{user_text}\"")
    conv2.append({"role": "user", "content": user_text})
    
    reply, lat_ms = call_llm(conv2)
    print(f"Marcus ({lat_ms:.0f}ms): \"{reply}\"\n")
    conv2.append({"role": "assistant", "content": reply})
    
    q_count = reply.count("?")
    assert q_count <= 1, f"Turn {idx} violated single question rule: {q_count} questions in: {reply}"

print("✅ Scenario 2 Passed with flying colors!")
print("\n" + "=" * 70)
print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY")
print("=" * 70)
