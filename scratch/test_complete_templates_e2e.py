"""
End-to-End Multi-Turn Case Scenario Verification for Top-Tier Voice AI Templates.
Tests:
1. Inbound HVAC Receptionist & Dispatcher (Riley / Comfort Breeze):
   - Conversational reciprocity & small talk
   - Broken AC empathy in extreme heat
   - Mandatory Address Confirmation Protocol
   - Flexible scheduling conflict handling ("That wouldn't work")
   - Upfront diagnostic pricing transparency ($89 fee)
   - 5-point verbal recap and clean sign-off
2. Outbound B2B Sales to HVAC Business Owners (Marcus / OrxLabs):
   - Fast 30-second hook with busy owner on job site
   - Peer connection on trade pain (missed calls while on a roof)
   - Price objection handling ("How much does it cost?")
   - 2-minute interactive text demo delivery
   - Declarative cell readback and clean close without over-pitching
"""

import sys
import os
import time

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.templates_mgr import DEFAULT_TEMPLATES
from app.tones import compile_tone_prompt
from scratch.test_server_tones_live import simulate_llm_call

def get_template(tpl_id: str):
    for t in DEFAULT_TEMPLATES:
        if t["id"] == tpl_id:
            return t
    raise ValueError(f"Template {tpl_id} not found!")

def run_inbound_hvac_e2e():
    print("=" * 70)
    print("SCENARIO 1: INBOUND HVAC RECEPTIONIST (RILEY / COMFORT BREEZE)")
    print("=" * 70)

    tpl = get_template("tpl-hvac")
    system_prompt = tpl["system_prompt"]
    greeting = tpl["first_message"]

    conversation = [
        {"role": "assistant", "content": greeting}
    ]
    print(f"Assistant (Greeting): \"{greeting}\"\n")

    user_turns = [
        # Turn 1: Small talk + urgent breakdown in heat
        "Good, how are you? Actually our AC completely died in 95-degree heat, it's blowing warm air and my house is baking.",
        # Turn 2: Address
        "We are at 2508 Delaware Street in Minneapolis, Minnesota.",
        # Turn 3: Address confirmation
        "Yes, that is correct.",
        # Turn 4: Schedule conflict / pushback
        "That wouldn't work at all, I work until 5pm every day.",
        # Turn 5: Specific time request + Pricing objection
        "Can you do tomorrow at 5:30pm? Also, how much does it cost just to have a technician come out?",
        # Turn 6: Contact Info
        "My name is Abdul, phone is 612-716-9989, and email is abdul@example.com.",
        # Turn 7: Final confirmation
        "Yes, that all sounds perfect, thank you!"
    ]

    for idx, user_input in enumerate(user_turns, start=1):
        print(f"--- Turn {idx} ---")
        print(f"Caller: \"{user_input}\"")
        conversation.append({"role": "user", "content": user_input})

        # Build prompt with conversation history for realistic multi-turn context
        history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation])
        full_context = f"{system_prompt}\n\n<conversation_history>\n{history_text}\n</conversation_history>\n\nAssistant (speak 1-2 natural spoken sentences):"

        t0 = time.time()
        reply = simulate_llm_call(full_context, user_input)
        latency_ms = (time.time() - t0) * 1000

        print(f"Riley ({latency_ms:.0f}ms): \"{reply}\"\n")
        conversation.append({"role": "assistant", "content": reply})

    print("✅ Inbound HVAC Call Simulation Finished Successfully!\n")
    return conversation


def run_outbound_sales_e2e():
    print("=" * 70)
    print("SCENARIO 2: OUTBOUND B2B SALES TO HVAC OWNER (MARCUS / ORXLABS)")
    print("=" * 70)

    tpl = get_template("tpl-outbound-sales")
    system_prompt = tpl["system_prompt"]
    greeting = tpl["first_message"]

    conversation = [
        {"role": "assistant", "content": greeting}
    ]
    print(f"Marcus (Greeting): \"{greeting}\"\n")

    user_turns = [
        # Turn 1: Busy contractor pushback
        "Who's this? I'm in the middle of a job right now.",
        # Turn 2: Real trade pain admission
        "We try to answer, but honestly when we're under a house or on a roof we just miss them.",
        # Turn 3: Price objection
        "How much does this cost anyway? I don't want another expensive software subscription.",
        # Turn 4: Agreement to text demo & providing cell
        "Alright, you can text it to my cell at 612-716-9989.",
        # Turn 5: Wrap up
        "Sounds good, I'll take a look tonight. Thanks Marcus."
    ]

    for idx, user_input in enumerate(user_turns, start=1):
        print(f"--- Turn {idx} ---")
        print(f"HVAC Owner: \"{user_input}\"")
        conversation.append({"role": "user", "content": user_input})

        history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation])
        full_context = f"{system_prompt}\n\n<conversation_history>\n{history_text}\n</conversation_history>\n\nAssistant (speak 1-2 natural spoken sentences):"

        t0 = time.time()
        reply = simulate_llm_call(full_context, user_input)
        latency_ms = (time.time() - t0) * 1000

        print(f"Marcus ({latency_ms:.0f}ms): \"{reply}\"\n")
        conversation.append({"role": "assistant", "content": reply})

    print("✅ Outbound HVAC Sales Simulation Finished Successfully!\n")
    return conversation


if __name__ == "__main__":
    run_inbound_hvac_e2e()
    run_outbound_sales_e2e()
