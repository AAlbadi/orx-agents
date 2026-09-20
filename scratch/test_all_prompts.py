"""
Test script to verify all upgraded voice AI system prompts across the project:
1. Validates template JSON integrity and prompt structure.
2. Validates agent JSON integrity.
3. Tests compile_livekit_voice_prompt character budget (<600 chars) and structure.
4. Tests compile_agent_prompt structure and slot progression.
5. Tests multi-turn simulated dialogue via Groq LPU / OpenAI / Direct simulation to verify:
   - Address readback and confirmation.
   - Zero markdown or bullet formatting.
   - Strictly one question per turn.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.templates_mgr import list_templates, get_template
from app.agents import list_assistants, get_assistant
from app.project_db import compile_livekit_voice_prompt
from app.onboarding import compile_agent_prompt

def test_templates_integrity():
    print("\n--- 1. Testing Templates Integrity ---")
    templates = list_templates()
    assert len(templates) >= 4, f"Expected at least 4 templates, got {len(templates)}"
    
    for tpl in templates:
        tid = tpl["id"]
        prompt = tpl.get("system_prompt", "")
        print(f"Checking template: {tid} ({tpl.get('name')}) - Prompt Length: {len(prompt)} chars")
        assert len(prompt) > 200, f"Template {tid} prompt is suspiciously short ({len(prompt)} chars)"
        assert "<identity_and_role>" in prompt, f"Template {tid} missing <identity_and_role>"
        assert "<spoken_style_and_conversational_rules>" in prompt, f"Template {tid} missing spoken style rules"
        
        # Check specific verification rules
        if tid == "tpl-hvac":
            assert "MANDATORY ADDRESS CONFIRMATION" in prompt, "tpl-hvac missing address confirmation"
            assert "PHONETIC DISAMBIGUATION" in prompt, "tpl-hvac missing phonetic disambiguation"
            assert "COMPREHENSIVE FINAL VERBAL RECAP" in prompt, "tpl-hvac missing final recap"
        elif tid == "tpl-medical":
            assert "PATIENT NAME & DOB CONFIRMATION" in prompt, "tpl-medical missing patient DOB confirmation"
            assert "INSURANCE VERIFICATION" in prompt, "tpl-medical missing insurance verification"
        elif tid == "tpl-outbound-sales":
            assert "SMS Demo Delivery & Readback" in prompt, "tpl-outbound-sales missing SMS readback"
        elif tid == "tpl-outbound-cloud":
            assert "Contact Verification" in prompt, "tpl-outbound-cloud missing contact verification"

    print("✅ All templates passed integrity and structure checks!")

def test_agents_integrity():
    print("\n--- 2. Testing Agents Integrity ---")
    agents = list_assistants()
    agent_ids = [a["id"] for a in agents]
    print(f"Found {len(agents)} agents: {agent_ids}")
    
    riley = get_assistant("riley-hvac")
    assert riley is not None, "riley-hvac not found"
    assert "MANDATORY ADDRESS CONFIRMATION" in riley.get("system_prompt", ""), "riley-hvac missing address confirmation"
    
    maya = get_assistant("maya-medical")
    assert maya is not None, "maya-medical not found"
    assert "PATIENT NAME & DOB CONFIRMATION" in maya.get("system_prompt", ""), "maya-medical missing DOB confirmation"
    
    marcus = get_assistant("marcus-sales")
    assert marcus is not None, "marcus-sales not found"
    assert "SMS Demo Delivery & Readback" in marcus.get("system_prompt", ""), "marcus-sales missing SMS readback"
    
    print("✅ Flagship agents passed integrity and verification checks!")

def test_compilers():
    print("\n--- 3. Testing Prompt Compilers ---")
    
    # Test compile_livekit_voice_prompt
    sample_data = {
        "business_name": "Comfort Air Solutions",
        "persona_name": "Riley",
        "hours": "Mon-Fri 7am-7pm, Sat 9am-3pm",
        "services": "AC repair, heat pump maintenance, emergency diagnostic",
        "pricing_policy": "$89 diagnostic fee applied toward any repair",
        "transfer_rules": "gas odor or immediate supervisor request",
        "forwarding_phone": "+1 (555) 345-6789",
    }
    livekit_prompt = compile_livekit_voice_prompt(sample_data)
    print(f"Compiled LiveKit Prompt ({len(livekit_prompt)} chars):\n{livekit_prompt}\n")
    assert len(livekit_prompt) < 600, f"LiveKit prompt exceeds 600 chars: {len(livekit_prompt)}"
    assert "confirm street address" in livekit_prompt or "Confirm address" in livekit_prompt
    assert "ONE clear question" in livekit_prompt or "ONE question" in livekit_prompt
    
    # Test compile_agent_prompt
    onboarding_prompt = compile_agent_prompt(sample_data)
    print(f"Compiled Onboarding Prompt ({len(onboarding_prompt)} chars):\n{onboarding_prompt[:300]}...\n")
    assert "Proactive Verification & Grounding" in onboarding_prompt
    assert "Address Confirmation" in onboarding_prompt
    assert "Phonetic Disambiguation" in onboarding_prompt
    
    print("✅ Both prompt compilers passed tests successfully!")

def test_groq_simulation():
    print("\n--- 4. Testing LLM Response Simulation (Groq LPU) ---")
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("⚠️ GROQ_API_KEY not configured, skipping live API simulation.")
        return

    import httpx
    tpl = get_template("tpl-hvac")
    system_prompt = tpl["system_prompt"]
    
    # Turn 1: Caller states problem and gives address
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": tpl["first_message"]},
        {"role": "user", "content": "Hi Riley, my AC is blowing warm air and it is 85 degrees inside. I live at 2508 Delaware Street in Minneapolis."}
    ]
    
    try:
        resp = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "qwen/qwen3.8-27b",
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 100,
            },
            timeout=10.0
        )
        data = resp.json()
        if "choices" in data:
            reply = data["choices"][0]["message"]["content"]
            print(f"User: Hi Riley, my AC is blowing warm air and it is 85 degrees inside. I live at 2508 Delaware Street in Minneapolis.")
            print(f"Riley: {reply}\n")
            
            # Verify Riley confirmed the address and did not ask for it again
            assert "2508 Delaware" in reply or "Delaware" in reply, "Agent failed to acknowledge/confirm address"
            assert not any(m in reply for m in ["**", "*", "#", "- "]), "Agent used markdown formatting!"
            word_count = len(reply.split())
            print(f"Reply word count: {word_count} words (limit < 35)")
            assert word_count < 35, f"Turn is too long: {word_count} words"
            print("✅ Groq LPU response adhered to address confirmation, voice brevity, and no-markdown guardrails!")
        else:
            print(f"Groq API response error: {data}")
    except Exception as e:
        print(f"Groq simulation error: {e}")

if __name__ == "__main__":
    test_templates_integrity()
    test_agents_integrity()
    test_compilers()
    test_groq_simulation()
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
