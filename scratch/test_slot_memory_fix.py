import asyncio
import json
from app.livekit_agent import ResilientVoiceAgent, _build_llm_service, _build_fallback_llm_service
from livekit.agents import llm

async def test_slot_memory_and_fallback():
    print("=== STARTING SLOT MEMORY & MODEL LOGIC TEST ===")
    
    # 1. Test Fallback Pairing Logic
    print("\n--- 1. Testing Model Selector & Fallback Pairing Logic ---")
    groq_primary = _build_llm_service("groq", "qwen/qwen3.8-27b")
    groq_fallback = _build_fallback_llm_service("groq")
    print(f"When primary is Groq ({groq_primary.model}): Fallback is {groq_fallback.model} (Gemini)")
    assert "gemini" in groq_fallback.model.lower()

    gemini_primary = _build_llm_service("gemini", "gemini-2.5-flash-lite")
    gemini_fallback = _build_fallback_llm_service("gemini")
    print(f"When primary is Gemini ({gemini_primary.model}): Fallback is {gemini_fallback.model} (Groq)")
    assert "qwen" in gemini_fallback.model.lower() or "groq" in gemini_fallback.model.lower()

    # 2. Simulate the Exact 47-Turn Call from data/calls.json
    print("\n--- 2. Replaying 47-Turn Call with SlotTracker ---")
    agent = ResilientVoiceAgent(
        instructions=(
            "You are Riley, receptionist for Comfort Breeze Heating and Air. "
            "Flow: 1. Issue. 2. Address. 3. Two times. 4. Name, phone, email. 5. Verbal recap."
        ),
        fallback_llm=groq_fallback,
        max_history_items=20,
    )
    
    with open("data/calls.json") as f:
        data = json.load(f)
    call = data["calls"][-1]
    
    # Feed turns 1 through 39 into agent
    for turn in call["transcript"][:39]:
        agent.record_turn(turn["speaker"], turn["text"])

    print("Slots tracked so far at turn 39:")
    for k, v in agent.slot_tracker.slots.items():
        print(f"  {k}: {v}")
        
    assert "address" in agent.slot_tracker.slots, "Address must be captured!"
    assert "Delaware" in agent.slot_tracker.slots["address"], "Delaware Street must be captured!"
    assert "appointment" in agent.slot_tracker.slots, "Appointment must be captured!"
    assert "name" in agent.slot_tracker.slots, "Name must be captured!"
    assert "phone" in agent.slot_tracker.slots, "Phone must be captured!"
    assert "email" in agent.slot_tracker.slots, "Email must be captured!"

    # 3. Verify Memory Injection Block
    print("\n--- 3. Verifying Injected Memory Block ---")
    memory_block = agent.slot_tracker.get_prompt_block()
    print(memory_block)
    assert "[CONFIRMED BOOKING DETAILS — DO NOT RE-ASK]" in memory_block
    assert "2508 Delaware Street" in memory_block
    assert "abdulazizalpadi91@gmail.com" in memory_block

    # 4. Test LLM Response at Turn 40
    print("\n--- 4. Testing LLM Response at Turn 40 (Spelling completed) ---")
    ctx = llm.ChatContext()
    ctx.add_message(role="system", content=agent.instructions + memory_block)
    ctx.add_message(role="assistant", content="Just to double-check, is that email address abdulazizalpadi91@gmail.com?")
    ctx.add_message(role="user", content="Yes. Exactly.")
    
    # Generate reply with Gemini
    chunks = []
    async with gemini_primary.chat(chat_ctx=ctx) as stream:
        async for chunk in stream:
            if hasattr(chunk, "delta") and chunk.delta and chunk.delta.content:
                chunks.append(chunk.delta.content)
    
    response = "".join(chunks).strip()
    print(f"Turn 40 Agent Response:\n'{response}'")
    
    # Assert that the agent did NOT ask for the issue or address again
    lower_resp = response.lower()
    assert "what is your service address" not in lower_resp, "Must NOT re-ask for address!"
    assert "what seems to be the issue" not in lower_resp, "Must NOT re-ask for issue!"
    assert "how can we help you with your hvac" not in lower_resp, "Must NOT restart flow!"
    print("✅ Verified! Agent did NOT restart flow or re-ask for collected info!")

    print("\n=== ALL TESTS PASSED! ===")

if __name__ == "__main__":
    asyncio.run(test_slot_memory_and_fallback())
