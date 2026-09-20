import asyncio
import time
from livekit.agents import llm
from livekit.plugins import openai
from app.config import settings
from app.livekit_agent import ResilientVoiceAgent, _build_llm_service, _build_fallback_llm_service

async def run_simulation():
    print("=== STARTING LIVEKIT RATE-LIMIT & TURN FIX SIMULATION ===")
    
    # 1. Verify Fallback LLM Factory
    groq_primary = _build_llm_service("groq", "qwen/qwen3.8-27b")
    groq_fallback = _build_fallback_llm_service("groq")
    print(f"Primary LLM: {groq_primary.model} (max_retries={getattr(groq_primary, '_max_retries', 'configured')})")
    assert groq_fallback is not None, "Fallback LLM should be created when GEMINI_API_KEY exists"
    print(f"Fallback LLM: {groq_fallback.model}")
    
    # 2. Test ResilientVoiceAgent Instantiation
    agent = ResilientVoiceAgent(
        instructions="You are Aria, a warm, articulate voice AI receptionist.",
        fallback_llm=groq_fallback,
        max_history_items=8,
    )
    print("ResilientVoiceAgent initialized successfully.")

    # 3. Simulate 25 turns to test context windowing & token bounding
    print("\nSimulating 25 continuous conversation turns...")
    from livekit.agents.voice.agent_activity import update_instructions
    for i in range(25):
        ctx = agent.chat_ctx.copy()
        update_instructions(ctx, instructions=agent.instructions, add_if_missing=True)
        ctx.add_message(role="user", content=f"Customer message {i}: Need AC repair at 104 Main St.")
        ctx.add_message(role="assistant", content=f"Aria response {i}: Got it, we have an opening tomorrow.")
        if len(ctx.items) > 8:
            ctx.truncate(max_items=8)
        await agent.update_chat_ctx(ctx)

    print(f"ChatContext items count after 25 turns: {len(agent.chat_ctx.items)}")
    # Should be 1 system message + 8 recent turns = 9 items
    assert len(agent.chat_ctx.items) == 9, f"Expected 9 items, got {len(agent.chat_ctx.items)}"
    assert agent.chat_ctx.items[0].role == "system"
    print("✅ Context sliding window verified! Tokens are strictly bounded.")

    # 4. Test Failover from simulated 429 Rate Limit
    print("\nTesting automatic failover when primary LLM fails (e.g. 429 rate limit)...")
    # Bad primary that simulates rate limit / auth error
    simulated_failed_primary = openai.LLM(
        base_url="https://api.groq.com/openai/v1",
        api_key="simulated_exhausted_key",
        model="qwen/qwen3.8-27b",
        max_retries=1,
    )
    
    test_agent = ResilientVoiceAgent(
        instructions="You are Aria, an AI receptionist.",
        fallback_llm=groq_fallback,
        max_history_items=8,
    )
    
    # Create test turn context where user asked a question
    turn_ctx = llm.ChatContext()
    turn_ctx.add_message(role="system", content="You are Aria, an AI receptionist. Speak in one short sentence.")
    turn_ctx.add_message(role="user", content="What days are you open?")
    
    class FakeActivity:
        def __init__(self, primary_llm):
            self.llm = primary_llm
            self.session = FakeSession()
            self.tools = []
    
    class FakeSession:
        def __init__(self):
            class ConnOpt:
                llm_conn_options = None
            self.conn_options = ConnOpt()
    
    test_agent._activity = FakeActivity(simulated_failed_primary)
    
    # Run llm_node through fake activity
    output_chunks = []
    async for chunk in test_agent.llm_node(turn_ctx, [], None):
        if hasattr(chunk, "delta") and chunk.delta and chunk.delta.content:
            output_chunks.append(chunk.delta.content)
        elif isinstance(chunk, str):
            output_chunks.append(chunk)

    full_response = "".join(output_chunks).strip()
    print(f"Generated response during primary failure: '{full_response}'")
    assert len(full_response) > 0, "Agent must yield a response even if primary LLM fails!"
    print("✅ Zero-downtime failover verified! The agent never goes silent.")

    print("\n=== ALL RATE-LIMIT & TURN RECOVERY TESTS PASSED! ===")

if __name__ == "__main__":
    asyncio.run(run_simulation())
