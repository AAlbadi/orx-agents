import asyncio
import time
from typing import AsyncGenerator, Optional
from livekit.agents import llm
from livekit.agents.voice import Agent
from livekit.agents.voice.agent import ModelSettings, NOT_GIVEN
from livekit.plugins import openai
from app.config import settings

class ResilientVoiceAgent(Agent):
    """Voice agent with automatic context truncation and multi-provider LLM fallback."""
    def __init__(self, *args, fallback_llm: Optional[llm.LLM] = None, max_history_items: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.fallback_llm = fallback_llm
        self.max_history_items = max_history_items

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        """Prune conversation context to prevent token explosion and rate limits."""
        if len(turn_ctx.items) > self.max_history_items:
            turn_ctx.truncate(max_items=self.max_history_items)

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[llm.ChatChunk | str, None]:
        # Enforce sliding window directly on chat_ctx before inference
        if len(chat_ctx.items) > self.max_history_items:
            chat_ctx.truncate(max_items=self.max_history_items)

        activity = self._get_activity_or_raise()
        primary_llm = activity.llm
        conn_options = activity.session.conn_options.llm_conn_options
        tool_choice = model_settings.tool_choice if model_settings else NOT_GIVEN

        has_yielded = False
        try:
            async with primary_llm.chat(
                chat_ctx=chat_ctx,
                tools=tools,
                tool_choice=tool_choice,
                conn_options=conn_options,
            ) as stream:
                async for chunk in stream:
                    has_yielded = True
                    yield chunk
            return
        except Exception as exc:
            print(f"[ResilientVoiceAgent] Primary LLM failed: {type(exc).__name__}: {exc}")

        # If primary failed without yielding, seamlessly fallback to secondary LLM
        if not has_yielded and self.fallback_llm:
            try:
                print("[ResilientVoiceAgent] Activating fallback LLM (Gemini)...")
                async with self.fallback_llm.chat(
                    chat_ctx=chat_ctx,
                    tools=tools,
                    tool_choice=tool_choice,
                    conn_options=conn_options,
                ) as stream:
                    async for chunk in stream:
                        has_yielded = True
                        yield chunk
                print("[ResilientVoiceAgent] Fallback LLM succeeded!")
                return
            except Exception as fb_exc:
                print(f"[ResilientVoiceAgent] Fallback LLM failed: {type(fb_exc).__name__}: {fb_exc}")

        if has_yielded:
            yield " ... excuse me, could you please repeat that?"
        else:
            yield "I'm sorry, I had a brief connection delay. Could you please repeat that?"

async def test_truncation_and_fallback():
    print("Testing ResilientVoiceAgent context truncation and fallback...")
    
    # 1. Test chat context truncation
    ctx = llm.ChatContext()
    ctx.add_message(role="system", content="System prompt instructions.")
    for i in range(25):
        ctx.add_message(role="user", content=f"Customer turn {i}")
        ctx.add_message(role="assistant", content=f"Assistant turn {i}")
    
    print(f"Initial context item count: {len(ctx.items)}")
    ctx.truncate(max_items=8)
    print(f"Truncated context item count: {len(ctx.items)}")
    assert len(ctx.items) == 9  # 1 system + 8 items
    assert ctx.items[0].role == "system"
    print("Truncation test passed!")
    
    # 2. Test fallback LLM call with Gemini
    fallback_llm = openai.LLM(
        base_url=settings.GEMINI_BASE_URL,
        api_key=settings.GEMINI_API_KEY,
        model="gemini-2.5-flash",
        temperature=0.6,
    )
    
    chunks = []
    async with fallback_llm.chat(chat_ctx=ctx) as stream:
        async for chunk in stream:
            if hasattr(chunk, "delta") and chunk.delta and chunk.delta.content:
                chunks.append(chunk.delta.content)
    
    result = "".join(chunks)
    print(f"Fallback response: '{result[:50]}...'")
    assert len(result) > 0
    print("Fallback LLM test passed successfully!")

if __name__ == "__main__":
    asyncio.run(test_truncation_and_fallback())
