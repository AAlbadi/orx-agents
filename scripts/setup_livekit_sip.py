"""Setup LiveKit SIP Dispatch Rule for Aria Voice Agent."""
import asyncio, os
from dotenv import load_dotenv
load_dotenv()
from livekit import api

LIVEKIT_URL    = os.getenv("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")
TRUNK_ID       = os.getenv("LIVEKIT_SIP_TRUNK_ID", "")

async def main():
    print(f"🔧 Connecting to: {LIVEKIT_URL}")
    print(f"🔀 SIP Trunk ID:  {TRUNK_ID}")
    lk = api.LiveKitAPI(url=LIVEKIT_URL, api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)

    # List existing rules
    try:
        rules = await lk.sip.list_sip_dispatch_rule(api.ListSIPDispatchRuleRequest())
        print(f"\n📋 Existing dispatch rules: {len(rules.items)}")
        for r in rules.items:
            print(f"  - {r.sip_dispatch_rule_id}: {r.name}")
    except Exception as e:
        print(f"⚠️  List error: {e}")

    # Create dispatch rule
    print(f"\n🚀 Creating dispatch rule...")
    try:
        rule = await lk.sip.create_sip_dispatch_rule(
            api.CreateSIPDispatchRuleRequest(
                name="Aria Inbound",
                trunk_ids=[TRUNK_ID],
                rule=api.SIPDispatchRule(
                    dispatch_rule_direct=api.SIPDispatchRuleDirect(
                        room_name="aria-phone",
                        pin="",
                    )
                ),
            )
        )
        print(f"✅ Dispatch rule created!")
        print(f"   ID:   {rule.sip_dispatch_rule_id}")
        print(f"   Name: {rule.name}")
        print(f"   Room: aria-phone")
        print(f"\n🎉 Done! Calling +1 (530) 977-1395 will connect to Aria.")
    except Exception as e:
        print(f"❌ Error: {e}")

    await lk.aclose()

asyncio.run(main())
