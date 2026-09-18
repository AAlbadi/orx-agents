import asyncio, json, time, base64
import websockets
import httpx
import audioop

async def run_real_scenario_test():
    print("==================================================")
    print("STARTING REAL-WORLD CALL SCENARIO TEST")
    print("==================================================")

    # 1. Synthesize realistic caller speech using Deepgram TTS
    print("Generating realistic caller speech...")
    from app.config import settings
    async with httpx.AsyncClient() as c:
        r = await c.post(
            "https://api.deepgram.com/v1/speak?model=aura-asteria-en&encoding=linear16&sample_rate=8000&container=none",
            headers={"Authorization": f"Token {settings.DEEPGRAM_API_KEY}", "Content-Type": "application/json"},
            json={"text": "Hi, my air conditioner is blowing warm air. Can you send someone today?"},
            timeout=10.0
        )
    if r.status_code != 200:
        print("Failed to generate caller audio:", r.text)
        return
    caller_pcm = r.content
    print(f"Caller audio ready: {len(caller_pcm)} bytes ({len(caller_pcm)/(8000*2):.2f}s)")

    caller_ulaw = audioop.lin2ulaw(caller_pcm, 2)

    uri = "ws://127.0.0.1:7860/ws"
    print(f"Connecting to {uri} as an inbound caller...")
    async with websockets.connect(uri) as ws:
        call_id = f"test-real-{int(time.time())}"
        stream_id = f"stream-{int(time.time())}"

        await ws.send(json.dumps({
            "event": "start",
            "start": {
                "streamId": stream_id,
                "callId": call_id,
                "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1}
            }
        }))

        await ws.send(json.dumps({
            "event": "media",
            "media": {"payload": base64.b64encode(b"\xff" * 160).decode()}
        }))

        greeting_received = False
        greeting_audio_bytes = 0

        print("Waiting for assistant greeting...")
        t_start = time.time()
        while time.time() - t_start < 6.0:
            try:
                msg_str = await asyncio.wait_for(ws.recv(), timeout=1.5)
                msg = json.loads(msg_str)
                event = msg.get("event")
                if event in ("media", "playAudio"):
                    greeting_received = True
                    payload = base64.b64decode(msg["media"]["payload"])
                    greeting_audio_bytes += len(payload)
                elif event == "transcript":
                    print(f"  Greeting transcript: [{msg.get('speaker')}] {msg.get('text')[:60]}...")
            except asyncio.TimeoutError:
                if greeting_received:
                    break

        print(f"Greeting received: {greeting_audio_bytes} bytes ({greeting_audio_bytes/8000:.2f}s audio)")
        await asyncio.sleep(0.5)

        print("Caller speaking: 'Hi, my air conditioner is blowing warm air. Can you send someone today?'")
        t_speech_start = time.time()
        for i in range(0, len(caller_ulaw), 160):
            chunk = caller_ulaw[i:i+160]
            if len(chunk) < 160:
                chunk += b"\xff" * (160 - len(chunk))
            await ws.send(json.dumps({
                "event": "media",
                "media": {"payload": base64.b64encode(chunk).decode()}
            }))
            await asyncio.sleep(0.02)

        t_speech_end = time.time()
        print(f"Caller finished speaking ({t_speech_end - t_speech_start:.2f}s)")

        print("Sending silence, waiting for assistant reply...")
        first_audio_time = None
        assistant_transcripts = []
        assistant_audio_bytes = 0

        for _ in range(350):
            silence_chunk = b"\xff" * 160
            await ws.send(json.dumps({
                "event": "media",
                "media": {"payload": base64.b64encode(silence_chunk).decode()}
            }))

            try:
                msg_str = await asyncio.wait_for(ws.recv(), timeout=0.03)
                msg = json.loads(msg_str)
                event = msg.get("event")
                if event in ("media", "playAudio"):
                    if first_audio_time is None:
                        first_audio_time = time.time()
                        ttfa = (first_audio_time - t_speech_end) * 1000
                        print(f"FIRST AUDIO RECEIVED! Time-to-First-Audio (TTFA): {ttfa:.0f}ms")
                    payload = base64.b64decode(msg["media"]["payload"])
                    assistant_audio_bytes += len(payload)
                elif event == "transcript":
                    speaker = msg.get("speaker")
                    text = msg.get("text")
                    print(f"  Live transcript: [{speaker}] {text}")
                    assistant_transcripts.append(f"[{speaker}] {text}")
            except asyncio.TimeoutError:
                pass

        print("==================================================")
        print("TEST RESULTS:")
        print(f"  Greeting received: {'YES' if greeting_received else 'NO'}")
        if first_audio_time:
            ttfa = (first_audio_time - t_speech_end) * 1000
            print(f"  Turn Latency (speech-end to audio): {ttfa:.0f}ms")
        else:
            print("  Turn Latency: NO AUDIO")
        print(f"  Total assistant reply audio: {assistant_audio_bytes} bytes ({assistant_audio_bytes/8000:.2f}s)")
        print(f"  Total transcripts: {len(assistant_transcripts)}")
        for t in assistant_transcripts:
            print(f"    {t}")
        has_nudge = any("are you still there" in t.lower() or "checking in" in t.lower() for t in assistant_transcripts)
        print(f"  Unwanted silence nudges: {'DETECTED (FAIL)' if has_nudge else 'NONE (PERFECT)'}")
        print("==================================================")

if __name__ == "__main__":
    asyncio.run(run_real_scenario_test())
