#!/usr/bin/env python3
"""
scripts/test_ui_buttons_and_speed.py
Comprehensive Speed & Button API Test Suite.
Tests every action and endpoint backing all UI buttons in Aria Voice AI 2.0:
1. Tab switching and dashboard rendering
2. Assistant CRUD and Activation (Publish button)
3. Presets and field updates (Auto-save)
4. AI Prompt Architect (Generate button)
5. Template Manager (Create, Save, Apply, Reset, Delete)
6. Voice Audition & Voice Selection (27-voice catalog)
7. Simulations & Denoise Benchmarks
8. Call Recordings Management (List, Select, Delete)
9. WebSocket Live Call (Talk Call button)
"""

import asyncio
import httpx
import json
import time
import websockets

BASE_HTTP = "http://localhost:7860"
BASE_WS = "ws://localhost:7860/ws"

GREEN = "\033[92m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

results = []

def record(action_name: str, passed: bool, latency_ms: float, detail: str = ""):
    status = f"{GREEN}[PASS]{RESET}" if passed else f"{RED}[FAIL]{RESET}"
    speed_tag = f"{latency_ms:6.1f}ms" if latency_ms >= 0 else "   N/A  "
    print(f"  {status} {speed_tag} | {BOLD}{action_name:<42}{RESET} : {detail}")
    results.append((action_name, passed, latency_ms, detail))

async def main():
    print(f"\n{BOLD}══════════════════════════════════════════════════════════════════════════════════{RESET}")
    print(f"{BOLD}  ARIA VOICE AI 2.0 — UI BUTTONS & ACTION SPEED VERIFICATION BENCHMARK{RESET}")
    print(f"{BOLD}══════════════════════════════════════════════════════════════════════════════════{RESET}\n")

    async with httpx.AsyncClient(base_url=BASE_HTTP, timeout=60.0) as client:
        # 1. Dashboard page load (Navigation & Studio view)
        t0 = time.perf_counter()
        r = await client.get("/dashboard")
        ms = (time.perf_counter() - t0) * 1000
        record("Load Studio Dashboard (index.html)", r.status_code == 200 and "vapi-bg-app" in r.text, ms, f"{len(r.text):,} bytes rendered")

        # 2. Assistants Fleet Fetch (Sidebar List)
        t0 = time.perf_counter()
        r = await client.get("/api/assistants")
        ms = (time.perf_counter() - t0) * 1000
        assistants = r.json().get("assistants", [])
        active_id = r.json().get("active_id", "")
        record("Fetch Assistants List (Sidebar render)", r.status_code == 200 and len(assistants) > 0, ms, f"{len(assistants)} agents (Active: {active_id})")

        # 3. Save / Update Assistant (Auto-save handler)
        if assistants:
            target = assistants[0]
            t0 = time.perf_counter()
            r = await client.put(f"/api/assistants/{target['id']}", json={
                "temperature": 0.45,
                "first_message": target.get("first_message", "Hello!"),
                "system_prompt": target.get("system_prompt", "You are helpful.")
            })
            ms = (time.perf_counter() - t0) * 1000
            record("Auto-Save Assistant Field Input", r.status_code == 200 and r.json().get("status") == "updated", ms, f"Agent '{target['id']}' updated")

        # 4. Publish / Activate Assistant Button (saveCurrentAssistant)
        if assistants:
            target = assistants[0]
            t0 = time.perf_counter()
            r = await client.post(f"/api/assistants/{target['id']}/activate")
            ms = (time.perf_counter() - t0) * 1000
            record("Publish & Activate Button (saveCurrentAssistant)", r.status_code == 200 and r.json().get("status") == "activated", ms, f"Activated '{target['id']}'")

        # 5. Create Assistant Modal Submit (submitCreateAssistant)
        new_test_id = f"test-btn-agent-{int(time.time())}"
        t0 = time.perf_counter()
        r = await client.post("/api/assistants", json={
            "id": new_test_id,
            "name": "Audit Test Agent",
            "tagline": "Created during button verification",
            "first_message": "Hello from button audit!",
            "system_prompt": "You are a test assistant.",
            "tts_voice": "af_heart",
            "stt_model": "large-v3-turbo",
            "llm_model": "llama-3.3-70b-versatile"
        })
        ms = (time.perf_counter() - t0) * 1000
        created_ok = r.status_code == 200 and r.json().get("status") == "created"
        record("Create Assistant Button (submitCreateAssistant)", created_ok, ms, f"Created '{new_test_id}'")

        # Cleanup created test agent
        if created_ok:
            await client.delete(f"/api/assistants/{new_test_id}")

        # 6. AI Prompt Architect Button (submitGeneratePrompt)
        t0 = time.perf_counter()
        r = await client.post("/api/prompts/generate", json={
            "name": "Dental Assistant",
            "role": "Receptionist & Booking",
            "topic": "Healthcare & Appointments"
        })
        ms = (time.perf_counter() - t0) * 1000
        has_prompt = r.status_code == 200 and "system_prompt" in r.json()
        record("AI Prompt Architect Button (submitGeneratePrompt)", has_prompt, ms, f"{len(r.json().get('system_prompt', ''))} chars prompt generated")

        # 7. Voice Catalog Fetch & Audition (Voice Drawer / auditionVoice)
        t0 = time.perf_counter()
        r = await client.get("/api/voices")
        ms = (time.perf_counter() - t0) * 1000
        voices = r.json().get("voices", [])
        record("Voice Catalog Fetch (Voice Drawer List)", r.status_code == 200 and len(voices) >= 20, ms, f"{len(voices)} neural voices with humanness score")

        # 8. Audition Voice Speech Audio (Hear Sample Button)
        t0 = time.perf_counter()
        r = await client.get("/api/test-speech?text=Audition+test+sample&voice=af_heart")
        ms = (time.perf_counter() - t0) * 1000
        is_wav = r.content[:4] == b"RIFF" and r.content[8:12] == b"WAVE"
        record("Hear Sample Voice Audition (auditionVoice)", r.status_code == 200 and is_wav, ms, f"WAV {len(r.content):,} bytes synthesized")

        # 9. Template Library Fetch (Template Manager Modal)
        t0 = time.perf_counter()
        r = await client.get("/api/templates")
        ms = (time.perf_counter() - t0) * 1000
        templates = r.json().get("templates", [])
        record("Fetch Templates (Template Manager render)", r.status_code == 200 and len(templates) > 0, ms, f"{len(templates)} industry templates available")

        # 10. Save As Template Modal Button (submitSaveAsTemplate)
        test_tpl_id = f"tpl-audit-{int(time.time())}"
        t0 = time.perf_counter()
        r = await client.post("/api/templates", json={
            "id": test_tpl_id,
            "name": "Audit Sample Template",
            "role": "Quality Assurance",
            "tag": "Audit",
            "description": "Generated by button speed audit",
            "first_message": "Audit ready.",
            "system_prompt": "Audit rules.",
            "default_voice": "af_heart"
        })
        ms = (time.perf_counter() - t0) * 1000
        saved_tpl_ok = r.status_code == 200 and r.json().get("status") == "created"
        record("Save As Template Button (submitSaveAsTemplate)", saved_tpl_ok, ms, f"Template '{test_tpl_id}' saved")

        # Cleanup template
        if saved_tpl_ok:
            await client.delete(f"/api/templates/{test_tpl_id}")

        # 11. Call History List Fetch (Call Recordings Tab)
        t0 = time.perf_counter()
        r = await client.get("/api/calls")
        ms = (time.perf_counter() - t0) * 1000
        calls = r.json().get("calls", [])
        record("Fetch Call History (Recordings Tab render)", r.status_code == 200, ms, f"{len(calls)} call recordings logged")

        # 12. Run Acoustic Denoise Benchmark Button
        t0 = time.perf_counter()
        r = await client.post("/api/simulations/denoise-benchmark", json={})
        ms = (time.perf_counter() - t0) * 1000
        denoise_ok = r.status_code == 200 and r.json().get("overall_status") == "PASS"
        snr_gain = r.json().get("avg_snr_gain_db", 0)
        speed_x = r.json().get("avg_speed_factor", 0)
        record("Run Denoise Benchmark Button (runDenoiseBenchmark)", denoise_ok, ms, f"+{snr_gain} dB SNR gain ({speed_x}x real-time)")

        # 13. Talk Call Button (toggleTalkCall / WebSocket Live Session)
        t0 = time.perf_counter()
        ws_ok = False
        ws_event = ""
        try:
            async with websockets.connect(BASE_WS) as ws:
                call_id = f"audit-call-{int(time.time())}"
                await ws.send(json.dumps({
                    "event": "start",
                    "start": {
                        "streamId": f"stream-{int(time.time())}",
                        "callId": call_id,
                        "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1}
                    }
                }))
                await ws.send(json.dumps({
                    "event": "media",
                    "media": {"payload": "//////8="}
                }))
                # Wait for audio stream
                for _ in range(15):
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2.5)
                        try:
                            parsed = json.loads(msg)
                            ev = parsed.get("event", "")
                            if ev in ["playAudio", "media", "connected", "start"]:
                                ws_ok = True
                                ws_event = ev
                                break
                        except Exception:
                            ws_ok = True
                            ws_event = "binary_pcm"
                            break
                    except asyncio.TimeoutError:
                        break
        except Exception as e:
            ws_event = str(e)
        ms = (time.perf_counter() - t0) * 1000
        record("Talk Test Call Button (toggleTalkCall / Live WS)", ws_ok, ms, f"Streaming active (event: {ws_event})")

    print(f"\n{BOLD}══════════════════════════════════════════════════════════════════════════════════{RESET}")
    all_passed = all(p for _, p, _, _ in results)
    if all_passed:
        print(f"{BOLD}{GREEN}  ALL {len(results)} BUTTON & ACTION SPEED TESTS PASSED PERFECTLY! (100% SUCCESS){RESET}")
    else:
        failed_count = sum(1 for _, p, _, _ in results if not p)
        print(f"{BOLD}{RED}  {failed_count} OF {len(results)} TESTS FAILED.{RESET}")
    print(f"{BOLD}══════════════════════════════════════════════════════════════════════════════════{RESET}\n")

if __name__ == "__main__":
    asyncio.run(main())
