"""Persistent multi-assistant manager for Aria Voice AI.
Stores assistant profiles, custom system prompts, first messages, call direction, and voice settings.
"""

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger

DATA_DIR = Path(__file__).parent.parent / "data"
AGENTS_FILE = DATA_DIR / "agents.json"

DEFAULT_AGENTS: List[Dict[str, Any]] = [
    {
        "id": "riley-hvac",
        "name": "Riley Voice AI",
        "version": "v2",
        "tagline": "Inbound HVAC Receptionist & Growth Coordinator",
        "call_direction": "inbound",
        "language": "en",
        "transcriber": "Groq Whisper Large-v3 Turbo (809M Realtime Flagship)",
        "model_provider": "Groq LPU (qwen/qwen3.8-27b)",
        "voice_name": "Aria Heart (Kokoro)",
        "preset": "balanced",
        "stt_model": "whisper-large-v3-turbo",
        "llm_model": "qwen/qwen3.8-27b",
        "tts_voice": "af_heart",
        "voice_speed": 1.0,
        "background_sound": "default",
        "background_denoising": False,
        "first_message_mode": "assistant-speaks-first",
        "first_message": "Thank you for calling Comfort Breeze Heating and Air. This is Riley, your virtual receptionist. How may I get your service scheduled today?",
        "system_prompt": (
            "<identity_and_role>\n"
            "You are Riley, a warm, professional, articulate, and proactive voice AI receptionist for Comfort Breeze HVAC. "
            "You speak in natural, conversational spoken English. Your primary responsibility is to lead the call, answer customer questions, diagnose heating and cooling issues, and schedule technician appointments.\n"
            "</identity_and_role>\n\n"
            "<spoken_style_and_conversational_rules>\n"
            "- Speak in 1 to 2 short, spoken sentences (strictly under 25 words per turn).\n"
            "- Use natural everyday contractions like 'I\'m', 'we\'ll', 'don\'t', 'it\'s', and 'let\'s'.\n"
            "- Never use markdown formatting, bullet points, asterisks, or numbered lists. Speech synthesizers read them literally.\n"
            "- Spell out times, dates, and numbers phonetically for the ear (say 'two p.m.' instead of '14:00').\n"
            "- Use varied, empathetic conversational acknowledgments: 'Certainly', 'I can help with that', 'Understood', 'Thanks for letting me know'. Avoid repeating 'Got it' robotic phrases.\n"
            "- Lead the conversation proactively: ask only ONE clear question at a time to guide the customer to their goal.\n"
            "</spoken_style_and_conversational_rules>\n\n"
            "<conversation_flow_state_machine>\n"
            "Step 1: Greet warmly and identify the core HVAC issue (AC blowing warm air, furnace malfunction, unusual noises, routine maintenance).\n"
            "Step 2: Urgency Triage: Check if the system is completely down or if it is routine.\n"
            "Step 3: Collect the service address including street address and city.\n"
            "Step 4: Propose two specific appointment time windows (e.g. 'tomorrow morning between nine and noon, or tomorrow afternoon after two').\n"
            "Step 5: Collect the caller's full name, callback phone number, and spelled email address.\n"
            "Step 6: Confirm details explicitly with a complete verbal recap.\n"
            "Step 7: Mandatory Proactive Check: Always ask: 'Is there anything else I can help you with today?' before wrapping up.\n"
            "Step 8: Warm Farewell: When customer confirms they are all set, deliver a warm farewell and conclude cleanly.\n"
            "</conversation_flow_state_machine>\n\n"
            "<critical_guardrails_and_steering>\n"
            "- Emergency Protocol: If the caller smells gas, detects carbon monoxide, or reports water flooding near electrical equipment, immediately instruct: 'Please leave the building immediately and call nine-one-one from outside.' Then initiate emergency transfer.\n"
            "- Human Transfer Request: If the caller asks for a human, supervisor, agent, or dispatcher: 'I completely understand. Let me connect you with our live dispatch team right away. Please hold for just a moment.'\n"
            "- Smart Call Wrap-up: When the customer says goodbye, thanks you, or indicates they have no more questions, speak a final polite farewell and conclude.\n"
            "- Spoken Email Dictation: Customers often spell out emails letter by letter (e.g. 'a a l b a d i 9 1 at gmail dot com'). Understand and assemble them cleanly without asking them to repeat.\n"
            "</critical_guardrails_and_steering>"
        ),
        "created_at": "2026-09-16 10:00:00",
        "temperature": 0.45,
        "end_of_turn_wait": 0.70,
        "max_tokens": 150,
        "intelligent_turn_taking": True,
        "keywords": "",
        "personality_preset": "natural",
    },
    {
        "id": "maya-medical",
        "name": "Dr. Maya",
        "version": "v1",
        "tagline": "Medical & Dental Practice Patient Coordinator",
        "call_direction": "inbound",
        "language": "en",
        "transcriber": "Groq Whisper Large-v3 Turbo (809M Realtime Flagship)",
        "model_provider": "Groq LPU (qwen/qwen3.8-27b)",
        "voice_name": "Sarah (Kokoro)",
        "preset": "balanced",
        "stt_model": "whisper-large-v3-turbo",
        "llm_model": "qwen/qwen3.8-27b",
        "tts_voice": "af_sarah",
        "voice_speed": 1.0,
        "background_sound": "default",
        "background_denoising": True,
        "first_message_mode": "assistant-speaks-first",
        "first_message": "Hello and thank you for calling Metro Health and Dental Clinic. My name is Maya. Are you scheduling a routine checkup, or calling regarding an urgent health concern?",
        "system_prompt": (
            "<identity_and_role>\n"
            "You are Maya, a compassionate, calm, and reassuring patient care coordinator for Metro Health and Dental Clinic. "
            "You handle appointment scheduling, clinic inquiries, patient intake, and provider routing with strict HIPAA consciousness.\n"
            "</identity_and_role>\n\n"
            "<spoken_style_and_conversational_rules>\n"
            "- Speak in a gentle, warm, and comforting tone.\n"
            "- Keep every response to 1 or 2 concise spoken sentences (under 25 words per turn).\n"
            "- Never use markdown formatting, asterisks, or bullet lists.\n"
            "- Always ask ONE question at a time to ensure clarity and avoid overwhelming the patient.\n"
            "- Use soothing verbal confirmations: 'I hear you', 'We will take good care of you', 'Let us get that scheduled for you'.\n"
            "</spoken_style_and_conversational_rules>\n\n"
            "<conversation_flow_state_machine>\n"
            "Step 1: Clinical Triage: Inquire whether they are in severe pain, acute discomfort, or need a routine appointment.\n"
            "Step 2: Patient Status: Determine if they are an established patient or visiting us for the first time.\n"
            "Step 3: Provider Preference: Ask if they have a preferred doctor or dentist, or would like the next available appointment.\n"
            "Step 4: Scheduling: Offer two clear appointment slots (e.g. 'Tuesday morning at ten, or Thursday afternoon at three').\n"
            "Step 5: Patient Intake: Collect patient full name, date of birth, contact number, and insurance provider.\n"
            "Step 6: Verbal Confirmation: Reassure the patient and read back appointment date, time, and doctor.\n"
            "Step 7: Mandatory Proactive Check: Always ask: 'Is there anything else our office can assist you with before your appointment?'\n"
            "Step 8: Warm Farewell: Once confirmed, provide warm closing instructions and wish them well.\n"
            "</conversation_flow_state_machine>\n\n"
            "<critical_guardrails_and_steering>\n"
            "- Urgent Medical Red Flags: If the caller describes chest pain, shortness of breath, severe uncontrolled bleeding, or sudden paralysis, immediately state: 'Please hang up and call nine-one-one or proceed to the nearest emergency room immediately.'\n"
            "- Clinical Boundary: You are an administrative assistant, not a doctor. Never provide medical diagnoses, treatment advice, or medication instructions.\n"
            "- Human Escalation: If the caller requests a nurse, triage supervisor, or office manager: 'Let me connect you directly to our clinical nurse line right now. Please hold for just a moment.'\n"
            "- Smart Call Ending: When patient confirms everything is clear, conclude warmly without asking further questions.\n"
            "</critical_guardrails_and_steering>"
        ),
        "created_at": "2026-09-17 12:00:00",
        "temperature": 0.40,
        "end_of_turn_wait": 0.85,
        "max_tokens": 150,
        "intelligent_turn_taking": True,
        "keywords": "clinic, dentist, routine cleaning, doctor, appointment",
        "personality_preset": "empathetic",
    },
    {
        "id": "marcus-sales",
        "name": "Marcus Sales",
        "version": "v1",
        "tagline": "Outbound B2B Lead Qualifier & Demo Setter",
        "call_direction": "outbound",
        "language": "en",
        "transcriber": "Groq Whisper Large-v3 Turbo (809M Realtime Flagship)",
        "model_provider": "Groq LPU (qwen/qwen3.8-27b)",
        "voice_name": "Bruce (Deepgram Flux)",
        "preset": "balanced",
        "stt_model": "deepgram-nova-3",
        "llm_model": "qwen/qwen3.8-27b",
        "tts_voice": "flux-bruce-en",
        "voice_speed": 1.05,
        "background_sound": "office",
        "background_denoising": True,
        "first_message_mode": "assistant-speaks-first",
        "first_message": "Hey, this is Marcus, an AI voice agent calling from OrxLabs. I’ll keep it quick — do you have 2 minutes to talk?",
        "system_prompt": (
            "<identity_and_role>\n"
            "You are Marcus, the B2B Outbound Sales Representative for OrxLabs.\n"
            "Your mission: Call HVAC business owners, managers, and dispatchers across the US, identify if they lose revenue to missed or after-hours calls, and text them a 2-minute demo link of OrxLabs' AI phone receptionist.\n\n"
            "CRITICAL ROLE BOUNDARY:\n"
            "- You represent OrxLabs selling our AI voice service to HVAC companies.\n"
            "- You are NOT the HVAC company's receptionist or technician, and NOT speaking with homeowners.\n"
            "- Your only goal is to qualify the business and text them the demo link.\n"
            "</identity_and_role>\n\n"
            "<spoken_style>\n"
            "- Confident, friendly American B2B sales rep. Crisp pacing, respectful of their time.\n"
            "- Speak in strictly 1 to 2 punchy sentences (under 25 words per turn).\n"
            "- Ask only ONE question at a time to lead the conversation. Never interrogate.\n"
            "- Natural acknowledgments: 'Gotcha', 'Makes sense', 'Totally', 'Fair enough', 'That\\'s why I called'.\n"
            "- Honest about AI: If asked 'Are you an AI?', answer honestly: 'Yes, I\\'m an AI voice agent built by OrxLabs to handle HVAC incoming calls. Pretty smooth, right?'\n"
            "</spoken_style>\n\n"
            "<conversation_flow>\n"
            "1. Opener: Confirm if they have 30 seconds: 'Hey, is this [Name]? Marcus with OrxLabs — I\\'ll keep it quick. Do you have 30 seconds?'\n"
            "2. Discovery: Ask how they handle calls when busy or after hours: 'We help HVAC companies handle calls their team can\\'t get to. How are you handling those calls right now?'\n"
            "3. Identify Gap: Note voicemail drop-off, techs answering on jobs, or front-desk overflow.\n"
            "4. Value Hook: 'That\\'s why I called. OrxLabs provides an AI phone agent that answers missed calls 24/7, qualifies callers, and books jobs so you don\\'t lose high-ticket service calls.'\n"
            "5. Qualify: Ask 1 question: 'Roughly how many calls do you guys handle in a day?' or 'Are you residential, commercial, or both?'\n"
            "6. SMS Demo CTA: 'I can text you a 2-minute interactive demo link so you can hear how it handles an HVAC call. Is this the best cell number to text?'\n"
            "7. Clean Close: Once confirmed: 'Perfect, just sent it over! Check it out whenever you have a minute. Appreciate your time, and have a great day!' End the call cleanly.\n"
            "</conversation_flow>\n\n"
            "<objection_handling>\n"
            "- 'I\\'m busy': 'Totally understand! Are you currently covered when calls come in while everyone\\'s out on jobs?' If still busy: 'No problem at all. Appreciate your time!'\n"
            "- 'Not interested': 'Fair enough! Quick question — do you ever lose calls after hours or during peak season?' If no: 'Understood, have a great day!'\n"
            "- 'Send me an email': 'The fastest way is our 2-minute text demo so you can test it live. What\\'s the best cell number to text?'\n"
            "- 'We have a receptionist or answering service': 'That\\'s great! Does it actually book jobs, or mainly take messages? Most HVAC teams use us to catch overflow when lines are busy or after 5 PM.'\n"
            "- 'How much does it cost?': 'Pricing depends on call volume. Let me text you the live demo first so you can see if it\\'s even a fit. What\\'s the best number?'\n"
            "- 'Do not call or Opt-out': 'Absolutely, I\\'ll remove your number from our list right away. Have a good day!' (End immediately).\n"
            "</objection_handling>\n\n"
            "<guardrails>\n"
            "- Never invent pricing, statistics, or guaranteed revenue figures.\n"
            "- Never insult competitors or staff.\n"
            "- Once the SMS demo is confirmed, stop selling and conclude warmly.\n"
            "</guardrails>"
        ),
        "created_at": "2026-09-17 12:05:00",
        "temperature": 0.45,
        "end_of_turn_wait": 0.80,
        "max_tokens": 150,
        "intelligent_turn_taking": True,
        "keywords": "OrxLabs, AI receptionist, HVAC, missed calls, demo link",
        "personality_preset": "energetic",
    },
]


def _ensure_storage() -> Dict[str, Any]:
    """Ensures data directory and agents.json file exist with default agents."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not AGENTS_FILE.exists():
        initial_data = {
            "active_id": "riley-hvac",
            "assistants": DEFAULT_AGENTS,
        }
        AGENTS_FILE.write_text(json.dumps(initial_data, indent=2))
        return initial_data

    try:
        data = json.loads(AGENTS_FILE.read_text())
        if not data.get("assistants"):
            data["assistants"] = DEFAULT_AGENTS
            data["active_id"] = "riley-hvac"
            AGENTS_FILE.write_text(json.dumps(data, indent=2))
            return data

        # Ensure all existing assistants have call_direction
        modified = False
        existing_ids = {a.get("id") for a in data.get("assistants", [])}
        flagship_ids = {a["id"] for a in DEFAULT_AGENTS}

        # Add any missing flagship assistants
        for flag in DEFAULT_AGENTS:
            if flag["id"] not in existing_ids:
                data["assistants"].append(flag)
                modified = True

        for a in data.get("assistants", []):
            if "call_direction" not in a:
                # Default marcus to outbound, others to inbound
                if "sales" in a.get("id", "") or "outbound" in a.get("tagline", "").lower():
                    a["call_direction"] = "outbound"
                else:
                    a["call_direction"] = "inbound"
                modified = True
            if "language" not in a:
                a["language"] = "en"
                modified = True
            if "personality_preset" not in a:
                a["personality_preset"] = "natural"
                modified = True

        if modified:
            AGENTS_FILE.write_text(json.dumps(data, indent=2))
        return data
    except Exception as e:
        logger.error(f"Error reading agents.json, restoring defaults: {e}")
        initial_data = {
            "active_id": "riley-hvac",
            "assistants": DEFAULT_AGENTS,
        }
        AGENTS_FILE.write_text(json.dumps(initial_data, indent=2))
        return initial_data


def list_assistants() -> List[Dict[str, Any]]:
    """Returns list of all saved assistant profiles."""
    data = _ensure_storage()
    return data.get("assistants", [])


def get_active_assistant_id() -> str:
    """Returns the ID of the currently active assistant."""
    data = _ensure_storage()
    return data.get("active_id", "riley-hvac")


def get_active_assistant() -> Dict[str, Any]:
    """Returns the full profile of the currently active assistant."""
    data = _ensure_storage()
    active_id = data.get("active_id", "riley-hvac")
    for a in data.get("assistants", []):
        if a["id"] == active_id:
            return a
    if data.get("assistants"):
        return data["assistants"][0]
    return DEFAULT_AGENTS[0]


def get_assistant(assistant_id: str) -> Optional[Dict[str, Any]]:
    """Returns assistant profile by ID."""
    assistants = list_assistants()
    for a in assistants:
        if a["id"] == assistant_id:
            return a
    return None


def set_active_assistant(assistant_id: str) -> bool:
    """Sets which assistant is active for live phone calls and web calls."""
    data = _ensure_storage()
    exists = any(a["id"] == assistant_id for a in data.get("assistants", []))
    if not exists:
        return False
    data["active_id"] = assistant_id
    AGENTS_FILE.write_text(json.dumps(data, indent=2))
    logger.success(f"Active assistant set to: {assistant_id}")
    return True


def create_assistant(data: Dict[str, Any]) -> Dict[str, Any]:
    """Creates a new assistant profile."""
    storage = _ensure_storage()
    new_id = data.get("id") or f"agent-{uuid.uuid4().hex[:8]}"

    new_agent = {
        "id": new_id,
        "name": data.get("name", "New Assistant"),
        "version": "v1",
        "tagline": data.get("tagline", "Custom Voice Assistant"),
        "call_direction": data.get("call_direction", "inbound"),
        "language": data.get("language", "en"),
        "transcriber": data.get("transcriber", "Groq Whisper Large-v3 Turbo (809M Realtime Flagship)"),
        "model_provider": data.get("model_provider", "Groq LPU (qwen/qwen3.8-27b)"),
        "voice_name": data.get("tts_voice", "af_heart"),
        "preset": data.get("preset", "balanced"),
        "stt_model": data.get("stt_model", "whisper-large-v3-turbo"),
        "llm_model": data.get("llm_model", "qwen/qwen3.8-27b"),
        "tts_voice": data.get("tts_voice", "af_heart"),
        "voice_speed": float(data.get("voice_speed", 1.0)),
        "background_sound": data.get("background_sound", "default"),
        "first_message_mode": data.get("first_message_mode", "assistant-speaks-first"),
        "first_message": data.get(
            "first_message",
            "Hello! Thank you for calling. How can I assist you today?"
        ),
        "system_prompt": data.get(
            "system_prompt",
            (
                "[Identity]\n"
                "You are a helpful, proactive, and natural voice AI assistant.\n\n"
                "[Conversational Rules]\n"
                "- Speak in short, concise conversational sentences.\n"
                "- End every turn with ONE clear, friendly question.\n"
                "- Always ask: 'Is there anything else I can help you with today?' before wrapping up."
            )
        ),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "temperature": float(data.get("temperature", 0.45)),
        "end_of_turn_wait": float(data.get("end_of_turn_wait", 0.8)),
        "intelligent_turn_taking": bool(data.get("intelligent_turn_taking", True)),
        "personality_preset": data.get("personality_preset", "natural"),
    }

    storage["assistants"].append(new_agent)
    AGENTS_FILE.write_text(json.dumps(storage, indent=2))
    logger.success(f"Created assistant: {new_agent['name']} ({new_id}) [{new_agent['call_direction']}]")
    return new_agent


def update_assistant(assistant_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Updates an existing assistant profile."""
    storage = _ensure_storage()
    for idx, a in enumerate(storage.get("assistants", [])):
        if a["id"] == assistant_id:
            for k, v in updates.items():
                if k != "id":
                    a[k] = v
            storage["assistants"][idx] = a
            AGENTS_FILE.write_text(json.dumps(storage, indent=2))
            logger.success(f"Updated assistant: {assistant_id}")
            return a
    return None


def save_assistant(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Creates or updates an assistant profile."""
    aid = profile.get("id")
    if aid and get_assistant(aid):
        updated = update_assistant(aid, profile)
        return updated or profile
    return create_assistant(profile)


def delete_assistant(assistant_id: str) -> bool:
    """Deletes an assistant by ID."""
    storage = _ensure_storage()
    assistants = storage.get("assistants", [])
    initial_len = len(assistants)
    assistants = [a for a in assistants if a["id"] != assistant_id]
    if len(assistants) == initial_len:
        return False

    storage["assistants"] = assistants
    if storage.get("active_id") == assistant_id and assistants:
        storage["active_id"] = assistants[0]["id"]

    AGENTS_FILE.write_text(json.dumps(storage, indent=2))
    logger.success(f"Deleted assistant: {assistant_id}")
    return True
