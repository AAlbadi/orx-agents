"""Call recording, transcript storage, and voice catalog management for Aria Voice AI."""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import soundfile as sf
from loguru import logger

DATA_DIR = Path(__file__).parent.parent / "data"
CALLS_FILE = DATA_DIR / "calls.json"
RECORDINGS_DIR = DATA_DIR / "recordings"

VOICE_CATALOG: List[Dict[str, Any]] = [
    # Deepgram Flux Conversational AI Voices (Ultra-Realistic Flagship)
    {
        "id": "flux-cliff-en",
        "name": "Cliff (Deepgram Flux) ★",
        "gender": "Male",
        "accent": "American",
        "humanness": 98,
        "mos": 4.68,
        "grade": "Grade A+",
        "style": "Conversational human prosody, micro-inflections, mature baritone",
        "tag": "Flagship Male",
        "provider": "deepgram",
        "sample_phrase": "Hello! Thank you for calling Comfort Breeze Heating and Air. This is Cliff, how may I assist you today?",
    },
    {
        "id": "flux-alexis-en",
        "name": "Alexis (Deepgram Flux)",
        "gender": "Female",
        "accent": "American",
        "humanness": 98,
        "mos": 4.69,
        "grade": "Grade A+",
        "style": "Natural, warm, expressive conversational coordinator with gentle cadence",
        "tag": "Conversational Female",
        "provider": "deepgram",
        "sample_phrase": "Hi there! Thank you for calling Comfort Breeze Heating and Air. How can I assist you with your booking today?",
    },
    {
        "id": "flux-sienna-en",
        "name": "Sienna (Deepgram Flux)",
        "gender": "Female",
        "accent": "American",
        "humanness": 98,
        "mos": 4.70,
        "grade": "Grade A+",
        "style": "Bright, engaging, friendly conversational receptionist",
        "tag": "Engaging & Bright",
        "provider": "deepgram",
        "sample_phrase": "Good morning! Thanks for reaching out to customer support. Let's get your appointment scheduled.",
    },
    {
        "id": "flux-cole-en",
        "name": "Cole (Deepgram Flux)",
        "gender": "Male",
        "accent": "American",
        "humanness": 98,
        "mos": 4.67,
        "grade": "Grade A+",
        "style": "Confident, crisp, professional advisor with relaxed conversational tone",
        "tag": "Confident Advisor",
        "provider": "deepgram",
        "sample_phrase": "Hey there, thanks for calling. I'd be happy to check technician availability for your area right away.",
    },
    {
        "id": "flux-colin-en",
        "name": "Colin (Deepgram Flux)",
        "gender": "Male",
        "accent": "British",
        "humanness": 98,
        "mos": 4.68,
        "grade": "Grade A+",
        "style": "Polished, distinguished British cadence with conversational charm",
        "tag": "British Gentleman",
        "provider": "deepgram",
        "sample_phrase": "Good day! Thank you for calling our concierge line. How may I be of assistance to you this afternoon?",
    },
    {
        "id": "flux-gemma-en",
        "name": "Gemma (Deepgram Flux)",
        "gender": "Female",
        "accent": "British",
        "humanness": 98,
        "mos": 4.69,
        "grade": "Grade A+",
        "style": "Articulate, welcoming British receptionist with natural warmth",
        "tag": "British Receptionist",
        "provider": "deepgram",
        "sample_phrase": "Hello and welcome! Thank you for getting in touch. I would love to help you book that appointment today.",
    },
    {
        "id": "flux-haley-en",
        "name": "Haley (Deepgram Flux)",
        "gender": "Female",
        "accent": "American",
        "humanness": 98,
        "mos": 4.67,
        "grade": "Grade A+",
        "style": "Upbeat, energetic, clear customer experience specialist",
        "tag": "Upbeat & Clear",
        "provider": "deepgram",
        "sample_phrase": "Hi, thanks for reaching out! Let's get your details confirmed so we can dispatch someone out to you.",
    },
    {
        "id": "flux-heather-en",
        "name": "Heather (Deepgram Flux)",
        "gender": "Female",
        "accent": "American",
        "humanness": 98,
        "mos": 4.68,
        "grade": "Grade A+",
        "style": "Calm, reassuring, empathetic patient care and service guide",
        "tag": "Reassuring & Calm",
        "provider": "deepgram",
        "sample_phrase": "Hello, thank you for reaching out today. Take your time, I'm here to walk you through the options.",
    },
    {
        "id": "flux-miles-en",
        "name": "Miles (Deepgram Flux)",
        "gender": "Male",
        "accent": "American",
        "humanness": 98,
        "mos": 4.66,
        "grade": "Grade A+",
        "style": "Modern, dynamic, approachable front-desk specialist",
        "tag": "Dynamic Modern",
        "provider": "deepgram",
        "sample_phrase": "Hey, thanks for calling in! Let me grab your account details so we can get this resolved quickly.",
    },
    {
        "id": "flux-sean-en",
        "name": "Sean (Deepgram Flux)",
        "gender": "Male",
        "accent": "British",
        "humanness": 98,
        "mos": 4.67,
        "grade": "Grade A+",
        "style": "Warm, relatable British cadence, friendly and articulate",
        "tag": "Warm British",
        "provider": "deepgram",
        "sample_phrase": "Good afternoon! Thanks for reaching out to us today. What can I help get sorted for you?",
    },
    {
        "id": "flux-bree-en",
        "name": "Bree (Deepgram Flux)",
        "gender": "Female",
        "accent": "American",
        "humanness": 98,
        "mos": 4.68,
        "grade": "Grade A+",
        "style": "Breezy, personable, vibrant conversational voice",
        "tag": "Vibrant & Warm",
        "provider": "deepgram",
        "sample_phrase": "Hi! Thanks for giving us a call. I can get a certified technician booked for you in just a couple minutes.",
    },
    {
        "id": "flux-brittany-en",
        "name": "Brittany (Deepgram Flux)",
        "gender": "Female",
        "accent": "American",
        "humanness": 98,
        "mos": 4.69,
        "grade": "Grade A+",
        "style": "Polished, helpful corporate receptionist with crisp diction",
        "tag": "Executive Concierge",
        "provider": "deepgram",
        "sample_phrase": "Good morning, thank you for calling. I'd be delighted to help connect you with our scheduling team.",
    },
    {
        "id": "flux-brooke-en",
        "name": "Brooke (Deepgram Flux)",
        "gender": "Female",
        "accent": "American",
        "humanness": 98,
        "mos": 4.67,
        "grade": "Grade A+",
        "style": "Friendly, natural, conversational dispatch and scheduling agent",
        "tag": "Natural Dispatch",
        "provider": "deepgram",
        "sample_phrase": "Hello, thanks for calling Comfort Breeze. What's the main issue you are experiencing with your unit today?",
    },
    {
        "id": "flux-bruce-en",
        "name": "Bruce (Deepgram Flux)",
        "gender": "Male",
        "accent": "American",
        "humanness": 98,
        "mos": 4.66,
        "grade": "Grade A+",
        "style": "Deep, trustworthy, authoritative service director voice",
        "tag": "Trustworthy Baritone",
        "provider": "deepgram",
        "sample_phrase": "Hello, this is Bruce. Thank you for calling. Let's see what service window works best for you.",
    },
    {
        "id": "flux-conor-en",
        "name": "Conor (Deepgram Flux)",
        "gender": "Male",
        "accent": "Irish",
        "humanness": 98,
        "mos": 4.69,
        "grade": "Grade A+",
        "style": "Authentic, charming Irish cadence with conversational flow",
        "tag": "Irish Charm",
        "provider": "deepgram",
        "sample_phrase": "Top of the morning to you! Thanks for ringing in. How can I help get everything sorted for you today?",
    },
    {
        "id": "flux-donovan-en",
        "name": "Donovan (Deepgram Flux)",
        "gender": "Male",
        "accent": "American",
        "humanness": 98,
        "mos": 4.67,
        "grade": "Grade A+",
        "style": "Smooth, resonant, thoughtful customer advisor",
        "tag": "Smooth Advisor",
        "provider": "deepgram",
        "sample_phrase": "Good day, thank you for calling. I'm here to ensure your service inquiry is handled smoothly.",
    },
    {
        "id": "flux-drew-en",
        "name": "Drew (Deepgram Flux)",
        "gender": "Male",
        "accent": "American",
        "humanness": 98,
        "mos": 4.68,
        "grade": "Grade A+",
        "style": "Warm, relatable, conversational specialist with clear pacing",
        "tag": "Relatable & Clear",
        "provider": "deepgram",
        "sample_phrase": "Hey there! Thanks for reaching out. What's the best phone number and address for your appointment?",
    },
    {
        "id": "flux-elise-en",
        "name": "Elise (Deepgram Flux)",
        "gender": "Female",
        "accent": "American",
        "humanness": 98,
        "mos": 4.70,
        "grade": "Grade A+",
        "style": "Sophisticated, gentle, articulate concierge persona",
        "tag": "Sophisticated Concierge",
        "provider": "deepgram",
        "sample_phrase": "Good afternoon, thank you for contacting us. I would be very happy to assist you with your booking.",
    },
    {
        "id": "flux-jack-en",
        "name": "Jack (Deepgram Flux)",
        "gender": "Male",
        "accent": "British",
        "humanness": 98,
        "mos": 4.68,
        "grade": "Grade A+",
        "style": "Direct, articulate British phone agent with professional delivery",
        "tag": "British Professional",
        "provider": "deepgram",
        "sample_phrase": "Hello, thanks for calling through. Let me look up technician availability in your district right away.",
    },
    {
        "id": "flux-kai-en",
        "name": "Kai (Deepgram Flux)",
        "gender": "Male",
        "accent": "Singaporean",
        "humanness": 98,
        "mos": 4.67,
        "grade": "Grade A+",
        "style": "Clear, cosmopolitan Singaporean accent with natural rhythm",
        "tag": "Singaporean International",
        "provider": "deepgram",
        "sample_phrase": "Hello, thank you for reaching our support center. How can I assist you with your inquiry today?",
    },
    {
        "id": "flux-kelsey-en",
        "name": "Kelsey (Deepgram Flux)",
        "gender": "Female",
        "accent": "American",
        "humanness": 98,
        "mos": 4.68,
        "grade": "Grade A+",
        "style": "Engaging, positive, friendly customer service agent",
        "tag": "Friendly Support",
        "provider": "deepgram",
        "sample_phrase": "Hi! Thanks for calling today. Let's get your appointment set up so our team can help you out.",
    },
    {
        "id": "flux-maeve-en",
        "name": "Maeve (Deepgram Flux)",
        "gender": "Female",
        "accent": "Irish",
        "humanness": 98,
        "mos": 4.71,
        "grade": "Grade A+",
        "style": "Enchanting, melodic Irish cadence with genuine warmth",
        "tag": "Irish Melodic",
        "provider": "deepgram",
        "sample_phrase": "Hello there! Thank you for calling us today. I'd be absolutely delighted to help get that arranged for you.",
    },
    {
        "id": "flux-marcelo-en",
        "name": "Marcelo (Deepgram Flux)",
        "gender": "Male",
        "accent": "Filipino",
        "humanness": 98,
        "mos": 4.67,
        "grade": "Grade A+",
        "style": "Courteous, hospitable Filipino accent with warm inflection",
        "tag": "Hospitable & Courteous",
        "provider": "deepgram",
        "sample_phrase": "Mabuhay and welcome! Thank you for calling. I will be very glad to help you schedule your appointment.",
    },
    {
        "id": "flux-marcus-en",
        "name": "Marcus (Deepgram Flux)",
        "gender": "Male",
        "accent": "American",
        "humanness": 98,
        "mos": 4.68,
        "grade": "Grade A+",
        "style": "Commanding, articulate, executive consultation baritone",
        "tag": "Commanding Executive",
        "provider": "deepgram",
        "sample_phrase": "Good morning, thank you for contacting our dispatch department. Let's confirm your service request details.",
    },
    {
        "id": "flux-meena-en",
        "name": "Meena (Deepgram Flux)",
        "gender": "Female",
        "accent": "Indian",
        "humanness": 98,
        "mos": 4.69,
        "grade": "Grade A+",
        "style": "Warm, highly articulate Indian accent with professional clarity",
        "tag": "Indian Professional",
        "provider": "deepgram",
        "sample_phrase": "Hello! Thank you for calling customer service. I am ready to help you with your booking and any questions.",
    },
    {
        "id": "flux-meghan-en",
        "name": "Meghan (Deepgram Flux)",
        "gender": "Female",
        "accent": "American",
        "humanness": 98,
        "mos": 4.68,
        "grade": "Grade A+",
        "style": "Friendly, cheerful, approachable customer care voice",
        "tag": "Cheerful Reception",
        "provider": "deepgram",
        "sample_phrase": "Hi there! Thanks for calling Comfort Breeze. What time of day works best for your service visit?",
    },
    {
        "id": "flux-naveen-en",
        "name": "Naveen (Deepgram Flux)",
        "gender": "Male",
        "accent": "Indian",
        "humanness": 98,
        "mos": 4.68,
        "grade": "Grade A+",
        "style": "Polished, courteous Indian cadence with crisp diction",
        "tag": "Indian Courteous",
        "provider": "deepgram",
        "sample_phrase": "Hello, good day! Thank you for reaching out. How may I be of service to you with your appointment today?",
    },
    {
        "id": "flux-paige-en",
        "name": "Paige (Deepgram Flux)",
        "gender": "Female",
        "accent": "American",
        "humanness": 98,
        "mos": 4.68,
        "grade": "Grade A+",
        "style": "Crisp, modern, pleasant scheduling assistant",
        "tag": "Modern Scheduling",
        "provider": "deepgram",
        "sample_phrase": "Hi! Thank you for calling in. I can check our earliest opening and get you on the schedule right away.",
    },
    {
        "id": "flux-priya-en",
        "name": "Priya (Deepgram Flux)",
        "gender": "Female",
        "accent": "Indian",
        "humanness": 98,
        "mos": 4.70,
        "grade": "Grade A+",
        "style": "Gentle, articulate, hospitable Indian cadence",
        "tag": "Gentle Hospitality",
        "provider": "deepgram",
        "sample_phrase": "Hello and welcome, thank you for calling. I would be happy to walk you through our service options.",
    },
    {
        "id": "flux-rufus-en",
        "name": "Rufus (Deepgram Flux)",
        "gender": "Male",
        "accent": "British",
        "humanness": 98,
        "mos": 4.69,
        "grade": "Grade A+",
        "style": "Distinguished, classic British RP gentleman narrator & concierge",
        "tag": "Distinguished British",
        "provider": "deepgram",
        "sample_phrase": "Good day to you. Rufus speaking. Allow me to coordinate your appointment with our senior specialists.",
    },
    {
        "id": "flux-sharon-en",
        "name": "Sharon (Deepgram Flux)",
        "gender": "Female",
        "accent": "Australian",
        "humanness": 98,
        "mos": 4.68,
        "grade": "Grade A+",
        "style": "Friendly, authentic Australian accent with warm sunny tone",
        "tag": "Aussie Warmth",
        "provider": "deepgram",
        "sample_phrase": "G'day! Thanks for calling Comfort Breeze. How can I help get your air conditioner sorted out today?",
    },
    {
        "id": "flux-tanner-en",
        "name": "Tanner (Deepgram Flux)",
        "gender": "Male",
        "accent": "British",
        "humanness": 98,
        "mos": 4.67,
        "grade": "Grade A+",
        "style": "Brisk, modern British consultant with engaging pacing",
        "tag": "Modern British",
        "provider": "deepgram",
        "sample_phrase": "Hello there, thanks for calling through. Let's get your details down and have a technician dispatched.",
    },
    {
        "id": "flux-wade-en",
        "name": "Wade (Deepgram Flux)",
        "gender": "Male",
        "accent": "American",
        "humanness": 98,
        "mos": 4.67,
        "grade": "Grade A+",
        "style": "Grounding, trustworthy, dependable field technician supervisor",
        "tag": "Dependable Supervisor",
        "provider": "deepgram",
        "sample_phrase": "Hey, thanks for calling our team. Tell me a bit about what's going on with your heating or cooling system.",
    },
    {
        "id": "flux-wes-en",
        "name": "Wes (Deepgram Flux)",
        "gender": "Male",
        "accent": "American",
        "humanness": 98,
        "mos": 4.68,
        "grade": "Grade A+",
        "style": "Casual, relatable, friendly neighborly conversational tone",
        "tag": "Neighborly & Friendly",
        "provider": "deepgram",
        "sample_phrase": "Hey there! Thanks for giving us a call. I'm Wes, and I'd be glad to help get that booked for you.",
    },
    # Deepgram Aura AI Voices

    {
        "id": "aura-asteria-en",
        "name": "Asteria (Deepgram Aura)",
        "gender": "Female",
        "accent": "American",
        "humanness": 96,
        "mos": 4.62,
        "grade": "Grade A+",
        "style": "Warm, articulate conversational receptionist with natural breathing",
        "tag": "Flagship Female",
        "provider": "deepgram",
        "sample_phrase": "Hi there! I would be glad to help get a certified HVAC technician scheduled for you.",
    },
    {
        "id": "aura-luna-en",
        "name": "Luna (Deepgram Aura)",
        "gender": "Female",
        "accent": "American",
        "humanness": 95,
        "mos": 4.59,
        "grade": "Grade A+",
        "style": "Gentle, soothing, patient customer care coordinator",
        "tag": "Patient & Calm",
        "provider": "deepgram",
        "sample_phrase": "Hello, thank you for reaching out. Take your time, I am here to walk you through everything step by step.",
    },
    {
        "id": "aura-stella-en",
        "name": "Stella (Deepgram Aura)",
        "gender": "Female",
        "accent": "American",
        "humanness": 95,
        "mos": 4.58,
        "grade": "Grade A+",
        "style": "Confident, crisp, modern corporate and commercial voice",
        "tag": "Modern Professional",
        "provider": "deepgram",
        "sample_phrase": "Good morning, thanks for reaching CloudScale Solutions. Let us get that demo scheduled for your team.",
    },
    {
        "id": "aura-athena-en",
        "name": "Athena (Deepgram Aura)",
        "gender": "Female",
        "accent": "American",
        "humanness": 94,
        "mos": 4.55,
        "grade": "Grade A",
        "style": "Articulate, authoritative, executive intelligence demeanor",
        "tag": "Executive Female",
        "provider": "deepgram",
        "sample_phrase": "Welcome to our executive concierge line. How may I direct your consultation today?",
    },
    {
        "id": "aura-hera-en",
        "name": "Hera (Deepgram Aura)",
        "gender": "Female",
        "accent": "American",
        "humanness": 93,
        "mos": 4.52,
        "grade": "Grade A",
        "style": "Direct, sharp, authoritative dispatcher and triage agent",
        "tag": "Direct & Sharp",
        "provider": "deepgram",
        "sample_phrase": "This is central dispatch. Please give me your service address and immediate issue description.",
    },
    {
        "id": "aura-orion-en",
        "name": "Orion (Deepgram Aura)",
        "gender": "Male",
        "accent": "American",
        "humanness": 95,
        "mos": 4.58,
        "grade": "Grade A+",
        "style": "Calm, reassuring, natural telephone support cadence",
        "tag": "Support Specialist",
        "provider": "deepgram",
        "sample_phrase": "Good morning, thanks for reaching support. Let us get that taken care of right away.",
    },
    {
        "id": "aura-arcas-en",
        "name": "Arcas (Deepgram Aura)",
        "gender": "Male",
        "accent": "American",
        "humanness": 94,
        "mos": 4.56,
        "grade": "Grade A",
        "style": "Energetic, youthful, friendly tech troubleshooter",
        "tag": "Youthful & Friendly",
        "provider": "deepgram",
        "sample_phrase": "Hey there! Thanks for calling. I can help troubleshoot your setup in just a couple minutes.",
    },
    {
        "id": "aura-perseus-en",
        "name": "Perseus (Deepgram Aura)",
        "gender": "Male",
        "accent": "American",
        "humanness": 94,
        "mos": 4.55,
        "grade": "Grade A",
        "style": "Deep, consultative, trustworthy advisor for sales and appointments",
        "tag": "Consultative Male",
        "provider": "deepgram",
        "sample_phrase": "Hello, thank you for reaching out today. I would be happy to review your options and find the best fit.",
    },
    {
        "id": "aura-angus-en",
        "name": "Angus (Deepgram Aura)",
        "gender": "Male",
        "accent": "Irish",
        "humanness": 94,
        "mos": 4.54,
        "grade": "Grade A",
        "style": "Authentic, charming Irish/British cadence, personable and relatable",
        "tag": "Authentic Irish",
        "provider": "deepgram",
        "sample_phrase": "Good day to you! Thanks for ringing us up. How can I help get things sorted for you today?",
    },
    {
        "id": "aura-orpheus-en",
        "name": "Orpheus (Deepgram Aura)",
        "gender": "Male",
        "accent": "American",
        "humanness": 93,
        "mos": 4.52,
        "grade": "Grade A",
        "style": "Warm, dynamic, expressive conversational narrator",
        "tag": "Dynamic & Warm",
        "provider": "deepgram",
        "sample_phrase": "Welcome aboard! We are excited to connect with you. What can I assist you with this afternoon?",
    },
    {
        "id": "aura-helios-en",
        "name": "Helios (Deepgram Aura)",
        "gender": "Male",
        "accent": "British",
        "humanness": 93,
        "mos": 4.51,
        "grade": "Grade A",
        "style": "Polished, distinguished British gentleman concierge",
        "tag": "British Concierge",
        "provider": "deepgram",
        "sample_phrase": "Good afternoon. It is our absolute pleasure to assist you with your booking today.",
    },
    {
        "id": "aura-zeus-en",
        "name": "Zeus (Deepgram Aura)",
        "gender": "Male",
        "accent": "American",
        "humanness": 94,
        "mos": 4.56,
        "grade": "Grade A",
        "style": "Commanding, resonant executive baritone, steady and confident",
        "tag": "Commanding Baritone",
        "provider": "deepgram",
        "sample_phrase": "This is executive operations. Let us get your service priority dispatched immediately.",
    },
    # Top Tier Kokoro Voices (Local RAM, Free)
    {
        "id": "af_heart",
        "name": "Aria Heart ★",
        "gender": "Female",
        "accent": "American",
        "humanness": 93,
        "mos": 4.48,
        "grade": "Grade A",
        "style": "Warm, lifelike cadence, emotive inflections (Flagship Default)",
        "tag": "Gold Standard",
        "sample_phrase": "Hi there! I'm Aria, your virtual voice assistant. I can schedule appointments, answer questions, or connect you with our team.",
    },
    {
        "id": "af_bella",
        "name": "Bella",
        "gender": "Female",
        "accent": "American",
        "humanness": 91,
        "mos": 4.41,
        "grade": "Grade A-",
        "style": "Confident, bright, upbeat front-desk & customer reception",
        "tag": "Top Receptionist",
        "sample_phrase": "Hello! Thanks for calling Comfort Breeze Heating and Air. How can I get your service appointment booked today?",
    },
    {
        "id": "bf_emma",
        "name": "Emma",
        "gender": "Female",
        "accent": "British",
        "humanness": 91,
        "mos": 4.40,
        "grade": "Grade A-",
        "style": "Polite, elegant, natural British RP accent. Highly articulated",
        "tag": "British Flagship",
        "sample_phrase": "Good afternoon. Thank you for calling our executive concierge. How may I be of assistance to you today?",
    },
    {
        "id": "am_adam",
        "name": "Adam",
        "gender": "Male",
        "accent": "American",
        "humanness": 91,
        "mos": 4.39,
        "grade": "Grade A-",
        "style": "Deep, reassuring baritone, consultative, trustworthy tone",
        "tag": "Top Male Voice",
        "sample_phrase": "Hello, thanks for reaching out. My name is Adam. What technical or billing issue can I help troubleshoot for you?",
    },
    {
        "id": "af_nicole",
        "name": "Nicole",
        "gender": "Female",
        "accent": "American",
        "humanness": 90,
        "mos": 4.38,
        "grade": "Grade A-",
        "style": "Crisp, patient, articulate phone support and dispatch triage",
        "tag": "Customer Support",
        "sample_phrase": "Thanks for calling support. Let me pull up your account details and check on that dispatch status for you.",
    },
    {
        "id": "af_aoede",
        "name": "Aoede",
        "gender": "Female",
        "accent": "American",
        "humanness": 90,
        "mos": 4.37,
        "grade": "Grade A-",
        "style": "Expressive, melodic, engaging storyteller and sales guide",
        "tag": "Expressive",
        "sample_phrase": "Welcome! We're excited to have you here today. I can walk you through our best plans and custom packages.",
    },

    # High Tier: Grade B+ (MOS 4.28 - 4.34 | 87 - 88% Humanness)
    {
        "id": "am_echo",
        "name": "Echo",
        "gender": "Male",
        "accent": "American",
        "humanness": 88,
        "mos": 4.34,
        "grade": "Grade B+",
        "style": "Clear, resonant, energetic customer engagement",
        "tag": "Dynamic",
        "sample_phrase": "Hi there, you're connected to the express dispatch line. Let's get a technician out to your location right away.",
    },
    {
        "id": "am_eric",
        "name": "Eric",
        "gender": "Male",
        "accent": "American",
        "humanness": 88,
        "mos": 4.33,
        "grade": "Grade B+",
        "style": "Steady, authoritative corporate representative",
        "tag": "Professional",
        "sample_phrase": "Good morning, this is Eric from corporate operations. I'd be happy to guide you through our onboarding process.",
    },
    {
        "id": "bf_isabella",
        "name": "Isabella",
        "gender": "Female",
        "accent": "British",
        "humanness": 88,
        "mos": 4.33,
        "grade": "Grade B+",
        "style": "Refined, warm UK accent for hospitality and luxury sales",
        "tag": "Sophisticated",
        "sample_phrase": "Good day. It is wonderful speaking with you. May I confirm your reservation details for this evening?",
    },
    {
        "id": "bm_george",
        "name": "George",
        "gender": "Male",
        "accent": "British",
        "humanness": 88,
        "mos": 4.32,
        "grade": "Grade B+",
        "style": "Classic, dignified British gentleman, distinguished tone",
        "tag": "British Classic",
        "sample_phrase": "Good afternoon. George here. I will ensure your inquiry is handled with the utmost care and precision.",
    },
    {
        "id": "af_sky",
        "name": "Sky",
        "gender": "Female",
        "accent": "American",
        "humanness": 88,
        "mos": 4.32,
        "grade": "Grade B+",
        "style": "Youthful, energetic, friendly lifestyle and wellness vibe",
        "tag": "Energetic",
        "sample_phrase": "Hey! Thanks for getting in touch. I'm ready to help you discover the perfect option for your weekend plans.",
    },
    {
        "id": "af_sarah",
        "name": "Sarah",
        "gender": "Female",
        "accent": "American",
        "humanness": 88,
        "mos": 4.31,
        "grade": "Grade B+",
        "style": "Gentle, soothing, empathetic patient coordinator tone",
        "tag": "Empathetic",
        "sample_phrase": "Hello. I understand how stressful this can be. Let me take down your information and arrange immediate help.",
    },
    {
        "id": "am_fenrir",
        "name": "Fenrir",
        "gender": "Male",
        "accent": "American",
        "humanness": 87,
        "mos": 4.30,
        "grade": "Grade B+",
        "style": "Authoritative, commanding, deep radio-host presence",
        "tag": "Commanding",
        "sample_phrase": "Welcome to emergency operations. State the nature of your request and I will deploy the necessary personnel.",
    },
    {
        "id": "am_michael",
        "name": "Michael",
        "gender": "Male",
        "accent": "American",
        "humanness": 87,
        "mos": 4.30,
        "grade": "Grade B+",
        "style": "Casual, relatable, neighborly conversationalist",
        "tag": "Conversational",
        "sample_phrase": "Hey there! Thanks for giving us a ring. What can I do to help you out around the house today?",
    },
    {
        "id": "af_jessica",
        "name": "Jessica",
        "gender": "Female",
        "accent": "American",
        "humanness": 87,
        "mos": 4.29,
        "grade": "Grade B+",
        "style": "Natural, modern conversational tone for tech companies",
        "tag": "Modern",
        "sample_phrase": "Hi, I'm Jessica. I can answer any questions regarding your billing cycle or upgrade options.",
    },
    {
        "id": "af_nova",
        "name": "Nova",
        "gender": "Female",
        "accent": "American",
        "humanness": 87,
        "mos": 4.29,
        "grade": "Grade B+",
        "style": "Articulate, crisp, executive concierge demeanor",
        "tag": "Corporate",
        "sample_phrase": "Good morning. I am Nova, your executive concierge. Allow me to route your call to the proper department.",
    },
    {
        "id": "bm_fable",
        "name": "Fable",
        "gender": "Male",
        "accent": "British",
        "humanness": 87,
        "mos": 4.28,
        "grade": "Grade B+",
        "style": "Warm, expressive British narrator and docent",
        "tag": "Storyteller",
        "sample_phrase": "Welcome. Let me guide you through the history and remarkable architecture of this historic estate.",
    },
    {
        "id": "bf_alice",
        "name": "Alice",
        "gender": "Female",
        "accent": "British",
        "humanness": 87,
        "mos": 4.28,
        "grade": "Grade B+",
        "style": "Bright, friendly British customer relations",
        "tag": "Bright UK",
        "sample_phrase": "Hello there! Thank you for contacting our support desk. I'd love to help you sort this out today.",
    },

    # Solid Tier: Grade B (MOS 4.20 - 4.26 | 85 - 86% Humanness)
    {
        "id": "af_kore",
        "name": "Kore",
        "gender": "Female",
        "accent": "American",
        "humanness": 86,
        "mos": 4.26,
        "grade": "Grade B",
        "style": "Soft-spoken, peaceful, mindfulness and wellness guide",
        "tag": "Gentle",
        "sample_phrase": "Welcome. Take a comfortable breath, and let's work through your wellness goals together.",
    },
    {
        "id": "am_liam",
        "name": "Liam",
        "gender": "Male",
        "accent": "American",
        "humanness": 86,
        "mos": 4.25,
        "grade": "Grade B",
        "style": "Youthful, modern, friendly tech support specialist",
        "tag": "Youthful",
        "sample_phrase": "Hey! Liam here. Let's see what we can do to get your network settings configured properly.",
    },
    {
        "id": "bf_lily",
        "name": "Lily",
        "gender": "Female",
        "accent": "British",
        "humanness": 86,
        "mos": 4.25,
        "grade": "Grade B",
        "style": "Subtle, quiet, soothing British tone",
        "tag": "Gentle UK",
        "sample_phrase": "Good day. Thank you for reaching out to us. How might I assist you with your booking today?",
    },
    {
        "id": "bm_daniel",
        "name": "Daniel",
        "gender": "Male",
        "accent": "British",
        "humanness": 86,
        "mos": 4.24,
        "grade": "Grade B",
        "style": "Clear, concise British logistics and dispatch",
        "tag": "Direct UK",
        "sample_phrase": "Good morning. Daniel speaking. I can confirm your delivery route and scheduled arrival window.",
    },
    {
        "id": "af_river",
        "name": "River",
        "gender": "Female",
        "accent": "American",
        "humanness": 85,
        "mos": 4.22,
        "grade": "Grade B",
        "style": "Easygoing, relaxed, contemporary casual tone",
        "tag": "Relaxed",
        "sample_phrase": "Hey there! Glad you stopped by. What can I look up or answer for you right now?",
    },
    {
        "id": "af_alloy",
        "name": "Alloy",
        "gender": "Female",
        "accent": "American",
        "humanness": 85,
        "mos": 4.21,
        "grade": "Grade B",
        "style": "Neutral, precise, direct instructional voice",
        "tag": "Direct",
        "sample_phrase": "System check initialized. All audio and speech recognition pipelines are operating within standard parameters.",
    },
    {
        "id": "am_onyx",
        "name": "Onyx",
        "gender": "Male",
        "accent": "American",
        "humanness": 85,
        "mos": 4.20,
        "grade": "Grade B",
        "style": "Gravelly, deep baritone security and dispatch",
        "tag": "Baritone",
        "sample_phrase": "Central dispatch online. Verified entry permitted. Proceed to your destination.",
    },
    {
        "id": "am_puck",
        "name": "Puck",
        "gender": "Male",
        "accent": "American",
        "humanness": 85,
        "mos": 4.20,
        "grade": "Grade B",
        "style": "Lively, spirited, engaging interactive character",
        "tag": "Playful",
        "sample_phrase": "Well hello there! Let's get right into the fun stuff and see what we can build today!",
    },
    {
        "id": "bm_lewis",
        "name": "Lewis",
        "gender": "Male",
        "accent": "British",
        "humanness": 85,
        "mos": 4.20,
        "grade": "Grade B",
        "style": "Measured, patient British customer service",
        "tag": "Calm UK",
        "sample_phrase": "Good day. Thank you for calling our offices. Please let me know how I can be of assistance.",
    },
]


def _ensure_storage() -> Dict[str, Any]:
    """Ensures recordings directory and calls.json file exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    if not CALLS_FILE.exists():
        initial_data = {"calls": []}
        CALLS_FILE.write_text(json.dumps(initial_data, indent=2))
        return initial_data

    try:
        return json.loads(CALLS_FILE.read_text())
    except Exception as e:
        logger.error(f"Error reading calls.json: {e}")
        initial_data = {"calls": []}
        CALLS_FILE.write_text(json.dumps(initial_data, indent=2))
        return initial_data


def list_calls() -> List[Dict[str, Any]]:
    """Returns list of past calls sorted by most recent first."""
    data = _ensure_storage()
    calls = data.get("calls", [])
    calls.sort(key=lambda c: c.get("started_at", ""), reverse=True)
    return calls


def get_call(call_id: str) -> Optional[Dict[str, Any]]:
    """Returns full call details including complete turn-by-turn transcript."""
    calls = list_calls()
    for c in calls:
        if c.get("call_id") == call_id:
            return c
    return None


def save_call_session(
    call_id: str,
    assistant_id: str,
    assistant_name: str,
    caller: Optional[str],
    called: Optional[str],
    started_at: Optional[str] = None,
    duration_seconds: float = 0.0,
    transcript: Optional[List[Dict[str, Any]]] = None,
    audio_pcm_bytes: Optional[bytes] = None,
    status: str = "completed",
    sample_rate: int = 16000,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Saves a completed call session:
    - Writes the 16-bit linear PCM audio to a standard .wav file (at 16kHz)
    - Stores turn-by-turn transcript with timestamps and latency
    """
    storage = _ensure_storage()
    wav_filename = f"{call_id}.wav"
    wav_path = RECORDINGS_DIR / wav_filename
    recording_url = f"/api/recordings/{wav_filename}"

    if audio_pcm_bytes and len(audio_pcm_bytes) > 0:
        try:
            samples = np.frombuffer(audio_pcm_bytes, dtype=np.int16).astype(np.float32)
            # Studio Broadcast Peak Normalization (-1.0 dBFS target)
            # Ensures call recordings have loud, clear, broadcast-grade audio for both caller and assistant
            max_val = float(np.max(np.abs(samples))) if len(samples) > 0 else 0.0
            if max_val > 80.0:
                target_peak = 32767.0 * 0.89  # -1.0 dBFS headroom
                norm_gain = min(4.0, max(1.0, target_peak / max_val))
                samples = np.clip(samples * norm_gain, -32768, 32767)

            samples_int16 = samples.astype(np.int16)
            sf.write(str(wav_path), samples_int16, sample_rate, format="WAV", subtype="PCM_16")
            logger.success(f"Saved call recording to {wav_path} ({len(samples_int16)/float(sample_rate):.2f}s audio at {sample_rate}Hz)")
        except Exception as e:
            logger.error(f"Error writing call recording WAV file: {e}")
            recording_url = ""
    else:
        recording_url = ""

    call_entry = {
        "call_id": call_id,
        "assistant_id": assistant_id,
        "assistant_name": assistant_name,
        "caller": caller or "Web Browser User",
        "called": called or "Aria Voice Line",
        "started_at": started_at or time.strftime("%Y-%m-%d %H:%M:%S"),
        "ended_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": round(duration_seconds, 1),
        "recording_url": recording_url,
        "transcript": transcript or [],
        "turn_count": len(transcript or []),
        "status": status,
    }

    if extra_metadata:
        for k, v in extra_metadata.items():
            if v is not None:
                call_entry[k] = v

    # Remove existing entry if updating
    calls = [c for c in storage.get("calls", []) if c.get("call_id") != call_id]
    calls.append(call_entry)
    storage["calls"] = calls
    CALLS_FILE.write_text(json.dumps(storage, indent=2))
    logger.success(f"Logged call session '{call_id}' with {len(transcript)} turns.")

    # Automated Post-Call Intelligence Extraction & Integrations
    try:
        from app.integrations import (
            extract_client_info,
            create_google_calendar_url,
            send_post_call_email,
            sync_to_calendar_webhook,
            get_integrations_settings,
        )

        cfg = get_integrations_settings()
        if cfg.get("auto_extract_client_info", True) and len(transcript) >= 2:
            import asyncio

            async def _run_post_call_pipeline():
                try:
                    from app.agents import get_assistant
                    asst = get_assistant(assistant_id) if assistant_id else None
                    asst_lang = asst.get("language", "en") if asst else "en"
                    info = await extract_client_info(transcript, caller=caller, called=called, language=asst_lang)
                    call_entry["extracted_info"] = info
                    cal_url = create_google_calendar_url(info, assistant_name=assistant_name)
                    call_entry["google_calendar_url"] = cal_url
                    call_entry["has_appointment"] = info.get("has_appointment", False)

                    # Update saved call entry in storage
                    s = _ensure_storage()
                    for idx, c in enumerate(s.get("calls", [])):
                        if c.get("call_id") == call_id:
                            s["calls"][idx] = call_entry
                            CALLS_FILE.write_text(json.dumps(s, indent=2))
                            break

                    # Automated post-call actions: Calendar webhook, Email notification, & HVAC 1-Tap SMS Dispatch
                    if info.get("has_appointment"):
                        await sync_to_calendar_webhook(info, call_id)
                        try:
                            from app.appointments import dispatch_hvac_booking
                            await dispatch_hvac_booking(info, call_entry)
                        except Exception as sms_ex:
                            logger.error(f"Error dispatching HVAC SMS: {sms_ex}")

                    if cfg.get("email_notifications_enabled") or cfg.get("notify_email"):
                        await send_post_call_email(info, call_entry)

                    logger.success(f"Post-call extraction & integrations completed for {call_id}")
                except Exception as ex:
                    logger.error(f"Error in post-call pipeline for {call_id}: {ex}")

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_run_post_call_pipeline())
            except RuntimeError:
                asyncio.run(_run_post_call_pipeline())
    except Exception as e:
        logger.error(f"Error initiating post-call integrations for {call_id}: {e}")

    return call_entry


def delete_call(call_id: str) -> bool:
    """Deletes a call record and associated audio recording file."""
    storage = _ensure_storage()
    calls = storage.get("calls", [])
    initial_len = len(calls)
    calls = [c for c in calls if c.get("call_id") != call_id]

    if len(calls) == initial_len:
        return False

    storage["calls"] = calls
    CALLS_FILE.write_text(json.dumps(storage, indent=2))

    wav_file = RECORDINGS_DIR / f"{call_id}.wav"
    if wav_file.exists():
        try:
            wav_file.unlink()
            logger.info(f"Deleted recording file {wav_file}")
        except Exception as e:
            logger.warning(f"Could not delete recording file: {e}")

    return True


def get_voice_catalog() -> List[Dict[str, Any]]:
    """Returns the full catalog of available Deepgram and Kokoro voices with metadata."""
    catalog = []
    for v in VOICE_CATALOG:
        item = dict(v)
        if "provider" not in item:
            item["provider"] = "deepgram" if (item["id"].startswith("flux-") or item["id"].startswith("aura-")) else "kokoro"
        catalog.append(item)
    return catalog


async def reextract_call_info(call_id: str) -> Optional[Dict[str, Any]]:
    """Re-runs client info extraction and calendar sync for an existing call."""
    call = get_call(call_id)
    if not call:
        return None

    from app.integrations import extract_client_info, create_google_calendar_url

    transcript = call.get("transcript", [])
    from app.agents import get_assistant
    asst_id = call.get("assistant_id")
    asst = get_assistant(asst_id) if asst_id else None
    asst_lang = asst.get("language", "en") if asst else "en"
    info = await extract_client_info(transcript, caller=call.get("caller"), called=call.get("called"), language=asst_lang)
    cal_url = create_google_calendar_url(info, assistant_name=call.get("assistant_name", "Riley"))

    call["extracted_info"] = info
    call["google_calendar_url"] = cal_url
    call["has_appointment"] = info.get("has_appointment", False)

    storage = _ensure_storage()
    for idx, c in enumerate(storage.get("calls", [])):
        if c.get("call_id") == call_id:
            storage["calls"][idx] = call
            CALLS_FILE.write_text(json.dumps(storage, indent=2))
            break
    return call
