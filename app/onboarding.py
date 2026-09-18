"""Onboarding, Industry Intelligence & Client Management Engine for ORX Agents.
Handles business discovery, smart questionnaires across any industry, prompt compilation,
client profile persistence, and tailored demo simulations.
"""

import json
import os
import time
import uuid
import re
import urllib.request
import urllib.parse
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger

from app.config import settings

DATA_DIR = Path(__file__).parent.parent / "data"
CLIENTS_FILE = DATA_DIR / "clients.json"

# ---------------------------------------------------------------------------
# Industry Knowledge Base & Smart Dynamic Questionnaires
# ---------------------------------------------------------------------------

INDUSTRIES: Dict[str, Dict[str, Any]] = {
    "hvac": {
        "id": "hvac",
        "name": "HVAC & Climate Control",
        "icon": "wind",
        "tagline": "Heating, AC Repair, Heat Pumps & Ductwork",
        "default_voice": "af_heart",
        "default_greeting": "Thank you for calling {business_name}. This is your virtual receptionist. How may I help get your heating or AC taken care of today?",
        "service_options": [
            "AC Repair & Installation",
            "Heating & Furnace Repair",
            "Seasonal System Tune-Ups",
            "Heat Pump Service",
            "Emergency Diagnostics",
            "Duct Cleaning & Air Quality",
            "Commercial HVAC Service"
        ],
        "emergency_triggers": [
            "Gas smell or carbon monoxide alert",
            "Active water leak from indoor AC unit",
            "No heat in freezing temperatures",
            "Caller specifically asks for owner",
            "Commercial account emergency"
        ],
        "pricing_preset": "We have an $89 diagnostic fee that is applied directly toward repair costs if approved.",
        "suggested_questions": [
            {
                "id": "hours",
                "label": "Operating & Emergency Hours",
                "placeholder": "e.g. Mon-Fri 7:30 AM to 6:00 PM, 24/7 emergency dispatch",
                "default": "Monday to Friday 8:00 AM to 6:00 PM. 24/7 on-call dispatch for emergencies.",
            },
            {
                "id": "transfer_rules",
                "label": "When should the agent transfer to your personal cell?",
                "placeholder": "e.g. Gas smell, total system failure in freezing weather, or caller asks for the owner",
                "default": "Immediate transfer for gas smell reports, water leaks from AC units, or commercial accounts.",
            },
            {
                "id": "diagnostic_fee",
                "label": "Diagnostic or Service Call Fee",
                "placeholder": "e.g. $89 diagnostic fee credited toward any repair",
                "default": "We have an $89 diagnostic fee that is applied directly toward repair costs if approved.",
            },
            {
                "id": "services",
                "label": "Core Services Offered",
                "placeholder": "e.g. AC repair, furnace maintenance, heat pumps, duct cleaning",
                "default": "AC repair, seasonal tune-ups, furnace replacement, heat pump installs, and duct cleaning.",
            },
        ],
    },
    "plumbing": {
        "id": "plumbing",
        "name": "Plumbing & Drain Services",
        "icon": "droplet",
        "tagline": "Leaks, Clogs, Water Heaters & Sewer Repair",
        "default_voice": "am_adam",
        "default_greeting": "Hello! Thanks for calling {business_name}. This is your virtual assistant. Do you have an active leak, or are you scheduling routine service?",
        "service_options": [
            "Emergency Drain Clearing",
            "Water Heater Repair & Replacement",
            "Burst Pipe & Leak Detection",
            "Toilet, Faucet & Sink Repair",
            "Sewer Line Camera Inspection & Jetting",
            "Garbage Disposal Installation",
            "Whole-Home Repiping",
            "Water Filtration & Softeners",
            "Commercial Plumbing Service"
        ],
        "emergency_triggers": [
            "Active water flooding or burst pipe",
            "Raw sewage backing up into building",
            "Water heater leaking or smoking",
            "Total loss of water supply to property",
            "Caller specifically asks for owner"
        ],
        "pricing_preset": "We have a $79 service dispatch fee that is applied directly toward repair costs if approved.",
        "suggested_questions": [
            {
                "id": "hours",
                "label": "Operating & Service Hours",
                "placeholder": "e.g. Mon-Sat 7:00 AM - 7:00 PM, 24/7 emergency line",
                "default": "Monday to Saturday 7:00 AM to 7:00 PM, with 24/7 emergency service available.",
            },
            {
                "id": "transfer_rules",
                "label": "When should the agent transfer immediately?",
                "placeholder": "e.g. Burst pipe, overflowing toilet, or water shutoff questions",
                "default": "Transfer immediately if caller reports a burst pipe, active ceiling water leak, or sewer backup.",
            },
            {
                "id": "emergency_guidance",
                "label": "First-aid advice while technician is dispatched",
                "placeholder": "e.g. Tell caller where to find the main water shut-off valve",
                "default": "Always instruct callers with active flooding to shut off their main water valve while we dispatch help.",
            },
            {
                "id": "services",
                "label": "Core Services Offered",
                "placeholder": "e.g. Drain snaking, water heater install, repiping, leak detection",
                "default": "Emergency drain clearing, water heater replacement, fixture installs, and repiping.",
            },
        ],
    },
    "electrical": {
        "id": "electrical",
        "name": "Electrical Services",
        "icon": "zap",
        "tagline": "Panel Upgrades, Rewiring, Lighting & EV Chargers",
        "default_voice": "am_adam",
        "default_greeting": "Thanks for calling {business_name}. My name is your virtual electrician assistant. How can we help power your home or business today?",
        "service_options": [
            "200-Amp Electrical Panel Upgrades",
            "EV Home Charger Installation",
            "Recessed Lighting & Ceiling Fans",
            "Outlet, GFCI & Switch Repair",
            "Whole-Home Rewiring & Safety Inspections",
            "Emergency Power Outage Diagnostics",
            "Standby Generator Installation",
            "Commercial Electrical Services"
        ],
        "emergency_triggers": [
            "Sparks, smoke, or electrical burning smell",
            "Total power loss isolated to property",
            "Breaker panel humming, hot to touch, or buzzing",
            "Downed overhead electrical wire",
            "Caller asks for licensed master electrician"
        ],
        "pricing_preset": "We have an $89 diagnostic inspection fee that is applied directly toward repair costs if approved.",
        "suggested_questions": [
            {
                "id": "hours",
                "label": "Electrician Service Hours",
                "placeholder": "e.g. Mon-Fri 7:00 AM - 6:00 PM, 24/7 emergency calls",
                "default": "Monday to Friday 7:00 AM to 6:00 PM, 24/7 on-call emergency electrical dispatch.",
            },
            {
                "id": "transfer_rules",
                "label": "When to transfer immediately?",
                "placeholder": "e.g. Smoke, sparks, burning breaker panel",
                "default": "Transfer immediately for reports of burning electrical odors, sparks, or buzzing breaker panels.",
            },
            {
                "id": "services",
                "label": "Core Electrical Services",
                "placeholder": "e.g. Panel upgrades, EV chargers, lighting",
                "default": "Panel upgrades, EV charger installations, rewiring, lighting, and emergency power restoration.",
            },
        ],
    },
    "roofing": {
        "id": "roofing",
        "name": "Roofing & Gutters",
        "icon": "home",
        "tagline": "Roof Repair, Replacement, Leak Stoppage & Storm Damage",
        "default_voice": "am_michael",
        "default_greeting": "Thank you for calling {business_name}. This is your virtual assistant. Do you have an active roof leak, or are you looking for an estimate?",
        "service_options": [
            "Emergency Roof Leak Repair",
            "Storm, Wind & Hail Damage Inspection",
            "Asphalt Shingle Roof Replacement",
            "Seamless Gutter Installation & Guards",
            "Metal & Tile Roof Installation",
            "Commercial Flat Roof & Silicone Coating",
            "Chimney Flashing & Skylight Repair",
            "Insurance Claim Restoration Assistance"
        ],
        "emergency_triggers": [
            "Water actively pouring through ceiling or sheetrock",
            "Tree limb punctured through roof deck",
            "Tarps required immediately due to active storm",
            "Insurance claims adjuster currently on site"
        ],
        "pricing_preset": "We provide a complimentary 21-point roof inspection and insurance claim estimate with zero obligation.",
        "suggested_questions": [
            {
                "id": "hours",
                "label": "Roofing Service & Inspection Hours",
                "placeholder": "e.g. Mon-Sat 7:00 AM - 7:00 PM, storm response 24/7",
                "default": "Monday to Saturday 7:00 AM to 7:00 PM. 24/7 rapid response for active storm damage.",
            },
            {
                "id": "transfer_rules",
                "label": "When to transfer immediately?",
                "placeholder": "e.g. Water pouring through roof, adjuster on site",
                "default": "Transfer immediately for active interior water intrusion or insurance adjusters on site.",
            },
            {
                "id": "services",
                "label": "Core Roofing Services",
                "placeholder": "e.g. Roof replacement, leak repair, gutters",
                "default": "Emergency leak repairs, full roof replacements, seamless gutters, and storm damage insurance inspections.",
            },
        ],
    },
    "dental_medical": {
        "id": "dental_medical",
        "name": "Dental Practice & Medical Clinics",
        "icon": "activity",
        "tagline": "Patient Intake, Cleaning Appointments & Triage",
        "default_voice": "af_sarah",
        "default_greeting": "Thank you for calling {business_name}. My name is Sarah. Are you scheduling a routine visit, or calling regarding urgent care?",
        "service_options": [
            "Routine Hygiene Cleaning & Exam",
            "Emergency Toothache & Dental Exam",
            "Dental Crowns, Bridges & Tooth Fillings",
            "Root Canal Therapy",
            "Invisalign & Clear Aligners",
            "Cosmetic Teeth Whitening",
            "Dental Implants & Restorations",
            "Pediatric & Family Dentistry"
        ],
        "emergency_triggers": [
            "Severe facial swelling or acute traumatic pain",
            "Knocked-out permanent adult tooth",
            "Post-surgical heavy bleeding beyond 2 hours",
            "Referring physician or hospital ER calling",
            "Severe medication allergic reaction"
        ],
        "pricing_preset": "We provide complimentary new patient consultations and bill in-network PPO insurance plans directly.",
        "suggested_questions": [
            {
                "id": "hours",
                "label": "Clinic Hours",
                "placeholder": "e.g. Mon-Thu 8:00 AM - 5:00 PM, Fri 8:00 AM - 1:00 PM",
                "default": "Monday through Thursday 8:00 AM to 5:00 PM, Friday 8:00 AM to 1:00 PM. Closed weekends.",
            },
            {
                "id": "transfer_rules",
                "label": "When to transfer to on-call nurse or doctor?",
                "placeholder": "e.g. Severe swelling, post-op bleeding, acute pain",
                "default": "Transfer immediately if patient had surgery in the last 48 hours or is experiencing severe facial pain.",
            },
            {
                "id": "insurance",
                "label": "Insurances Accepted & New Patient Policy",
                "placeholder": "e.g. We accept Delta Dental, Cigna, MetLife, and cash self-pay discounts",
                "default": "We accept major PPO insurance plans including Delta Dental, MetLife, and Blue Cross, plus flexible self-pay plans.",
            },
            {
                "id": "services",
                "label": "Services Offered",
                "placeholder": "e.g. Cleanings, whitening, crowns, emergency exams",
                "default": "Comprehensive exams, cleanings, root canals, Invisalign, teeth whitening, and emergency dental care.",
            },
        ],
    },
    "legal": {
        "id": "legal",
        "name": "Law Firm & Legal Counsel",
        "icon": "briefcase",
        "tagline": "Client Intake, Case Evaluation & Confidential Screening",
        "default_voice": "af_bella",
        "default_greeting": "Thank you for calling {business_name}. This is the intake coordinator. How can we assist with your legal matter today?",
        "service_options": [
            "Complimentary Case Evaluation",
            "Personal Injury & Auto Accident Claims",
            "Family Law, Divorce & Child Custody",
            "Criminal Defense & Traffic Offenses",
            "Estate Planning, Wills & Revocable Trusts",
            "Business Formation, LLCs & Contracts",
            "Real Estate Closings & Lease Disputes",
            "Civil Litigation & Dispute Resolution"
        ],
        "emergency_triggers": [
            "Caller currently in police custody or jail booking",
            "Court deadline or emergency hearing within 24 hours",
            "Presiding judge or opposing counsel calling",
            "Existing retained client with urgent matter",
            "Law enforcement or government subpoena"
        ],
        "pricing_preset": "We provide a complimentary, confidential 20-minute case evaluation with zero upfront fee.",
        "suggested_questions": [
            {
                "id": "hours",
                "label": "Office Hours & Consultation Windows",
                "placeholder": "e.g. Mon-Fri 9:00 AM - 5:30 PM",
                "default": "Monday through Friday 9:00 AM to 5:30 PM. Consultations available in-person or via secure video.",
            },
            {
                "id": "transfer_rules",
                "label": "When to transfer directly to attorney?",
                "placeholder": "e.g. Active court deadlines within 48h, existing retained clients, opposing counsel",
                "default": "Transfer immediately if caller is an existing client with an active case or opposing counsel calling regarding court dates.",
            },
            {
                "id": "consultation_policy",
                "label": "Free Consultation / Intake Policy",
                "placeholder": "e.g. Free 20-minute case evaluation for personal injury",
                "default": "We provide a complimentary, confidential 20-minute consultation for all new inquiries.",
            },
            {
                "id": "services",
                "label": "Practice Areas",
                "placeholder": "e.g. Personal injury, business law, estate planning, criminal defense",
                "default": "Personal injury, business formation, contracts, estate planning, and civil litigation.",
            },
        ],
    },
    "auto": {
        "id": "auto",
        "name": "Auto Repair & Detailing",
        "icon": "tool",
        "tagline": "Mechanics, Brake Service, Oil Changes & Body Shop",
        "default_voice": "am_adam",
        "default_greeting": "Thanks for calling {business_name}. How can we help get your vehicle running smoothly or serviced today?",
        "service_options": [
            "Brake Pad & Rotor Replacement",
            "Synthetic Oil Change & Filter Service",
            "Computer Check Engine Light Diagnostic",
            "Transmission Fluid & Clutch Repair",
            "Tire Mounting, Balancing & 4-Wheel Alignment",
            "AC System Recharge & Climate Repair",
            "Battery, Starter & Alternator Diagnostics",
            "Suspension, Shocks & Struts Replacement"
        ],
        "emergency_triggers": [
            "Tow truck driver en route delivering vehicle",
            "Vehicle stalled or stranded in active traffic",
            "Customer currently approving repair order on hoist",
            "Insurance claims appraiser calling for teardown"
        ],
        "pricing_preset": "We offer an $89 complete digital vehicle scan that is credited directly toward your repair.",
        "suggested_questions": [
            {
                "id": "hours",
                "label": "Shop Hours & Key Drop-off",
                "placeholder": "e.g. Mon-Fri 7:30 AM - 5:30 PM, early bird night drop box available",
                "default": "Monday through Friday 7:30 AM to 5:30 PM, with early bird / after-hours secure key drop box.",
            },
            {
                "id": "transfer_rules",
                "label": "When to transfer to service advisor?",
                "placeholder": "e.g. Tow truck arriving, vehicle already on lift, estimate approval",
                "default": "Transfer immediately if a tow truck driver is en route or a customer is approving a work order.",
            },
            {
                "id": "services",
                "label": "Core Services",
                "placeholder": "e.g. Brakes, transmission, check engine light, tires, oil changes",
                "default": "Complete computer diagnostics, brake pad and rotor replacement, suspension, transmission repair, and scheduled maintenance.",
            },
        ],
    },
    "restaurant": {
        "id": "restaurant",
        "name": "Restaurant & Hospitality",
        "icon": "coffee",
        "tagline": "Reservations, Private Events, Hours & Catering",
        "default_voice": "am_michael",
        "default_greeting": "Good day! Thank you for calling {business_name}. How may I help with table reservations or dining information?",
        "service_options": [
            "Dinner Table Reservations (1-6 guests)",
            "Large Group Dining & Buyouts (7+ guests)",
            "Private Dining Room Bookings",
            "Full-Service Offsite Catering",
            "Daily Specials & Allergy Questions",
            "Online Takeout & Curbside Pickup",
            "Gift Cards & Event Hosting"
        ],
        "emergency_triggers": [
            "Food delivery vendor or purveyor at loading dock",
            "Event host calling regarding tonight's reservation",
            "Urgent severe allergen inquiry from seated table",
            "Health inspector or municipal official on site"
        ],
        "pricing_preset": "Full seasonal dining menus and custom private catering quotes are provided upon request.",
        "suggested_questions": [
            {
                "id": "hours",
                "label": "Dining Hours & Kitchen Close Times",
                "placeholder": "e.g. Tue-Sun 5:00 PM - 10:00 PM, closed Mondays",
                "default": "Tuesday through Sunday 5:00 PM to 10:00 PM. Weekend brunch Saturday and Sunday 10:00 AM to 2:30 PM.",
            },
            {
                "id": "transfer_rules",
                "label": "When to transfer to host stand or manager?",
                "placeholder": "e.g. Parties of 8 or more, private buyout inquiries, dietary emergencies",
                "default": "Transfer parties of 8 or larger, wedding/buyout inquiries, or vendors to the event director.",
            },
            {
                "id": "parking_and_dress",
                "label": "Parking & Dress Code",
                "placeholder": "e.g. Complimentary valet parking, smart casual dress code",
                "default": "Smart casual attire. Complimentary valet parking is available at the main entrance.",
            },
        ],
    },
    "realestate": {
        "id": "realestate",
        "name": "Real Estate & Property Management",
        "icon": "home",
        "tagline": "Property Showings, Buyer Qualification & Tenant Inquiries",
        "default_voice": "bf_emma",
        "default_greeting": "Hello! Thank you for calling {business_name}. Are you looking to buy, sell, or inquire about a rental property today?",
        "service_options": [
            "Private Home & Condo Showings",
            "Free Home Valuation & Market Analysis",
            "First-Time Homebuyer Consultations",
            "Residential Property Management",
            "Commercial Real Estate Leasing",
            "Open House Schedule & Tour Info",
            "Relocation & Neighborhood Consultations"
        ],
        "emergency_triggers": [
            "Pre-approved buyer submitting written offer today",
            "Title company or escrow officer on closing deadline",
            "Homeowner client listing property immediately",
            "Lockbox or showing access issue at occupied property"
        ],
        "pricing_preset": "We provide complimentary property valuation reports and buyer consultations with zero upfront fees.",
        "suggested_questions": [
            {
                "id": "hours",
                "label": "Showing & Office Hours",
                "placeholder": "e.g. Daily 9:00 AM - 7:00 PM by appointment",
                "default": "Daily 9:00 AM to 7:00 PM for private showings and consultations.",
            },
            {
                "id": "transfer_rules",
                "label": "When to transfer to listing agent?",
                "placeholder": "e.g. Pre-approved buyers ready to submit an offer, property owners listing",
                "default": "Transfer immediately for callers looking to list a home or buyers pre-approved with proof of funds.",
            },
            {
                "id": "services",
                "label": "Services & Focus Markets",
                "placeholder": "e.g. Residential homes, luxury condos, commercial leases",
                "default": "Residential home buying and selling, luxury relocation, and comprehensive property management.",
            },
        ],
    },
    "salon_spa": {
        "id": "salon_spa",
        "name": "Salon, Spa & Aesthetics",
        "icon": "scissors",
        "tagline": "Hair, Nail, Lash & Massage Appointments",
        "default_voice": "af_bella",
        "default_greeting": "Hi! Thanks for calling {business_name}. How can I assist you with scheduling your appointment or beauty service today?",
        "service_options": [
            "Signature Haircut, Wash & Blowout",
            "Custom Balayage, Highlights & Color",
            "Keratin Treatment & Deep Conditioning",
            "Bridal & Special Event Styling",
            "Gel & Acrylic Manicures and Pedicures",
            "Eyelash Extensions & Brow Lamination",
            "Deep Tissue & Swedish Massage Therapy",
            "Hydrating Facials & Chemical Peels"
        ],
        "emergency_triggers": [
            "Client running 15+ minutes late for booked appointment",
            "Bridal party coordinator needing same-day adjustment",
            "Stylist or therapist calling out sick",
            "Beauty supply distributor delivery"
        ],
        "pricing_preset": "Transparent tier-based service pricing is quoted during scheduling based on hair length and stylist level.",
        "suggested_questions": [
            {
                "id": "hours",
                "label": "Salon Hours",
                "placeholder": "e.g. Tue-Sat 9:00 AM - 7:00 PM",
                "default": "Tuesday through Saturday 9:00 AM to 7:00 PM. Closed Sundays and Mondays.",
            },
            {
                "id": "transfer_rules",
                "label": "When to transfer to stylist or manager?",
                "placeholder": "e.g. Complex color corrections, bridal parties, same-day cancellations",
                "default": "Transfer bridal parties or requests for custom color correction consultations to the front desk coordinator.",
            },
            {
                "id": "cancellation_policy",
                "label": "Cancellation & Deposit Policy",
                "placeholder": "e.g. 24-hour cancellation notice required",
                "default": "We require 24 hours notice for cancellations to avoid a 50 percent rebooking fee.",
            },
        ],
    },
    "general": {
        "id": "general",
        "name": "Professional Services & Other Businesses",
        "icon": "shield",
        "tagline": "Agencies, Consultants, Contractors & Local Shops",
        "default_voice": "af_heart",
        "default_greeting": "Thank you for calling {business_name}. How may I help you today?",
        "service_options": [
            "General Service & On-Site Repairs",
            "Free In-Person or Phone Estimate",
            "Routine Scheduled Preventative Maintenance",
            "Urgent Same-Day Dispatch",
            "Project Planning & Consultation",
            "Commercial Accounts & Maintenance Contracts"
        ],
        "emergency_triggers": [
            "Active hazard or time-critical emergency",
            "Caller specifically asks for the business owner",
            "High-priority corporate contract account",
            "Municipal official or inspector calling"
        ],
        "pricing_preset": "We provide upfront, transparent estimates before any work commences with zero surprise fees.",
        "suggested_questions": [
            {
                "id": "hours",
                "label": "Standard Business Hours",
                "placeholder": "e.g. Monday to Friday 9:00 AM - 5:00 PM",
                "default": "Monday to Friday 9:00 AM to 5:00 PM.",
            },
            {
                "id": "transfer_rules",
                "label": "When to transfer to owner/cell phone?",
                "placeholder": "e.g. High-priority inquiries, urgent client requests, or direct referrals",
                "default": "Transfer immediately when caller has a time-sensitive emergency or asks specifically for the business owner.",
            },
            {
                "id": "services",
                "label": "Services Offered",
                "placeholder": "e.g. Full-service consulting, emergency repairs, custom projects",
                "default": "Professional consultations, custom project quotes, and direct client support.",
            },
        ],
    },
}

# ---------------------------------------------------------------------------
# Business Auto-Discovery & Address Enrichment Engine
# ---------------------------------------------------------------------------

def scrape_business_intelligence(query_text: str) -> Optional[Dict[str, Any]]:
    """Fetches real-world business directory search snippets and extracts the exact
    operating company name, phone, trade, and hours using Firecrawl keyless search and Groq LLM.
    """
    clean_q = query_text.strip()
    if not clean_q:
        return None

    snippets = []

    # 1. Primary: Firecrawl search (keyless, high-reliability)
    try:
        resp = requests.post(
            "https://api.firecrawl.dev/v1/search",
            json={"query": f'"{clean_q}"'},
            headers={"Content-Type": "application/json"},
            timeout=4.0
        )
        if resp.status_code == 200:
            for item in resp.json().get("data", []):
                t = item.get("title", "")
                d = item.get("description", "")
                if t or d:
                    snippets.append(f"{t}: {d}")

        if not snippets:
            resp = requests.post(
                "https://api.firecrawl.dev/v1/search",
                json={"query": clean_q},
                headers={"Content-Type": "application/json"},
                timeout=4.0
            )
            if resp.status_code == 200:
                for item in resp.json().get("data", []):
                    t = item.get("title", "")
                    d = item.get("description", "")
                    if t or d:
                        snippets.append(f"{t}: {d}")
    except Exception as e:
        logger.debug(f"Firecrawl search error: {e}")

    # 2. Fallback: DuckDuckGo Lite POST
    if not snippets:
        try:
            url = "https://lite.duckduckgo.com/lite/"
            data = urllib.parse.urlencode({"q": clean_q}).encode()
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Content-Type": "application/x-www-form-urlencoded",
            }
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                if resp.status == 200:
                    html = resp.read().decode("utf-8", errors="ignore")
                    raw_snippets = re.findall(r"class=[\"\x27]?result-snippet[\"\x27]?[^>]*>(.*?)</td>", html, re.DOTALL)
                    for s in raw_snippets[:5]:
                        clean = re.sub(r"<[^<]+?>", "", s).strip()
                        if clean:
                            snippets.append(clean)
        except Exception as e:
            logger.debug(f"DDG Lite fallback error: {e}")

    if not snippets:
        return None

    combined_text = "\n".join(snippets[:6])

    # 3. Use Groq LLM for entity resolution
    groq_key = settings.GROQ_API_KEY
    if groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            prompt = f"""You are an expert US business entity resolver.
Given these real web directory snippets for "{query_text}", extract the real operating business:
- business_name: Actual operating company name (e.g. "Same Day Heating & Air Conditioning"). Never return generic "San Diego Services".
- phone: Real phone formatted e.g. "(619) 503-3355" or "" if not found.
- industry: One of [hvac, plumbing, electrical, roofing, dental_medical, auto, legal, restaurant, realestate, salon_spa, general].
- hours: Real business hours (e.g. "Open 24 hours" or "Mon-Fri 8am-5pm") or "".
- formatted_address: Complete street address with city, state, zip.

Snippets:
{combined_text}

Output MUST be a single raw JSON object only. No markdown, no triple backticks."""

            completion = client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=250
            )
            raw = completion.choices[0].message.content.strip()
            raw = re.sub(r"^```json\s*", "", raw)
            raw = re.sub(r"^```\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            if data.get("business_name") and "unknown" not in data["business_name"].lower() and "services" != data["business_name"].lower():
                return data
        except Exception as e:
            logger.debug(f"Groq business extraction error: {e}")

    # Heuristic regex extraction fallback from snippets
    phone_match = re.search(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", combined_text)
    phone_str = phone_match.group(0) if phone_match else ""

    hours_str = ""
    if "open 24 hours" in combined_text.lower():
        hours_str = "Open 24 hours"
    elif "24/7" in combined_text:
        hours_str = "24/7 Emergency Service"

    return {
        "phone": phone_str,
        "hours": hours_str,
    }


def auto_discover_business(query_text: str, location_hint: Optional[str] = None) -> Dict[str, Any]:
    """Smart auto-enrichment engine that finds details from an address or business name.
    Attempts real web intelligence resolution, Google Maps, OpenStreetMap geocoding,
    and robust heuristic parsing fallback.
    """
    clean_query = query_text.strip()
    if not clean_query:
        return {"success": False, "message": "Query cannot be empty"}

    result: Dict[str, Any] = {
        "success": True,
        "query": clean_query,
        "business_name": "",
        "formatted_address": clean_query,
        "city": "",
        "state": "",
        "postcode": "",
        "inferred_industry": "general",
        "timezone": "America/New_York",
        "phone_placeholder": "+1 (555) 000-0000",
        "confidence": "high",
        "detected_features": [],
    }

    # 1. First priority: Live Web Intelligence Scraping (Finds exact company, phone, hours)
    intel = scrape_business_intelligence(clean_query)
    if intel and intel.get("business_name"):
        result["business_name"] = intel["business_name"]
        if intel.get("formatted_address"):
            result["formatted_address"] = intel["formatted_address"]
        if intel.get("phone"):
            result["phone"] = intel["phone"]
            result["phone_placeholder"] = intel["phone"]
            result["detected_features"].append(f"Phone: {intel['phone']}")
        if intel.get("industry") and intel["industry"] in INDUSTRIES:
            result["inferred_industry"] = intel["industry"]
            result["detected_features"].append(f"Trade: {INDUSTRIES[intel['industry']]['name']}")
        if intel.get("hours"):
            result["hours"] = intel["hours"]
            result["detected_features"].append("Operating hours verified via web directory")
        result["detected_features"].append(f"Verified {intel['business_name']} via Web Intelligence")
        result["source"] = "web_intelligence"

        # If hours still not resolved, set industry default
        if not result.get("hours"):
            ind_key = result.get("inferred_industry", "general")
            ind_def = INDUSTRIES.get(ind_key, INDUSTRIES["general"])
            result["hours"] = ind_def["suggested_questions"][0]["default"]
        return result

    # 2. Heuristic industry detection based on keywords
    query_lower = clean_query.lower()
    if any(k in query_lower for k in ["hvac", "heating", "air condition", "cooling", "furnace", "duct"]):
        result["inferred_industry"] = "hvac"
        result["detected_features"].append("Identified HVAC & Climate Control")
    elif any(k in query_lower for k in ["plumb", "drain", "rooter", "water heater", "pipe"]):
        result["inferred_industry"] = "plumbing"
        result["detected_features"].append("Identified Plumbing & Drains")
    elif any(k in query_lower for k in ["electric", "wiring", "breaker", "panel", "ev charger", "lighting"]):
        result["inferred_industry"] = "electrical"
        result["detected_features"].append("Identified Electrical Services")
    elif any(k in query_lower for k in ["roof", "shingle", "gutter", "siding"]):
        result["inferred_industry"] = "roofing"
        result["detected_features"].append("Identified Roofing & Gutters")
    elif any(k in query_lower for k in ["dental", "dentist", "ortho", "clinic", "doctor", "chiro", "md", "smile"]):
        result["inferred_industry"] = "dental_medical"
        result["detected_features"].append("Identified Dental / Medical Practice")
    elif any(k in query_lower for k in ["law", "legal", "attorney", "esq", "counsel", "advocate"]):
        result["inferred_industry"] = "legal"
        result["detected_features"].append("Identified Legal Practice")
    elif any(k in query_lower for k in ["auto", "tire", "mechanic", "transmission", "brake", "detailing", "garage"]):
        result["inferred_industry"] = "auto"
        result["detected_features"].append("Identified Automotive Services")
    elif any(k in query_lower for k in ["restaurant", "bistro", "cafe", "grill", "pizza", "sushi", "dining", "bar"]):
        result["inferred_industry"] = "restaurant"
        result["detected_features"].append("Identified Restaurant & Dining")
    elif any(k in query_lower for k in ["salon", "spa", "barber", "hair", "nails", "lash", "massage"]):
        result["inferred_industry"] = "salon_spa"
        result["detected_features"].append("Identified Salon & Spa")
    elif any(k in query_lower for k in ["realty", "real estate", "properties", "brokerage", "homes"]):
        result["inferred_industry"] = "realestate"
        result["detected_features"].append("Identified Real Estate")

    # 3. Check if Google Maps / Places API key is configured
    google_key = settings.GOOGLE_MAPS_API_KEY
    if google_key:
        try:
            # Google Places Text Search
            g_search_url = (
                f"https://maps.googleapis.com/maps/api/place/textsearch/json?"
                f"query={urllib.parse.quote(clean_query)}&key={google_key}"
            )
            req = urllib.request.Request(g_search_url, headers={"User-Agent": "ORXAgents/1.0"})
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                if resp.status == 200:
                    g_data = json.loads(resp.read().decode())
                    results = g_data.get("results", [])
                    if results:
                        top_place = results[0]
                        place_id = top_place.get("place_id")
                        result["business_name"] = top_place.get("name", "")
                        result["formatted_address"] = top_place.get("formatted_address", "")
                        result["source"] = "google_maps"
                        result["detected_features"].append("Verified via Google Maps")

                        # Map Google Place types to ORX industries
                        g_types = top_place.get("types", [])
                        if any(t in g_types for t in ["plumber"]):
                            result["inferred_industry"] = "plumbing"
                        elif any(t in g_types for t in ["electrician"]):
                            result["inferred_industry"] = "electrical"
                        elif any(t in g_types for t in ["roofing_contractor"]):
                            result["inferred_industry"] = "roofing"
                        elif any(t in g_types for t in ["dentist", "doctor", "health", "hospital"]):
                            result["inferred_industry"] = "dental_medical"
                        elif any(t in g_types for t in ["lawyer"]):
                            result["inferred_industry"] = "legal"
                        elif any(t in g_types for t in ["car_repair", "car_dealer"]):
                            result["inferred_industry"] = "auto"
                        elif any(t in g_types for t in ["restaurant", "cafe", "bar", "meal_takeaway"]):
                            result["inferred_industry"] = "restaurant"
                        elif any(t in g_types for t in ["beauty_salon", "hair_care", "spa"]):
                            result["inferred_industry"] = "salon_spa"
                        elif any(t in g_types for t in ["real_estate_agency"]):
                            result["inferred_industry"] = "realestate"

                        if place_id:
                            details_url = (
                                f"https://maps.googleapis.com/maps/api/place/details/json?"
                                f"place_id={place_id}&fields=name,formatted_address,formatted_phone_number,international_phone_number,opening_hours,website,rating&key={google_key}"
                            )
                            d_req = urllib.request.Request(details_url, headers={"User-Agent": "ORXAgents/1.0"})
                            with urllib.request.urlopen(d_req, timeout=3.5) as d_resp:
                                if d_resp.status == 200:
                                    d_data = json.loads(d_resp.read().decode()).get("result", {})
                                    if d_data.get("formatted_phone_number"):
                                        result["phone"] = d_data.get("formatted_phone_number")
                                        result["phone_placeholder"] = d_data.get("formatted_phone_number")
                                        result["detected_features"].append(f"Phone: {result['phone']}")
                                    
                                    op_hours = d_data.get("opening_hours", {})
                                    weekday_text = op_hours.get("weekday_text", [])
                                    if weekday_text:
                                        result["operating_hours_raw"] = weekday_text
                                        result["hours"] = "; ".join(weekday_text[:3])
                                        result["detected_features"].append("Operating hours synced from Google Maps")
                                    
                                    if d_data.get("website"):
                                        result["website"] = d_data.get("website")
                                    if d_data.get("rating"):
                                        result["rating"] = d_data.get("rating")

                        return result
        except Exception as e:
            logger.warning(f"Google Maps API lookup failed: {e}. Falling back to OpenStreetMap.")

    # 4. Fallback to OpenStreetMap Nominatim geocoding (with suite stripping)
    try:
        # Strip suite/apt/unit numbers so Nominatim can geocode building
        cleaned_geo = re.sub(r"(?i)\b(ste|suite|apt|unit|fl|floor|bldg|building|#)\.?\s*[a-z0-9\-]+", "", clean_query)
        cleaned_geo = re.sub(r"\s+", " ", cleaned_geo).strip()

        url = (
            f"https://nominatim.openstreetmap.org/search?"
            f"q={urllib.parse.quote(cleaned_geo)}&format=json&addressdetails=1&extratags=1&namedetails=1&countrycodes=us&limit=1"
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ORXAgents-Onboarding/1.0 (agents.orxlabs; support@orxlabs.com)"}
        )
        with urllib.request.urlopen(req, timeout=2.5) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                if data and len(data) > 0:
                    item = data[0]
                    addr = item.get("address", {})
                    road = addr.get("road", "")
                    house_number = addr.get("house_number", "")
                    city = addr.get("city") or addr.get("town") or addr.get("village") or addr.get("county", "")
                    state = addr.get("state", "")
                    postcode = addr.get("postcode", "")
                    country = addr.get("country", "")

                    formatted_parts = [p for p in [f"{house_number} {road}".strip(), city, state, postcode, country] if p]
                    if formatted_parts:
                        result["formatted_address"] = ", ".join(formatted_parts)
                    result["city"] = city
                    result["state"] = state
                    result["postcode"] = postcode
                    result["detected_features"].append(f"Located address via OpenStreetMap in {city or state or 'USA'}")

                    # Extract POI name if present
                    poi_name = item.get("name") or (item.get("namedetails") or {}).get("name")
                    if poi_name and not result.get("business_name"):
                        result["business_name"] = poi_name

                    # Parse extratags for real phone, opening hours, and category
                    extratags = item.get("extratags") or {}
                    osm_phone = extratags.get("phone") or extratags.get("contact:phone")
                    if osm_phone and not result.get("phone"):
                        result["phone"] = osm_phone
                        result["phone_placeholder"] = osm_phone
                        result["detected_features"].append(f"Phone on record: {osm_phone}")

                    osm_hours = extratags.get("opening_hours")
                    if osm_hours and not result.get("hours"):
                        result["hours"] = osm_hours
                        result["detected_features"].append("Operating hours verified via OpenStreetMap")

                    craft = (extratags.get("craft") or "").lower()
                    amenity = (extratags.get("amenity") or "").lower()
                    shop = (extratags.get("shop") or "").lower()
                    office = (extratags.get("office") or "").lower()

                    if any(c in [craft, shop] for c in ["plumber"]):
                        result["inferred_industry"] = "plumbing"
                    elif any(c in [craft, shop] for c in ["electrician", "electrical"]):
                        result["inferred_industry"] = "electrical"
                    elif any(c in [craft, shop] for c in ["roofer", "roofing"]):
                        result["inferred_industry"] = "roofing"
                    elif any(c in [craft, shop] for c in ["hvac", "heating"]):
                        result["inferred_industry"] = "hvac"
                    elif any(c in [amenity, shop] for c in ["dentist", "doctors", "clinic"]):
                        result["inferred_industry"] = "dental_medical"
                    elif any(c in [shop, craft] for c in ["car_repair", "tyres", "car"]):
                        result["inferred_industry"] = "auto"
                    elif any(c in [amenity] for c in ["restaurant", "cafe", "fast_food", "bar"]):
                        result["inferred_industry"] = "restaurant"
                    elif any(c in [shop] for c in ["hairdresser", "beauty", "massage"]):
                        result["inferred_industry"] = "salon_spa"
                    elif any(c in [office] for c in ["lawyer"]):
                        result["inferred_industry"] = "legal"
                    elif any(c in [office] for c in ["estate_agent"]):
                        result["inferred_industry"] = "realestate"
    except Exception as e:
        logger.debug(f"Nominatim lookup skipped or timed out: {e}")

    # Fallback to web search phone/hours if captured
    if intel:
        if not result.get("phone") and intel.get("phone"):
            result["phone"] = intel["phone"]
            result["phone_placeholder"] = intel["phone"]
        if not result.get("hours") and intel.get("hours"):
            result["hours"] = intel["hours"]

    # Extract or infer business name if not yet set
    if not result["business_name"]:
        parts = [p.strip() for p in clean_query.split(",")]
        if not parts[0][0].isdigit():
            result["business_name"] = parts[0]
        else:
            street_clean = re.sub(r"(?i)\b(ste|suite|apt|unit|fl|floor|bldg|building|#)\.?\s*[a-z0-9\-]+", "", parts[0]).strip()
            result["business_name"] = street_clean or parts[0]

    # Suggest industry hours & phone if not extracted
    if not result.get("hours"):
        ind_key = result.get("inferred_industry", "general")
        ind_def = INDUSTRIES.get(ind_key, INDUSTRIES["general"])
        result["hours"] = ind_def["suggested_questions"][0]["default"]

    return result


def search_places_autocomplete(query_text: str) -> List[Dict[str, Any]]:
    """Returns matching US businesses and addresses using Firecrawl live search and OpenStreetMap."""
    clean_query = query_text.strip()
    if not clean_query or len(clean_query) < 2:
        return []

    matches: List[Dict[str, Any]] = []

    # 1. Firecrawl Live US Business & Trade Directory Search
    if len(clean_query) >= 3:
        try:
            resp = requests.post(
                "https://api.firecrawl.dev/v1/search",
                json={"query": f"{clean_query} USA"},
                timeout=2.8
            )
            if resp.status_code == 200:
                for item in resp.json().get("data", [])[:4]:
                    raw_title = item.get("title", "")
                    title = raw_title.split("|")[0].split("-")[0].strip()
                    desc = item.get("description", "").replace("\n", " ")
                    clean_desc = re.sub(r"\s+", " ", desc).strip()
                    if title and len(title) > 2 and "404" not in title.lower():
                        matches.append({
                            "title": title,
                            "address": clean_desc[:110] if clean_desc else f"{title}, USA",
                            "place_id": f"fc_{len(matches)}",
                            "source": "verified_business"
                        })
        except Exception as e:
            logger.debug(f"Firecrawl autocomplete search error: {e}")

    # 2. Google Places Text Search (Strictly US)
    google_key = settings.GOOGLE_MAPS_API_KEY
    if google_key and not matches:
        try:
            us_query = clean_query if "usa" in clean_query.lower() or "united states" in clean_query.lower() else f"{clean_query}, USA"
            url = (
                f"https://maps.googleapis.com/maps/api/place/textsearch/json?"
                f"query={urllib.parse.quote(us_query)}&region=us&key={google_key}"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "ORXAgents/1.0"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    for item in data.get("results", [])[:5]:
                        addr = item.get("formatted_address", "")
                        if "usa" in addr.lower() or any(st in addr for st in [", AL", ", AK", ", AZ", ", AR", ", CA", ", CO", ", CT", ", DE", ", FL", ", GA", ", HI", ", ID", ", IL", ", IN", ", IA", ", KS", ", KY", ", LA", ", ME", ", MD", ", MA", ", MI", ", MN", ", MS", ", MO", ", MT", ", NE", ", NV", ", NH", ", NJ", ", NM", ", NY", ", NC", ", ND", ", OH", ", OK", ", OR", ", PA", ", RI", ", SC", ", SD", ", TN", ", TX", ", UT", ", VT", ", VA", ", WA", ", WV", ", WI", ", WY"]):
                            matches.append({
                                "title": item.get("name", ""),
                                "address": addr,
                                "place_id": item.get("place_id", ""),
                                "rating": item.get("rating"),
                                "source": "google_maps_us",
                            })
                    if matches:
                        return matches
        except Exception as e:
            logger.warning(f"Google Places autocomplete error: {e}")

    # 3. OpenStreetMap / Nominatim US-Only Search (with suite stripping)
    if not matches or len(matches) < 2:
        try:
            cleaned_geo = re.sub(r"(?i)\b(ste|suite|apt|unit|fl|floor|bldg|building|#)\.?\s*[a-z0-9\-]+", "", clean_query)
            cleaned_geo = re.sub(r"\s+", " ", cleaned_geo).strip()
            url = (
                f"https://nominatim.openstreetmap.org/search?"
                f"q={urllib.parse.quote(cleaned_geo)}&format=json&addressdetails=1&countrycodes=us&limit=3"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "ORXAgents-Onboarding/1.0 (agents.orxlabs; support@orxlabs.com)"})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    osm_data = json.loads(resp.read().decode())
                    for item in osm_data:
                        addr_dict = item.get("address", {})
                        road = addr_dict.get("road", "")
                        house_no = addr_dict.get("house_number", "")
                        city = addr_dict.get("city") or addr_dict.get("town") or addr_dict.get("village", "")
                        state = addr_dict.get("state", "")
                        postcode = addr_dict.get("postcode", "")

                        parts = item.get("display_name", "").split(",")
                        title = item.get("name") or (f"{house_no} {road}".strip() if house_no else parts[0].strip())

                        us_addr_parts = [p for p in [f"{house_no} {road}".strip(), city, f"{state} {postcode}".strip(), "USA"] if p]
                        formatted_us_addr = ", ".join(us_addr_parts) if len(us_addr_parts) > 1 else item.get("display_name", "")

                        matches.append({
                            "title": title,
                            "address": formatted_us_addr,
                            "place_id": item.get("place_id", ""),
                            "source": "us_directory",
                        })
        except Exception as e:
            logger.debug(f"OSM fallback search: {e}")

    if not matches:
        matches.append({
            "title": clean_query.title(),
            "address": f"{clean_query.title()}, USA",
            "place_id": "gen_1",
            "source": "us_directory",
        })

    return matches

# ---------------------------------------------------------------------------
# Client Profile Persistence
# ---------------------------------------------------------------------------

def _ensure_clients_storage() -> Dict[str, Any]:
    """Ensures clients storage file exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CLIENTS_FILE.exists():
        storage = {"clients": {}}
        CLIENTS_FILE.write_text(json.dumps(storage, indent=2))
        return storage
    try:
        return json.loads(CLIENTS_FILE.read_text())
    except Exception:
        storage = {"clients": {}}
        CLIENTS_FILE.write_text(json.dumps(storage, indent=2))
        return storage

def save_client_profile(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """Saves or updates a business client profile."""
    storage = _ensure_clients_storage()
    client_id = profile_data.get("id") or f"cli_{uuid.uuid4().hex[:10]}"
    profile_data["id"] = client_id

    if "created_at" not in profile_data:
        profile_data["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    profile_data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Ensure phone number assigned
    if not profile_data.get("assigned_phone"):
        profile_data["assigned_phone"] = settings.PLIVO_PHONE_NUMBER or "+1 (833) 420-5227"

    # Default polar status
    if "polar_status" not in profile_data:
        profile_data["polar_status"] = "trial_ready"
    if "plan" not in profile_data:
        profile_data["plan"] = "starter"

    # Compile dynamic prompt
    profile_data["compiled_prompt"] = compile_agent_prompt(profile_data)

    storage["clients"][client_id] = profile_data
    CLIENTS_FILE.write_text(json.dumps(storage, indent=2))
    logger.info(f"Saved client profile '{client_id}' for {profile_data.get('business_name')}")

    # Auto-provision dedicated project database & custom prompt workspace upon onboarding
    try:
        from app.project_db import create_project
        create_project(profile_data, trigger_source="onboarding")
    except Exception as pr_ex:
        logger.warning(f"Could not auto-provision dedicated project DB: {pr_ex}")

    return profile_data

def get_client_profile(client_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a client profile by ID."""
    storage = _ensure_clients_storage()
    return storage.get("clients", {}).get(client_id)

def get_latest_client_profile() -> Optional[Dict[str, Any]]:
    """Gets the most recently saved or created client profile."""
    storage = _ensure_clients_storage()
    clients = list(storage.get("clients", {}).values())
    if not clients:
        default_profile = {
            "id": "cli_demo",
            "business_name": "Apex Climate & Air",
            "industry": "hvac",
            "address": "742 Evergreen Terrace, Springfield",
            "forwarding_phone": "+1 (555) 234-5678",
            "assigned_phone": settings.PLIVO_PHONE_NUMBER or "+1 (833) 420-5227",
            "hours": "Mon-Fri 7:30 AM to 6:00 PM, 24/7 Emergency Dispatch",
            "transfer_rules": "Transfer immediately for gas smell, water leak, or caller asking for owner.",
            "services": "AC Repair, Furnace Maintenance, Heat Pump Installs, Duct Cleaning.",
            "custom_qa": [
                {"question": "Do you offer financing?", "answer": "Yes, zero percent interest financing for up to 24 months on new installations."}
            ],
            "polar_status": "active",
            "plan": "starter",
            "connection_status": "pending",
        }
        return save_client_profile(default_profile)
    clients.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return clients[0]


def verify_client_connection(client_id: Optional[str] = None, carrier: str = "verizon") -> Dict[str, Any]:
    """Marks a client's carrier call forwarding as verified and active."""
    profile = get_client_profile(client_id) if client_id else get_latest_client_profile()
    if not profile:
        profile = get_latest_client_profile()
        if not profile:
            return {"success": False, "message": "Client not found"}

    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    profile["connection_status"] = "verified"
    profile["carrier"] = carrier.capitalize()
    profile["verified_at"] = now_str
    save_client_profile(profile)
    logger.info(f"Verified connection for client '{profile.get('id')}' ({profile.get('business_name')}) via {carrier}")
    return {
        "success": True,
        "connection_status": "verified",
        "carrier": carrier.capitalize(),
        "verified_at": now_str,
        "assigned_phone": profile.get("assigned_phone"),
        "forwarding_phone": profile.get("forwarding_phone"),
        "business_name": profile.get("business_name"),
    }


def disconnect_client_connection(client_id: Optional[str] = None) -> Dict[str, Any]:
    """Pauses or disconnects call forwarding for client."""
    profile = get_client_profile(client_id) if client_id else get_latest_client_profile()
    if not profile:
        return {"success": False, "message": "Client not found"}

    profile["connection_status"] = "paused"
    profile["disconnected_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_client_profile(profile)
    logger.info(f"Paused connection for client '{profile.get('id')}'")
    return {"success": True, "connection_status": "paused"}

# ---------------------------------------------------------------------------
# Prompt Compiler for Any Industry
# ---------------------------------------------------------------------------

def compile_agent_prompt(profile: Dict[str, Any]) -> str:
    """Compiles a production-grade, natural-sounding voice AI prompt strictly
    adapted to the user's business, industry, hours, transfer rules, and custom Q&As.
    """
    biz_name = profile.get("business_name", "Our Company")
    industry_id = profile.get("industry", "general")
    address = profile.get("address", "")
    forwarding_phone = profile.get("forwarding_phone", "")
    hours = profile.get("hours", "Monday through Friday 8:00 AM to 6:00 PM")
    transfer_rules = profile.get("transfer_rules", "Transfer immediately if caller requests human staff or has an emergency.")
    services = profile.get("services", "Full residential and commercial services.")
    custom_qa = profile.get("custom_qa", [])

    persona_name = profile.get("persona_name", "Riley")
    pricing_policy = profile.get("pricing_policy", "We provide upfront estimates and our diagnostic fee is applied directly to repairs.")
    booking_action = profile.get("booking_action", "Offer to schedule an arrival time window and collect the customer street address.")
    shift_mode = profile.get("shift_mode", "24/7 Answering & Overflow")

    custom_qa_lines = ""
    if custom_qa:
        for idx, qa in enumerate(custom_qa, 1):
            q = qa.get("question", "").strip()
            a = qa.get("answer", "").strip()
            if q and a:
                custom_qa_lines += f"- Caller Question: \"{q}\" -> Answer: \"{a}\"\n"

    prompt = f"""[Identity & Purpose]
You are {persona_name}, the friendly, highly efficient, and trusted voice receptionist for {biz_name}.
Your job is to answer customer phone calls, provide clear details on our services, schedule appointments or bookings, and connect callers to human staff when necessary.
Our address is {address or 'available upon booking'}.
Our business operating hours: {hours}.
Shift Coverage: {shift_mode}.

[Spoken Voice Rules - Strictly Followed]
- Speak in natural, everyday conversational American English.
- Keep every answer to 1 to 2 short spoken sentences (strictly under 20 words per response).
- Use natural contractions like "I'm", "we'll", "don't", "it's", and "let's".
- Never use markdown formatting, bullet points, asterisks, or numbered lists.
- Speak numbers phonetically: say "nine a.m." instead of "09:00", say "eighty-nine dollars" instead of "$89".
- Ask only ONE single question at a time so the caller is never overwhelmed.

[Our Core Services]
{services}

[Pricing & Diagnostic Policy]
{pricing_policy}

[Scheduling & Booking Protocol]
{booking_action}

[Call Transfer & Escalation Rules]
Transfer trigger conditions:
{transfer_rules}
If a transfer condition is met: Say "I am connecting you with our on-call team right now. Please hold for just a moment." and execute transfer to {forwarding_phone or 'the owner cell'}.

[Business FAQ & Knowledge]
{custom_qa_lines if custom_qa_lines else '- Provide helpful, concise answers and offer to schedule service.'}

[Turn Ending]
Always end your turn with either a helpful booking question or a confirmation."""
    return prompt.strip()

# ---------------------------------------------------------------------------
# Live Conversational Demo Engine
# ---------------------------------------------------------------------------

def simulate_agent_turn(client_profile: Dict[str, Any], user_message: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """Generates a real-time conversational response from the client's tailored voice agent.
    Uses Groq LLM if configured; otherwise uses smart prompt-aware heuristics.
    """
    biz_name = client_profile.get("business_name", "Apex Services")
    system_prompt = client_profile.get("compiled_prompt") or compile_agent_prompt(client_profile)

    msg_lower = user_message.lower()
    is_transfer = False
    if any(k in msg_lower for k in ["transfer", "human", "speak to owner", "manager", "emergency", "gas smell", "burst pipe", "leak"]):
        is_transfer = True

    groq_key = settings.GROQ_API_KEY
    if groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            messages = [{"role": "system", "content": system_prompt}]
            if history:
                for h in history[-4:]:
                    messages.append({"role": h["role"], "content": h["content"]})
            messages.append({"role": "user", "content": user_message})

            completion = client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=messages,
                temperature=0.5,
                max_tokens=100,
            )
            reply = completion.choices[0].message.content.strip()
            reply = reply.replace("*", "").replace("#", "").replace("- ", "")
            return {
                "response": reply,
                "is_transfer": is_transfer,
                "business_name": biz_name,
            }
        except Exception as e:
            logger.warning(f"Groq turn simulation fallback: {e}")

    # Fallback Responses
    if is_transfer:
        reply = f"I understand completely. I am transferring you directly to our on-call team at {client_profile.get('forwarding_phone', 'our main line')} right now. Please hold for one second."
    elif any(k in msg_lower for k in ["hour", "open", "close", "time", "sunday", "weekend"]):
        reply = f"We are open {client_profile.get('hours', 'Monday through Friday from 8 a.m. to 6 p.m.')}. Would you like to schedule a service visit?"
    elif any(k in msg_lower for k in ["price", "cost", "fee", "rate", "quote"]):
        reply = f"We provide upfront estimates for all our services, and our standard diagnostic fee is applied to your repair. What type of service do you need?"
    elif any(k in msg_lower for k in ["where", "address", "location"]):
        addr = client_profile.get("address", "")
        reply = f"We are located at {addr or 'the address on file'} and we dispatch technicians directly to your location. Where are you located?"
    elif any(k in msg_lower for k in ["book", "schedule", "appointment", "come over", "visit"]):
        reply = f"I can get that scheduled right away for {biz_name}! What day and time window works best for you?"
    else:
        reply = f"Thanks for checking with {biz_name}. We can certainly take care of that for you. Would you like me to book a technician or answer any other questions?"

    return {
        "response": reply,
        "is_transfer": is_transfer,
        "business_name": biz_name,
    }

# ---------------------------------------------------------------------------
# Polar Subscription & Checkout Helper
# ---------------------------------------------------------------------------

def create_polar_checkout_session(plan_id: str, client_id: str, success_url: str) -> Dict[str, Any]:
    """Generates a Polar.sh checkout session or test checkout payload."""
    token = settings.POLAR_ACCESS_TOKEN
    org_id = settings.POLAR_ORGANIZATION_ID
    product_id = settings.POLAR_PRODUCT_ID_GROWTH if plan_id == "growth" else settings.POLAR_PRODUCT_ID_STARTER
    custom_checkout_link = settings.POLAR_CHECKOUT_URL

    # 1. Fastest 60-second method: Direct Polar buy link
    if custom_checkout_link and "buy.polar.sh" in custom_checkout_link:
        sep = "&" if "?" in custom_checkout_link else "?"
        return {
            "mode": "polar_payment_link",
            "checkout_url": f"{custom_checkout_link}{sep}client_id={client_id}",
            "product_id": product_id,
            "plan": plan_id,
            "amount": "$49/mo" if plan_id == "starter" else "$99/mo",
        }

    # 2. Polar Python SDK Checkout Session
    if token:
        try:
            from polar_sdk import Polar
            polar_client = Polar(access_token=token)
            checkout = polar_client.checkouts.create(request={
                "products": [product_id],
                "success_url": success_url,
                "metadata": {"client_id": client_id, "plan": plan_id},
            })
            if checkout and checkout.url:
                return {
                    "mode": "polar_live",
                    "checkout_url": checkout.url,
                    "checkout_id": checkout.id,
                }
        except Exception as e:
            logger.warning(f"Polar SDK checkout error: {e}. Falling back to test checkout.")

    test_checkout_url = f"{success_url}?session_id=polar_chk_{uuid.uuid4().hex[:12]}&client_id={client_id}&plan={plan_id}&status=success"
    return {
        "mode": "polar_test_ready",
        "checkout_url": test_checkout_url,
        "product_id": product_id,
        "plan": plan_id,
        "amount": "$49/mo" if plan_id == "starter" else "$99/mo",
    }
