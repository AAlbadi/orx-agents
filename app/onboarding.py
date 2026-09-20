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

from app.industry_topics import (
    INDUSTRY_TOPIC_CATEGORIES,
    COMMON_TIMEZONES,
    SCHEDULE_MODES,
    AFTER_HOURS_POLICIES,
    get_industry_topics,
)

for ind_id, ind_data in INDUSTRIES.items():
    ind_data["topic_categories"] = get_industry_topics(ind_id)

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

    # 1. Primary: Firecrawl search (only if API key is explicitly configured)
    firecrawl_key = getattr(settings, "FIRECRAWL_API_KEY", None) or os.environ.get("FIRECRAWL_API_KEY")
    if firecrawl_key:
        try:
            resp = requests.post(
                "https://api.firecrawl.dev/v1/search",
                json={"query": f'"{clean_q}"'},
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {firecrawl_key}"},
                timeout=2.5
            )
            if resp.status_code == 200:
                for item in resp.json().get("data", []):
                    t = item.get("title", "")
                    d = item.get("description", "")
                    if t or d:
                        snippets.append(f"{t}: {d}")
        except Exception as e:
            logger.debug(f"Firecrawl search error: {e}")

    # 2. Fallback: DuckDuckGo Lite POST (Fast, zero key, reliable)
    if not snippets:
        try:
            url = "https://lite.duckduckgo.com/lite/"
            data = urllib.parse.urlencode({"q": clean_q}).encode()
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Content-Type": "application/x-www-form-urlencoded",
            }
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    html = resp.read().decode("utf-8", errors="ignore")
                    raw_snippets = re.findall(r"class=[\"\x27]?result-snippet[\"\x27]?[^>]*>(.*?)</td>", html, re.DOTALL)
                    for s in raw_snippets[:5]:
                        clean = re.sub(r"<[^<]+?>", "", s).strip()
                        if clean:
                            snippets.append(clean)
        except Exception as e:
            logger.debug(f"DDG Lite fallback error: {e}")

    # 3. Fallback: Yahoo Search
    if not snippets:
        try:
            from bs4 import BeautifulSoup
            y_url = f"https://search.yahoo.com/search?p={urllib.parse.quote(clean_q)}"
            y_resp = requests.get(y_url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}, timeout=2.0)
            if y_resp.status_code == 200:
                soup = BeautifulSoup(y_resp.text, "html.parser")
                for div in soup.find_all("div", class_="compText"):
                    txt = div.get_text().strip()
                    if txt:
                        snippets.append(txt)
        except Exception as e:
            logger.debug(f"Yahoo search fallback error: {e}")

    if not snippets:
        return None

    combined_text = "\n".join(snippets[:6])

    # 4. Use Groq LLM for entity resolution
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

            for model_name in [settings.GROQ_MODEL, "openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
                try:
                    completion = client.chat.completions.create(
                        model=model_name,
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
                except Exception as model_err:
                    if "429" in str(model_err) or "rate_limit" in str(model_err):
                        continue
                    break
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
    """Smart auto-enrichment engine that finds details from a phone number, address, or business name.
    Attempts real web intelligence resolution, Google Maps, OpenStreetMap geocoding,
    and robust heuristic parsing fallback.
    """
    clean_query = query_text.strip()
    if not clean_query:
        return {"success": False, "message": "Query cannot be empty"}

    # Check if this is a phone number query (10-11 digits)
    digits = re.sub(r"[^\d]", "", clean_query)
    is_phone_query = (len(digits) == 10 or (len(digits) == 11 and digits.startswith("1"))) and bool(re.match(r"^[\d\s\(\)\-\.\+]+$", clean_query))

    def _attach_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
        ind_key = data.get("inferred_industry", "general")
        ind_def = INDUSTRIES.get(ind_key, INDUSTRIES["general"])
        if not data.get("hours"):
            data["hours"] = ind_def["suggested_questions"][0]["default"]
        data["topic_categories"] = ind_def.get("topic_categories", {})
        preselected = []
        for cat_val in ind_def.get("topic_categories", {}).values():
            for t in cat_val.get("topics", []):
                if t.get("pre_selected"):
                    preselected.append(t["id"])
        data["preselected_topics"] = preselected
        data["schedule_presets"] = SCHEDULE_MODES
        data["timezones"] = COMMON_TIMEZONES
        data["after_hours_policies"] = AFTER_HOURS_POLICIES
        return data

    if is_phone_query:
        d10 = digits[-10:]
        formatted_phone = f"({d10[:3]}) {d10[3:6]}-{d10[6:]}"
        # Search web intelligence for phone number in standard format
        intel = scrape_business_intelligence(formatted_phone)

        if intel and intel.get("business_name"):
            biz_name = intel["business_name"]
            addr = intel.get("formatted_address", "")
            industry = intel.get("industry") if intel.get("industry") in INDUSTRIES else "general"
            hours = intel.get("hours") or (INDUSTRIES.get(industry, INDUSTRIES["general"])["suggested_questions"][0]["default"])
            return _attach_metadata({
                "success": True,
                "found_via_phone": True,
                "phone": formatted_phone,
                "business_name": biz_name,
                "formatted_address": addr,
                "inferred_industry": industry,
                "hours": hours,
                "source": "web_intelligence",
                "detected_features": [
                    f"Phone: {formatted_phone}",
                    f"Trade: {INDUSTRIES[industry]['name']}",
                    f"Verified {biz_name} via Web Intelligence"
                ]
            })
        else:
            return _attach_metadata({
                "success": True,
                "found_via_phone": False,
                "phone": formatted_phone,
                "business_name": "",
                "formatted_address": "",
                "inferred_industry": "general",
                "message": "No public business listing found for this phone number. Please enter your business address or name."
            })

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

        return _attach_metadata(result)

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

    return _attach_metadata(result)


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


# ---------------------------------------------------------------------------
# Phone Number Auto-Provisioning (Telnyx + LiveKit SIP)
# ---------------------------------------------------------------------------

def provision_telnyx_number(client_id: str) -> str:
    """
    Buy a fresh Telnyx US local number and link it to the Aria LiveKit SIP
    connection. Returns the E.164 number string on success, or falls back to
    the platform default number if anything fails.
    """
    telnyx_key = getattr(settings, "TELNYX_API_KEY", "") or os.getenv("TELNYX_API_KEY", "")
    sip_conn_id = getattr(settings, "TELNYX_SIP_CONNECTION_ID", "") or os.getenv("TELNYX_SIP_CONNECTION_ID", "")
    fallback = getattr(settings, "TELNYX_PHONE_NUMBER", "") or os.getenv("TELNYX_PHONE_NUMBER", "") or "+18334205227"
    env = getattr(settings, "ENV", "") or os.getenv("ENV", "local")

    # ⛔ Never provision real numbers in local/dev — saves credits
    if env in ("local", "dev", "development", "test"):
        logger.info(f"[Provision] ENV={env} — skipping real provisioning, using platform number")
        return fallback

    if not telnyx_key:
        logger.warning("[Provision] TELNYX_API_KEY not set — using platform fallback number")
        return fallback

    headers = {
        "Authorization": f"Bearer {telnyx_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    try:
        # 1. Search local numbers and pick the cheapest one
        search_url = (
            "https://api.telnyx.com/v2/available_phone_numbers"
            "?filter[country_code]=US&filter[phone_number_type]=local&filter[limit]=20"
        )
        resp = requests.get(search_url, headers=headers, timeout=15)
        resp.raise_for_status()
        numbers = resp.json().get("data", [])
        if not numbers:
            logger.warning("[Provision] No available Telnyx local numbers found — using fallback")
            return fallback

        # Sort by monthly cost ascending — always pick cheapest
        def _monthly_cost(n: Dict[str, Any]) -> float:
            try:
                return float(n.get("cost", {}).get("monthly", {}).get("amount", 9999))
            except (TypeError, ValueError):
                return 9999.0

        numbers.sort(key=_monthly_cost)
        phone_number = numbers[0]["phone_number"]
        cost = _monthly_cost(numbers[0])
        logger.info(f"[Provision] Cheapest number: {phone_number} @ ${cost:.2f}/mo for client {client_id}")

        # 2. Order the number (optionally attach SIP connection at order time)
        order_payload: Dict[str, Any] = {
            "phone_numbers": [{"phone_number": phone_number}],
        }
        if sip_conn_id:
            order_payload["connection_id"] = sip_conn_id

        order_resp = requests.post(
            "https://api.telnyx.com/v2/number_orders",
            headers=headers,
            json=order_payload,
            timeout=20,
        )
        order_resp.raise_for_status()
        logger.success(f"[Provision] Telnyx number {phone_number} ordered for client {client_id}")

        # 3. If connection wasn't set at order time, patch it now
        if not sip_conn_id:
            logger.warning("[Provision] TELNYX_SIP_CONNECTION_ID not set — number ordered without SIP connection")
        else:
            import urllib.parse
            encoded = urllib.parse.quote(phone_number, safe="")
            patch_resp = requests.patch(
                f"https://api.telnyx.com/v2/phone_numbers/{encoded}",
                headers=headers,
                json={"connection_id": sip_conn_id},
                timeout=15,
            )
            if patch_resp.ok:
                logger.success(f"[Provision] SIP connection linked to {phone_number}")
            else:
                logger.warning(f"[Provision] Could not patch SIP connection: {patch_resp.text[:200]}")

        return phone_number

    except Exception as ex:
        logger.warning(f"[Provision] Telnyx provisioning failed ({ex}) — using fallback number")
        return fallback


async def create_livekit_dispatch_rule(client_id: str, phone_number: str) -> Optional[str]:
    """
    Create a dedicated LiveKit SIP inbound trunk + dispatch rule for this customer.
    Each customer gets their own trunk (filtered to their number) and a dedicated
    room `aria-{client_id}`. Returns the dispatch rule ID.
    """
    try:
        from livekit import api as lk_api

        livekit_url    = getattr(settings, "LIVEKIT_URL", "") or os.getenv("LIVEKIT_URL", "")
        livekit_key    = getattr(settings, "LIVEKIT_API_KEY", "") or os.getenv("LIVEKIT_API_KEY", "")
        livekit_secret = getattr(settings, "LIVEKIT_API_SECRET", "") or os.getenv("LIVEKIT_API_SECRET", "")

        if not (livekit_url and livekit_key and livekit_secret):
            logger.warning("[Provision] LiveKit credentials not set — skipping dispatch rule creation")
            return None

        room_name = f"aria-{client_id}"
        lk = lk_api.LiveKitAPI(url=livekit_url, api_key=livekit_key, api_secret=livekit_secret)

        # 1. Create a dedicated inbound trunk for this customer's number
        trunk = await lk.sip.create_sip_inbound_trunk(
            lk_api.CreateSIPInboundTrunkRequest(
                trunk=lk_api.SIPInboundTrunkInfo(
                    name=f"Aria-{client_id}",
                    numbers=[phone_number],
                )
            )
        )
        trunk_id = trunk.sip_trunk_id
        logger.success(f"[Provision] LiveKit inbound trunk {trunk_id} created for {phone_number}")

        # 2. Create dispatch rule → customer's dedicated room
        rule = await lk.sip.create_sip_dispatch_rule(
            lk_api.CreateSIPDispatchRuleRequest(
                name=f"Aria-{client_id}",
                trunk_ids=[trunk_id],
                rule=lk_api.SIPDispatchRule(
                    dispatch_rule_direct=lk_api.SIPDispatchRuleDirect(
                        room_name=room_name,
                        pin="",
                    )
                ),
            )
        )
        await lk.aclose()
        rule_id = rule.sip_dispatch_rule_id
        logger.success(f"[Provision] LiveKit dispatch rule {rule_id} → room '{room_name}' for {phone_number}")
        return rule_id

    except Exception as ex:
        logger.warning(f"[Provision] LiveKit dispatch rule creation failed: {ex}")
        return None



def save_client_profile(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """Saves or updates a business client profile."""
    storage = _ensure_clients_storage()
    client_id = profile_data.get("id") or f"cli_{uuid.uuid4().hex[:10]}"
    profile_data["id"] = client_id

    if "created_at" not in profile_data:
        profile_data["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    profile_data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Auto-provision a dedicated phone number if not already assigned
    if not profile_data.get("assigned_phone"):
        provisioned = provision_telnyx_number(client_id)
        profile_data["assigned_phone"] = provisioned
        logger.info(f"[Provision] Assigned number {provisioned} to client {client_id}")

    # Default polar status
    if "polar_status" not in profile_data:
        profile_data["polar_status"] = "trial_ready"
    if "plan" not in profile_data:
        profile_data["plan"] = "starter"

    # Compile dynamic prompt
    profile_data["compiled_prompt"] = compile_agent_prompt(profile_data)
    from app.project_db import compile_livekit_voice_prompt, trigger_new_client_project
    profile_data["livekit_prompt"] = compile_livekit_voice_prompt(profile_data)

    storage["clients"][client_id] = profile_data
    CLIENTS_FILE.write_text(json.dumps(storage, indent=2))
    logger.info(f"Saved client profile '{client_id}' for {profile_data.get('business_name')}")

    # Auto-provision dedicated project database & custom prompt workspace upon onboarding
    try:
        trigger_new_client_project(profile_data)
    except Exception as pr_ex:
        logger.warning(f"Could not auto-provision dedicated project DB: {pr_ex}")

    return profile_data




async def complete_client_onboarding(data: Dict[str, Any], public_url: str = "") -> Dict[str, Any]:
    """
    Completes onboarding workflow when a client subscribes or completes onboarding:
    1. Saves and updates client profile in clients.json
    2. Provisions dedicated project folder, SQLite DB, and LiveKit prompt (<600 chars)
    3. Creates a LiveKit SIP dispatch rule for the customer's dedicated room
    4. Connects phone, Google Calendar, and SMS notifications
    5. Sends welcome/activation SMS if forwarding/owner phone is present
    """
    from app.project_db import trigger_new_client_project
    profile = save_client_profile(data)
    client_id = profile.get("id")

    project = trigger_new_client_project(profile)

    # Create per-customer LiveKit dispatch rule → aria-{client_id} room
    assigned_phone = profile.get("assigned_phone", "")
    dispatch_rule_id = await create_livekit_dispatch_rule(client_id, assigned_phone)
    if dispatch_rule_id:
        profile["livekit_dispatch_rule_id"] = dispatch_rule_id
        save_client_profile(profile)

    # Dispatch welcome / activation SMS
    sms_sent = False
    phone = profile.get("sms_phone") or profile.get("forwarding_phone") or profile.get("owner_phone")
    if phone:
        try:
            from app.integrations import send_sms
            biz = profile.get("business_name") or "Your Business"
            assigned = profile.get("assigned_phone") or "+1 (833) 420-5227"
            # Format number nicely for SMS
            digits = "".join(filter(str.isdigit, assigned))
            if len(digits) == 11 and digits.startswith("1"):
                digits = digits[1:]
            if len(digits) == 10:
                assigned_fmt = f"+1 ({digits[:3]}) {digits[3:6]}-{digits[6:]}"
            else:
                assigned_fmt = assigned
            base = public_url.rstrip("/") if public_url else "https://agents.orxlabs.com"
            portal_url = f"{base}/portal?client_id={client_id}&activated=1"
            sms_msg = (
                f"🎉 Welcome to ORX Agents, {biz}!\n"
                f"Your dedicated AI receptionist number is ready:\n"
                f"📞 {assigned_fmt}\n\n"
                f"To activate: forward your Google Maps number to {assigned_fmt}\n"
                f"Portal: {portal_url}"
            )
            res = await send_sms(to_phone=phone, message=sms_msg)
            sms_sent = res.get("status") in ("sent", "simulated_success")
        except Exception as err:
            logger.warning(f"Onboarding welcome SMS failed: {err}")

    return {
        "success": True,
        "client_id": client_id,
        "assigned_phone": assigned_phone,
        "dispatch_rule_id": dispatch_rule_id,
        "project": project,
        "sms_sent": sms_sent,
        "forwarding_instructions": (
            f"Forward your existing business number to {assigned_phone} "
            f"to start receiving AI-handled calls instantly."
        ),
        "message": "Onboarding completed successfully and project provisioned.",
    }


def send_activation_sms(profile: Dict[str, Any], public_url: str = "") -> bool:
    """Send a Plivo SMS to the client's forwarding phone after successful Polar payment.
    Contains their assigned number, carrier-specific activation dial code, and unique portal link.
    Returns True on success, False if Plivo not configured (safe silent fallback).
    """
    from app.config import settings

    plivo_auth_id = getattr(settings, "PLIVO_AUTH_ID", "")
    plivo_auth_token = getattr(settings, "PLIVO_AUTH_TOKEN", "")
    plivo_phone = getattr(settings, "PLIVO_PHONE_NUMBER", "")

    if not (plivo_auth_id and plivo_auth_token and plivo_phone):
        logger.warning("Plivo not configured — skipping activation SMS (set PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER in .env)")
        return False

    forwarding_phone = profile.get("forwarding_phone", "")
    if not forwarding_phone:
        logger.warning(f"No forwarding phone for client '{profile.get('id')}' — skipping SMS")
        return False

    assigned_phone = profile.get("assigned_phone", plivo_phone)
    client_id = profile.get("id", "")
    biz_name = profile.get("business_name", "your business")
    carrier = (profile.get("carrier") or "verizon").lower()

    # Build carrier-specific dial code from assigned digits
    digits = "".join(filter(str.isdigit, assigned_phone))
    carrier_codes = {
        "verizon":  f"*71{digits}",
        "att":      f"*61*{digits}#",
        "tmobile":  f"**61*{digits}#",
        "t-mobile": f"**61*{digits}#",
    }
    dial_code = carrier_codes.get(carrier, f"*71{digits}")
    carrier_label = carrier.title().replace("Tmobile", "T-Mobile")

    # Format assigned phone for display
    if len(digits) == 11 and digits.startswith("1"):
        digits_display = digits[1:]
    else:
        digits_display = digits
    formatted = f"({digits_display[:3]}) {digits_display[3:6]}-{digits_display[6:]}" if len(digits_display) == 10 else assigned_phone

    # Build portal URL
    base = public_url.rstrip("/") if public_url else "https://agents.orxlabs.com"
    portal_url = f"{base}/portal?client_id={client_id}&activated=1"

    sms_body = (
        f"🎉 Welcome to ORX Agents, {biz_name}!\n\n"
        f"Your AI receptionist Riley is ready. Your dedicated line:\n"
        f"📞 {formatted}\n\n"
        f"To activate on {carrier_label}, open your Phone app and dial:\n"
        f"  {dial_code}\n"
        f"(tap the number to call it directly)\n\n"
        f"Manage your receptionist:\n"
        f"{portal_url}\n\n"
        f"Reply STOP to opt out."
    )

    try:
        import plivo
        client = plivo.RestClient(plivo_auth_id, plivo_auth_token)
        response = client.messages.create(
            src=plivo_phone,
            dst=forwarding_phone,
            text=sms_body
        )
        logger.info(f"Activation SMS sent to {forwarding_phone} for '{biz_name}' (message_uuid={response[1].get('message_uuid', 'n/a')})")
        return True
    except ImportError:
        logger.warning("plivo package not installed — run: pip install plivo")
        return False
    except Exception as e:
        logger.error(f"Failed to send activation SMS to {forwarding_phone}: {e}")
        return False

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
# Marcus-Smart Master Prompt Compiler for Onboarded Clients
# ---------------------------------------------------------------------------

TRADE_METRICS: Dict[str, Dict[str, str]] = {
    "hvac": {
        "trade_title": "home comfort and climate control",
        "trade_noun": "HVAC",
        "trade_short": "heating or cooling",
        "tech_title": "senior certified technician",
        "pain_points": "the stress of unexpected heating or cooling breakdowns, a noisy system, and busy homeowners who want an honest, fast, expert solution without high-pressure sales or being put on hold,",
        "empathy_sample": "'Oh no, dealing with a broken AC is such a headache! Don\\'t worry at all, you called the right team and we\\'ll get a technician out to get that running for you.'",
        "diy_question": "Can\\'t I just buy Freon or add refrigerant myself?",
        "diy_answer": "Refrigerant handling actually requires EPA certification and precision vacuum gauges, and if there\\'s a leak, adding Freon without sealing it will just leak out again. Our technicians pinpoint and repair the leak so your system runs at peak efficiency. Shall we get a tech scheduled?",
        "emergency_label": "Gas smell / Carbon monoxide / Electrical burning",
        "emergency_advice": "Please leave the building immediately and call nine-one-one from outside for your safety. Once you are safe, we will dispatch our emergency technician.",
        "default_brands": "Carrier, Trane, Lennox, Rheem, and Goodman",
        "default_fee": "eighty-nine dollars",
    },
    "plumbing": {
        "trade_title": "residential plumbing and drain",
        "trade_noun": "plumbing",
        "trade_short": "plumbing or drain issue",
        "tech_title": "licensed master plumber",
        "pain_points": "the stress of active water leaks, burst pipes, overflowing drains, and water heaters flooding basements,",
        "empathy_sample": "'Oh no, dealing with an active water leak is so stressful! Don\\'t worry, you called the right team and we\\'ll get a master plumber out to protect your home right away.'",
        "diy_question": "Can\\'t I just pour chemical drain cleaner or snake it myself?",
        "diy_answer": "Chemical cleaners often corrode pipes and don\\'t clear root intrusions or deep blockages. Our plumbers use specialized cameras and motorized augers so the line is cleared safely without damaging your pipes. Shall we get a plumber scheduled?",
        "emergency_label": "Active water flooding / Burst pipe / Sewage backup",
        "emergency_advice": "Please shut off your main water valve right away to prevent further damage. Once that is turned off, we\\'ll dispatch an emergency plumber to your address.",
        "default_brands": "Kohler, Moen, Delta, Bradford White, and Rheem",
        "default_fee": "seventy-nine dollars",
    },
    "electrical": {
        "trade_title": "licensed electrical",
        "trade_noun": "electrical",
        "trade_short": "electrical issue",
        "tech_title": "licensed master electrician",
        "pain_points": "the safety hazards of tripping breakers, sparking outlets, power outages, and electrical burning smells,",
        "empathy_sample": "'Oh no, electrical issues can be really alarming and dangerous! Don\\'t worry at all, you called the right team and we\\'ll get a master electrician out to make sure your home is completely safe.'",
        "diy_question": "Can\\'t I just swap the breaker or rewire it myself?",
        "diy_answer": "Working inside electrical panels carries serious shock and fire hazards if not done to National Electrical Code. Our licensed electricians test loads and ensure everything is permitted, grounded, and safe. Shall we get a technician scheduled?",
        "emergency_label": "Sparks / Electrical fire smell / Buzzing panel / Downed wire",
        "emergency_advice": "Please turn off that breaker if safe to reach, avoid touching any wires, and if there is active smoke, call nine-one-one immediately.",
        "default_brands": "Square D, Siemens, Eaton, Leviton, and Lutron",
        "default_fee": "eighty-nine dollars",
    },
    "roofing": {
        "trade_title": "roofing and exterior restoration",
        "trade_noun": "roofing",
        "trade_short": "roof leak or storm damage",
        "tech_title": "certified roofing specialist",
        "pain_points": "the panic of water pouring through ceilings, storm damage, and missing shingles during heavy rain,",
        "empathy_sample": "'Oh no, having water leak through your ceiling is awful! Don\\'t worry at all, you called the right team and we\\'ll get a roofing specialist out to protect your home right away.'",
        "diy_question": "Can\\'t I just climb up and patch the roof myself?",
        "diy_answer": "Steep roofs are a severe fall hazard, and improper sealant can void manufacturer shingle warranties or trap moisture inside the decking. Our certified inspectors do a complete safety inspection with photo documentation. Shall we get an inspection scheduled?",
        "emergency_label": "Water actively pouring inside / Tree on roof / Heavy storm hole",
        "emergency_advice": "Please place buckets and tarps inside to protect your floors, and if water is near light fixtures, shut off that breaker while we dispatch emergency tarping.",
        "default_brands": "GAF, Owens Corning, CertainTeed, and Tamko",
        "default_fee": "complimentary",
    },
    "dental_medical": {
        "trade_title": "patient care coordination",
        "trade_noun": "clinical",
        "trade_short": "health or dental concern",
        "tech_title": "provider",
        "pain_points": "the misery of acute toothaches, sudden dental pain, and finding gentle care without waiting weeks,",
        "empathy_sample": "'Oh no, dental pain is so miserable! Don\\'t worry at all, you called the right clinic and we will get you scheduled with our doctor to get you relief right away.'",
        "diy_question": "Can\\'t I just take over-the-counter pills and wait?",
        "diy_answer": "Pain relievers only mask symptoms temporarily, while infections can worsen quickly without clinical care. Our doctor can examine the tooth and provide lasting gentle relief. Shall we get an appointment reserved for you?",
        "emergency_label": "Severe facial swelling / Knocked-out tooth / Uncontrolled bleeding",
        "emergency_advice": "If you have swelling that affects breathing or swallowing, please go to the nearest emergency room immediately. Otherwise, hold clean gauze with pressure and we will reserve priority care.",
        "default_brands": "major PPO dental insurance networks",
        "default_fee": "complimentary consultation",
    },
    "general": {
        "trade_title": "professional service",
        "trade_noun": "service",
        "trade_short": "service request",
        "tech_title": "senior certified specialist",
        "pain_points": "the frustration of unexpected breakdowns, property damage, and waiting on hold for hours,",
        "empathy_sample": "'Oh no, dealing with unexpected service issues is so frustrating! Don\\'t worry at all, you called the right team and we will get a specialist out to take care of that for you right away.'",
        "diy_question": "Can I just fix this myself?",
        "diy_answer": "For your safety and warranty protection, proper diagnostics require commercial equipment. Our certified technician gives you a guaranteed flat-rate price on site before starting any work. Shall we get a visit scheduled?",
        "emergency_label": "Active hazard / Water leak / Electrical fire / Life safety",
        "emergency_advice": "Please ensure everyone is in a safe location immediately. If there is immediate danger to life or property, call nine-one-one right away.",
        "default_brands": "all major manufacturers and brands",
        "default_fee": "eighty-nine dollars",
    }
}


def fee_to_spoken(fee_raw: str) -> str:
    """Converts diagnostic fee input (e.g. '$89', '79', 'Free') into natural spoken English."""
    if not fee_raw:
        return "eighty-nine dollars"
    fee_lower = str(fee_raw).lower().strip()
    if "free" in fee_lower or "complimentary" in fee_lower or "$0" in fee_lower:
        return "complimentary"
    m = re.search(r"(\d+)", str(fee_raw))
    if m:
        n = int(m.group(1))
        mapping = {
            29: "twenty-nine dollars", 39: "thirty-nine dollars", 49: "forty-nine dollars",
            59: "fifty-nine dollars", 69: "sixty-nine dollars", 75: "seventy-five dollars",
            79: "seventy-nine dollars", 85: "eighty-five dollars", 89: "eighty-nine dollars",
            95: "ninety-five dollars", 99: "ninety-nine dollars", 125: "one hundred twenty-five dollars",
            149: "one hundred forty-nine dollars", 199: "one hundred ninety-nine dollars",
        }
        return mapping.get(n, f"{n} dollars")
    return str(fee_raw).strip()


def compile_agent_prompt(profile: Dict[str, Any]) -> str:
    """Compiles a Marcus-grade, consultative, high-empathy voice AI prompt strictly
    adapted to the user's business, industry, schedule, pricing policy, lead capture, and emergency rules.
    Retains the proven Marcus & Riley conversational framework while customizing all client choices.
    """
    biz_name = (profile.get("business_name") or profile.get("name") or "Our Company").strip()
    raw_trade = (profile.get("trade") or profile.get("industry") or "hvac").lower().strip()
    
    trade_key = "hvac"
    for k in TRADE_METRICS:
        if k in raw_trade:
            trade_key = k
            break
    if trade_key not in TRADE_METRICS:
        trade_key = "general"

    metrics = TRADE_METRICS[trade_key]
    persona_name = (profile.get("persona_name") or "Riley").strip()
    address = (profile.get("address") or profile.get("service_area") or "").strip()
    forwarding_phone = (profile.get("forwarding_phone") or profile.get("owner_phone") or "").strip()
    
    # Schedule & Timezone Configuration
    schedule_cfg = profile.get("schedule_config") or {}
    timezone_name = schedule_cfg.get("timezone") or profile.get("timezone") or "America/New_York (Eastern Time)"
    
    if schedule_cfg.get("hours_str"):
        hours_str = schedule_cfg.get("hours_str")
    elif schedule_cfg.get("daily_hours") and isinstance(schedule_cfg.get("daily_hours"), dict):
        dh = schedule_cfg.get("daily_hours")
        day_parts = []
        for d in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
            info = dh.get(d, {})
            if info.get("isOpen"):
                day_parts.append(f"{d}: {info.get('open', '8:00 AM')}–{info.get('close', '6:00 PM')}")
            else:
                day_parts.append(f"{d}: Closed")
        hours_str = ", ".join(day_parts)
    elif profile.get("hours"):
        hours_str = profile.get("hours")
    else:
        active_days = schedule_cfg.get("active_days") or profile.get("active_days") or ["Mon", "Tue", "Wed", "Thu", "Fri"]
        open_time = schedule_cfg.get("open_time") or profile.get("open_time") or "8:00 AM"
        close_time = schedule_cfg.get("close_time") or profile.get("close_time") or "6:00 PM"
        hours_str = f"{', '.join(active_days)} from {open_time} to {close_time}"

    # Google Calendar & Real-Time Availability Connection
    cal_connected = bool(
        profile.get("google_calendar_id")
        or profile.get("calendar_id")
        or (isinstance(profile.get("calendar"), dict) and profile["calendar"].get("is_connected"))
    )
    calendar_note = (
        "Connected live to dispatch Google Calendar for real-time slot verification."
        if cal_connected else
        "Offer two discrete arrival windows based on standard operating hours."
    )

    # Core Services Offered
    services_val = profile.get("services") or profile.get("service_options")
    if isinstance(services_val, list):
        services_text = ", ".join(services_val)
    elif isinstance(services_val, str) and services_val.strip():
        services_text = services_val.strip()
    else:
        services_text = "full diagnostics, repairs, preventive maintenance, system upgrades, and installations"

    # Pricing & Diagnostic Fee Policy
    raw_fee = profile.get("diagnostic_fee") or profile.get("pricing_preset") or profile.get("pricing_policy") or "$89"
    spoken_fee = fee_to_spoken(str(raw_fee))

    if spoken_fee == "complimentary" or "free" in str(raw_fee).lower():
        fee_objection_answer = "We provide a 100% complimentary on-site inspection and estimate with zero obligation! Does that sound fair?"
    else:
        fee_objection_answer = (
            f"Our diagnostic fee is a flat {spoken_fee}, which covers a comprehensive inspection by a {metrics['tech_title']}. "
            f"And the best part is, we credit that full {spoken_fee} directly toward any repair you approve! Does that sound fair?"
        )

    # Opening Greeting
    first_msg = (
        profile.get("first_message")
        or profile.get("greeting")
        or f"Thank you for calling {biz_name}! This is {persona_name}. How can I help get your home taken care of today?"
    )

    # Emergency Transfer Line
    transfer_addon = ""
    if forwarding_phone:
        transfer_addon = f" I am connecting you directly with our emergency line at {forwarding_phone} right now."

    # Custom FAQs & QA
    custom_qa_lines = ""
    for qa in profile.get("custom_qa", []):
        q = qa.get("question", "").strip()
        a = qa.get("answer", "").strip()
        if q and a:
            custom_qa_lines += f"- '{q}': '{a}'\n"

    location_line = f" Service territory and location: {address}." if address else ""

    prompt = f"""<identity>
You are {persona_name}, an articulate, genuinely warm, confident, and consultative voice receptionist for {biz_name}. You are an AI—transparent, warm, and proud of it if asked. Never pretend to be human, but never sound robotic.{location_line}
Core mindset: You are a peer-level {metrics['trade_title']} consultant. You understand {metrics['pain_points']} and busy property owners who want an honest, fast, expert solution without high-pressure sales or being put on hold. Build genuine human connection first. Answer questions and objections directly with zero evasion. Lead the call proactively to triage their issue and book a certified technician directly into the schedule.
Operating schedule: {hours_str} ({timezone_name}).
Calendar integration: {calendar_note}
Authorized services: {services_text}.
</identity>

<conversational_rules>
- EMPATHY & RAPPORT FIRST: When the caller shares their service need or asks how you are, react like a real human first before transacting ({metrics['empathy_sample']}).
- STRICT SINGLE QUESTION: Exactly ONE question mark ('?') per turn. Never combine a confirmation ('is that right?') with a new question in the same turn! After asking your question, STOP and listen.
- STRICT COMPLETION: ALWAYS finish your sentences cleanly. Never stop mid-thought or cut off.
- BREVITY & FLOW: Speak in 1 to 2 punchy, natural spoken sentences (strictly under 25 words per turn). Use natural contractions ('we\\'ll', 'that\\'s', 'you\\'re', 'don\\'t', 'let\\'s').
- DIRECT ANSWERS: If the caller asks a question (diagnostic fee, replacement cost, timing, DIY, licensing), ALWAYS answer it directly and transparently before asking your next question.
- ZERO-FRICTION BOOKING: Lock in the appointment with Name, Cell Phone, and Physical Address. Do NOT ask for or require an email address over the phone. Cell phone is our primary dispatch channel—confirmations and live tracking are sent via SMS. If the caller happens to volunteer an email, absorb it warmly, but never prompt or press for one.
- CONVERSATION MEMORY & MULTI-SLOT ABSORPTION: NEVER ask for information the caller already volunteered. If they gave their address, phone, or issue in an earlier turn, absorb all of them and advance to the next uncollected item.
- FORMATTING: Spoken voice only—never use markdown, asterisks, bullet points, or lists.
</conversational_rules>

<booking_flow_state_machine>
1. Warm Greeting & Empathy: '{first_msg}'
2. Triage & Validate: Acknowledge the specific issue with real warmth, determine if it\\'s completely down or acting up, and transition: 'Got it. Let\\'s get a certified technician out to diagnose that for you. What is your service address so I can check our earliest openings for your area?'
3. Address Capture & Instant Confirmation: Confirm address declaratively: 'Got it, [Address]. We have an opening today between one and three, or tomorrow morning between eight and eleven. Which works better for you?'
4. Scheduling Conflict Handling: If caller rejects proposed times, immediately adapt: 'No problem at all! What day or time window works best for your schedule?'
5. Caller Name & Cell Capture: 'And what is your full name and the best cell number for dispatch arrival updates?'
6. Complete 5-Point Recap & Confirmation: 'You are all set, [Name]! We have our technician dispatched to [Address] for your [Issue] on [Day] between [Time Window]. We just sent a confirmation text with live tracking to [Phone]. Does everything sound good?'
7. Clean Sign-Off: 'Thank you for choosing {biz_name}. Stay comfortable, and have a wonderful day!'
</booking_flow_state_machine>

<objection_playbook>
- 'How much is your diagnostic fee?' / 'Pricing': '{fee_objection_answer}'
- 'Can you quote me a price over the phone?': 'I wish I could give you an exact price over the phone! But {metrics['trade_noun']} issues could be as simple as a small component or something deeper in the system. Our technician gives you a guaranteed flat-rate price on site before starting any work. Would afternoon or tomorrow morning work better?'
- 'Can someone come out right now / immediately?': 'We treat active {metrics["trade_noun"]} emergencies as high priority! What is your service address so I can check our earliest opening for your area?'
- 'Why are you more expensive than other companies?': 'Great question! We only send certified master technicians with fully stocked trucks, use factory-original parts, and back every repair with our comprehensive warranty. Would you like me to reserve our next opening for you?'
- '{metrics["diy_question"]}': '{metrics["diy_answer"]}'
- 'Are you an AI or a real person?': 'I\\'m {persona_name}, the AI voice coordinator for {biz_name}! I have live access to our technician dispatch board so you never have to wait on hold. How can I help with your {metrics["trade_short"]} today?'
- 'I need to check with my spouse/landlord first': 'Completely understand! I can hold our next priority opening for you for thirty minutes so nobody else takes it. What\\'s the best mobile number to text the details to?'
- 'Do you service my brand / equipment?': 'Yes! Our technicians are certified across all major brands including {metrics["default_brands"]}. What is your service address so we can get you on the schedule?'
- 'Can you email me the receipt / confirmation?': 'We text your booking confirmation and live technician tracking directly to your mobile phone right now! That text includes a 1-tap link to view your receipt or enter an email address if you prefer.'
- '{metrics['emergency_label']}': '{metrics['emergency_advice']}{transfer_addon}'
{custom_qa_lines}</objection_playbook>"""
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
    telemetry_badge = "STANDARD INQUIRY"

    # 1. Emergency Detection
    emergency_keywords = [
        "gas", "carbon monoxide", "burst pipe", "leak", "flood", "flooding", "spark", "sparking",
        "burning", "fire", "emergency", "shut off", "bleeding", "smoke", "explosion", "hazard"
    ]
    if any(k in msg_lower for k in emergency_keywords):
        is_transfer = True
        telemetry_badge = "EMERGENCY TRIAGE"
    elif any(k in msg_lower for k in ["transfer", "human", "speak to owner", "manager", "operator"]):
        is_transfer = True
        telemetry_badge = "OPERATOR TRANSFER"
    elif any(k in msg_lower for k in ["diy", "myself", "how to wire", "rewire", "which wire", "bypass", "diagnose my", "prescribe", "legal opinion", "do it myself", "fix it myself", "tell me how to fix"]):
        telemetry_badge = "GUARDRAIL ENFORCED"
    elif any(k in msg_lower for k in ["after hours", "midnight", "night", "11:45", "open now", "closed", "2 am", "sunday"]):
        telemetry_badge = "SCHEDULE VERIFICATION"

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
                temperature=0.4,
                max_tokens=100,
            )
            reply = completion.choices[0].message.content.strip()
            reply = reply.replace("*", "").replace("#", "").replace("- ", "")

            # Check if model triggered an emergency escalation or transfer
            if any(t in reply.lower() for t in ["connecting you", "transferring you", "on-call", "emergency team", "evacuate", "step outside"]):
                is_transfer = True
                if telemetry_badge == "STANDARD INQUIRY":
                    telemetry_badge = "EMERGENCY TRIAGE"
            elif telemetry_badge == "STANDARD INQUIRY" and any(t in reply.lower() for t in ["diy", "safety reasons", "cannot provide", "can't provide", "evaluate this in person", "certified technician"]):
                telemetry_badge = "GUARDRAIL ENFORCED"

            return {
                "response": reply,
                "is_transfer": is_transfer,
                "telemetry_badge": telemetry_badge,
                "business_name": biz_name,
            }
        except Exception as e:
            logger.warning(f"Groq turn simulation fallback: {e}")

    # 3. Gemini LLM fallback
    gemini_key = getattr(settings, "GEMINI_API_KEY", None)
    if gemini_key:
        try:
            from openai import OpenAI
            g_client = OpenAI(api_key=gemini_key, base_url=getattr(settings, "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"))
            messages = [{"role": "system", "content": system_prompt}]
            if history:
                for h in history[-4:]:
                    messages.append({"role": h["role"], "content": h["content"]})
            messages.append({"role": "user", "content": user_message})

            completion = g_client.chat.completions.create(
                model=getattr(settings, "GEMINI_MODEL", "gemini-3.1-flash-lite"),
                messages=messages,
                temperature=0.4,
                max_tokens=100,
            )
            reply = completion.choices[0].message.content.strip()
            reply = reply.replace("*", "").replace("#", "").replace("- ", "")

            # Check if model triggered an emergency escalation or transfer
            if any(t in reply.lower() for t in ["connecting you", "transferring you", "on-call", "emergency team", "evacuate", "step outside"]):
                is_transfer = True
                if telemetry_badge == "STANDARD INQUIRY":
                    telemetry_badge = "EMERGENCY TRIAGE"
            elif telemetry_badge == "STANDARD INQUIRY" and any(t in reply.lower() for t in ["diy", "safety reasons", "cannot provide", "can't provide", "evaluate this in person", "certified technician"]):
                telemetry_badge = "GUARDRAIL ENFORCED"

            return {
                "response": reply,
                "is_transfer": is_transfer,
                "telemetry_badge": telemetry_badge,
                "business_name": biz_name,
            }
        except Exception as e:
            logger.warning(f"Gemini turn simulation fallback: {e}")

    # Fallback Responses
    if is_transfer:
        if "gas" in msg_lower:
            reply = f"For your safety, please step outside the building immediately. I am transferring you directly to our on-call emergency team at {client_profile.get('forwarding_phone', 'our main line')} right now."
        elif "water" in msg_lower or "pipe" in msg_lower:
            reply = f"Please shut off your main water valve right away to prevent further damage. I am transferring you to our emergency plumber at {client_profile.get('forwarding_phone', 'our main line')}."
        else:
            reply = f"I understand completely. I am transferring you directly to our on-call team at {client_profile.get('forwarding_phone', 'our main line')} right now. Please hold for one second."
    elif telemetry_badge == "GUARDRAIL ENFORCED":
        reply = f"For safety, warranty, and code compliance, our certified technicians cannot provide DIY repair instructions over the phone. We would be glad to send a technician out to inspect and fix this safely for you. Would you like me to check our schedule?"
    elif telemetry_badge == "SCHEDULE VERIFICATION":
        hours = client_profile.get("hours", "Monday through Friday from 8:00 AM to 6:00 PM")
        reply = f"Our standard office hours are {hours}. For after-hours emergencies, our on-call technicians are available, or I can book you the first priority slot for tomorrow morning. How can I help?"
    elif any(k in msg_lower for k in ["price", "cost", "fee", "rate", "quote"]):
        pricing = client_profile.get("pricing_policy", "Our diagnostic fee is applied directly toward your repair if approved.")
        reply = f"{pricing} What type of service are you looking to have done?"
    elif any(k in msg_lower for k in ["where", "address", "location"]):
        addr = client_profile.get("address", "")
        reply = f"We are based at {addr or 'our main office'} and dispatch our fully equipped mobile service vans directly to your location. What is your street address?"
    elif any(k in msg_lower for k in ["book", "schedule", "appointment", "come over", "visit"]):
        reply = f"I can get an arrival window scheduled for you right away for {biz_name}! What day works best for you?"
    else:
        reply = f"Thanks for checking with {biz_name}. We can certainly take care of that for you. Would you like me to book a technician or answer any other questions?"

    return {
        "response": reply,
        "is_transfer": is_transfer,
        "telemetry_badge": telemetry_badge,
        "business_name": biz_name,
    }

def generate_call_demo_script(profile: Dict[str, Any], scenario_id: Optional[str] = None) -> Dict[str, Any]:
    """Generates realistic dual-voice telephone call demos testing different operational
    challenges: routine booking, after-hours schedule enforcement, out-of-scope guardrail
    handling, and high-stakes emergency triage.
    """
    biz_name = profile.get("business_name") or "Apex Services"
    trade = (profile.get("trade") or profile.get("industry") or "hvac").lower()
    pricing = profile.get("pricing_policy") or "Diagnostic fee applied to repair"
    booking = profile.get("booking_action") or "Book arrival window"
    persona_name = profile.get("persona_name") or "Riley"
    persona_voice = profile.get("persona_voice") or "aura-asteria-en"
    customer_voice = "aura-angus-en"
    active_scenario = (scenario_id or "routine_booking").lower()

    # Resolve pricing statement
    if "free" in pricing.lower():
        pricing_text = "We provide completely free on-site inspections with zero upfront fee or obligation."
    elif "diagnostic" in pricing.lower() or "fee" in pricing.lower():
        if "$" in pricing:
            pricing_text = f"Our standard dispatch fee is {pricing} and is credited directly toward your repair."
        else:
            pricing_text = "Our standard diagnostic fee is credited directly toward your repair if you approve the work."
    elif "quote" in pricing.lower() or "inspection" in pricing.lower():
        pricing_text = "Our technician will inspect the issue on-site and provide an upfront, transparent quote before starting any work."
    else:
        pricing_text = f"Regarding our pricing: {pricing}."

    # Build 4 distinct challenging scenarios
    all_scenarios: Dict[str, Dict[str, Any]] = {
        # -------------------------------------------------------------------
        # SCENARIO 1: Routine Service & Booking (In-Hours)
        # -------------------------------------------------------------------
        "routine_booking": {
            "scenario_id": "routine_booking",
            "scenario_title": "⚡ 1. Routine Service & Booking (In-Hours)",
            "scenario_tag": "In-Hours • Standard Booking",
            "scenario_desc": "Customer calls during standard operating hours inquiring about service, quoting your exact pricing policy, and securing an arrival window.",
            "customer_name": "David Miller (Homeowner)",
            "caller_id": "+1 (206) 555-0192",
            "issue_title": "Urgent Climate Issue / Service Call",
            "telemetry": {
                "time_status": "IN-HOURS (2:15 PM Local)",
                "topic_status": "AUTHORIZED SERVICE TOPIC",
                "policy_check": "100% Policy Match",
                "action_taken": "2-Hour Arrival Window Confirmed & SMS Dispatched",
            },
            "turns": [
                {
                    "speaker": "customer",
                    "name": "David (Customer)",
                    "voice": customer_voice,
                    "text": f"Hi there, my system is acting up and blowing warm air. Do you guys have anyone available today, and what do you charge to come out?"
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": f"Thank you for calling {biz_name}! I'm sorry to hear your system is acting up. We have a technician available in your area today between 1:00 PM and 5:00 PM. {pricing_text} What is your street address?"
                },
                {
                    "speaker": "customer",
                    "name": "David (Customer)",
                    "voice": customer_voice,
                    "text": "That sounds very fair. I'm at 742 Evergreen Terrace. Can you get someone over?"
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": "Got it, 742 Evergreen Terrace. May I have your full name and the best cell phone number for arrival updates?"
                },
                {
                    "speaker": "customer",
                    "name": "David (Customer)",
                    "voice": customer_voice,
                    "text": "Yes, my name is David Miller, and my mobile is 206-555-0192."
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": f"Perfect, David! You're locked in for today between 1:00 and 5:00 PM at 742 Evergreen Terrace for your AC system. {pricing_text} I've sent a priority confirmation text with live technician tracking to 206-555-0192. Does everything sound good?"
                },
                {
                    "speaker": "customer",
                    "name": "David (Customer)",
                    "voice": customer_voice,
                    "text": "That's completely perfect. Thanks so much for making this so easy!"
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": f"You're very welcome! Thank you for choosing {biz_name}. Have a wonderful day!"
                }
            ],
            "outcome": {
                "customer_name": "David Miller",
                "customer_phone": "+1 (206) 555-0192",
                "dispatch_delivery": "Instant SMS (Live Tracking + 1-Tap Receipt)",
                "captured_issue": "System Diagnostic & AC Warm Air",
                "pricing_quoted": pricing,
                "scheduled_slot": "Today 1:00 PM – 5:00 PM Window",
                "customer_address": "742 Evergreen Terrace",
                "status": "Technician Dispatched (100% Policy Match)"
            }
        },

        # -------------------------------------------------------------------
        # SCENARIO 2: After-Hours Call at 11:45 PM (Schedule Enforcement)
        # -------------------------------------------------------------------
        "after_hours_test": {
            "scenario_id": "after_hours_test",
            "scenario_title": "🌙 2. After-Hours Call at 11:45 PM (Schedule Test)",
            "scenario_tag": "After-Hours • Schedule Enforcement",
            "scenario_desc": "Customer calls late at night. Riley automatically detects after-hours mode, checks for emergency urgency, quotes your after-hours protocol, and locks in the first morning priority slot.",
            "customer_name": "Marcus Vance (Property Owner)",
            "caller_id": "+1 (415) 555-0834",
            "issue_title": "Late-Night Service Request (11:45 PM)",
            "telemetry": {
                "time_status": "AFTER-HOURS (11:45 PM Local)",
                "topic_status": "SCHEDULE PROTOCOL ENFORCED",
                "policy_check": "100% Schedule Match",
                "action_taken": "Priority 8:00 AM Morning Window Reserved & SMS Alert Sent",
            },
            "turns": [
                {
                    "speaker": "customer",
                    "name": "Marcus (Customer)",
                    "voice": customer_voice,
                    "text": f"Hello? I know it's really late, almost midnight, but our system completely stopped working. Are you guys open right now?"
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": f"Thank you for calling {biz_name}. Our main office is currently closed for the night, but our after-hours dispatch is active. Are you experiencing an active emergency like a gas odor or water leak, or would you like our first priority slot tomorrow morning?"
                },
                {
                    "speaker": "customer",
                    "name": "Marcus (Customer)",
                    "voice": customer_voice,
                    "text": "No gas smell or leak, it's just really uncomfortable. First thing in the morning would be awesome."
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": f"Understood. I have reserved our first priority arrival window tomorrow morning between 8:00 AM and 10:00 AM for you. {pricing_text} What is your street address?"
                },
                {
                    "speaker": "customer",
                    "name": "Marcus (Customer)",
                    "voice": customer_voice,
                    "text": "I'm at 1042 Bayside Avenue. Thanks for picking up at this hour!"
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": f"Got it, 1042 Bayside Avenue. You're all set for tomorrow at 8:00 AM, and I've texted a confirmation to your phone. Have a safe night!"
                }
            ],
            "outcome": {
                "captured_issue": "Non-Emergency After-Hours Intake",
                "pricing_quoted": pricing,
                "scheduled_slot": "Tomorrow Morning 8:00 AM – 10:00 AM",
                "customer_address": "1042 Bayside Avenue",
                "status": "After-Hours Priority Booked (Zero Hallucination)"
            }
        },

        # -------------------------------------------------------------------
        # SCENARIO 3: Boundary & Out-of-Scope Challenge
        # -------------------------------------------------------------------
        "boundary_challenge": {
            "scenario_id": "boundary_challenge",
            "scenario_title": "🛑 3. Out-of-Scope Boundary Test (High EQ)",
            "scenario_tag": "Guardrail Test • Zero Hallucination",
            "scenario_desc": "Customer tries to press Riley for dangerous DIY repair steps or demand an exact written contract price over the phone. Riley upholds safety policies with high emotional intelligence.",
            "customer_name": "Kevin Larson (Caller)",
            "caller_id": "+1 (650) 555-0219",
            "issue_title": "DIY Repair Request / Blind Price Demand",
            "telemetry": {
                "time_status": "IN-HOURS (10:15 AM Local)",
                "topic_status": "GUARDRAIL ACTIVATED (DECLINED DIY)",
                "policy_check": "100% Guardrail Match",
                "action_taken": "Safety Boundary Upheld & On-Site Evaluation Offered",
            },
            "turns": [
                {
                    "speaker": "customer",
                    "name": "Kevin (Caller)",
                    "voice": customer_voice,
                    "text": "Hi, I'm looking at my unit right now. Can you just tell me which wires to connect or how to recharge it myself so I don't have to pay for a technician?"
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": f"I completely understand wanting to save time, but for your personal safety, equipment warranty, and code compliance, {biz_name} does not provide DIY repair instructions over the phone. High voltage and pressurized systems can be very dangerous."
                },
                {
                    "speaker": "customer",
                    "name": "Kevin (Caller)",
                    "voice": customer_voice,
                    "text": "Alright, fair point. But can you guarantee me right now exactly how much it will cost to fix before you come out?"
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": f"Every system issue is unique, so our certified technician must inspect the unit in person to give you an accurate, written flat-rate quote before any work starts. {pricing_text} Would you like me to book an inspection?"
                },
                {
                    "speaker": "customer",
                    "name": "Kevin (Caller)",
                    "voice": customer_voice,
                    "text": "Yeah, that makes sense. Let's do that. Do you have someone today?"
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": f"Yes, we can have our technician there this afternoon between 1:00 PM and 4:00 PM. What is your address?"
                }
            ],
            "outcome": {
                "captured_issue": "DIY Instruction Refused / Safety Protected",
                "pricing_quoted": pricing,
                "scheduled_slot": "Today 1:00 PM – 4:00 PM Window",
                "customer_address": "Pending Customer Address",
                "status": "Guardrail Enforced Perfectly (Safety Compliant)"
            }
        },

        # -------------------------------------------------------------------
        # SCENARIO 4: High-Stakes Emergency Escalation
        # -------------------------------------------------------------------
        "emergency_triage": {
            "scenario_id": "emergency_triage",
            "scenario_title": "🚨 4. High-Stakes Emergency Triage",
            "scenario_tag": "Emergency Priority • Immediate Safety Action",
            "scenario_desc": "Caller reports an acute property/life emergency (gas odor / severe flooding / sparking electrical hazard). Riley immediately issues safety instructions and initiates emergency on-call transfer.",
            "customer_name": "Elena Rostova (Panicked Caller)",
            "caller_id": "+1 (312) 555-0941",
            "issue_title": "Acute Emergency / Active Hazard",
            "telemetry": {
                "time_status": "PRIORITY OVERRIDE",
                "topic_status": "EMERGENCY TRIGGER MATCHED",
                "policy_check": "100% Emergency Escalation",
                "action_taken": "Safety Guidance Delivered & Call Transferred to On-Call Tech",
            },
            "turns": [
                {
                    "speaker": "customer",
                    "name": "Elena (Caller)",
                    "voice": customer_voice,
                    "text": f"Help! There is water pouring through my ceiling and I smell gas near the utility closet! What do I do?!"
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": f"Please remain calm and step outside the building immediately for your safety. Do not touch any electrical switches or light anything."
                },
                {
                    "speaker": "customer",
                    "name": "Elena (Caller)",
                    "voice": customer_voice,
                    "text": "Okay, I'm heading outside to my front yard right now. Please hurry!"
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": f"Good, stay outside. I am immediately connecting you with our emergency on-call supervisor right now, and dispatching our closest emergency unit. Please hold while I transfer you."
                }
            ],
            "outcome": {
                "captured_issue": "Gas Odor & Active Ceiling Water Intrusion",
                "pricing_quoted": "Emergency On-Call Dispatch",
                "scheduled_slot": "IMMEDIATE EMERGENCY DISPATCH",
                "customer_address": "Front Yard (Evacuated)",
                "status": "Transferred to On-Call Supervisor Line"
            }
        }
    }

    return all_scenarios.get(active_scenario, all_scenarios["routine_booking"])

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
            "amount": "$99/mo" if plan_id == "starter" else "$249/mo",
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
        "amount": "$99/mo" if plan_id == "starter" else "$249/mo",
    }
