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
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple
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

    # Normalize Schedule, Timezone & Night-Call configuration
    sched_cfg = profile_data.get("schedule_config") or {}
    if sched_cfg and isinstance(sched_cfg, dict):
        if not profile_data.get("timezone") and sched_cfg.get("timezone"):
            profile_data["timezone"] = sched_cfg["timezone"]
        if not profile_data.get("after_hours_action"):
            profile_data["after_hours_action"] = sched_cfg.get("afterHoursPolicy") or sched_cfg.get("after_hours_policy") or "book_morning"
        if not profile_data.get("night_action"):
            profile_data["night_action"] = profile_data["after_hours_action"]
        raw_cov = str(sched_cfg.get("mode") or sched_cfg.get("answering_coverage") or profile_data.get("answering_coverage") or "always_24_7").lower()
        if "after" in raw_cov:
            profile_data["answering_coverage"] = "after_hours"
        elif "overflow" in raw_cov:
            profile_data["answering_coverage"] = "overflow"
        else:
            profile_data["answering_coverage"] = "always_24_7"

        raw_delay = sched_cfg.get("overflowDelaySecs")
        if raw_delay is None:
            raw_delay = sched_cfg.get("overflow_delay_secs")
        if raw_delay is None:
            raw_delay = profile_data.get("overflow_delay_seconds")
        if raw_delay is None:
            raw_delay = 15
        try:
            profile_data["overflow_delay_seconds"] = int(raw_delay)
        except Exception:
            profile_data["overflow_delay_seconds"] = 15

        if not profile_data.get("hours") and (sched_cfg.get("hours_str") or sched_cfg.get("hoursStr")):
            profile_data["hours"] = sched_cfg.get("hours_str") or sched_cfg.get("hoursStr")

    # Global normalization fallback
    raw_cov = str(profile_data.get("answering_coverage") or "always_24_7").lower()
    if "after" in raw_cov:
        profile_data["answering_coverage"] = "after_hours"
    elif "overflow" in raw_cov:
        profile_data["answering_coverage"] = "overflow"
    else:
        profile_data["answering_coverage"] = "always_24_7"

    global_delay = profile_data.get("overflow_delay_seconds")
    if global_delay is None:
        global_delay = 15
    try:
        profile_data["overflow_delay_seconds"] = int(global_delay)
    except Exception:
        profile_data["overflow_delay_seconds"] = 15

    if profile_data.get("timezone"):
        profile_data["timezone"] = clean_timezone(profile_data["timezone"])

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
            sms_res = await send_activation_sms(profile, public_url=public_url)
            sms_sent = sms_res.get("status") in ("sent", "simulated_success") or sms_res.get("success", False)
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


def build_activation_sms_message(profile: Dict[str, Any], public_url: str = "") -> str:
    """Builds an actionable 1-tap activation SMS containing:
    1. Dedicated AI receptionist phone number
    2. Exact carrier dialing codes (*71, *61*, **61*, *72) with clickable tel: dialer links
    3. Direct 1-tap connect link to open dialer directly from mobile SMS
    4. Explicit note to link Google Calendar for live arrival window conflict prevention
    5. Portal management link
    """
    assigned_phone = profile.get("assigned_phone") or "+1 (833) 420-5227"
    client_id = profile.get("id") or profile.get("client_id") or ""
    biz_name = profile.get("business_name") or "your business"
    carrier = (profile.get("carrier") or "verizon").lower()

    digits = "".join(filter(str.isdigit, assigned_phone))
    if len(digits) == 11 and digits.startswith("1"):
        digits_10 = digits[1:]
    elif len(digits) == 10:
        digits_10 = digits
    else:
        digits_10 = digits

    formatted = f"({digits_10[:3]}) {digits_10[3:6]}-{digits_10[6:]}" if len(digits_10) == 10 else assigned_phone

    carrier_codes = {
        "verizon": f"*71{digits_10}",
        "att": f"*61*{digits_10}#",
        "tmobile": f"**61*{digits_10}#",
        "t-mobile": f"**61*{digits_10}#",
        "landline": f"*72{digits_10}",
        "other": f"*72{digits_10}",
    }
    primary_code = carrier_codes.get(carrier, f"*71{digits_10}")
    primary_code_encoded = primary_code.replace("#", "%23")
    primary_label = carrier.title().replace("Tmobile", "T-Mobile")

    raw_base = public_url or getattr(settings, "PUBLIC_URL", "") or "https://agents.orxlabs.com"
    base = raw_base.strip().rstrip("/")
    connect_url = f"{base}/portal?client_id={client_id}&connect=1&carrier={carrier}"
    portal_url = f"{base}/portal?client_id={client_id}&activated=1"

    sms_body = (
        f"🎉 Welcome to ORX Agents, {biz_name}!\n\n"
        f"Your AI receptionist Riley is live on your dedicated line:\n"
        f"📞 {formatted}\n\n"
        f"👉 CLICK TO CONNECT NOW (1-Tap Call):\n"
        f"Tap the number below for your mobile carrier. It opens your phone dialer directly—just press Call! (Your cell rings first; Riley only answers missed calls):\n"
        f"• {primary_label}: tel:{primary_code_encoded} (or dial {primary_code})\n"
        f"• Other carriers: tel:*71{digits_10} (Verizon) | tel:*61*{digits_10}%23 (AT&T) | tel:**61*{digits_10}%23 (T-Mobile)\n\n"
        f"📲 Or click this link to connect directly from SMS:\n"
        f"{connect_url}\n\n"
        f"📅 GOOGLE CALENDAR:\n"
        f"Please connect your Google Calendar in your portal (takes 10 secs with zero passwords) so Riley checks live arrival windows and prevents double-booking!\n\n"
        f"Access Your Portal:\n{portal_url}\n\n"
        f"Reply STOP to opt out."
    )
    return sms_body


async def send_activation_sms(profile: Dict[str, Any], public_url: str = "") -> Dict[str, Any]:
    """Send an activation SMS to the client's SMS/forwarding phone after successful payment.
    Contains their assigned number, carrier-specific activation dial code, clickable dialer link,
    direct SMS connect link, and Google Calendar sync instructions.
    Uses app.integrations.send_sms with multi-provider (Twilio, Plivo, Telnyx) and simulated fallback.
    """
    from app.integrations import send_sms

    sms_target = profile.get("sms_phone") or profile.get("forwarding_phone") or profile.get("owner_phone") or ""
    if not sms_target:
        logger.warning(f"No SMS or forwarding phone for client '{profile.get('id')}' — skipping SMS")
        return {"status": "skipped", "error": "No recipient phone number on file"}

    sms_body = build_activation_sms_message(profile, public_url=public_url)
    res = await send_sms(to_phone=sms_target, message=sms_body)
    logger.info(f"Activation SMS dispatched to {sms_target} for '{profile.get('business_name')}': {res.get('status')}")
    return res


def send_activation_sms_sync(profile: Dict[str, Any], public_url: str = "") -> Dict[str, Any]:
    """Synchronous execution wrapper for send_activation_sms."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, send_activation_sms(profile, public_url)).result()
        else:
            return asyncio.run(send_activation_sms(profile, public_url))
    except Exception as e:
        logger.error(f"Error in send_activation_sms_sync: {e}")
        return {"status": "error", "error": str(e)}

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
            "sms_phone": "+1 (555) 234-5678",
            "hours": "Mon-Fri 7:30 AM to 6:00 PM, 24/7 Emergency Dispatch",
            "transfer_rules": "Transfer immediately for gas smell, water leak, or caller asking for owner.",
            "emergency_triggers": "Gas smell, water leak, carbon monoxide, furnace breakdown",
            "after_hours_action": "book_morning",
            "pricing_policy": "$89 diagnostic fee credited toward repair",
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


def get_client_by_assigned_phone(phone_number: str) -> Optional[Dict[str, Any]]:
    """Finds client profile by assigned_phone, forwarding_phone, owner_phone, or sms_phone digits."""
    if not phone_number:
        return None
    digits = "".join(filter(str.isdigit, str(phone_number)))
    if not digits:
        return None
    digits_10 = digits[-10:] if len(digits) >= 10 else digits

    storage = _ensure_clients_storage()
    for cid, profile in storage.get("clients", {}).items():
        for field in ("assigned_phone", "forwarding_phone", "owner_phone", "sms_phone", "phone"):
            val = profile.get(field) or ""
            c_digits = "".join(filter(str.isdigit, str(val)))
            c_10 = c_digits[-10:] if len(c_digits) >= 10 else c_digits
            if c_10 and c_10 == digits_10:
                return profile

    # Check project directories as fallback
    try:
        from app.project_db import list_projects, get_project
        for proj in list_projects():
            meta = proj.get("meta", {})
            for field in ("assigned_phone", "forwarding_phone", "owner_phone", "sms_phone"):
                val = meta.get(field) or ""
                c_digits = "".join(filter(str.isdigit, str(val)))
                c_10 = c_digits[-10:] if len(c_digits) >= 10 else c_digits
                if c_10 and c_10 == digits_10:
                    cid = proj.get("id") or proj.get("client_id")
                    p = get_client_profile(cid)
                    if p:
                        return p
                    return {**meta, "id": cid, "client_id": cid}
    except Exception:
        pass
    return None


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
    profile["forwarding_setup_at"] = now_str
    save_client_profile(profile)

    # Sync to project DB
    try:
        from app.project_db import create_project
        create_project(profile, trigger_source="carrier_verified")
    except Exception as ex:
        logger.warning(f"Project DB sync error during carrier verification: {ex}")

    logger.info(f"Verified connection for client '{profile.get('id')}' ({profile.get('business_name')}) via {carrier} at {now_str}")
    return {
        "success": True,
        "connection_status": "verified",
        "carrier": carrier.capitalize(),
        "verified_at": now_str,
        "assigned_phone": profile.get("assigned_phone"),
        "forwarding_phone": profile.get("forwarding_phone"),
        "sms_phone": profile.get("sms_phone") or profile.get("forwarding_phone"),
        "business_name": profile.get("business_name"),
    }


def disconnect_client_connection(client_id: Optional[str] = None) -> Dict[str, Any]:
    """Pauses or disconnects call forwarding for client."""
    profile = get_client_profile(client_id) if client_id else get_latest_client_profile()
    if not profile:
        return {"success": False, "message": "Client not found"}

    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    profile["connection_status"] = "paused"
    profile["disconnected_at"] = now_str
    save_client_profile(profile)

    try:
        from app.project_db import create_project
        create_project(profile, trigger_source="carrier_paused")
    except Exception as ex:
        logger.warning(f"Project DB sync error during disconnect: {ex}")

    logger.info(f"Paused connection for client '{profile.get('id')}' at {now_str}")
    return {"success": True, "connection_status": "paused", "disconnected_at": now_str}

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
    "auto": {
        "trade_title": "automotive repair and maintenance",
        "trade_noun": "auto repair",
        "trade_short": "vehicle issue",
        "tech_title": "ASE-certified master mechanic",
        "pain_points": "the stress of check engine lights, strange grinding noises, brake failures, and being stranded without transportation,",
        "empathy_sample": "'Oh no, car trouble is so stressful and inconvenient! Don\\'t worry at all, you called the right shop and we will get our master mechanic to inspect your vehicle and get you back on the road safely.'",
        "diy_question": "Can\\'t I just clear the check engine code myself or replace the sensor?",
        "diy_answer": "Clearing the code only turns off the dash light without fixing the underlying fuel, ignition, or emissions fault, which can ruin your catalytic converter. Our digital scan pinpoints the root cause so you don\\'t spend money guessing on parts. Shall we get your vehicle checked in?",
        "emergency_label": "Vehicle stalled in traffic / Brake failure / Tow truck arriving / Steam from hood",
        "emergency_advice": "Please pull completely over to the shoulder, turn on your hazard lights, and stay inside the vehicle away from traffic while we coordinate with the tow driver.",
        "default_brands": "Ford, Toyota, Honda, Chevrolet, BMW, and all domestic and import makes",
        "default_fee": "eighty-nine dollars",
    },
    "legal": {
        "trade_title": "legal counsel and client intake",
        "trade_noun": "legal matter",
        "trade_short": "legal matter",
        "tech_title": "attorney",
        "pain_points": "the anxiety of facing an unexpected accident, legal dispute, court deadline, or family matter without clear guidance,",
        "empathy_sample": "'I understand this is a very stressful and overwhelming situation for you. You called the right firm, and our attorneys are here to protect your rights and guide you through every step.'",
        "diy_question": "Can\\'t I just use an online legal template or handle this myself?",
        "diy_answer": "Generic online forms often omit mandatory statutory clauses and local court filings, which can result in dismissed claims or severe liability. Our attorneys tailor every filing specifically to your jurisdiction. Would you like to schedule a confidential consultation?",
        "emergency_label": "Arrest or police custody / Impending 24-hour court deadline / Opposing counsel calling",
        "emergency_advice": "Please remember that you have the right to remain silent and to speak with an attorney before answering questions. We are escalating your details to our on-call counsel right now.",
        "default_brands": "state bar certified attorneys and legal advisors",
        "default_fee": "complimentary consultation",
    },
    "restaurant": {
        "trade_title": "dining and hospitality reservations",
        "trade_noun": "dining",
        "trade_short": "table reservation",
        "tech_title": "hospitality manager",
        "pain_points": "the frustration of securing prime dining times, dietary accommodations, and organizing group dinners,",
        "empathy_sample": "'We would love to host you and your guests! You\\'re going to have an incredible dining experience with us.'",
        "diy_question": "Can we just show up without a reservation?",
        "diy_answer": "Walk-ins are welcomed on a first-come, first-served basis at our bar area, but tables fill quickly during peak hours. Locking in your reservation ensures your table is ready the moment you arrive. Shall we hold a table for you?",
        "emergency_label": "Severe food allergy alert / Large event buyout / VIP guest arrival",
        "emergency_advice": "Please inform your server the moment you arrive so our executive chef can prepare your meal with dedicated cookware.",
        "default_brands": "farm-to-table seasonal dining",
        "default_fee": "complimentary reservation",
    },
    "realestate": {
        "trade_title": "real estate advisory and client representation",
        "trade_noun": "real estate",
        "trade_short": "property inquiry",
        "tech_title": "licensed real estate agent",
        "pain_points": "the stress of missing out on hot listings, navigating mortgage approvals, and scheduling private property tours on short notice,",
        "empathy_sample": "'Buying or selling a home is such an exciting milestone, but it can definitely feel overwhelming! You called the right team, and our licensed agents will help you find the perfect property.'",
        "diy_question": "Can\\'t I just tour the home directly with the seller or go to an open house?",
        "diy_answer": "Listing agents represent the seller\\'s financial interest, not yours. Having our dedicated buyer\\'s agent protects your negotiating power, verifies title disclosures, and costs you zero out-of-pocket fees. Shall we set up a private showing?",
        "emergency_label": "Time-sensitive written offer deadline / Pre-approved buyer ready / Seller listing consultation",
        "emergency_advice": "We are flagging your offer deadline directly to our senior broker so your contract can be drafted and submitted before the deadline.",
        "default_brands": "MLS, Zillow Premier, Realtor.com, and local board of realtors",
        "default_fee": "complimentary consultation",
    },
    "salon_spa": {
        "trade_title": "salon, spa, and beauty concierge",
        "trade_noun": "beauty service",
        "trade_short": "salon appointment",
        "tech_title": "master stylist or aesthetician",
        "pain_points": "the frustration of bad haircuts, uneven color jobs, and struggling to find a trusted stylist who listens,",
        "empathy_sample": "'We would love to pamper you! You\\'re in wonderful hands with our master stylists, and we will make sure you leave looking and feeling amazing.'",
        "diy_question": "Can\\'t I just use box dye or trim my own bangs?",
        "diy_answer": "Box dyes contain harsh metallic salts that react unpredictably with previous treatments and often require expensive color corrections later. Our colorists formulate custom blends matched to your hair texture. Shall we book your appointment?",
        "emergency_label": "Same-day bridal party change / Severe allergic reaction / Urgent appointment adjustment",
        "emergency_advice": "Please let us know immediately so we can adjust our schedule or perform a complimentary patch test before your service.",
        "default_brands": "Olaplex, Redken, Kérastase, and Aveda",
        "default_fee": "transparent tier pricing",
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


def fee_to_spoken(fee_raw) -> str:
    """Converts diagnostic fee input (e.g. '$89', '79', 'Free', 89) into natural spoken English."""
    if fee_raw is None or fee_raw == "":
        return "eighty-nine dollars"
    fee_lower = str(fee_raw).lower().strip()
    if "free" in fee_lower or "complimentary" in fee_lower or "$0" in fee_lower or fee_lower == "0":
        return "complimentary"
    m = re.search(r"(\d+)", str(fee_raw))
    if m:
        n = int(m.group(1))
        if n == 0:
            return "complimentary"
        ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        if n < 20:
            words = ones[n]
        elif n < 100:
            words = tens[n // 10] + ("-" + ones[n % 10] if n % 10 else "")
        elif n < 1000:
            rem = n % 100
            h = ones[n // 100] + " hundred"
            if rem == 0:
                words = h
            elif rem < 20:
                words = f"{h} {ones[rem]}".strip()
            else:
                words = f"{h} {tens[rem // 10]}" + ("-" + ones[rem % 10] if rem % 10 else "")
        else:
            words = str(n)
        return f"{words} dollars"
    return "eighty-nine dollars"
def clean_timezone(tz_raw: Optional[str]) -> str:
    """Extract standard IANA timezone name from strings like 'Local (Asia/Muscat)' or 'America/New_York (Eastern Time)'."""
    if not tz_raw:
        return "America/New_York"
    m = re.search(r"([A-Za-z]+/[A-Za-z_]+)", str(tz_raw))
    if m:
        candidate = m.group(1)
        try:
            ZoneInfo(candidate)
            return candidate
        except Exception:
            pass
    try:
        ZoneInfo(str(tz_raw).strip())
        return str(tz_raw).strip()
    except Exception:
        return "America/New_York"


def parse_time_to_minutes(time_str: Optional[str]) -> int:
    """Converts a standard time string like '8:00 AM' or '6:00 PM' to minutes from midnight."""
    if not time_str:
        return 0
    m = re.search(r"(\d{1,2}):(\d{2})\s*(AM|PM)?", str(time_str), re.I)
    if not m:
        return 0
    hours = int(m.group(1))
    mins = int(m.group(2))
    ampm = (m.group(3) or "").upper()
    if ampm == "PM" and hours != 12:
        hours += 12
    elif ampm == "AM" and hours == 12:
        hours = 0
    return hours * 60 + mins


def evaluate_client_schedule_status(profile: Dict[str, Any], test_now: Optional[datetime] = None, current_time: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Evaluates in real-time whether a business is currently OPEN, AFTER-HOURS, or CLOSED TODAY
    based on the client's configured timezone and daily hours schedule.
    Determines next available appointment window and the exact night-call rule to execute.
    """
    sched_cfg = profile.get("schedule_config") or {}
    if isinstance(sched_cfg, str):
        try:
            sched_cfg = json.loads(sched_cfg)
        except Exception:
            sched_cfg = {}
    if not isinstance(sched_cfg, dict):
        sched_cfg = {}
    tz_raw = profile.get("timezone") or sched_cfg.get("timezone") or "America/New_York"
    tz_name = clean_timezone(tz_raw)
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz_name = "America/New_York"
        tz = ZoneInfo(tz_name)

    now = current_time or test_now or datetime.now(tz)
    if now.tzinfo is None:
        now = now.replace(tzinfo=tz)
    else:
        now = now.astimezone(tz)

    days_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    full_day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_name = days_order[now.weekday()]
    now_mins = now.hour * 60 + now.minute
    now_display = now.strftime("%A, %B %d, %Y at %I:%M %p").replace(" 0", " ")

    # Parse and normalize daily hours
    raw_dh = sched_cfg.get("daily_hours") or sched_cfg.get("dailyHours") or {}
    daily_hours = {}
    for idx, d in enumerate(days_order):
        full_d = full_day_names[idx]
        info = (
            raw_dh.get(d)
            or raw_dh.get(d.lower())
            or raw_dh.get(full_d)
            or raw_dh.get(full_d.lower())
            or {}
        )
        if not info and d in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
            daily_hours[d] = {"isOpen": True, "open": "8:00 AM", "close": "6:00 PM"}
        elif not info and d == "Sat":
            daily_hours[d] = {"isOpen": True, "open": "9:00 AM", "close": "3:00 PM"}
        elif not info and d == "Sun":
            daily_hours[d] = {"isOpen": False, "open": "10:00 AM", "close": "2:00 PM"}
        else:
            daily_hours[d] = {
                "isOpen": bool(info.get("isOpen", info.get("is_open", True))),
                "open": info.get("open", "8:00 AM"),
                "close": info.get("close", "6:00 PM"),
            }

    today_info = daily_hours.get(day_name, {"isOpen": True, "open": "8:00 AM", "close": "6:00 PM"})
    is_open = False
    reason = ""

    if not today_info["isOpen"]:
        is_open = False
        reason = f"Closed all day {now.strftime('%A')}"
        status_code = "CLOSED_TODAY"
    else:
        open_mins = parse_time_to_minutes(today_info["open"])
        close_mins = parse_time_to_minutes(today_info["close"])
        if now_mins < open_mins:
            is_open = False
            reason = f"Before hours (Opens today at {today_info['open']})"
            status_code = "BEFORE_HOURS"
        elif now_mins > close_mins:
            is_open = False
            reason = f"After hours (Closed for the day at {today_info['close']})"
            status_code = "AFTER_HOURS"
        else:
            is_open = True
            reason = f"Open now (Until {today_info['close']})"
            status_code = "OPEN"

    # Determine next opening window
    next_opening = "tomorrow morning"
    curr_idx = now.weekday()
    for offset in range(1, 8):
        check_idx = (curr_idx + offset) % 7
        check_day = days_order[check_idx]
        check_day_full = full_day_names[check_idx]
        check_info = daily_hours[check_day]
        if check_info["isOpen"]:
            if offset == 1:
                next_opening = f"tomorrow morning ({check_day_full}) between {check_info['open']} and 11:00 AM"
            else:
                next_opening = f"{check_day_full} morning between {check_info['open']} and 11:00 AM"
            break

    # Determine night-call policy
    after_policy = (
        profile.get("after_hours_action")
        or profile.get("night_action")
        or sched_cfg.get("afterHoursPolicy")
        or sched_cfg.get("after_hours_policy")
        or "book_earliest_slot"
    ).lower().strip()

    forwarding_phone = (
        profile.get("forwarding_phone")
        or profile.get("owner_phone")
        or profile.get("phone")
        or "our on-call line"
    )

    raw_booking = str(profile.get("booking_action") or profile.get("booking_mode") or "").strip()
    is_text_details_mode = any(k in raw_booking.lower() for k in ["text", "reach out", "callback", "owner schedules", "lead capture", "details"])

    if "transfer" in after_policy or "emergency" in after_policy:
        night_title = "WARM TRANSFER EMERGENCIES; MORNING CONTACT FOR ROUTINE"
        night_inst = (
            f"You are taking an after-hours call. If the caller reports an active emergency (gas smell, water leak, flood, sparks, no heat in freezing weather), "
            f"warmly transfer and connect them immediately to our on-call technician at {forwarding_phone}. "
            + (f"For routine non-emergency requests, reassure them that our service team will reach out first thing tomorrow morning to coordinate the earliest time." if is_text_details_mode else
               f"For routine non-emergency requests, reassure them and reserve our first priority arrival window for {next_opening}.")
        )
    elif is_text_details_mode:
        night_title = "CAPTURE CALLER DETAILS & SERVICE TEAM CONTACTS CALLER TO SCHEDULE"
        night_inst = (
            f"You are taking an after-hours call. Inform the caller that our dispatch office is closed for the evening. "
            f"Collect their full name, cell number, service address, and issue. Reassure them that their details have been dispatched to our service team who will reach out first thing in the morning to coordinate the best appointment time. "
            f"NEVER offer or quote specific calendar arrival windows yourself!"
        )
    elif "sms" in after_policy or "alert" in after_policy or "lead_capture" in after_policy:
        night_title = "CAPTURE CALLER DETAILS & SEND INSTANT PRIORITY SMS ALERT"
        night_inst = (
            f"You are taking an after-hours call. Inform the caller that our dispatch office is closed for the evening. "
            f"Collect their full name, cell number, service address, and issue. Reassure them that a priority SMS alert has been dispatched to our on-call manager who will follow up first thing in the morning."
        )
    else:
        night_title = "LOCK IN EARLIEST MORNING ARRIVAL WINDOW"
        night_inst = (
            f"You are taking an after-hours call. Warmly reassure the caller: 'Our office is closed for the night, but don\\'t worry—I can reserve our very first priority arrival window for {next_opening} so you are first on our technician\\'s schedule!' "
            f"Collect their address and name to confirm the morning priority reservation."
        )

    raw_mode = str(sched_cfg.get("mode") or profile.get("answering_coverage") or "always_24_7").lower()
    if "after" in raw_mode:
        norm_mode = "after_hours"
    elif "overflow" in raw_mode:
        norm_mode = "overflow"
    else:
        norm_mode = "always_24_7"

    raw_delay = sched_cfg.get("overflowDelaySecs")
    if raw_delay is None:
        raw_delay = sched_cfg.get("overflow_delay_secs")
    if raw_delay is None:
        raw_delay = profile.get("overflow_delay_seconds")
    if raw_delay is None:
        raw_delay = 15
    try:
        norm_delay = int(raw_delay)
    except Exception:
        norm_delay = 15

    return {
        "timezone": tz_name,
        "local_time": now_display,
        "day_name": day_name,
        "is_open": is_open,
        "status_code": status_code,
        "reason": reason,
        "next_opening": next_opening,
        "night_policy": after_policy,
        "night_action": after_policy,
        "night_title": night_title,
        "night_instruction": night_inst,
        "answering_mode": norm_mode,
        "overflow_delay_seconds": norm_delay
    }


def determine_call_answering_decision(profile: Dict[str, Any], current_time: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Evaluates whether and when the AI receptionist should answer an incoming call based on the client's
    configured answering coverage rule:
      1. 'always_24_7': 24/7 hours — answers immediately day and night (0s delay).
      2. 'after_hours' (or 'after_hours_only'): Answers ONLY after hours when office is closed (0s delay).
         During open business hours, does not intercept; transfers to office.
      3. 'overflow': Answers after configured delay (e.g. 15s / 3 rings) to let business cell phone ring first.
    """
    sched_cfg = profile.get("schedule_config") or {}
    if isinstance(sched_cfg, str):
        try:
            sched_cfg = json.loads(sched_cfg)
        except Exception:
            sched_cfg = {}
    if not isinstance(sched_cfg, dict):
        sched_cfg = {}

    raw_cov = str(profile.get("answering_coverage") or sched_cfg.get("mode") or sched_cfg.get("answering_coverage") or "always_24_7").lower()
    if "after" in raw_cov:
        mode = "after_hours"
    elif "overflow" in raw_cov:
        mode = "overflow"
    else:
        mode = "always_24_7"

    raw_delay = profile.get("overflow_delay_seconds")
    if raw_delay is None:
        raw_delay = sched_cfg.get("overflowDelaySecs")
    if raw_delay is None:
        raw_delay = sched_cfg.get("overflow_delay_secs")
    if raw_delay is None:
        raw_delay = 15
    try:
        overflow_delay = int(raw_delay)
    except Exception:
        overflow_delay = 15

    sched_status = evaluate_client_schedule_status(profile, current_time=current_time)
    is_open = sched_status.get("is_open", True)
    biz_name = profile.get("business_name") or "Our Company"
    forwarding_phone = profile.get("forwarding_phone") or profile.get("owner_phone") or "our main office line"

    if mode == "always_24_7":
        return {
            "mode": "always_24_7",
            "mode_title": "⚡ 24/7 Always Active",
            "should_answer": True,
            "pickup_delay_seconds": 0,
            "action": "ANSWER_IMMEDIATELY",
            "reason": "Configured for 24/7 continuous call answering day and night.",
            "is_after_hours": not is_open,
            "schedule_status": sched_status,
            "greeting_type": "standard_daytime" if is_open else "after_hours_protocol",
            "directive": "Answer all calls immediately without delay. If currently open, execute standard daytime booking protocol; if closed/after-hours, strictly enforce the configured Night-Call rule."
        }
    elif mode == "after_hours":
        if not is_open:
            return {
                "mode": "after_hours",
                "mode_title": "🌙 After-Hours Only (Office Closed)",
                "should_answer": True,
                "pickup_delay_seconds": 0,
                "action": "ANSWER_AFTER_HOURS",
                "reason": f"Office is currently closed ({sched_status['status_code']} at {sched_status['local_time']}). AI receptionist active.",
                "is_after_hours": True,
                "schedule_status": sched_status,
                "greeting_type": "after_hours_protocol",
                "directive": "The main office is closed for the evening or weekend. Answer immediately and strictly execute the configured Night-Call rule."
            }
        else:
            return {
                "mode": "after_hours",
                "mode_title": "🌙 After-Hours Only (Office Open)",
                "should_answer": False,
                "pickup_delay_seconds": 0,
                "action": "BYPASS_OR_TRANSFER_TO_OFFICE",
                "reason": f"Office is currently OPEN ({sched_status['local_time']}). AI receptionist will not intercept call.",
                "is_after_hours": False,
                "schedule_status": sched_status,
                "greeting_type": "office_open_transfer",
                "directive": f"The main office is open. If connected, inform the caller the front desk is available and connect them directly to our office team at {forwarding_phone}."
            }
    else:  # overflow
        return {
            "mode": "overflow",
            "mode_title": f"📞 Overflow Backup ({overflow_delay}s Delay)",
            "should_answer": True,
            "pickup_delay_seconds": overflow_delay,
            "action": "ANSWER_AFTER_DELAY" if overflow_delay > 0 else "ANSWER_ON_BUSY",
            "reason": f"Wait {overflow_delay} seconds for primary cell phone/team to answer. If unanswered, AI receptionist picks up.",
            "is_after_hours": not is_open,
            "schedule_status": sched_status,
            "greeting_type": "overflow_greeting",
            "directive": f"You are answering as the overflow backup receptionist because the team was busy on ladders or jobs. Reassure the caller warmly that you can assist them immediately."
        }


HUMAN_TRANSFER_POLICIES = {
    "life_safety_emergencies": {
        "id": "life_safety_emergencies",
        "title": "🚨 Life & Safety Emergencies Only (Recommended)",
        "instruction": "Transfer the call immediately ONLY when there is an active life, health, or major property safety emergency (e.g. gas leak, burst pipe flooding, electrical sparks/fire, carbon monoxide). For all routine inquiries, pricing questions, and standard appointments, handle them autonomously.",
        "trigger_keywords": ["gas", "carbon monoxide", "burst pipe", "leak", "flood", "flooding", "spark", "sparking", "burning", "fire", "emergency", "shut off", "bleeding", "smoke", "explosion", "hazard", "freezing", "no heat"]
    },
    "caller_demands_human": {
        "id": "caller_demands_human",
        "title": "👤 When Caller Demands Human or Manager",
        "instruction": "Transfer the call immediately if the caller explicitly asks or demands to speak with a human, manager, owner, or live representative, OR if there is an active life/safety emergency. Otherwise handle standard triage and bookings autonomously.",
        "trigger_keywords": ["gas", "carbon monoxide", "burst pipe", "leak", "flood", "flooding", "spark", "sparking", "burning", "fire", "emergency", "shut off", "smoke", "hazard", "transfer", "human", "speak to owner", "manager", "operator", "real person", "representative", "someone else", "speak to a person", "talk to a person", "talk to human"]
    },
    "unresolved_queries": {
        "id": "unresolved_queries",
        "title": "🔍 Complex or Out-of-Scope Technical Inquiries",
        "instruction": "Transfer the call if the caller asks complex custom technical questions outside your authorized services, OR if there is an active emergency, OR if they need a specialist consultation. Otherwise handle standard triage and bookings autonomously.",
        "trigger_keywords": ["gas", "carbon monoxide", "burst pipe", "leak", "flood", "flooding", "spark", "sparking", "burning", "fire", "emergency", "shut off", "smoke", "hazard", "custom engineering", "out of scope", "specialist consult", "commercial blueprint", "speak with a technician", "technical expert", "engineering spec"]
    },
    "never_transfer_take_message": {
        "id": "never_transfer_take_message",
        "title": "📴 Never Transfer — Take Detailed Message & Send SMS Alert",
        "instruction": "DO NOT transfer or patch the live call to a phone number under any circumstances. If the caller requests a manager, has an emergency, or has an out-of-scope question, warmly reassure them, take down their full name, mobile number, service address, and issue description, and explain that the owner and dispatch team have been alerted with top priority and will contact them directly right away via phone or SMS.",
        "trigger_keywords": []
    }
}


def determine_human_transfer_policy(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Resolves the client's human transfer policy from profile or onboarding settings."""
    raw = str(
        profile.get("transfer_to_human_policy")
        or profile.get("human_transfer_policy")
        or profile.get("transfer_policy")
        or profile.get("when_to_transfer")
        or "life_safety_emergencies"
    ).lower().strip()

    if "never" in raw or "message" in raw or "no_transfer" in raw:
        return HUMAN_TRANSFER_POLICIES["never_transfer_take_message"]
    elif "demand" in raw or "ask" in raw or "person" in raw or "manager" in raw or "human" in raw or "operator" in raw:
        return HUMAN_TRANSFER_POLICIES["caller_demands_human"]
    elif "unresolved" in raw or "complex" in raw or "scope" in raw or "technical" in raw:
        return HUMAN_TRANSFER_POLICIES["unresolved_queries"]
    else:
        return HUMAN_TRANSFER_POLICIES["life_safety_emergencies"]


def compile_agent_prompt(profile: Dict[str, Any]) -> str:
    """Compiles a Marcus-grade, consultative, high-empathy voice AI prompt strictly
    adapted to the user's business, industry, schedule, pricing policy, lead capture, and emergency rules.
    Retains the proven Marcus & Riley conversational framework while customizing all client choices.
    """
    biz_name = (profile.get("business_name") or profile.get("name") or "Our Company").strip()
    raw_trade = (profile.get("trade") or profile.get("industry") or "hvac").lower().strip()
    
    trade_key = "hvac"
    trade_map = {
        "hvac": ["hvac", "ac", "air conditioning", "heating", "cooling", "furnace", "climate"],
        "plumbing": ["plumb", "drain", "water heater", "sewer", "pipe"],
        "electrical": ["electr", "panel", "wiring", "breaker", "lighting", "generator"],
        "roofing": ["roof", "gutter", "shingle"],
        "dental_medical": ["dental", "medical", "clinic", "doctor", "hygiene", "dentist", "health", "teeth"],
        "auto": ["auto", "car", "mechanic", "vehicle", "brake", "transmission", "motor", "tire"],
        "legal": ["legal", "law", "attorney", "lawyer", "counsel", "litigation"],
        "restaurant": ["restaurant", "dining", "cafe", "bistro", "catering", "hospitality", "food"],
        "realestate": ["realestate", "real estate", "realty", "realtor", "property", "tenant", "listing"],
        "salon_spa": ["salon", "spa", "hair", "barber", "nail", "lash", "massage", "aesthetic", "beauty"],
    }
    for tk, aliases in trade_map.items():
        if any(a in raw_trade for a in aliases):
            trade_key = tk
            break
    if trade_key not in TRADE_METRICS and raw_trade in TRADE_METRICS:
        trade_key = raw_trade
    elif trade_key not in TRADE_METRICS:
        trade_key = "general"

    biz_name = harmonize_business_name_for_trade(biz_name, trade_key)

    metrics = TRADE_METRICS[trade_key]
    hours = profile.get("hours") or "Monday to Friday 8:00 AM to 6:00 PM"
    raw_addr = (profile.get("address") or profile.get("formatted_address") or "").strip()
    raw_city = (profile.get("city") or "").strip()
    clean_addr, clean_city = sanitize_address_and_city(raw_addr, raw_city)
    persona_name = (profile.get("persona_name") or "Riley").strip()
    forwarding_phone = (profile.get("forwarding_phone") or profile.get("business_phone") or profile.get("phone") or "").strip()

    # Schedule & Timezone Configuration
    schedule_cfg = profile.get("schedule_config") or {}
    if isinstance(schedule_cfg, str):
        try:
            schedule_cfg = json.loads(schedule_cfg)
        except Exception:
            schedule_cfg = {}
    if not isinstance(schedule_cfg, dict):
        schedule_cfg = {}
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

    # Dynamic Real-Time Schedule & Night-Call Status Evaluation
    sched_eval = evaluate_client_schedule_status(profile)
    night_rule_title = sched_eval["night_title"]
    night_rule_instructions = sched_eval["night_instruction"]
    timezone_name = sched_eval["timezone"]
    current_time_str = sched_eval["local_time"]
    current_status_str = f"{sched_eval['status_code']} ({sched_eval['reason']})"
    next_opening_str = sched_eval["next_opening"]

    # Dynamic Answering Coverage Decision
    answering_decision = determine_call_answering_decision(profile)
    coverage_mode_title = answering_decision["mode_title"]
    coverage_rule_reason = answering_decision["reason"]
    coverage_pickup_delay = answering_decision["pickup_delay_seconds"]
    coverage_directive = answering_decision["directive"]

    # Dynamic Human Transfer Policy Resolution
    human_transfer = determine_human_transfer_policy(profile)
    human_transfer_id = human_transfer["id"]
    human_transfer_title = human_transfer["title"]
    human_transfer_instruction = human_transfer["instruction"]

    # Emergency Triggers Configuration
    emergency_raw = profile.get("emergency_triggers") or profile.get("transfer_rules")
    if isinstance(emergency_raw, list):
        emergency_triggers_str = ", ".join(emergency_raw)
    elif isinstance(emergency_raw, str) and emergency_raw.strip():
        emergency_triggers_str = emergency_raw.strip()
    else:
        emergency_triggers_str = metrics.get("emergency_label", "Gas odor, carbon monoxide, active flooding, burst pipes, electrical sparks or fire")

    # Pricing & Diagnostic Fee Policy
    pricing_policy_str = str(profile.get("pricing_policy") or "").lower().strip()
    raw_fee = (
        profile.get("fee_amount")
        or profile.get("diagnostic_fee")
        or profile.get("pricing_policy")
        or profile.get("pricing_preset")
        or "Diagnostic fee credited toward repair"
    )
    raw_fee_str = str(raw_fee).lower().strip()
    spoken_fee = fee_to_spoken(str(raw_fee))
    if not spoken_fee and profile.get("pricing_policy"):
        spoken_fee = fee_to_spoken(str(profile.get("pricing_policy")))

    if "free" in pricing_policy_str or "complimentary" in pricing_policy_str or "$0" in pricing_policy_str or "free" in raw_fee_str or "complimentary" in raw_fee_str or "$0" in raw_fee_str:
        fee_objection_answer = "We provide a 100% complimentary on-site inspection and estimate with zero obligation! Does that sound fair?"
    elif "upfront" in pricing_policy_str or "inspection after" in pricing_policy_str or "upfront" in raw_fee_str or "inspection after" in raw_fee_str:
        fee_objection_answer = f"Our {metrics['tech_title']} evaluates the system in person and gives you a guaranteed upfront flat-rate price before any work begins! Does that sound fair?"
    elif spoken_fee:
        fee_objection_answer = (
            f"Our diagnostic fee is a flat {spoken_fee}, which covers a comprehensive inspection by a {metrics['tech_title']}. "
            f"And the best part is, we credit that full {spoken_fee} directly toward any repair you approve! Does that sound fair?"
        )
    else:
        fee_objection_answer = (
            f"Our diagnostic visit fee covers a comprehensive inspection by a {metrics['tech_title']}. "
            f"And the best part is, we credit that full fee directly toward any repair you approve! Does that sound fair?"
        )

    # Opening Greeting
    first_msg = (
        profile.get("first_message")
        or profile.get("greeting")
        or f"Thank you for calling {biz_name}! This is {persona_name}. How can I help you today?"
    )
    if any(p in first_msg.lower() for p in ["comfortable today", "taken care of today", "scheduled today", "active leak", "power your", "routine visit", "matter today", "dining information", "view a property"]):
        first_msg = f"Thank you for calling {biz_name}! This is {persona_name}. How can I help you today?"

    # Emergency Transfer Line
    transfer_addon = ""
    if forwarding_phone and human_transfer_id != "never_transfer_take_message":
        transfer_addon = f" I am connecting you directly with our emergency line at {forwarding_phone} right now."
    elif human_transfer_id == "never_transfer_take_message":
        transfer_addon = f" I am taking down your details right now, and our service team will follow up directly with top priority."

    # Custom FAQs & QA
    custom_qa_lines = ""
    for qa in profile.get("custom_qa", []):
        q = qa.get("question", "").strip()
        a = qa.get("answer", "").strip()
        if q and a:
            custom_qa_lines += f"- '{q}': '{a}'\n"

    if clean_addr and clean_addr != "Local Area":
        location_line = f" Service territory and location: based in {clean_city} ({clean_addr})."
    elif clean_city and clean_city != "the local area":
        location_line = f" Service territory: {clean_city} area and surrounding communities."
    else:
        location_line = ""

    # Booking Mode Resolution (Rule 1 vs Rule 2)
    raw_booking = str(profile.get("booking_action") or profile.get("booking_mode") or "").strip()
    is_text_details_mode = any(k in raw_booking.lower() for k in ["text", "reach out", "callback", "owner schedules", "lead capture", "details"])

    if is_text_details_mode:
        core_rule = (
            f"Core Rule (OWNER-CALLBACK DISPATCH MODE - Rule 2): You are a consultative {metrics['trade_short']} advisor. Never assume the caller wants an appointment right away. "
            f"Always explore and diagnose their symptoms first. For scheduling, our service team reviews each service inquiry and contacts the caller directly "
            f"to coordinate the best appointment time. Capture their street address, full name, and mobile number so our team can follow up immediately. "
            f"CRITICAL: NEVER offer or quote specific calendar arrival windows yourself—explain that our service team will reach out directly to arrange the best time!"
        )
        calendar_note = "Owner Callback Dispatch Mode: Collect service address and caller contact info; service team reaches out directly to arrange appointment time."
        booking_flow_str = f"""<booking_flow_state_machine>
1. Warm Greeting: '{first_msg}'
2. Symptom Discovery & Triage First:
   - For REPAIRS / BREAKDOWNS / PROBLEMS: React with warm empathy, but do NOT jump to booking or ask for an address yet: 'Oh no, dealing with {metrics["trade_short"]} trouble is such a headache! What seems to be happening with the system—is it blowing warm air, making a strange sound, or completely shut off?'
   - For NEW INSTALLATIONS / REPLACEMENTS / QUOTES: Acknowledge enthusiastically: 'We\\'d love to help you with a new system installation! That\\'s a great project, and we provide free in-person estimates. Would you like our specialist to reach out to coordinate a free estimate visit?'
3. Owner-Callback Dispatch Offer: Once they describe symptoms or confirm interest, explain that our service team will contact them directly to coordinate the best appointment time: 'Got it, that definitely sounds like something our technician should inspect to diagnose properly. I can take down your service details right now, and our service team will reach out directly to arrange the best time to come take a look! What is your service address so we know your location?'
4. Service Address Capture & Immediate Confirmation: Collect their street address: 'Great! What is your service address so I can route this to our team?' Immediately read it back: 'Got it — so I have [Address]. Did I get that right?' Once confirmed, NEVER re-ask for the address!
5. Caller Full Name & Cell Capture: 'And what is your full name and the best cell number for our team to call and text you?' Read number back digit-by-digit and confirm.
6. Complete 4-Point Owner-Callback Recap: Deliver the complete recap directly without asking for any information again: 'You are all set, [Name]! I have dispatched your details for [Confirmed Address] directly to our service team. We just sent a confirmation text to [Phone], and our team will follow up shortly to coordinate your visit time. Does that sound like a plan?'
7. Clean Sign-Off: 'Thank you for choosing {biz_name}. Have a wonderful day!'
CRITICAL DIRECTIVE: DO NOT OFFER OR PROMISE SPECIFIC CALENDAR TIME SLOTS (e.g. do NOT say 'between one and three' or 'tomorrow morning'). The business owner/team will review the request and text/call the customer to schedule!
</booking_flow_state_machine>"""
        obj_quote = f"- 'Can you quote me a price over the phone?': 'I wish I could give you an exact price over the phone! But {metrics['trade_noun']} issues could be as simple as a small component or something deeper in the system. Our technician gives you a guaranteed flat-rate price on site before starting any work. I can take down your address and our team will reach out to schedule—what is your address?'"
        obj_dispatch = f"- 'Can someone come out right now / immediately?': 'Our technicians are currently out on scheduled routes, but I can send your details directly to our service team right now so they can contact you shortly to coordinate the earliest time! What is your service address?'"
        obj_when = f"- 'When can someone come out?': 'Our team coordinates arrival times directly with homeowners based on daily routes. Once I take your address and phone number, our team will reach out right away to arrange the best time to come out! What is your street address?'"
    else:
        core_rule = (
            f"Core Rule: You are a consultative home comfort advisor. Never assume the caller wants an appointment right away. "
            f"Always explore and diagnose their symptoms first before offering to schedule. Never claim immediate dispatch or sending the closest technician—we schedule arrival windows for our team to come out later today or tomorrow."
        )
        booking_flow_str = f"""<booking_flow_state_machine>
1. Warm Greeting: '{first_msg}'
2. Symptom Discovery & Triage First:
   - For REPAIRS / BREAKDOWNS / PROBLEMS: React with warm empathy, but do NOT jump to booking or ask for an address yet: 'Oh no, dealing with {metrics["trade_short"]} trouble is such a headache! What seems to be happening with the system—is it blowing warm air, making a strange sound, or completely shut off?'
   - For NEW INSTALLATIONS / REPLACEMENTS / QUOTES: Acknowledge enthusiastically: 'We\\'d love to help you with a new system installation! That\\'s a great project, and we provide free in-person estimates. Would you like to check our available times for a comfort consultant to come out and provide a free estimate?'
3. Consultative Scheduling Offer: Once they describe symptoms or confirm interest in an estimate, acknowledge with expert knowledge. Propose checking available arrival windows for the team to come out later today or tomorrow: 'Got it, that definitely sounds like something one of our technicians should inspect to diagnose properly. We can get you on the schedule so our team can come out and take care of that for you. Would you like to check our available appointment times?'
4. Service Address Capture & Immediate Confirmation: Only when caller agrees to check times or schedule, collect their address: 'Great! What is your service address so I can check our schedule for your area?' Immediately read it back: 'Got it — so I have [Address]. Did I get that right?' Once confirmed, NEVER re-ask for the address!
5. Arrival Window Selection: Offer two clear arrival windows for the team to come out: 'Got it! We have an opening today between one and three, or tomorrow morning between eight and eleven. Which arrival window works better for your schedule?'
6. Scheduling Conflict Handling: If caller rejects proposed times, immediately adapt: 'No problem at all! What day or time window works best for your schedule?'
7. Caller Name & Cell Capture: 'And what is your full name and the best cell number for dispatch arrival updates?' Read number back and confirm.
8. Complete 5-Point Recap & Confirmation: All details are confirmed. Deliver the complete recap directly without asking for any information again: 'You are all set, [Name]! We have our team scheduled for [Address] for your [Issue] on [Day] between [Time Window]. We just sent a confirmation text with arrival tracking to [Phone]. Does everything sound good?'
9. Clean Sign-Off: 'Thank you for choosing {biz_name}. Have a wonderful day!'
</booking_flow_state_machine>"""
        obj_quote = f"- 'Can you quote me a price over the phone?': 'I wish I could give you an exact price over the phone! But {metrics['trade_noun']} issues could be as simple as a small component or something deeper in the system. Our technician gives you a guaranteed flat-rate price on site before starting any work. Would afternoon or tomorrow morning work better?'"
        obj_dispatch = f"- 'Can someone come out right now / immediately?': 'Our technicians are currently out on scheduled routes with homeowners, so we don\\'t have an immediate truck roll right this second. But we can reserve our earliest priority opening for you today or tomorrow! Would you like me to check available arrival windows?'"
        obj_when = f"- 'When can someone come out?': 'We have an opening today between one and three, or tomorrow morning between eight and eleven. Which arrival window works better for your schedule?'"

    prompt = f"""<identity>
You are {persona_name}, an articulate, genuinely warm, confident, and consultative voice receptionist for {biz_name}. You are an AI—transparent, warm, and proud of it if asked. Never pretend to be human, but never sound robotic.{location_line}
Core mindset: You are a peer-level {metrics['trade_title']} consultant. You understand {metrics['pain_points']} and busy property owners who want an honest, fast, expert solution without high-pressure sales or being put on hold. Build genuine human connection first. Answer questions and objections directly with zero evasion.
{core_rule}
Operating schedule: {hours_str} ({timezone_name}).
Calendar integration: {calendar_note}
Pricing policy: {raw_fee}.
Authorized services: {services_text}.
</identity>

<conversational_rules>
- NEVER DUMP OPERATING HOURS OR SERVICE TERRITORY UNPROMPTED: Do not recite operating hours (e.g. 'We\'re open Mon-Fri 8:00 AM - 6:00 PM, and we cover the entire...') or recite service territories in routine answers or greetings. Only mention hours if the caller specifically asks when you are open or calls after hours. Only mention service territory if the caller specifically asks if you service their neighborhood or city.
- EMPATHY & RAPPORT FIRST: When the caller shares their service need or asks how you are, react like a real human first before transacting ({metrics['empathy_sample']}).
- STRICT SINGLE QUESTION: Exactly ONE question mark ('?') per turn. Never combine a confirmation ('is that right?') with a new question in the same turn! After asking your question, STOP and listen.
- STRICT COMPLETION: ALWAYS finish your sentences cleanly. Never stop mid-thought or cut off.
- BREVITY & FLOW: Speak in 1 to 2 punchy, natural spoken sentences (strictly under 25 words per turn). Use natural contractions ('we\\'ll', 'that\\'s', 'you\\'re', 'don\\'t', 'let\\'s').
- DIRECT ANSWERS: If the caller asks a question (diagnostic fee, replacement cost, timing, DIY, licensing), ALWAYS answer it directly and transparently before asking your next question.
- ZERO-FRICTION BOOKING: Lock in the appointment with Name, Cell Phone, and Physical Address. Do NOT ask for or require an email address over the phone. Cell phone is our primary dispatch channel—confirmations and live tracking are sent via SMS. If the caller happens to volunteer an email, absorb it warmly, but never prompt or press for one.
- CONVERSATION MEMORY & MULTI-SLOT ABSORPTION: NEVER ask for information the caller already volunteered. If they gave their address, phone, or issue in an earlier turn, absorb all of them and advance to the next uncollected item.
- FORMATTING: Spoken voice only—never use markdown, asterisks, bullet points, or lists.
</conversational_rules>

<field_confirmation_protocol>
ADDRESS CAPTURE & CONFIRMATION PROTOCOL:
- When the caller provides their service address, IMMEDIATELY read it back verbatim and ask ONLY for confirmation: 'Got it — so I have [Full Address]. Did I get that right?' (STOP and wait for confirmation).
- Once confirmed by the caller, that address is PERMANENTLY LOCKED.
- CRITICAL RULE: NEVER ASK FOR THE SERVICE ADDRESS AGAIN UNDER ANY CIRCUMSTANCES.
- When completing the booking in the final recap, ALWAYS use the confirmed address directly: 'You\\'re all set, [Name]! We have our team scheduled for [Confirmed Address]...' NEVER ask 'And just to confirm, what is your service address?'.

PHONE NUMBER CAPTURE & CONFIRMATION PROTOCOL:
- When the caller provides their mobile number, IMMEDIATELY read it back digit-by-digit and ask ONLY for confirmation: 'Perfect — I have seven one three, five five five, zero one eight four. Did I get that right?' (STOP and wait for confirmation).
- Once confirmed, proceed to the recap.
</field_confirmation_protocol>

<human_transfer_policy>
Transfer Policy: {human_transfer_title}
Directive: {human_transfer_instruction}
Forwarding Phone: {forwarding_phone if forwarding_phone else "Emergency Dispatch Line"}
</human_transfer_policy>

<call_termination_directives>
1. 15-Minute Safety Cap: All calls are strictly capped at 15 minutes to preserve line availability.
2. Natural Farewell: When caller or agent says sign-off ("thanks that's all", "goodbye", "have a wonderful day"), conclude gracefully and end call.
3. Immediate Polite Farewell on Caller Opt-out / Disinterest: If the caller states they are not interested, calling by mistake, shopping around and do not want an appointment, or declines service, respond politely ("No problem at all! If you ever need assistance in the future, don't hesitate to give us a call. Have a wonderful day!") and end call.
4. Automated System / AI Robocall Loop Detection: If the caller is detected to be an automated voicemail greeting, IVR bot, or AI, state "Automated system detected. Ending call." and hang up immediately.
5. Keypad Selection IVR Options: If the incoming line presents a keypad menu ("press 1 for sales", "select from the following options"), state "This direct line does not support automated keypad menu selections. Goodbye." and terminate call immediately.
6. Transfer Completed: When transferring the caller under the Human Transfer Policy, inform them clearly ("I am connecting you directly with our dispatch team right now. Please stay on the line.") and execute the transfer handoff.
</call_termination_directives>

<answering_coverage_policy>
Coverage Mode: {coverage_mode_title}
Pickup Rule: {coverage_rule_reason}
Pickup Delay: {coverage_pickup_delay} seconds
Current Operational Status: {current_status_str}
Operational Answering Directive: {coverage_directive}
</answering_coverage_policy>

<after_hours_policy>
Active Night Rule: {night_rule_title}
Configured Timezone: {timezone_name}
Operating Schedule: {hours_str}
Current Real-Time Clock: {current_time_str} ({timezone_name})
Live Operational Status: {current_status_str}
Next Available Dispatch Opening: {next_opening_str}
Night-Call Rule: {night_rule_instructions}
Operational Directive: If Live Operational Status indicates CLOSED_TODAY, AFTER_HOURS, or BEFORE_HOURS, or if the caller calls outside standard business hours, you are handling an after-hours call. Strictly execute the Night-Call Rule above!
</after_hours_policy>

<emergency_triage_rules>
Recognized Emergency Triggers: {emergency_triggers_str}
Immediate Action: Express priority empathy, provide safety instructions, and escalate without asking non-critical questions.{transfer_addon}
</emergency_triage_rules>

{booking_flow_str}

<objection_playbook>
- 'How much is your diagnostic fee?' / 'Pricing': '{fee_objection_answer}'
{obj_quote}
{obj_dispatch}
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

def simulate_agent_turn(client_profile: Dict[str, Any], user_message: str, history: Optional[List[Dict[str, str]]] = None, test_now: Optional[datetime] = None) -> Dict[str, Any]:
    """Generates a real-time conversational response from the client's tailored voice agent.
    Uses Groq LLM if configured; otherwise uses smart prompt-aware heuristics that faithfully
    mirror all onboarding rules (after-hours action, emergency triggers, pricing policy).
    """
    raw_biz = (client_profile.get("business_name") or client_profile.get("name") or "Apex Services").strip()
    raw_trade = (client_profile.get("trade") or client_profile.get("industry") or "hvac").lower().strip()
    trade_key = "general"
    trade_map = {
        "hvac": ["hvac", "ac", "air conditioning", "heating", "cooling", "furnace", "climate"],
        "plumbing": ["plumb", "drain", "water heater", "sewer", "pipe"],
        "electrical": ["electr", "panel", "wiring", "breaker", "lighting", "generator"],
        "roofing": ["roof", "gutter", "shingle"],
        "dental_medical": ["dental", "medical", "clinic", "doctor", "hygiene", "dentist", "health", "teeth"],
        "auto": ["auto", "car", "mechanic", "vehicle", "brake", "transmission", "motor", "tire"],
        "legal": ["legal", "law", "attorney", "lawyer", "counsel", "litigation"],
        "restaurant": ["restaurant", "dining", "cafe", "bistro", "catering", "hospitality", "food"],
        "realestate": ["realestate", "real estate", "realty", "realtor", "property", "tenant", "listing"],
        "salon_spa": ["salon", "spa", "hair", "barber", "nail", "lash", "massage", "aesthetic", "beauty"],
    }
    for tk, aliases in trade_map.items():
        if any(a in raw_trade for a in aliases):
            trade_key = tk
            break
    biz_name = harmonize_business_name_for_trade(raw_biz, trade_key)
    persona_name = client_profile.get("persona_name", "Riley")
    forwarding = client_profile.get("forwarding_phone") or client_profile.get("owner_phone") or "our on-call line"

    # Evaluate schedule & answering decision
    sched_eval = evaluate_client_schedule_status(client_profile, test_now=test_now)
    answering_decision = determine_call_answering_decision(client_profile, current_time=test_now)

    # If client configured after-hours only and office is OPEN, do not intercept -> transfer to front desk
    if answering_decision["mode"] == "after_hours" and not answering_decision["should_answer"]:
        return {
            "response": f"Thank you for calling {biz_name}! Our main office is open right now ({sched_eval['local_time']}). Let me connect you directly with our front desk team at {forwarding} right now.",
            "is_transfer": True,
            "telemetry_badge": "OFFICE OPEN TRANSFER",
            "business_name": biz_name,
        }

    system_prompt = client_profile.get("compiled_prompt") or compile_agent_prompt(client_profile)

    msg_lower = user_message.lower()
    msg_norm = re.sub(r'[^a-z0-9\s]', ' ', msg_lower)
    msg_norm = re.sub(r'\s+', ' ', msg_norm).strip()
    is_transfer = False
    telemetry_badge = "STANDARD INQUIRY"

    # 1. AI-to-AI Robocall & Voicemail Loop Termination
    robocall_indicators = [
        "leave a message after the tone", "at the tone", "record your message",
        "you have reached the voicemail", "is not available to take your call",
        "mailbox is full", "automated voice", "virtual assistant",
        "i am an ai", "i'm an ai", "i am an artificial intelligence",
        "i am a language model", "press 1 to speak with an agent",
        "all of our agents are currently busy", "please hold for the next available representative",
        "your call is important to us", "press 1 to accept", "this is an automated call"
    ]
    if any(k in msg_lower or k in msg_norm for k in robocall_indicators):
        return {
            "response": "Automated system detected. Ending call.",
            "is_transfer": False,
            "is_terminal": True,
            "termination_reason": "ai_bot_loop_detected",
            "telemetry_badge": "AI ROBOCALL TERMINATION",
            "business_name": biz_name,
        }

    # 2. Interactive Keypad Selection IVR Options Termination
    keypad_ivr_indicators = [
        "press 1", "press 2", "press 3", "press 4", "press 0",
        "press one", "press two", "press three", "press four",
        "press pound", "press star", "for sales", "for service",
        "for billing", "for english", "for spanish",
        "select from the following options", "listen carefully to the following options",
        "main menu", "to repeat this menu"
    ]
    if any(k in msg_lower or k in msg_norm for k in keypad_ivr_indicators):
        return {
            "response": "This direct line does not support automated keypad menu selections. Goodbye.",
            "is_transfer": False,
            "is_terminal": True,
            "termination_reason": "keypad_ivr_detected",
            "telemetry_badge": "KEYPAD IVR TERMINATION",
            "business_name": biz_name,
        }

    # 3. Natural Caller Farewell & Sign-Off Termination
    caller_farewells = [
        "goodbye", "bye", "bye for now", "have a good day", "have a great day",
        "have a wonderful day", "that's all thank you", "that's all, thank you",
        "that is all thank you", "that is all, thank you", "that's all thanks",
        "that is all thanks", "that's everything thank you", "that is everything",
        "no that's all", "no that is all", "no that's everything",
        "all set thank you", "all set, thank you", "thanks for your help bye",
        "thank you bye", "thank you, bye"
    ]
    if any(k in msg_lower or k in msg_norm for k in caller_farewells):
        return {
            "response": f"Thank you for choosing {biz_name}! Have a wonderful day!",
            "is_transfer": False,
            "is_terminal": True,
            "termination_reason": "natural_farewell",
            "telemetry_badge": "CALL COMPLETED",
            "business_name": biz_name,
        }

    # 4. Emergency Detection & Human Transfer Policy Resolution
    emergency_keywords = [
        "gas", "carbon monoxide", "burst pipe", "leak", "flood", "flooding", "spark", "sparking",
        "burning", "fire", "emergency", "shut off", "bleeding", "smoke", "explosion", "hazard", "freezing", "no heat"
    ]
    custom_triggers = client_profile.get("emergency_triggers") or client_profile.get("transfer_rules")
    if isinstance(custom_triggers, list):
        for ct in custom_triggers:
            if isinstance(ct, str) and ct.strip():
                emergency_keywords.append(ct.strip().lower())
    elif isinstance(custom_triggers, str) and custom_triggers.strip():
        for ct in custom_triggers.split(","):
            if ct.strip():
                emergency_keywords.append(ct.strip().lower())

    transfer_policy = determine_human_transfer_policy(client_profile)
    pol_id = transfer_policy["id"]

    if pol_id == "never_transfer_take_message":
        is_transfer = False
        if any(k in msg_lower for k in emergency_keywords):
            telemetry_badge = "HIGH PRIORITY INTAKE"
        elif any(k in msg_lower for k in ["transfer", "human", "speak to owner", "manager", "operator", "real person"]):
            telemetry_badge = "MESSAGE INTAKE"
        elif any(k in msg_lower for k in ["diy", "myself", "how to wire", "rewire", "which wire", "bypass", "diagnose my", "prescribe", "legal opinion", "do it myself", "fix it myself", "tell me how to fix"]):
            telemetry_badge = "GUARDRAIL ENFORCED"
        elif any(k in msg_lower for k in ["after hours", "midnight", "night", "11:45", "open now", "closed", "2 am", "sunday", "outside hours", "emergency line"]):
            telemetry_badge = "SCHEDULE VERIFICATION"
    elif pol_id == "caller_demands_human":
        if any(k in msg_lower for k in emergency_keywords):
            is_transfer = True
            telemetry_badge = "EMERGENCY TRIAGE"
        elif any(k in msg_lower for k in ["transfer", "human", "speak to owner", "manager", "operator", "real person", "representative", "someone else", "speak to a person", "talk to a person", "talk to human"]):
            is_transfer = True
            telemetry_badge = "OPERATOR TRANSFER"
        elif any(k in msg_lower for k in ["diy", "myself", "how to wire", "rewire", "which wire", "bypass", "diagnose my", "prescribe", "legal opinion", "do it myself", "fix it myself", "tell me how to fix"]):
            telemetry_badge = "GUARDRAIL ENFORCED"
        elif any(k in msg_lower for k in ["after hours", "midnight", "night", "11:45", "open now", "closed", "2 am", "sunday", "outside hours", "emergency line"]):
            telemetry_badge = "SCHEDULE VERIFICATION"
    elif pol_id == "unresolved_queries":
        if any(k in msg_lower for k in emergency_keywords):
            is_transfer = True
            telemetry_badge = "EMERGENCY TRIAGE"
        elif any(k in msg_lower for k in ["custom engineering", "out of scope", "blueprint", "commercial spec", "specialist consult", "technical expert", "engineering spec"]):
            is_transfer = True
            telemetry_badge = "SPECIALIST TRANSFER"
        elif any(k in msg_lower for k in ["transfer", "human", "speak to owner", "manager"]):
            is_transfer = False
            telemetry_badge = "STANDARD INQUIRY"
        elif any(k in msg_lower for k in ["diy", "myself", "how to wire", "rewire", "which wire", "bypass", "diagnose my", "prescribe", "legal opinion", "do it myself", "fix it myself", "tell me how to fix"]):
            telemetry_badge = "GUARDRAIL ENFORCED"
        elif any(k in msg_lower for k in ["after hours", "midnight", "night", "11:45", "open now", "closed", "2 am", "sunday", "outside hours", "emergency line"]):
            telemetry_badge = "SCHEDULE VERIFICATION"
    else:  # life_safety_emergencies (default)
        if any(k in msg_lower for k in emergency_keywords):
            is_transfer = True
            telemetry_badge = "EMERGENCY TRIAGE"
        elif any(k in msg_lower for k in ["transfer", "human", "speak to owner", "manager", "operator"]):
            is_transfer = False
            telemetry_badge = "STANDARD INQUIRY"
        elif any(k in msg_lower for k in ["diy", "myself", "how to wire", "rewire", "which wire", "bypass", "diagnose my", "prescribe", "legal opinion", "do it myself", "fix it myself", "tell me how to fix"]):
            telemetry_badge = "GUARDRAIL ENFORCED"
        elif any(k in msg_lower for k in ["after hours", "midnight", "night", "11:45", "open now", "closed", "2 am", "sunday", "outside hours", "emergency line"]):
            telemetry_badge = "SCHEDULE VERIFICATION"

    groq_key = settings.GROQ_API_KEY
    greeting_msg = f"Thank you for calling {biz_name}! This is {persona_name}. How can I help you today?"

    if groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            messages = [{"role": "system", "content": system_prompt}]
            if history:
                for h in history[-4:]:
                    messages.append({"role": h["role"], "content": h["content"]})
            else:
                messages.append({"role": "assistant", "content": greeting_msg})
            messages.append({"role": "user", "content": user_message})

            completion = client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=messages,
                temperature=0.4,
                max_tokens=600,
            )
            reply = completion.choices[0].message.content.strip()
            reply = reply.replace("*", "").replace("#", "").replace("- ", "")

            if reply and len(reply) >= 10:
                # Check if model triggered an emergency escalation or transfer
                if any(t in reply.lower() for t in ["connecting you", "transferring you", "on-call", "emergency team", "evacuate", "step outside"]):
                    if pol_id != "never_transfer_take_message":
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
            else:
                messages.append({"role": "assistant", "content": greeting_msg})
            messages.append({"role": "user", "content": user_message})

            completion = g_client.chat.completions.create(
                model=getattr(settings, "GEMINI_MODEL", "gemini-2.5-flash"),
                messages=messages,
                temperature=0.4,
                max_tokens=600,
            )
            reply = completion.choices[0].message.content.strip()
            reply = reply.replace("*", "").replace("#", "").replace("- ", "")

            if reply and len(reply) >= 10:
                # Check if model triggered an emergency escalation or transfer
                if any(t in reply.lower() for t in ["connecting you", "transferring you", "on-call", "emergency team", "evacuate", "step outside"]):
                    if pol_id != "never_transfer_take_message":
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

    # Heuristic Fallback Responses Faithfully Mirroring Onboarding Rules
    forwarding = client_profile.get("forwarding_phone") or client_profile.get("owner_phone") or "our on-call line"
    after_hours_action = (client_profile.get("after_hours_action") or client_profile.get("night_action") or "book_morning").lower()

    # Dynamic schedule and night-call status
    raw_b = str(client_profile.get("booking_action") or client_profile.get("booking_mode") or "").lower()
    is_text_details_mode = any(k in raw_b for k in ["text", "reach out", "callback", "owner schedules", "lead capture", "details"])
    sched_eval = evaluate_client_schedule_status(client_profile)
    is_closed = not sched_eval.get("is_open", True)
    night_act = sched_eval.get("night_action", "book_earliest_slot")
    # Dynamic answering decision
    answering_decision = determine_call_answering_decision(client_profile)
    if answering_decision["mode"] == "after_hours" and not answering_decision["should_answer"]:
        return {
            "response": f"Thank you for calling {biz_name}! Our main office is open right now ({sched_eval['local_time']}). Let me connect you directly with our front desk team at {forwarding} right now.",
            "is_transfer": True,
            "telemetry_badge": "OFFICE OPEN TRANSFER",
            "business_name": biz_name,
        }

    if is_transfer:
        if "gas" in msg_lower:
            reply = f"For your safety, please step outside immediately. I am transferring you directly to our on-call emergency team at {forwarding} right now."
        elif "water" in msg_lower or "pipe" in msg_lower or "flood" in msg_lower:
            reply = f"Please shut off your main water valve right away to prevent property damage! I am transferring you directly to our emergency technician at {forwarding}."
        elif "specialist" in telemetry_badge.lower():
            reply = f"That is a specialized technical request. Let me connect you directly with our senior specialist at {forwarding} right now. Please stay on the line."
        else:
            reply = f"I understand completely. I am transferring you directly to our team at {forwarding} right now. Please stay on the line."
    elif pol_id == "never_transfer_take_message" and (telemetry_badge in ["HIGH PRIORITY INTAKE", "MESSAGE INTAKE"] or any(k in msg_lower for k in emergency_keywords)):
        reply = f"I have marked your request with top priority! I can take down your service address and issue right now, and our service manager at {biz_name} will follow up with you directly via phone or text right away! What is your street address?"
    elif telemetry_badge == "GUARDRAIL ENFORCED":
        reply = f"For your safety and warranty protection, our certified technicians cannot give DIY repair instructions over the phone. We'd be glad to send a technician out to inspect and fix this safely for you. Shall we get a visit scheduled?"
    elif telemetry_badge == "SCHEDULE VERIFICATION" or (is_closed and any(k in msg_lower for k in ["come", "book", "schedule", "appointment", "visit", "tonight", "available", "open", "tonight", "right now", "immediately"])):
        telemetry_badge = "SCHEDULE VERIFICATION"
        if "transfer" in night_act or "tech" in night_act:
            reply = f"Our standard office hours are {hours}, and our office is currently closed for the night ({sched_eval['local_time']}). For urgent emergencies, I can connect you directly to our on-call technician at {forwarding}, or schedule our earliest regular visit for {sched_eval['next_opening']}. Is this an active emergency?"
        elif is_text_details_mode or "sms" in night_act or "lead" in night_act:
            reply = f"Our standard office hours are {hours}, and our office is currently closed for the night ({sched_eval['local_time']}). I can take down your service details right now, and our service team at {biz_name} will reach out to you first thing tomorrow morning to coordinate the best appointment time! What is your street address?"
        else:
            reply = f"Our standard office hours are {hours}. Since our office is currently closed for the night ({sched_eval['local_time']}), I can lock in our very first priority arrival window for {sched_eval['next_opening']} for you! Would that work?"
    elif any(k in msg_lower for k in ["price", "cost", "fee", "rate", "quote"]):
        pricing_rule = client_profile.get("pricing_policy") or client_profile.get("diagnostic_fee") or "Diagnostic fee credited toward repair"
        if "free" in pricing_rule.lower() or "complimentary" in pricing_rule.lower():
            reply = f"We provide a 100% complimentary on-site inspection and estimate with zero obligation! What service are you looking to have done?"
        elif "upfront" in pricing_rule.lower():
            reply = f"Our certified technician inspects everything on-site first and provides an upfront guaranteed flat-rate price before starting any work. What service do you need help with?"
        else:
            fee_num = "".join(filter(str.isdigit, pricing_rule))
            if fee_num:
                reply = f"Our diagnostic fee is ${fee_num}, which covers a complete on-site inspection by our certified technician, and we credit that full fee toward any repair you approve! What issue are you experiencing?"
            else:
                reply = f"Our diagnostic fee covers a complete on-site inspection by our certified technician, and we credit that fee directly toward any repair you approve! What issue are you experiencing?"
    elif any(k in msg_lower for k in ["where", "address", "location"]):
        addr = client_profile.get("address", "")
        clean_addr, clean_city = sanitize_address_and_city(addr, client_profile.get("city", ""))
        reply = f"We are based in {clean_city} and dispatch our fully equipped service units directly to your location. What is your street address?"
    elif any(k in msg_lower for k in ["install", "new ac", "new unit", "replacement", "no ac", "don't have an ac", "need an ac", "new roof", "re-roof", "repipe", "panel replacement", "upgrade"]):
        if trade_key == "roofing":
            reply = f"We can definitely take care of that! We offer comprehensive on-site roof inspections and estimates. Would you like to check our available times for an estimate?"
        elif trade_key == "plumbing":
            reply = f"We can definitely take care of that! We provide on-site assessments and upfront quotes for all installations and fixture replacements. Would you like to check available appointment times?"
        elif trade_key == "electrical":
            reply = f"We can definitely help with that! Our licensed electricians provide full on-site assessments and transparent pricing. Would you like to schedule a consultation?"
        elif trade_key == "auto":
            reply = f"We can certainly assist with parts installation and major vehicle services. Would you like to schedule a time to drop off your vehicle?"
        elif trade_key == "legal":
            reply = f"We can schedule an initial legal consultation with one of our attorneys to evaluate your case. Would morning or afternoon suit you best?"
        elif trade_key == "dental_medical":
            reply = f"We would be delighted to welcome you to our practice! Would you like to schedule an initial consultation and examination?"
        elif trade_key == "realestate":
            reply = f"We can definitely assist you with your property goals! Would you like to connect with our listing specialist or view available properties?"
        elif trade_key == "salon_spa":
            reply = f"We would love to book your appointment! What specific treatments or services are you looking to receive?"
        elif trade_key == "restaurant":
            reply = f"We would love to accommodate you! Are you looking to reserve a table, place an order, or discuss private event catering?"
        else:
            reply = f"We can definitely take care of that! We offer free in-person estimates where our specialist inspects your setup and provides exact options. Would you like to check our available times for a free consultation?"
    elif any(k in msg_lower for k in ["not working", "broken", "warm air", "shut off", "rattle", "noise", "cooling", "trouble", "issue", "problem", "leak", "won't turn", "damaged", "stain", "fault"]):
        trade_trouble_prompts = {
            "roofing": "Dealing with roof damage or a leak can be stressful! Can you tell me a bit more—is water actively dripping inside, or did you notice missing shingles or storm damage?",
            "plumbing": "Plumbing issues can be so disruptive! What seems to be happening—is there an active water leak, a backed-up drain, or an issue with your water heater?",
            "electrical": "Electrical problems require careful attention! What kind of issue are you seeing—is a circuit breaker constantly tripping, an outlet dead, or lights flickering?",
            "hvac": "Oh no, dealing with heating or AC trouble is such a headache! What seems to be happening—is it blowing the wrong temperature, making a strange sound, or completely shut off?",
            "auto": "Car trouble is always frustrating! What seems to be going on with the vehicle—is there a check engine light, an unusual sound, or a drivability issue?",
            "dental_medical": "I am sorry to hear you are having discomfort. Could you describe the symptoms or dental issue you are experiencing so we can best assist you?",
            "legal": "I understand this is an important matter. Could you briefly share what legal issue or dispute you are seeking counsel for?",
            "realestate": "I'd be glad to help resolve any property inquiries. What specific question or issue do you have regarding the property or listing?",
            "restaurant": "We are sorry to hear there is an issue with your order or reservation. Can you tell me what happened so we can make it right immediately?",
            "salon_spa": "I would be happy to help adjust or resolve any questions about your booking or service. What can I do for you today?",
            "general": "Oh no, dealing with trouble is such a headache! What seems to be happening, and how can our team best help you today?"
        }
        reply = trade_trouble_prompts.get(trade_key, trade_trouble_prompts["general"])
    elif any(k in msg_lower for k in ["book", "schedule", "appointment", "come over", "visit", "when can someone", "tour", "reservation"]):
        raw_b = str(client_profile.get("booking_action") or "").lower()
        if any(k in raw_b for k in ["text", "reach out", "callback", "owner schedules", "lead capture", "details"]):
            reply = f"I can take down your contact details and request right now, and our team at {biz_name} will reach out to you shortly to coordinate the best time! What is the best phone number and address for you?"
        else:
            reply = f"I can get an arrival window scheduled for you right away with {biz_name}! We have openings today between one and three, or tomorrow morning between eight and eleven. Which works better for you?"
    else:
        trade_greeting_prompts = {
            "roofing": f"Thanks for calling {biz_name}, this is {persona_name}! We specialize in roof inspections, leak repairs, and full replacements. How can we help you today?",
            "plumbing": f"Thanks for calling {biz_name}, this is {persona_name}! We handle all plumbing repairs, drain cleanings, and water heaters. How can we help you today?",
            "electrical": f"Thanks for calling {biz_name}, this is {persona_name}! We handle all residential and commercial electrical service and panel upgrades. How can we assist you today?",
            "hvac": f"Thanks for calling {biz_name}, this is {persona_name}! We can certainly take care of that for you. What seems to be going on with your heating or cooling today?",
            "auto": f"Thanks for calling {biz_name}, this is {persona_name}! We provide complete auto repair, maintenance, and diagnostics. How can we help with your vehicle today?",
            "dental_medical": f"Thanks for calling {biz_name}, this is {persona_name}! How may our care team assist you with an appointment or inquiry today?",
            "legal": f"Thanks for calling {biz_name}, this is {persona_name}! How may our legal team assist you today?",
            "realestate": f"Thanks for calling {biz_name}, this is {persona_name}! How can our realty team assist you with buying, selling, or touring today?",
            "restaurant": f"Thanks for calling {biz_name}, this is {persona_name}! How can we assist you with reservations, our menu, or dining today?",
            "salon_spa": f"Thanks for calling {biz_name}, this is {persona_name}! How may we assist you with scheduling your appointment today?",
            "general": f"Thanks for calling {biz_name}, this is {persona_name}! We can certainly take care of that for you. What can we help you with today?"
        }
        reply = trade_greeting_prompts.get(trade_key, trade_greeting_prompts["general"])

    return {
        "response": reply,
        "is_transfer": is_transfer,
        "telemetry_badge": telemetry_badge,
        "business_name": biz_name,
    }

def sanitize_address_and_city(raw_addr: str, raw_city: str = "") -> Tuple[str, str]:
    """Cleans addresses and extracts municipal city names.
    Strips real-estate & search listing metadata like Property type, Year built, Last sold,
    MLS, Zestimate, Sq Ft descriptions, etc., and returns (clean_address, clean_city).
    """
    clean_addr = (raw_addr or "").strip()
    clean_addr = re.split(r"[·|•\n\r]", clean_addr)[0].strip()
    meta_pat = r"(?i)\s*(Property type|Year built|Last sold|Single Family|Multi Family|Condo|Townhouse|Beds?|Baths?|Sq\.?\s*Ft|Square Feet|Zestimate|MLS|Est\.\s*Value|Lot Size).*$"
    clean_addr = re.sub(meta_pat, "", clean_addr).strip()
    clean_addr = clean_addr.rstrip(" ,.-")

    clean_city = (raw_city or "").strip()
    clean_city = re.split(r"[·|•\n\r]", clean_city)[0].strip()
    clean_city = re.sub(meta_pat, "", clean_city).strip()
    clean_city = clean_city.rstrip(" ,.-")

    listing_desc_pat = r"(?i)\b(offering|featuring|features|spacious|gorgeous|stunning|welcome to|luxury|charming|renovated|remodeled|living space|elevator|levels|stories|finished|open concept|custom built|master suite|chef'?s kitchen)\b"
    street_suffix_pat = r"(?i)\b(street|st|avenue|ave|boulevard|blvd|road|rd|drive|dr|lane|ln|court|ct|circle|cir|way|place|pl|terrace|ter|parkway|pkwy|highway|hwy|route|rt)\b"

    if re.search(listing_desc_pat, clean_addr):
        if re.search(r"^(offering|featuring|features|welcome|spacious|gorgeous|stunning|luxury|charming)\b", clean_addr, re.I) or not re.search(street_suffix_pat, clean_addr):
            clean_addr = ""

    target_for_city = clean_city if clean_city and not re.search(r"^\d+\s+", clean_city) and "," not in clean_city else clean_addr

    city = ""
    if target_for_city:
        parts = [p.strip() for p in target_for_city.split(",") if p.strip()]
        if len(parts) >= 3:
            cand = parts[1]
        elif len(parts) == 2:
            cand = parts[0] if re.search(r"^[A-Za-z]{2}(\s+\d{5})?$", parts[1].strip()) else parts[1]
        else:
            cand = parts[0]
        
        cand = re.sub(r"\b[A-Za-z]{2}\b\s*\d{5}(-\d{4})?$", "", cand).strip()
        cand = re.sub(r"\b\d{5}(-\d{4})?$", "", cand).strip()
        cand = re.sub(r"\b[A-Za-z]{2}$", "", cand).strip()
        cand = cand.rstrip(" ,.-")
        if cand and not re.search(r"^\d+", cand) and len(cand) < 40 and not re.search(listing_desc_pat, cand):
            city = cand.title()

    if not city or len(city) < 2:
        city = "the local area"

    if not clean_addr:
        clean_addr = f"{city} Area" if city != "the local area" else "Local Area"

    return clean_addr, city

TRADE_KEYWORDS = {
    "hvac": [r"\bheating\s*(&|and)?\s*cooling\b", r"\bhvac\b", r"\bair\s+conditioning\b", r"\bfurnace\b"],
    "plumbing": [r"\bplumbing\b", r"\bplumber(s)?\b", r"\bdrain(s)?\b", r"\brooter\b"],
    "electrical": [r"\belectrical\b", r"\belectric(ians?)?\b"],
    "roofing": [r"\broofing\b", r"\broof(s)?\b", r"\bgutter(s)?\b"],
    "dental_medical": [r"\bdental\b", r"\bdentist(ry)?\b", r"\bclinic\b", r"\borthodontic(s)?\b"],
    "auto": [r"\bauto(motive)?\b", r"\bcar\s+repair\b", r"\bmechanic(s)?\b", r"\bmotors?\b"],
    "legal": [r"\blaw(\s+firm|\s+group)?\b", r"\blegal\b", r"\battorney(s)?\b"],
    "restaurant": [r"\brestaurant\b", r"\bbistro\b", r"\bcafe\b", r"\bgrill\b", r"\bdining\b"],
    "realestate": [r"\breal\s*estate\b", r"\brealty\b", r"\brealtor(s)?\b", r"\bproperties\b"],
    "salon_spa": [r"\bsalon\b", r"\bspa\b", r"\bbarber(shop)?\b", r"\bhair\b"],
}

TRADE_REPLACEMENT_SUFFIX = {
    "hvac": "Heating & Cooling",
    "plumbing": "Plumbing & Drain",
    "electrical": "Electrical Services",
    "roofing": "Roofing & Restorations",
    "dental_medical": "Family Dental Care",
    "auto": "Auto Repair & Diagnostics",
    "legal": "Law Group",
    "restaurant": "Bistro & Dining",
    "realestate": "Realty & Property Group",
    "salon_spa": "Salon & Spa",
    "general": "Services",
}

def harmonize_business_name_for_trade(biz_name: str, target_trade: str) -> str:
    """Ensures business name doesn't contradict the active trade in simulation scripts."""
    if not biz_name or biz_name.strip().lower() in ("our company", "my business", "apex services"):
        return biz_name or "Our Company"

    # If the business name is composed mostly of phone digits (e.g. "(454) 545-4554"), default to trade standard
    clean_digits = re.sub(r"\D", "", biz_name)
    non_phone_chars = re.sub(r"[\d\s\(\)\-\+\.]", "", biz_name)
    if len(clean_digits) >= 7 and len(non_phone_chars) == 0:
        trade_defaults = {
            "hvac": "Comfort Breeze Heating and Air",
            "plumbing": "Apex Plumbing & Drain",
            "electrical": "VoltCraft Electric",
            "roofing": "Apex Roofing Solutions",
            "dental_medical": "Gentle Dental Care",
            "auto": "Precision Auto Care",
            "legal": "Apex Legal Group",
            "restaurant": "The Bistro Grill",
            "realestate": "Apex Realty Partners",
            "salon_spa": "Luxe Beauty Studio",
            "general": "Apex Services"
        }
        return trade_defaults.get(target_trade, "Comfort Breeze Heating and Air")
    
    # Check if biz_name contains keywords for ANY trade other than target_trade
    for trade, patterns in TRADE_KEYWORDS.items():
        if trade == target_trade:
            continue
        for pat in patterns:
            m = re.search(pat, biz_name, re.I)
            if m:
                new_suffix = TRADE_REPLACEMENT_SUFFIX.get(target_trade, "Services")
                prefix = biz_name[:m.start()].strip()
                prefix = re.sub(r"[\s,&/-]+$", "", prefix).strip()
                if prefix:
                    return f"{prefix} {new_suffix}"
                else:
                    return f"{new_suffix}"
    return biz_name


def phone_to_spoken_words(phone_str: str) -> str:
    digit_words = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
    }
    digits = [c for c in str(phone_str) if c.isdigit()]
    if len(digits) == 11 and digits[0] == '1':
        digits = digits[1:]
    if len(digits) == 10:
        area = " ".join(digit_words[d] for d in digits[0:3])
        prefix = " ".join(digit_words[d] for d in digits[3:6])
        line = " ".join(digit_words[d] for d in digits[6:10])
        return f"{area}, {prefix}, {line}"
    return " ".join(digit_words.get(d, d) for d in digits)


def generate_call_demo_script(profile: Dict[str, Any], scenario_id: Optional[str] = None) -> Dict[str, Any]:
    """Generates realistic dual-voice telephone call demos for ANY of the 11 business types:
    HVAC, Plumbing, Electrical, Roofing, Dental/Medical, Auto Repair, Law Firm,
    Restaurant, Real Estate, Salon & Spa, and General / Professional Services.
    
    Agent ALWAYS answers first on Ring 1 using the business greeting.
    Scripts sound natural, articulate, warm, and consultative.
    Never robotic, never reads lists of hours/territories out of nowhere.
    """
    raw_trade  = (profile.get("trade") or profile.get("industry") or "hvac").lower().strip()
    booking    = profile.get("booking_action") or "Book arrival window"
    persona_name  = profile.get("persona_name") or "Riley"
    persona_voice = profile.get("persona_voice") or "flux-heather-en"
    customer_voice = "flux-bruce-en"
    active_scenario = (scenario_id or "routine_booking").lower()
    if active_scenario in ("after_hours", "afterhours", "after-hours", "after_hours_test"):
        active_scenario = "after_hours_test"
    elif active_scenario in ("daytime", "routine", "booking", "routine_booking"):
        active_scenario = "routine_booking"

    trade_map = {
        "hvac": ["hvac", "ac", "air conditioning", "heating", "cooling", "furnace", "climate"],
        "plumbing": ["plumb", "drain", "water heater", "sewer", "pipe"],
        "electrical": ["electr", "panel", "wiring", "breaker", "lighting", "generator"],
        "roofing": ["roof", "gutter", "shingle"],
        "dental_medical": ["dental", "medical", "clinic", "doctor", "hygiene", "dentist", "health", "teeth"],
        "auto": ["auto", "car", "mechanic", "vehicle", "brake", "transmission", "motor", "tire"],
        "legal": ["legal", "law", "attorney", "lawyer", "counsel", "litigation"],
        "restaurant": ["restaurant", "dining", "cafe", "bistro", "catering", "hospitality", "food"],
        "realestate": ["realestate", "real estate", "realty", "realtor", "property", "tenant", "listing"],
        "salon_spa": ["salon", "spa", "hair", "barber", "nail", "lash", "massage", "aesthetic", "beauty"],
        "general": ["general", "other", "consult", "contractor", "facility", "maintenance"],
    }
    trade_key = "general"
    for tk, aliases in trade_map.items():
        if any(a in raw_trade for a in aliases):
            trade_key = tk
            break

    # Harmonize business name so e.g. "Seattle Heating & Cooling" becomes "Seattle Roofing & Restorations" for roofing
    raw_biz = (profile.get("business_name") or profile.get("name") or "Our Company").strip()
    biz_name = harmonize_business_name_for_trade(raw_biz, trade_key)

    DEFAULT_TRADE_PRICING = {
        "hvac": "$89 diagnostic fee applied toward repair",
        "plumbing": "$79 service fee credited to approved repair",
        "electrical": "$89 diagnostic inspection fee applied toward repair",
        "roofing": "Complimentary 21-point roof inspection with zero obligation",
        "dental_medical": "Complimentary consultation & in-network PPO billing",
        "auto": "$65 digital OBD2 scan credited toward approved repair",
        "legal": "Free initial consultation & 100% contingency fee",
        "restaurant": "Complimentary table reservation (Zero fee)",
        "realestate": "100% Free buyer representation & market valuation",
        "salon_spa": "Transparent tier-based pricing & free consultation",
        "general": "Complimentary on-site consultation & written estimate",
    }
    raw_pricing = profile.get("pricing_policy")
    if not raw_pricing or raw_pricing.strip().lower() in [
        "diagnostic fee applied to repair",
        "diagnostic fee credited toward repair",
        "standard service rate",
        ""
    ]:
        pricing = DEFAULT_TRADE_PRICING.get(trade_key, "Diagnostic fee applied to repair") if trade_key != "hvac" else (raw_pricing or "$89 diagnostic fee applied toward repair")
    else:
        pricing = raw_pricing.strip()

    # Dynamic Hours resolution
    configured_hours = profile.get("hours")
    if configured_hours and len(configured_hours) > 4:
        hours_line = f"open {configured_hours}"
    else:
        hours_defaults = {
            "hvac": "open Monday through Saturday, 7:00 AM to 7:00 PM",
            "plumbing": "open Monday through Saturday, 7:00 AM to 7:00 PM",
            "electrical": "open Monday through Friday 7:00 AM to 6:00 PM, and Saturday 8:00 AM to 4:00 PM",
            "roofing": "open Monday through Saturday, 7:00 AM to 7:00 PM",
            "dental_medical": "open Monday through Thursday 8:00 AM to 5:00 PM, and Friday 8:00 AM to 1:00 PM",
            "auto": "open Monday through Friday, 7:30 AM to 5:30 PM",
            "legal": "open Monday through Friday, 9:00 AM to 5:30 PM",
            "restaurant": "open Tuesday through Sunday from 5:00 PM to 10:00 PM",
            "realestate": "open daily from 9:00 AM to 7:00 PM for private showings and consultations",
            "salon_spa": "open Tuesday through Saturday, 9:00 AM to 7:00 PM",
            "general": "open Monday through Friday, 8:00 AM to 6:00 PM",
        }
        hours_line = hours_defaults.get(trade_key, "open Monday through Friday, 8:00 AM to 6:00 PM")

    # Dynamic Clean City & Coverage Resolution (Strips property metadata snippets cleanly)
    raw_addr = (profile.get("address") or profile.get("query") or "").strip()
    raw_city = (profile.get("city") or "").strip()
    clean_addr, city_disp = sanitize_address_and_city(raw_addr, raw_city)

    # Dynamic Custom Services from Onboarding Playbook
    raw_services = profile.get("services") or profile.get("service_options") or []
    if isinstance(raw_services, str):
        services_list = [s.strip() for s in raw_services.split(",") if s.strip()]
    elif isinstance(raw_services, list):
        services_list = [str(s).strip() for s in raw_services if str(s).strip()]
    else:
        services_list = []

    if len(services_list) >= 3:
        custom_services_str = f"{services_list[0]}, {services_list[1]}, and {services_list[2]}"
        primary_service = services_list[0]
    elif len(services_list) == 2:
        custom_services_str = f"{services_list[0]} and {services_list[1]}"
        primary_service = services_list[0]
    elif len(services_list) == 1:
        custom_services_str = services_list[0]
        primary_service = services_list[0]
    else:
        custom_services_str = ""
        primary_service = ""

    # Dynamic spoken fee resolution
    raw_fee = profile.get("fee_amount") or profile.get("diagnostic_fee")
    if not raw_fee and "$" in pricing:
        m_fee = re.search(r"\$(\d+)", pricing)
        if m_fee:
            raw_fee = int(m_fee.group(1))
    if raw_fee is not None and str(raw_fee).strip() != "":
        spoken_fee = fee_to_spoken(raw_fee)
    else:
        spoken_fee = fee_to_spoken(89)

    if "free" in pricing.lower() or "complimentary" in pricing.lower() or "$0" in pricing or spoken_fee == "complimentary":
        fee_agent_answer = "Our initial inspection and estimate are completely complimentary with zero obligation! Does that sound fair?"
        ah_fee_agent_answer = "Glad there's no safety hazard! Here's what we can do — I'll reserve our very first priority slot tomorrow morning between eight and ten AM so a technician is at your door first thing. Our initial inspection is completely complimentary with zero upfront cost! Does that work for you?"
        pricing_spoken = "The initial consultation is completely free, with no obligation whatsoever."
        pricing_text = "and our consultation and initial assessment are completely free — zero upfront cost, no obligation."
    elif "upfront" in pricing.lower() or "inspection after" in pricing.lower():
        fee_agent_answer = "Our technician evaluates the system on-site and provides a guaranteed upfront flat-rate price before any work begins! Does that sound fair?"
        ah_fee_agent_answer = "Glad there's no safety hazard! Here's what we can do — I'll reserve our very first priority slot tomorrow morning between eight and ten AM so a technician evaluates on-site and provides a guaranteed upfront price before any work begins. Does that work for you?"
        pricing_spoken = "Our technician provides a guaranteed upfront price after evaluating on-site."
        pricing_text = "our technician evaluates everything on-site and provides a guaranteed upfront price before any work starts."
    else:
        fee_agent_answer = (
            f"Our diagnostic fee is a flat {spoken_fee}, which covers a full comprehensive inspection of your system by a certified technician. "
            f"And the best part is, if you decide to move forward with the repair, we credit that full {spoken_fee} directly toward the cost of the repair! Does that sound fair?"
        )
        ah_fee_agent_answer = (
            f"Glad there's no safety hazard! Here's what we can do — I'll reserve our very first priority slot tomorrow morning between eight and ten AM so a technician is at your door first thing. "
            f"Our diagnostic fee is a flat {spoken_fee}, and we credit that full amount directly toward the repair! Does that work for you?"
        )
        pricing_spoken = f"The diagnostic fee is a flat {spoken_fee}, and it's credited back to you if you move forward with the repair."
        pricing_text = f"the initial diagnostic fee is a flat {spoken_fee}, and that fee gets credited right back toward your repair if you move forward."

    is_text_details_booking = any(k in booking.lower() for k in ["text", "reach out", "callback", "owner schedules", "lead capture", "details"])
    owner_phone_display = profile.get("sms_phone") or profile.get("phone") or "+1 (555) 234-5678"

    # ════════════════════════════════════════════════════════════════════════
    # TRADE DATA DEFINITIONS FOR ALL 11 TRADES (DOMAIN SPECIFIC & HUMAN-LIKE)
    # ════════════════════════════════════════════════════════════════════════
    TRADE_DEMO_DATA = {
        "hvac": {
            "title": "AC Not Cooling — Diagnostic Service",
            "caller_name": "David (Homeowner)",
            "caller_name_full": "David Thompson",
            "caller_phone": "+1 (713) 555-0184",
            "caller_phone_raw": "713-555-0184",
            "caller_phone_display": "(713) 555-0184",
            "customer_address": "419 Maple Drive",
            "location_display": "419 Maple Drive",
            "routine_slot": "Tomorrow 8:00 AM – 11:00 AM Window",
            "routine_status": "Service Window Confirmed — SMS Sent",
            "owner_alert_title": "⚡ New Service Job Booked",
            "booking_confirm_line": f"You're all set, David! We have a certified technician scheduled for 419 Maple Drive tomorrow morning between eight and eleven. We're texting confirmation and tracking to (713) 555-0184. Does everything sound good?",
            "sms_body_routine": f"Hi David! You're confirmed with {biz_name} for tomorrow, 8–11 AM at 419 Maple Drive. Our technician will text 15 min before arrival. Reply anytime with questions.",
            "daytime_cust_problem": "My AC isn't working.",
            "daytime_agent_service": "Oh no, dealing with AC trouble is such a headache! What seems to be happening with the system—is it blowing warm air, making a strange sound, or completely shut off?",
            "daytime_cust_clarify": "Don't know, it's just not blowing cold air.",
            "daytime_agent_consult": "I understand, it's frustrating when you can't quite pinpoint the issue! Got it, that definitely sounds like something one of our technicians should inspect to diagnose properly. We can get you on the schedule so our team can come out and take care of that for you. Would you like to check our available appointment times?",
            "ah_slot": "Tomorrow 8:00 AM – 10:00 AM Priority Window",
            "ah_status": "After-Hours Priority Booked — Tech Alerted",
            "owner_alert_ah_title": "🌙 After-Hours HVAC Service Request",
            "ah_booking_line": f"You're all set, Marcus! I've reserved our priority eight to ten AM window for you tomorrow at 1042 Bayside Avenue, and alerted our on-call team. Confirmation and arrival tracking are on your phone. Does everything sound good?",
            "sms_body_ah": f"Hi Marcus! You're confirmed with {biz_name} for tomorrow, 8–10 AM at 1042 Bayside Avenue. Our technician will text 15 min before arrival. Reply anytime.",
            "ah_cust_problem": "Oh wow, someone actually picked up. Look, it's almost midnight and my AC completely stopped working. It's eighty-two degrees inside and I've got my kids here. I didn't think anyone would answer at this hour.",
            "ah_agent_triage": "I understand, having the AC stop working late at night with kids in the house is stressful. Let's make sure you're safe first — are you noticing any burning smell or strange noises from the unit?",
            "ah_cust_clarify": "No, no smell or anything — it just stopped blowing cold air entirely. What can you actually do for me right now? And what's the cost?",
        },
        "plumbing": {
            "title": "Water Heater Issue & Diagnostic",
            "caller_name": "David (Homeowner)",
            "caller_name_full": "David Miller",
            "caller_phone": "+1 (512) 555-0198",
            "caller_phone_raw": "512-555-0198",
            "caller_phone_display": "(512) 555-0198",
            "customer_address": "724 Oak Crest Lane",
            "location_display": "724 Oak Crest Lane",
            "routine_slot": "Tomorrow 8:00 AM – 11:00 AM Window",
            "routine_status": "Plumber Dispatched — SMS Sent",
            "owner_alert_title": "⚡ New Plumbing Job Booked",
            "booking_confirm_line": f"You're all set, David! We have a licensed plumber scheduled for 724 Oak Crest Lane tomorrow morning between eight and eleven. We're texting confirmation and tracking to (512) 555-0198. Does everything sound good?",
            "sms_body_routine": f"Hi David! You're confirmed with {biz_name} for tomorrow, 8–11 AM at 724 Oak Crest Lane. Our plumber will text 15 min before arrival. Reply anytime with questions.",
            "daytime_cust_problem": "My water heater isn't working.",
            "daytime_agent_service": "Oh no, dealing with water heater trouble is definitely stressful! What seems to be happening—is it leaking water, making a strange sound, or not producing hot water?",
            "daytime_cust_clarify": "Don't know, there's water pooling around the bottom and no hot water.",
            "daytime_agent_consult": "I understand, that definitely sounds like something our licensed plumber should inspect to diagnose properly. We can get you on the schedule so our team can take care of that for you. Would you like to check our available appointment times?",
            "ah_slot": "Tomorrow 8:00 AM – 10:00 AM Priority Window",
            "ah_status": "After-Hours Urgent Triage — Plumber Alerted",
            "owner_alert_ah_title": "🌙 After-Hours Plumbing Alert",
            "ah_booking_line": f"You're all set, Marcus! I've reserved our priority eight to ten AM window for you tomorrow at 1042 Bayside Avenue, and alerted our master plumber. Confirmation is on your phone. Does everything sound good?",
            "sms_body_ah": f"Hi Marcus! You're confirmed with {biz_name} for tomorrow, 8–10 AM at 1042 Bayside Avenue. Our plumber will text 15 min before arrival. Reply anytime.",
            "ah_cust_problem": "Hello? I know it's late, almost midnight, but my kitchen sink is completely backed up and gurgling dirty water into the basin. Can someone help?",
            "ah_agent_triage": "I hear you — a late-night kitchen backup is definitely frustrating! First, is water overflowing onto your cabinets or floor, or is it contained inside the sink bowl?",
            "ah_cust_clarify": "It's sitting in the sink for now, not overflowing yet. What can you do for me right now? And what's the cost?",
        },
        "electrical": {
            "title": "Circuit Breaker Tripping & Diagnostic",
            "caller_name": "Robert (Property Owner)",
            "caller_name_full": "Robert Davis",
            "caller_phone": "+1 (404) 555-0177",
            "caller_phone_raw": "404-555-0177",
            "caller_phone_display": "(404) 555-0177",
            "customer_address": "812 Highland View",
            "location_display": "812 Highland View",
            "routine_slot": "Tomorrow 8:00 AM – 11:00 AM Window",
            "routine_status": "Electrician Dispatched — SMS Sent",
            "owner_alert_title": "⚡ New Electrical Job Booked",
            "booking_confirm_line": f"You're all set, Robert! We have a certified electrician scheduled for 812 Highland View tomorrow morning between eight and eleven. We're texting confirmation and tracking to (404) 555-0177. Does everything sound good?",
            "sms_body_routine": f"Hi Robert! You're confirmed with {biz_name} for tomorrow, 8–11 AM at 812 Highland View. Our electrician will text 15 min before arrival. Reply anytime with questions.",
            "daytime_cust_problem": "My breaker keeps tripping.",
            "daytime_agent_service": "Oh no, dealing with electrical issues is so frustrating! What seems to be happening—is a breaker tripping, lights flickering, or did a whole room lose power?",
            "daytime_cust_clarify": "Don't know, half the kitchen just lost power whenever we turn on an appliance.",
            "daytime_agent_consult": "I understand, that definitely sounds like something our certified electrician should inspect for safety. We can get you on the schedule so our team can come out. Would you like to check our available appointment times?",
            "ah_slot": "Tomorrow 8:00 AM Priority Safety Slot",
            "ah_status": "After-Hours Hazard Triage — Electrician Alerted",
            "owner_alert_ah_title": "🌙 After-Hours Electrical Hazard Alert",
            "ah_booking_line": f"You're all set, Marcus! I've reserved our first priority eight AM slot tomorrow at 1042 Bayside Avenue, and our master electrician is on alert. Confirmation is on your phone. Does everything sound good?",
            "sms_body_ah": f"Hi Marcus! You're confirmed with {biz_name} for tomorrow at 8:00 AM at 1042 Bayside Avenue. Our electrician will text 15 min before arrival. Reply anytime.",
            "ah_cust_problem": "Hi! It's almost midnight and my breaker box started making a faint buzzing sound, and the living room lights are flickering. I'm really nervous about an electrical fire.",
            "ah_agent_triage": "Thank you for calling — safety is our top priority. First, do you see any visible sparks, smoke, or is the panel hot to the touch?",
            "ah_cust_clarify": "No smoke or sparks, but the breaker switch feels warm. What can you do for me right now? And what's the cost?",
        },
        "roofing": {
            "title": "Ceiling Leak & Shingle Inspection",
            "caller_name": "Tom (Homeowner)",
            "caller_name_full": "Tom Reynolds",
            "caller_phone": "+1 (303) 555-0162",
            "caller_phone_raw": "303-555-0162",
            "caller_phone_display": "(303) 555-0162",
            "customer_address": "518 Pine Valley Road",
            "location_display": "518 Pine Valley Road",
            "routine_slot": "Tomorrow 8:00 AM – 11:00 AM Window",
            "routine_status": "Roof Inspection Booked — SMS Sent",
            "owner_alert_title": "⚡ New Roof Inspection Booked",
            "booking_confirm_line": f"You're all set, Tom! We have our roofing specialist scheduled for 518 Pine Valley Road tomorrow morning between eight and eleven. We're texting confirmation and tracking to (303) 555-0162. Does everything sound good?",
            "sms_body_routine": f"Hi Tom! You're confirmed with {biz_name} for tomorrow, 8–11 AM at 518 Pine Valley Road. Our specialist will text 15 min before arrival. Reply anytime with questions.",
            "daytime_cust_problem": "I think my roof is leaking.",
            "daytime_agent_service": "Oh no, dealing with a roof leak is definitely stressful! What seems to be happening—are you seeing water dripping from the ceiling, loose shingles, or water stains?",
            "daytime_cust_clarify": "Don't know, I found shingles in the yard and noticed a small water ring spreading on my upstairs ceiling.",
            "daytime_agent_consult": "I understand, that definitely needs prompt inspection before water can cause further drywall damage. We offer comprehensive roof evaluations. Would you like to check our available times?",
            "ah_slot": "Tomorrow 7:30 AM First Light Roof Response",
            "ah_status": "Roof Leak Triage — Crew Alerted for 7:30 AM",
            "owner_alert_ah_title": "🌙 After-Hours Roof Leak Emergency Dispatched",
            "ah_booking_line": f"You're all set, Marcus! I've locked in our emergency crew for first light tomorrow at seven-thirty AM at 1042 Bayside Avenue to inspect and tarp the area. Confirmation is on your phone. Does everything sound good?",
            "sms_body_ah": f"Hi Marcus! You're confirmed with {biz_name} for tomorrow at 7:30 AM at 1042 Bayside Avenue for emergency tarping and inspection. Reply anytime with questions.",
            "ah_cust_problem": "Hello? It's late at night and water is actively dripping through my upstairs bedroom ceiling into a bowl. What can I do right now?",
            "ah_agent_triage": "I'm sorry you're dealing with a leak late at night. First, is the water dripping near any ceiling light fixtures or electrical switches?",
            "ah_cust_clarify": "No, it's about four feet away from the light, right in the center of the drywall. What can you do for me right now? And what's the cost?",
        },
        "dental_medical": {
            "title": "Acute Toothache & Emergency Care",
            "caller_name": "Emily (Patient)",
            "caller_phone": "+1 (214) 555-0145",
            "customer_address": "Central Clinic (Suite 200)",
            "location_display": "Central Clinic (Suite 200)",
            "routine_slot": "Today at 1:30 PM (Doctor Chair)",
            "routine_status": "Doctor Chair Reserved — Digital Chart Sent",
            "owner_alert_title": "🦷 New Patient Appointment Confirmed",
            "booking_confirm_line": f"You're all set, Emily! I've reserved our doctor chair for you today at one-thirty PM at {biz_name}. Please arrive ten minutes early. I just texted your confirmation and digital intake link!",
            "sms_body_routine": f"Hi Emily! Your appointment with {biz_name} is confirmed for today at 1:30 PM at our clinic. Please arrive 10 min early with your ID and insurance card. Reply anytime with questions.",
            "daytime_cust_problem": "Hi! I have a severe throbbing toothache on my lower right molar that kept me awake all night. It hurts when I drink anything cold. Do you have any emergency appointments today?",
            "daytime_agent_service": "I am so sorry you are dealing with that pain — toothaches can be very uncomfortable. We reserve emergency chairs daily for urgent dental evaluations. Are you noticing any facial swelling or fever?",
            "daytime_cust_price_q": "No visible swelling yet, just sharp shooting pain. How much is the emergency exam? And do you take insurance?",
            "daytime_agent_price_a": f"We accept all major PPO insurances. For self-pay, {pricing_spoken} That includes digital X-rays and a gentle doctor evaluation with upfront pricing before any treatment.",
            "daytime_cust_area_q": f"Where is your clinic located in {city_disp}? I want to make sure it's not too far to drive with this toothache.",
            "daytime_agent_area_a": f"Our clinic is located right in {city_disp} with dedicated patient parking. We have an opening today at one-thirty PM, or tomorrow morning at eight. Would one-thirty today work for you?",
            "daytime_cust_name_turn": "Today at one-thirty is perfect. Emily Watson, cell is 214-555-0145. If the tooth needs a root canal or crown, do you do that in-house or refer me out to an endodontist?",
            "daytime_agent_scope_a": f"Our doctors perform root canals, crowns, and gentle extractions right here under one roof" + (f", including {custom_services_str}" if custom_services_str else "") + ", so you won't need to be referred out.",
            "ah_slot": "Tomorrow 8:00 AM First Emergency Chair",
            "ah_status": "After-Hours Dental Triage — Morning Chair Confirmed",
            "owner_alert_ah_title": "🌙 After-Hours Urgent Dental Patient Triage",
            "ah_booking_line": f"You're all set, Marcus! I've reserved our very first emergency chair tomorrow morning at eight AM at {biz_name}. Our doctor will have your chart ready. I just texted your confirmation link.",
            "sms_body_ah": f"Hi Marcus! Your urgent dental appointment with {biz_name} is confirmed for tomorrow at 8:00 AM. Please apply a cold compress tonight. Reply anytime if pain worsens.",
            "ah_cust_problem": "Hi, I'm calling late because I was eating dinner and bit into an olive pit — my back crown cracked in half and now the nerve is exposed and throbbing. I don't know what to do at midnight.",
            "ah_agent_triage": "Ouch, an exposed nerve is terribly painful! Let's make sure you're safe: are you having any continuous bleeding, or trouble swallowing?",
            "ah_cust_clarify": "No bleeding, just intense sensitivity when air hits it. Can I take ibuprofen?",
            "ah_agent_solution": f"Yes, you can take over-the-counter ibuprofen with water, and gently cover the sharp edge with clean sugar-free gum or wax tonight. I'm reserving our first eight AM emergency chair tomorrow so our doctor can treat that tooth first thing. {pricing_spoken}",
            "ah_cust_address_q": "That would be a lifesaver. Will the doctor be there right at eight?",
            "ah_agent_address_a": "Yes, our clinical team opens at seven-forty-five and our doctor will see you right at eight. Can I get your full name and best cell number?",
            "ah_cust_info_turn": "Marcus Vance, cell is 415-555-0834. Do I need to bring my insurance card?",
            "ah_agent_scope_a": "Yes, bring your dental insurance card and photo ID. I just texted you our instant digital intake link so you can finish paperwork from your phone tonight.",
        },
        "auto": {
            "title": "Brake Shaking & Computer Scan",
            "caller_name": "Chris (Driver)",
            "caller_phone": "+1 (615) 555-0133",
            "customer_address": "Service Bay 1 (Vehicle Drop-Off)",
            "location_display": "Service Bay 1 (Vehicle Drop-Off)",
            "routine_slot": "Today at 1:00 PM (Bay Inspection)",
            "routine_status": "Bay Reserved — Work Order Dispatched",
            "owner_alert_title": "🚗 New Vehicle Service Drop-Off Booked",
            "booking_confirm_line": f"You're all set, Chris! I've reserved Bay 1 for you today at one o'clock at {biz_name}. Our service advisor will check you right in. I just texted your confirmation to your mobile!",
            "sms_body_routine": f"Hi Chris! Your vehicle inspection at {biz_name} is confirmed for today at 1:00 PM. Pull right into Bay 1 and our service advisor will check you in. Reply anytime with questions.",
            "daytime_cust_problem": "Hey there! My car shakes violently whenever I step on the brakes on the highway, and my check engine light just started flashing. Can I bring it in today?",
            "daytime_agent_service": "A flashing check engine light and severe brake vibration should definitely be evaluated before driving further. Can you safely nurse the car into our shop, or do you need a tow?",
            "daytime_cust_price_q": "I can nurse it over slowly. But what do you charge for the computer scan and brake inspection? Dealerships want two hundred dollars just to look.",
            "daytime_agent_price_a": f"Dealership prices can be crazy — {pricing_spoken} We scan all computer codes, inspect the brakes, and text a digital report with upfront pricing before touching a thing.",
            "daytime_cust_area_q": f"Where is your shop located in {city_disp}, and can I drop it off around one o'clock?",
            "daytime_agent_area_a": f"Our shop is located right in {city_disp} with quick drop-off. We have an inspection bay open today at one o'clock, or tomorrow morning at eight. Would one o'clock work for your schedule?",
            "daytime_cust_name_turn": "One o'clock works great. Chris Martinez, cell is 615-555-0133. It's a 2018 Honda Accord. Do you guys do full transmission and engine repairs too if it's more than just brakes?",
            "daytime_agent_scope_a": f"Yes, we do everything from brakes to full engine and transmission repairs" + (f", {custom_services_str}" if custom_services_str else "") + " with warranty on all parts and labor.",
            "ah_slot": "Tomorrow 8:00 AM First Diagnostic Slot",
            "ah_status": "Overnight Key Drop Scheduled — Tech Alerted",
            "owner_alert_ah_title": "🌙 Overnight Vehicle Drop-Off Scheduled",
            "ah_booking_line": f"You're all set, Marcus! Have the tow truck drop your car in our secure lot tonight and put keys in our drop box by Bay 1. Our technician will inspect it at eight AM tomorrow. Details are on your phone.",
            "sms_body_ah": f"Hi Marcus! Your overnight vehicle drop-off at {biz_name} is logged. Please place keys in our secure drop box by Bay 1. Our technician will inspect your vehicle at 8:00 AM tomorrow. Reply anytime.",
            "ah_cust_problem": "Hello! My car broke down in a parking lot late tonight — the engine died and when I turn the key it just clicks rapidly. I'm stranded and don't know what to do with my car.",
            "ah_agent_triage": "I'm sorry you're stranded late at night! First, are you in a safe, well-lit location away from active road traffic?",
            "ah_cust_clarify": "Yes, I'm inside a lighted grocery store lot. But I need to get the car towed somewhere safe tonight so it doesn't get ticketed.",
            "ah_agent_solution": f"You can have your tow truck drop the car in our secure fenced lot tonight! Put the keys in our night drop box by Bay 1. Our technician will pull it into the bay first thing at eight AM. {pricing_spoken}",
            "ah_cust_address_q": "Can you text me your exact shop address and key drop instructions?",
            "ah_agent_address_a": "Absolutely! I'm texting you our direct shop address and key drop instructions right now. Can I get your name and vehicle make?",
            "ah_cust_info_turn": "Marcus Vance, 415-555-0834, it's a 2019 Toyota Camry.",
            "ah_agent_scope_a": "Got it, Marcus! Your Camry will be first in line when our techs clock in at eight AM. We'll call you with the digital report before touching a thing.",
        },
        "legal": {
            "title": "Accident Injury Consultation",
            "caller_name": "Amanda (Client)",
            "caller_phone": "+1 (312) 555-0155",
            "customer_address": "Confidential Legal Consultation (Phone / Office)",
            "location_display": "Confidential Legal Consultation (Phone / Office)",
            "routine_slot": "Today at 2:00 PM (Attorney Intake)",
            "routine_status": "Consultation Confirmed — Calendar Invite Sent",
            "owner_alert_title": "⚖️ New Case Consultation Confirmed",
            "booking_confirm_line": f"You're all set, Amanda! I've confirmed your confidential consultation for today at two PM with our senior attorney directly by phone. I just texted confirmation to your cell!",
            "sms_body_routine": f"Hi Amanda! Your confidential consultation with {biz_name} is confirmed for today at 2:00 PM. Our attorney will call you directly at this number. Reply to this text anytime with questions.",
            "daytime_cust_problem": "Hello, I was involved in a car accident yesterday where another driver ran a red light and hit my driver side. The other driver's insurance adjuster has already called three times trying to get me to sign a settlement. Do I need an attorney?",
            "daytime_agent_service": "I am very sorry to hear about your accident. You should definitely avoid signing anything or giving recorded statements until an attorney reviews your file. Are you currently receiving medical care for your injuries?",
            "daytime_cust_price_q": "Yes, my neck and shoulder are in a lot of pain. But how much does it cost to talk to an attorney? I don't have thousands of dollars for legal fees right now.",
            "daytime_agent_price_a": f"You don't need to worry about legal fees at all — {pricing_spoken} We work on contingency, meaning zero out-of-pocket costs and zero fee unless we successfully recover money for you.",
            "daytime_cust_area_q": f"I live here in {city_disp}. Can I do the consultation over phone or video today? It's really hard for me to drive right now with my shoulder.",
            "daytime_agent_area_a": f"Yes, our firm serves clients throughout {city_disp} by phone, video, or in our office. We have a consultation opening today at two PM, or tomorrow morning. Would two PM today work for you?",
            "daytime_cust_name_turn": "Two PM today works fine. Amanda Jenkins, cell is 312-555-0155. Will your firm handle my medical bills and dealing with the insurance company directly?",
            "daytime_agent_scope_a": f"Yes, our attorneys take over all communication with insurance adjusters and hospital billing" + (f", {custom_services_str}" if custom_services_str else "") + ", so you can just focus on healing.",
            "ah_slot": "Tomorrow 8:00 AM Priority Arraignment Review",
            "ah_status": "Urgent Detention Triage — On-Call Counsel Alerted",
            "owner_alert_ah_title": "🌙 After-Hours Urgent Legal Intake Alert",
            "ah_booking_line": f"You're all set, Marcus. I have logged the details and alerted our on-call counsel. An attorney will review the docket and call you at eight AM tomorrow. Confirmation is on your phone.",
            "sms_body_ah": f"Hi Marcus! Your urgent legal inquiry has been routed to our on-call counsel at {biz_name}. An attorney will review the booking details and contact you tomorrow at 8:00 AM. Reply anytime.",
            "ah_cust_problem": "Hello? I'm calling late because my brother was just pulled over and arrested by police tonight, and he's currently being held at the county detention facility. I need a defense attorney immediately.",
            "ah_agent_triage": "I understand this is an urgent and stressful situation. Has he already gone through booking, and do you know what specific charges are listed?",
            "ah_cust_clarify": "He just finished booking about twenty minutes ago. They told me his bond hearing is tomorrow morning.",
            "ah_agent_solution": f"Understood. Advise him not to answer questions until counsel is present. I'm alerting our on-call criminal defense attorney right now, and scheduling an eight AM priority review before his hearing. {pricing_spoken}",
            "ah_cust_address_q": "Can someone definitely review his case before the morning hearing?",
            "ah_agent_address_a": "Yes, our on-call counsel handles morning arraignments regularly. Can I get his full legal name, date of birth, and your mobile number?",
            "ah_cust_info_turn": "Marcus Vance is his name, and my cell is 415-555-0834.",
            "ah_agent_scope_a": "I've logged Marcus Vance in our priority intake. I'm texting you a confirmation and our counsel will review his docket first thing in the morning.",
        },
        "restaurant": {
            "title": "Anniversary Dinner Reservation (6 Guests)",
            "caller_name": "Jessica (Guest)",
            "caller_phone": "+1 (212) 555-0129",
            "customer_address": "Main Dining Room (Table for 6)",
            "location_display": "Main Dining Room (Table for 6)",
            "routine_slot": "Friday at 7:15 PM (6 Guests)",
            "routine_status": "Table Reserved — Host Team Confirmed",
            "owner_alert_title": "🍽️ New Dining Reservation Confirmed",
            "booking_confirm_line": f"You're all set, Jessica! I've reserved a wonderful table for six this Friday at seven-fifteen PM at {biz_name}, with complimentary champagne flutes noted. I just texted your confirmation!",
            "sms_body_routine": f"Hi Jessica! Your reservation for 6 guests at {biz_name} is confirmed for Friday at 7:15 PM. Complimentary valet parking is available at the main entrance. Reply anytime to modify your reservation.",
            "daytime_cust_problem": "Hi! We're celebrating my parents' 40th wedding anniversary this Friday evening, and we'd love to reserve a table for six guests around seven PM. Do you have availability?",
            "daytime_agent_service": "Congratulations to your parents on their 40th anniversary — forty years is such a wonderful milestone! We would love to host your family celebration. We have a wonderful table this Friday at seven-fifteen PM. Would that time work well for your group?",
            "daytime_cust_price_q": "Seven-fifteen is great. Is there any deposit or reservation fee required? And do you have a dress code?",
            "daytime_agent_price_a": "There is zero reservation fee or deposit! Our dining room is smart casual, and we have complimentary valet right at the front door.",
            "daytime_cust_area_q": f"My father has a severe peanut allergy and my mother is strictly gluten-free. We're coming in from {city_disp} — can your kitchen accommodate both of them safely?",
            "daytime_agent_area_a": f"Our chef takes allergies very seriously! We have dedicated allergy cookware and note everything on your ticket in advance. Can I get your name and cell number for the reservation?",
            "daytime_cust_name_turn": "Jessica Campbell, cell is 212-555-0129. Could we arrange a complimentary champagne toast or special dessert for their anniversary?",
            "daytime_agent_scope_a": "Absolutely! I've noted complimentary champagne flutes for the anniversary table and our chef's signature dessert. We can't wait to host you!",
            "ah_slot": "Tomorrow 8:30 AM Event Director Callback",
            "ah_status": "Private Event Request Logged — GM Alerted",
            "owner_alert_ah_title": "🌙 Large Party Dining Inquiry Captured",
            "ah_booking_line": f"You're all set, Marcus! I've flagged your private event inquiry for our Event Director. They'll call you at eight-thirty AM tomorrow with menus. Confirmation is on your phone.",
            "sms_body_ah": f"Hi Marcus! Your private dining request for 18 guests at {biz_name} has been received. Our Event Director will call you tomorrow morning at 8:30 AM with preset menu options. Reply anytime.",
            "ah_cust_problem": "Good evening! I know you're closed for the night, but I'm organizing a corporate celebration dinner for eighteen colleagues tomorrow night and our original venue cancelled on us. Can you host a group of eighteen?",
            "ah_agent_triage": "I understand how stressful a last-minute cancellation can be! Let's see how we can help. Would you prefer a semi-private area or our private wine dining room for your team of eighteen?",
            "ah_cust_clarify": "The private wine room would be incredible. We'd want appetizers, dinner, and cocktails starting around seven PM.",
            "ah_agent_solution": "Our private wine room comfortably seats up to twenty-two guests! I am flagging your event parameters for our General Manager and Event Director for an eight-thirty AM morning callback to confirm the menu.",
            "ah_cust_address_q": "Can you text me confirmation that this inquiry was received by management?",
            "ah_agent_address_a": "Yes! I am sending an instant SMS receipt right now so you know your request is first in line for morning review.",
            "ah_cust_info_turn": "Marcus Vance, cell is 415-555-0834, company is Vance Capital.",
            "ah_agent_scope_a": "Logged under Vance Capital, Marcus. Our Event Director will call you first thing tomorrow morning to finalize your team's dinner.",
        },
        "realestate": {
            "title": "Private Home Showing & Buyer Tour",
            "caller_name": "Brandon (Buyer)",
            "caller_phone": "+1 (206) 555-0182",
            "customer_address": "842 Highland Park Boulevard",
            "location_display": "842 Highland Park Boulevard",
            "routine_slot": "Today at 4:30 PM (Private Showing)",
            "routine_status": "Showing Confirmed — Lockbox Access Ready",
            "owner_alert_title": "🏡 New Property Showing Booked",
            "booking_confirm_line": f"You're all set, Brandon! I have locked in your private showing for today at four-thirty PM at 842 Highland Park Boulevard. Our agent will meet you at the front door. Confirmation is on your phone!",
            "sms_body_routine": f"Hi Brandon! Your private showing of 842 Highland Park Blvd with {biz_name} is confirmed for today at 4:30 PM. Your agent will meet you at the front door. Reply anytime with questions.",
            "daytime_cust_problem": "Hello! I saw your featured listing for the four-bedroom craftsman home on Highland Park Boulevard. It looks beautiful online — is that home still actively on the market, and can I tour it today?",
            "daytime_agent_service": "Hello! Yes, that craftsman home on Highland Park Boulevard is actively on the market. We can certainly coordinate a private walkthrough. Are you currently pre-approved with a lender, or looking to purchase with cash?",
            "daytime_cust_price_q": "I have a verified pre-approval letter from Chase Bank. What fees do you charge buyers to represent us on a purchase?",
            "daytime_agent_price_a": f"Good news — {pricing_spoken} Buyer representation is 100% free to you because the seller covers all broker commissions. You get full MLS access, contract drafting, and negotiation with zero fees.",
            "daytime_cust_area_q": f"That's great to hear. We're looking specifically in {city_disp if city_disp.lower().startswith('the ') else f'the {city_disp}'} market for the school district — does your team handle showings and offer negotiations throughout {city_disp}?",
            "daytime_agent_area_a": f"Yes, absolutely! We coordinate showings throughout {city_disp}. We have a showing window open today at four-thirty PM, or tomorrow at ten AM. Which time works better for you?",
            "daytime_cust_name_turn": "Four-thirty today works great. Brandon Cole, cell is 206-555-0182. If we love the house, do you help us draft the purchase offer and negotiate inspection repairs?",
            "daytime_agent_scope_a": "Yes! Our agents handle competitive offers, inspection repair negotiations, and guide you through every step all the way to closing.",
            "ah_slot": "Tomorrow 9:00 AM Listing Consultation",
            "ah_status": "Home Seller Lead Logged — Managing Broker Alerted",
            "owner_alert_ah_title": "🌙 New Home Seller Lead Captured",
            "ah_booking_line": f"You're all set, Marcus! I've booked your listing consultation for tomorrow morning at nine AM at 1042 Bayside Avenue. Our managing broker will bring neighborhood comps. Confirmation is on your phone.",
            "sms_body_ah": f"Hi Marcus! Your listing consultation with {biz_name} is confirmed for tomorrow at 9:00 AM at 1042 Bayside Avenue. Our broker will bring full neighborhood comps. Reply anytime.",
            "ah_cust_problem": "Hi, I know it's eleven PM, but my wife and I just accepted job transfers out of state and need to list our four-bedroom home for sale within the next two weeks. We need a top listing agent.",
            "ah_agent_triage": "Congratulations on the new job opportunities! Relocating on a two-week timeline requires fast, experienced listing execution. What neighborhood or city is your home located in?",
            "ah_cust_clarify": f"It's a custom four-bedroom home on Bayside Avenue in {city_disp}. We want to know what it's worth and get it staged quickly.",
            "ah_agent_solution": f"Bayside Avenue is a high-demand market! I'm alerting our managing broker right now to prepare a complimentary market analysis, and scheduling a priority nine AM consultation tomorrow. {pricing_spoken}",
            "ah_cust_address_q": "Can the broker come out to the property tomorrow morning?",
            "ah_agent_address_a": f"Yes, our managing broker can meet you on site tomorrow at nine AM with recent neighborhood comps. Can I get your full name and best cell number?",
            "ah_cust_info_turn": "Marcus Vance, cell is 415-555-0834, address is 1042 Bayside Avenue.",
            "ah_agent_scope_a": "I've locked in Marcus Vance for 1042 Bayside Avenue. Our broker will have professional photography and staging plans ready for your morning meeting.",
        },
        "salon_spa": {
            "title": "Full Balayage & Styling Appointment",
            "caller_name": "Ashley (Client)",
            "caller_phone": "+1 (310) 555-0164",
            "customer_address": "Salon Styling Chair (Master Colorist)",
            "location_display": "Salon Styling Chair (Master Colorist)",
            "routine_slot": "Thursday at 2:00 PM (Stylist Chair)",
            "routine_status": "Chair Reserved — Stylist Schedule Confirmed",
            "owner_alert_title": "✨ New Salon Appointment Booked",
            "booking_confirm_line": f"You're all set, Ashley! I've reserved our master colorist for you this Thursday at two PM at {biz_name}. Please arrive five minutes early to relax. I just texted your confirmation!",
            "sms_body_routine": f"Hi Ashley! Your styling appointment at {biz_name} is confirmed for Thursday at 2:00 PM with our master colorist. Please arrive 5 min early. Reply to this text anytime if you need to adjust.",
            "daytime_cust_problem": "Hi! I have a big wedding to attend this weekend and my hair desperately needs a fresh blonde balayage, toner, and a blowout. Do you have any openings with a master colorist this Thursday?",
            "daytime_agent_service": "Hello! We would love to take care of you and have you looking gorgeous for the wedding! We have an opening with our master colorist this Thursday at two PM. How long is your hair right now, and is it color-treated?",
            "daytime_cust_price_q": "My hair is about shoulder-length, with grown-out highlights. What is your pricing for a full balayage and tone? Some salons quote one number and then charge twice as much at checkout.",
            "daytime_agent_price_a": "We believe in 100% transparent pricing — a full balayage, toner, and signature blowout for shoulder-length hair is between one eighty and two ten, with zero hidden add-on fees.",
            "daytime_cust_area_q": "That's very fair! Where is your salon located, and is parking easy?",
            "daytime_agent_area_a": f"Our boutique salon is located in {city_disp} with convenient parking right out front. Can I grab your name and cell number to hold that Thursday two PM chair?",
            "daytime_cust_name_turn": "Ashley Miller, mobile is 310-555-0164. Do you also do lash extensions and bridal makeup if I want to add that on?",
            "daytime_agent_scope_a": "Yes! We do lash lifts, brow laminations, facials, and special event makeup so you can bundle everything in one visit.",
            "ah_slot": "Tomorrow 8:30 AM Bridal Coordinator Callback",
            "ah_status": "Bridal Party Request Logged — Coordinator Alerted",
            "owner_alert_ah_title": "🌙 Bridal Party Schedule Adjustment Request",
            "ah_booking_line": f"You're all set, Marcus! I've logged the timeline update for our Bridal Coordinator. She'll text and call you by eight-thirty AM to ensure all chairs are shifted earlier. Confirmation is on your phone.",
            "sms_body_ah": f"Hi Marcus! Your bridal party timeline request for {biz_name} has been received. Our Bridal Coordinator will contact you tomorrow by 8:30 AM to finalize the morning schedule. Reply anytime.",
            "ah_cust_problem": "Hello! It's almost midnight, but I'm the maid of honor for a Saturday wedding and two of our bridesmaids need their hair styling times moved earlier. Can someone help us adjust the bridal party schedule?",
            "ah_agent_triage": "Bridal mornings can be hectic — don't worry, we manage wedding party schedules all the time! What time does the bridal party need to be finished by on Saturday?",
            "ah_cust_clarify": "The photographer arrives at eleven-thirty AM, so we need all hair and makeup completed by eleven AM sharp.",
            "ah_agent_solution": "I'm logging your timeline adjustment for our Bridal Coordinator for first-thing morning review to shift your chairs earlier so everyone is photo-ready before eleven AM. I'm texting you a receipt right now.",
            "ah_cust_address_q": "Thank you so much! Will someone call me first thing in the morning?",
            "ah_agent_address_a": "Yes, our Bridal Coordinator will text and call you by eight-thirty AM. Can I have your name and the bride's party name?",
            "ah_cust_info_turn": "Marcus Vance for the Vance-Taylor Wedding, cell is 415-555-0834.",
            "ah_agent_scope_a": "Logged under the Vance-Taylor wedding party. Sleep well tonight — our team will have your wedding morning schedule locked down first thing.",
        },
        "general": {
            "title": "Facility Repair & On-Site Consultation",
            "caller_name": "David (Property Owner)",
            "caller_name_full": "David Thompson",
            "caller_phone": "+1 (713) 555-0184",
            "caller_phone_raw": "713-555-0184",
            "caller_phone_display": "(713) 555-0184",
            "customer_address": "419 Commercial Boulevard",
            "location_display": "419 Commercial Boulevard",
            "routine_slot": "Tomorrow 8:00 AM – 11:00 AM Window",
            "routine_status": "Consultation Confirmed — SMS Sent",
            "owner_alert_title": "⚡ New Commercial Service Visit Booked",
            "booking_confirm_line": f"You're all set, David! We have our service specialist scheduled for 419 Commercial Boulevard tomorrow morning between eight and eleven. We're texting confirmation and tracking to (713) 555-0184. Does everything sound good?",
            "sms_body_routine": f"Hi David! You're confirmed with {biz_name} for tomorrow, 8–11 AM at 419 Commercial Boulevard. Our service specialist will text 15 min before arrival. Reply to this text anytime with questions.",
            "daytime_cust_problem": "We have an urgent maintenance issue at our facility.",
            "daytime_agent_service": "Oh no, dealing with maintenance issues is definitely stressful! What seems to be happening with the facility—is equipment down, a plumbing issue, or an electrical issue?",
            "daytime_cust_clarify": "Don't know, half the building's workspace lighting and power outlets suddenly shut off.",
            "daytime_agent_consult": "I understand, that definitely sounds like something our specialist should inspect to diagnose properly. We can get you on the schedule so our team can take care of that for you. Would you like to check our available appointment times?",
            "ah_slot": "Tomorrow 8:00 AM Priority Maintenance Slot",
            "ah_status": "After-Hours Request Logged — Supervisor Alerted",
            "owner_alert_ah_title": "🌙 After-Hours Commercial Facility Alert",
            "ah_booking_line": f"You're all set, Marcus! I've reserved our priority eight AM slot tomorrow at 1042 Bayside Avenue, and alerted our on-call supervisor. Confirmation is on your phone. Does everything sound good?",
            "sms_body_ah": f"Hi Marcus! You're confirmed with {biz_name} for tomorrow morning at 8:00 AM at 1042 Bayside Avenue. Our specialist will text 15 min before arrival. Reply anytime with questions.",
            "ah_cust_problem": "Hello? It's late, almost midnight, but we have an urgent maintenance issue at our building and need to know if someone can come out first thing in the morning.",
            "ah_agent_triage": "Thank you for calling our twenty-four-seven line. First, is there an active life-safety hazard, water leak, or electrical danger at the property?",
            "ah_cust_clarify": "No immediate life safety hazard, but we need it resolved before staff arrives tomorrow morning. What can you do for me right now? And what's the cost?",
        }
    }

    tdata = TRADE_DEMO_DATA.get(trade_key, TRADE_DEMO_DATA["general"])

    # ── Booking language variants ────────────────────────────────────────────
    if is_text_details_booking:
        booking_confirm_line = (
            f"I've got all your details, {tdata['caller_name'].split()[0]}! "
            f"I just sent confirmation to your cell, and one of our specialists will call you within the hour to coordinate arrival. "
            f"You're all set!"
        )
        routine_slot   = "Specialist Callback Within the Hour"
        routine_status = "Inquiry Dispatched to Team — Callback Alert Sent"
        owner_alert_title = "⚡ New Booking — Team Callback Requested"
    else:
        booking_confirm_line = tdata["booking_confirm_line"]
        routine_slot   = tdata["routine_slot"]
        routine_status = tdata["routine_status"]
        owner_alert_title = tdata["owner_alert_title"]

    sms_body_routine = tdata["sms_body_routine"]

    # ── After-hours booking language ─────────────────────────────────────────
    if is_text_details_booking:
        ah_booking_line = (
            f"I've got your info logged, Marcus, and alerted our on-call team. "
            f"Someone will reach out first thing in the morning to get you taken care of. "
            f"I just texted confirmation to your cell as well."
        )
        ah_slot   = "Team Priority Callback First Thing Tomorrow Morning"
        ah_status = "After-Hours Request — Team Alerted via SMS"
        owner_alert_ah_title = "🌙 After-Hours Request Dispatched"
    else:
        ah_booking_line = tdata["ah_booking_line"]
        ah_slot   = tdata["ah_slot"]
        ah_status = tdata["ah_status"]
        owner_alert_ah_title = tdata["owner_alert_ah_title"]

    sms_body_ah = tdata["sms_body_ah"]
    # ════════════════════════════════════════════════════════════════════════
    # BUILD ALL SCENARIOS FOR SELECTED TRADE
    # ════════════════════════════════════════════════════════════════════════
    all_scenarios: Dict[str, Dict[str, Any]] = {
        "routine_booking": {
            "scenario_id": "routine_booking",
            "scenario_title": "⚡ Daytime Service Call",
            "scenario_tag": "In-Hours • Real Booking",
            "scenario_desc": (
                f"{persona_name} picks up on the first ring, handles every question warmly — "
                "scope, territory, pricing — then locks in the booking smoothly."
            ),
            "customer_name": tdata["caller_name"],
            "caller_id": tdata["caller_phone"],
            "issue_title": tdata["title"],
            "telemetry": {
                "time_status": "IN-HOURS (2:15 PM Local)",
                "topic_status": "AUTHORIZED SERVICE TOPIC",
                "policy_check": "100% Policy Match",
                "action_taken": owner_alert_title,
            },
            "turns": [
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": f"Thank you for calling {biz_name}! This is {persona_name}. How can I help you today?"
                },
                {
                    "speaker": "customer",
                    "name": tdata["caller_name"],
                    "voice": customer_voice,
                    "text": tdata["daytime_cust_problem"]
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": tdata["daytime_agent_service"]
                },
                {
                    "speaker": "customer",
                    "name": tdata["caller_name"],
                    "voice": customer_voice,
                    "text": tdata.get("daytime_cust_clarify", "Don't know, it's just not working properly.")
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": tdata.get("daytime_agent_consult", f"I understand, it's frustrating when you can't quite pinpoint the issue! Got it, that definitely sounds like something one of our technicians should inspect to diagnose properly. We can get you on the schedule so our team can come out and take care of that for you. Would you like to check our available appointment times?")
                },
                {
                    "speaker": "customer",
                    "name": tdata["caller_name"],
                    "voice": customer_voice,
                    "text": "Yeah. But, like, what's the cost?"
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": fee_agent_answer
                },
                {
                    "speaker": "customer",
                    "name": tdata["caller_name"],
                    "voice": customer_voice,
                    "text": "Yeah, I guess. That sounds fair."
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": "Great! What is your service address so I can check our schedule for your area?"
                },
                {
                    "speaker": "customer",
                    "name": tdata["caller_name"],
                    "voice": customer_voice,
                    "text": f"I'm at {tdata['customer_address']} in {city_disp}."
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": f"Got it — so I have {tdata['customer_address']} in {city_disp}. Did I get that right?"
                },
                {
                    "speaker": "customer",
                    "name": tdata["caller_name"],
                    "voice": customer_voice,
                    "text": "Yes."
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": "We have an opening today between one and three, or tomorrow morning between eight and eleven. Which arrival window works better for your schedule?"
                },
                {
                    "speaker": "customer",
                    "name": tdata["caller_name"],
                    "voice": customer_voice,
                    "text": "Tomorrow morning works best."
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": "Perfect, we'll get you set for tomorrow morning between eight and eleven. And what is your full name and the best cell number for dispatch arrival updates?"
                },
                {
                    "speaker": "customer",
                    "name": tdata["caller_name"],
                    "voice": customer_voice,
                    "text": f"{tdata.get('caller_name_full', 'David Thompson')}, phone number is {tdata.get('caller_phone_display', '(713) 555-0184')}."
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": f"Got it, {tdata.get('caller_name_full', 'David').split()[0]}! Perfect — I have {phone_to_spoken_words(tdata['caller_phone'])}. Did I get that right?"
                },
                {
                    "speaker": "customer",
                    "name": tdata["caller_name"],
                    "voice": customer_voice,
                    "text": "Yes, that's it!"
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": booking_confirm_line
                },
                {
                    "speaker": "customer",
                    "name": tdata["caller_name"],
                    "voice": customer_voice,
                    "text": "Yes, sounds great! Goodbye."
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": f"Thank you for choosing {biz_name}! Have a wonderful day!"
                }
            ],
            "outcome": {
                "customer_name": tdata["caller_name"].split(" (")[0],
                "customer_phone": tdata["caller_phone"],
                "dispatch_delivery": "Instant SMS — Arrival Window + Live Tracking",
                "captured_issue": tdata["title"],
                "pricing_quoted": pricing,
                "scheduled_slot": routine_slot,
                "customer_address": tdata.get("location_display", tdata["customer_address"]),
                "status": routine_status,
                "sms_preview": {
                    "to_phone": tdata["caller_phone"],
                    "sender_label": f"{biz_name} Dispatch",
                    "message_body": sms_body_routine,
                },
                "owner_dispatch": {
                    "title": owner_alert_title,
                    "details": f"{tdata['caller_name']} • {tdata.get('location_display', tdata['customer_address'])} • {tdata['title']} • {pricing}",
                    "phone": owner_phone_display,
                },
            },
        },
        "after_hours_test": {
            "scenario_id": "after_hours_test",
            "scenario_title": "🌙 After-Hours Call",
            "scenario_tag": "After-Hours • Schedule Enforcement",
            "scenario_desc": (
                f"{persona_name} answers at 11:45 PM, reassures the caller, "
                "triages urgency, and secures the morning priority slot — all without waking the owner."
            ),
            "customer_name": "Marcus Vance (Caller)",
            "caller_id": "+1 (415) 555-0834",
            "issue_title": f"Late-Night {tdata['title']}",
            "telemetry": {
                "time_status": "AFTER-HOURS (11:45 PM Local)",
                "topic_status": "SCHEDULE PROTOCOL ENFORCED",
                "policy_check": "100% Schedule Match",
                "action_taken": owner_alert_ah_title,
            },
            "turns": [
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": f"Thank you for calling {biz_name}, this is {persona_name}. Our office is closed for the evening, but our after-hours line is active. How can I help you?"
                },
                {
                    "speaker": "customer",
                    "name": "Marcus (Caller)",
                    "voice": customer_voice,
                    "text": tdata["ah_cust_problem"]
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": tdata["ah_agent_triage"]
                },
                {
                    "speaker": "customer",
                    "name": "Marcus (Caller)",
                    "voice": customer_voice,
                    "text": tdata["ah_cust_clarify"]
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": ah_fee_agent_answer
                },
                {
                    "speaker": "customer",
                    "name": "Marcus (Caller)",
                    "voice": customer_voice,
                    "text": "Okay, that works. Let's reserve that morning priority slot."
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": "Great! What is your address so I can confirm you're in our service area and log it for our on-call team?"
                },
                {
                    "speaker": "customer",
                    "name": "Marcus (Caller)",
                    "voice": customer_voice,
                    "text": f"I'm at 1042 Bayside Avenue in {city_disp}."
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": f"Got it — so I have 1042 Bayside Avenue in {city_disp}. Did I get that right?"
                },
                {
                    "speaker": "customer",
                    "name": "Marcus (Caller)",
                    "voice": customer_voice,
                    "text": "Yes, that's right."
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": "And what is your full name and the best cell number for the priority dispatch confirmation?"
                },
                {
                    "speaker": "customer",
                    "name": "Marcus (Caller)",
                    "voice": customer_voice,
                    "text": "Marcus Vance, cell is 415-555-0834."
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": f"Perfect, Marcus — I have {phone_to_spoken_words('+1 (415) 555-0834')}. Did I get that right?"
                },
                {
                    "speaker": "customer",
                    "name": "Marcus (Caller)",
                    "voice": customer_voice,
                    "text": "Yes, that's it!"
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": ah_booking_line
                },
                {
                    "speaker": "customer",
                    "name": "Marcus (Caller)",
                    "voice": customer_voice,
                    "text": "Yes, sounds great! Thank you so much for answering at midnight."
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": "Rest easy tonight, Marcus. We've got you covered first thing! Have a good night."
                }
            ],
            "outcome": {
                "customer_name": "Marcus Vance",
                "customer_phone": "+1 (415) 555-0834",
                "dispatch_delivery": "Instant SMS Confirmation + Priority Dispatch Alert",
                "captured_issue": f"Late-Night {tdata['title']}",
                "pricing_quoted": pricing,
                "scheduled_slot": ah_slot,
                "customer_address": "1042 Bayside Avenue",
                "status": ah_status,
                "sms_preview": {
                    "to_phone": "+1 (415) 555-0834",
                    "sender_label": f"{biz_name} After-Hours",
                    "message_body": sms_body_ah,
                },
                "owner_dispatch": {
                    "title": owner_alert_ah_title,
                    "details": f"Marcus Vance • 1042 Bayside Avenue • Late-Night {tdata['title']} • {pricing}",
                    "phone": owner_phone_display,
                },
            },
        },
        "boundary_challenge": {
            "scenario_id": "boundary_challenge",
            "scenario_title": "🛑 3. Out-of-Scope Boundary Test",
            "scenario_tag": "Guardrail Test • Zero Hallucination",
            "scenario_desc": (
                f"{persona_name} gracefully declines unsafe DIY guidance or blind price demands "
                "while still moving the caller toward an authorized appointment."
            ),
            "customer_name": "Kevin Larson (Caller)",
            "caller_id": "+1 (650) 555-0219",
            "issue_title": "Out-of-Scope / Blind Price Demand",
            "telemetry": {
                "time_status": "IN-HOURS (10:15 AM Local)",
                "topic_status": "GUARDRAIL ACTIVATED",
                "policy_check": "100% Guardrail Match",
                "action_taken": "Safety Boundary Upheld — Professional Service Offered",
            },
            "turns": [
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": f"Thank you for calling {biz_name}, this is {persona_name}! How can I help you today?"
                },
                {
                    "speaker": "customer",
                    "name": "Kevin (Caller)",
                    "voice": customer_voice,
                    "text": "Yeah hi. Can you just tell me how to do this repair myself over the phone so I don't have to pay for a service visit?"
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": f"I completely understand wanting to save money, and I respect that! But for your personal safety, equipment warranties, and code compliance, {biz_name} cannot provide DIY repair instructions over the phone. I'd hate for anything to go sideways or void your warranty. What we can do is have our specialist inspect it and give you an upfront written flat-rate quote before any work starts."
                },
                {
                    "speaker": "customer",
                    "name": "Kevin (Caller)",
                    "voice": customer_voice,
                    "text": "Fair enough. Can you guarantee me an exact price right now before you come out?"
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": f"Every situation is unique, so our specialist needs to evaluate the system in person to give you an accurate, guaranteed flat-rate price. {pricing_text}"
                },
                {
                    "speaker": "customer",
                    "name": "Kevin (Caller)",
                    "voice": customer_voice,
                    "text": "Okay that's actually fair. Do you have availability this afternoon?"
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": "Yes! We have an opening this afternoon. What is your address and name, and I'll lock that in right now."
                },
            ],
            "outcome": {
                "captured_issue": "DIY Instruction Declined — Professional Assessment Booked",
                "pricing_quoted": pricing,
                "scheduled_slot": "Today Afternoon Window",
                "customer_address": "Pending Customer Address",
                "status": "Guardrail Enforced — Booking Secured",
            },
        },
        "emergency_triage": {
            "scenario_id": "emergency_triage",
            "scenario_title": "🚨 4. Emergency Triage",
            "scenario_tag": "Emergency Priority • Immediate Action",
            "scenario_desc": (
                f"{persona_name} detects an active emergency, issues life-safety guidance, "
                "and escalates to on-call support immediately."
            ),
            "customer_name": "Elena Rostova (Panicked Caller)",
            "caller_id": "+1 (312) 555-0941",
            "issue_title": "Active Property Emergency",
            "telemetry": {
                "time_status": "PRIORITY OVERRIDE",
                "topic_status": "EMERGENCY TRIGGER MATCHED",
                "policy_check": "100% Emergency Protocol",
                "action_taken": "Safety Guidance Delivered + Priority Dispatch Initiated",
            },
            "turns": [
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": f"Thank you for calling {biz_name}, this is {persona_name}! How can I help you?"
                },
                {
                    "speaker": "customer",
                    "name": "Elena (Caller)",
                    "voice": customer_voice,
                    "text": "Help! There's an active emergency at my property right now and I don't know what to do!"
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": "Please remain calm. If there is any active fire, gas odor, or risk to life, please step outside to a safe area immediately and call nine-one-one. Are you in a safe spot right now?"
                },
                {
                    "speaker": "customer",
                    "name": "Elena (Caller)",
                    "voice": customer_voice,
                    "text": "Yes, I am outside in the front yard now. Please send help!"
                },
                {
                    "speaker": "agent",
                    "name": f"{persona_name} (Receptionist)",
                    "voice": persona_voice,
                    "text": "Good — stay outside in the safe area. I am alerting our emergency on-call supervisor right now and dispatching our closest unit directly to your location. What is your exact address?"
                },
            ],
            "outcome": {
                "captured_issue": "Active Hazard / Emergency Triage",
                "pricing_quoted": "Emergency On-Call Dispatch",
                "scheduled_slot": "IMMEDIATE EMERGENCY ESCALATION",
                "customer_address": "Front Yard (Safe Area)",
                "status": "On-Call Supervisor Dispatched Immediately",
            },
        },
    }

    return all_scenarios.get(active_scenario, all_scenarios["routine_booking"])


# ---------------------------------------------------------------------------
# Polar Subscription & Checkout Helper
# ---------------------------------------------------------------------------

def create_polar_checkout_session(plan_id: str, client_id: str, success_url: str, customer_email: Optional[str] = None) -> Dict[str, Any]:
    """Generates a Polar.sh checkout session or test checkout payload."""
    import urllib.request
    import json as _json

    token = settings.POLAR_ACCESS_TOKEN
    org_id = settings.POLAR_ORGANIZATION_ID
    product_id = settings.POLAR_PRODUCT_ID_GROWTH if (plan_id == "growth" and settings.POLAR_PRODUCT_ID_GROWTH) else settings.POLAR_PRODUCT_ID_STARTER
    custom_checkout_link = settings.POLAR_CHECKOUT_URL

    # Ensure success_url is a fully qualified URL required by Polar API
    if success_url and not success_url.startswith(("http://", "https://")):
        base_public = (getattr(settings, "PUBLIC_URL", "") or "https://agents.orxlabs.com").rstrip("/")
        if not success_url.startswith("/"):
            success_url = f"/{success_url}"
        success_url = f"{base_public}{success_url}"

    # Append placeholder {CHECKOUT_SESSION_ID} if not present so Polar redirects back with session id
    if "{CHECKOUT_SESSION_ID}" not in success_url and "session_id=" not in success_url:
        joiner = "&" if "?" in success_url else "?"
        success_url = f"{success_url}{joiner}session_id={{CHECKOUT_SESSION_ID}}&client_id={client_id}&plan={plan_id}&status=success"

    # 1. Fastest method: Direct Polar buy link if configured
    if custom_checkout_link and "buy.polar.sh" in custom_checkout_link:
        sep = "&" if "?" in custom_checkout_link else "?"
        return {
            "mode": "polar_payment_link",
            "checkout_url": f"{custom_checkout_link}{sep}client_id={client_id}",
            "product_id": product_id,
            "plan": plan_id,
            "amount": "$20/mo" if plan_id == "starter" else "$249/mo",
        }

    # 2. Direct Polar REST API Checkout Session (robust against SDK schema discrepancies)
    if token and product_id:
        def _call_polar_api(include_email: bool = True):
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            req_body: Dict[str, Any] = {
                "products": [product_id],
                "success_url": success_url,
                "metadata": {"client_id": client_id, "plan": plan_id},
            }
            if include_email and customer_email and "@" in customer_email and not customer_email.endswith("@example.com"):
                req_body["customer_email"] = customer_email.strip()

            req = urllib.request.Request(
                "https://api.polar.sh/v1/checkouts/custom/",
                data=_json.dumps(req_body).encode("utf-8"),
                headers=headers
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                return _json.loads(resp.read().decode("utf-8"))

        try:
            try:
                chk_data = _call_polar_api(include_email=True)
            except Exception as email_err:
                if customer_email:
                    logger.warning(f"Polar checkout with prefilled email failed ({email_err}). Retrying without customer_email...")
                    chk_data = _call_polar_api(include_email=False)
                else:
                    raise email_err

            chk_url = chk_data.get("url")
            chk_id = chk_data.get("id")
            if chk_url:
                logger.success(f"Generated live Polar checkout session '{chk_id}' for client '{client_id}': {chk_url}")
                return {
                    "mode": "polar_live",
                    "checkout_url": chk_url,
                    "checkout_id": chk_id,
                    "product_id": product_id,
                    "plan": plan_id,
                    "amount": "$20/mo" if plan_id == "starter" else "$249/mo",
                }
        except Exception as api_err:
            logger.warning(f"Polar REST checkout error: {api_err}. Trying polar_sdk fallback...")
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
                        "product_id": product_id,
                        "plan": plan_id,
                        "amount": "$20/mo" if plan_id == "starter" else "$249/mo",
                    }
            except Exception as e:
                logger.warning(f"Polar SDK checkout error: {e}. Falling back to test checkout.")

    test_checkout_url = f"{success_url}?session_id=polar_chk_{uuid.uuid4().hex[:12]}&client_id={client_id}&plan={plan_id}&status=success"
    return {
        "mode": "polar_test_ready",
        "checkout_url": test_checkout_url,
        "product_id": product_id,
        "plan": plan_id,
        "amount": "$20/mo" if plan_id == "starter" else "$249/mo",
    }


def report_usage_event_to_polar(client_id: str, minutes: float, event_name: str = "call_minute") -> bool:
    """Ingests a usage event into Polar for metered billing aggregation.
    Polar sums these events during the billing cycle and automatically bills overage above 50 minutes at $0.25/min."""
    token = settings.POLAR_ACCESS_TOKEN
    if not token or minutes <= 0:
        return False

    try:
        from polar_sdk import Polar
        polar_client = Polar(access_token=token)
        polar_client.events.ingest(request={
            "events": [
                {
                    "name": event_name,
                    "external_customer_id": client_id,
                    "metadata": {
                        "minutes": round(float(minutes), 2),
                        "source": "livekit_call",
                        "timestamp": int(time.time()),
                    }
                }
            ]
        })
        logger.info(f"Reported {round(minutes, 2)}m usage event to Polar for client '{client_id}'")
        return True
    except Exception as e:
        logger.warning(f"Polar usage event ingestion notice: {e}")
        return False
