"""Industry Topic Catalog, Smart Operating Schedules & Guardrail Definitions.
Provides comprehensive categorized topic lists for each business type with pre-selected
defaults, custom write-ins, smart operating time presets, and out-of-scope boundary rules.
"""

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Default Operating Schedule Presets & Timezone Catalog
# ---------------------------------------------------------------------------

COMMON_TIMEZONES = [
    {"value": "America/New_York", "label": "Eastern Time (US & Canada - ET)", "offset": "UTC-4/5"},
    {"value": "America/Chicago", "label": "Central Time (US & Canada - CT)", "offset": "UTC-5/6"},
    {"value": "America/Denver", "label": "Mountain Time (US & Canada - MT)", "offset": "UTC-6/7"},
    {"value": "America/Los_Angeles", "label": "Pacific Time (US & Canada - PT)", "offset": "UTC-7/8"},
    {"value": "America/Phoenix", "label": "Arizona (No DST - MST)", "offset": "UTC-7"},
    {"value": "America/Anchorage", "label": "Alaska Time (AKT)", "offset": "UTC-8/9"},
    {"value": "Pacific/Honolulu", "label": "Hawaii Time (HST)", "offset": "UTC-10"},
    {"value": "Europe/London", "label": "London / GMT / BST", "offset": "UTC+0/1"},
    {"value": "Europe/Paris", "label": "Central European Time (CET)", "offset": "UTC+1/2"},
    {"value": "Asia/Dubai", "label": "Gulf Standard Time (GST)", "offset": "UTC+4"},
]

SCHEDULE_MODES = [
    {
        "id": "always_24_7",
        "name": "24/7/365 Always Answer",
        "icon": "⚡",
        "badge": "Maximum Revenue",
        "desc": "Answers every incoming customer call instantly in <2 seconds, day or night, weekends and holidays.",
    },
    {
        "id": "after_hours_only",
        "name": "After-Hours & Weekends Only",
        "icon": "🌙",
        "badge": "Night Shield",
        "desc": "Turns on automatically when your office closes. Dispatches emergencies and captures night leads.",
    },
    {
        "id": "overflow",
        "name": "Overflow & Missed Calls",
        "icon": "📞",
        "badge": "Zero Missed Calls",
        "desc": "Your office phone rings first (3 rings / 15 seconds). Riley answers only when your staff is busy on another line.",
    },
    {
        "id": "custom_schedule",
        "name": "Custom Operating Schedule",
        "icon": "🕒",
        "badge": "Exact Hours",
        "desc": "Define exact local operating hours per day. Riley adapts behavior between open hours and closed hours.",
    },
]

AFTER_HOURS_POLICIES = [
    {
        "id": "book_earliest_slot",
        "label": "Book Earliest Open Slot & Text Instant Confirmation (Recommended)",
        "badge": "Highest Lead Conversion",
        "desc": "Riley locks in the caller for your earliest available arrival window next business morning (based on your operating hours), secures name and address, and texts instant confirmation so you never lose the job to a competitor.",
    },
    {
        "id": "emergency_transfer",
        "label": "Urgent Calls Alert My Cell; Book Others for Earliest Slot",
        "badge": "Fast Emergency Response",
        "desc": "Active emergencies transfer to your cell phone immediately. All routine service requests are booked into your earliest morning arrival window.",
    },
    {
        "id": "lead_capture_sms_link",
        "label": "Capture Full Lead & Text Caller VIP Self-Booking Link",
        "badge": "Instant SMS Follow-up",
        "desc": "Collects full caller details, texts you an urgent lead notification, and immediately texts the caller a self-scheduling booking link.",
    },
]

# ---------------------------------------------------------------------------
# Comprehensive Categorized Answerable Topics by Industry
# ---------------------------------------------------------------------------

INDUSTRY_TOPIC_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "hvac": {
        "core_services": {
            "title": "Core Services & Work Scopes",
            "desc": "Services Riley can describe, quote standard dispatch for, and schedule:",
            "topics": [
                {"id": "ac_repair", "label": "AC Repair & Troubleshooting", "desc": "Diagnose AC blowing warm air, frozen coils, or electrical failure", "pre_selected": True},
                {"id": "heating_furnace", "label": "Heating & Furnace Repair", "desc": "Fix pilot lights, faulty igniters, blowers, and burner issues", "pre_selected": True},
                {"id": "tune_up", "label": "Seasonal System Tune-Up", "desc": "Comprehensive 21-point heating and cooling preventative tune-up", "pre_selected": True},
                {"id": "heat_pump", "label": "Heat Pump Service & Install", "desc": "Service mini-splits, dual-fuel systems, and inverter heat pumps", "pre_selected": True},
                {"id": "emergency_diag", "label": "Emergency Diagnostics", "desc": "Rapid same-day technician dispatch for urgent climate failures", "pre_selected": True},
                {"id": "duct_cleaning", "label": "Duct Cleaning & Air Quality", "desc": "Clean supply vents, return plenums, and install UV air purifiers", "pre_selected": False},
                {"id": "thermostat_smart", "label": "Smart Thermostat Installation", "desc": "Install Nest, Ecobee, and multi-zone climate controllers", "pre_selected": False},
                {"id": "commercial_hvac", "label": "Commercial Rooftop HVAC", "desc": "Service commercial packaged units, chillers, and RTUs", "pre_selected": False},
            ],
        },
        "pricing_topics": {
            "title": "Pricing, Fees & Estimates",
            "desc": "How Riley answers questions regarding service fees and costs:",
            "topics": [
                {"id": "diagnostic_fee", "label": "Diagnostic Fee Credited to Repair", "desc": "Explain that the diagnostic fee is credited directly toward repairs if approved", "pre_selected": True},
                {"id": "free_replacement_quote", "label": "Free On-Site Replacement Estimates", "desc": "Offer free zero-obligation quotes for full system replacements", "pre_selected": False},
                {"id": "financing_options", "label": "Financing & Monthly Payment Plans", "desc": "Mention low-interest monthly financing options for major repairs or new units", "pre_selected": False},
                {"id": "upfront_transparent", "label": "Upfront Flat-Rate Pricing", "desc": "Assure caller that technician provides written quote before doing any work", "pre_selected": False},
            ],
        },
        "booking_topics": {
            "title": "Scheduling & Availability",
            "desc": "Arrival window policies and dispatch timing:",
            "topics": [
                {"id": "arrival_windows", "label": "2-4 Hour Arrival Windows", "desc": "Book morning (8-12) or afternoon (1-5) arrival windows with 30-min call-ahead", "pre_selected": True},
                {"id": "same_day_dispatch", "label": "Same-Day Priority Dispatch", "desc": "Confirm availability for urgent same-day service calls", "pre_selected": True},
                {"id": "weekend_service", "label": "Weekend & Saturday Service", "desc": "Explain Saturday availability and on-call technician coverage", "pre_selected": False},
                {"id": "reschedule_policy", "label": "Easy Rescheduling via SMS", "desc": "Explain how customers can easily change their booking slot via SMS", "pre_selected": False},
            ],
        },
        "emergency_topics": {
            "title": "Emergency & Urgent Triage",
            "desc": "Urgent life/property situations that trigger immediate escalation:",
            "topics": [
                {"id": "gas_smell", "label": "Gas Smell / Carbon Monoxide Alert", "desc": "Instruct immediate evacuation of premises and transfer to on-call phone", "pre_selected": True},
                {"id": "no_heat_freezing", "label": "No Heat in Freezing Temperatures", "desc": "Treat as urgent freeze-protection emergency and dispatch fastest tech", "pre_selected": True},
                {"id": "indoor_water_leak", "label": "Active Water Leak from Indoor AC Unit", "desc": "Instruct shutting off thermostat and placing towels; dispatch tech", "pre_selected": True},
                {"id": "commercial_priority", "label": "Commercial Account Priority Outage", "desc": "Direct priority routing for server rooms and commercial facilities", "pre_selected": False},
            ],
        },
        "policies_credentials": {
            "title": "Trust, Credentials & Guarantees",
            "desc": "Credentials Riley mentions when callers ask about trust and qualifications:",
            "topics": [
                {"id": "licensed_insured", "label": "Licensed, Bonded & Insured", "desc": "Confirm state mechanical contractor license and comprehensive liability insurance", "pre_selected": True},
                {"id": "satisfaction_warranty", "label": "100% Satisfaction / Workmanship Warranty", "desc": "Highlight warranty on all replacement parts and labor", "pre_selected": True},
                {"id": "background_checked", "label": "EPA-Certified & Background Checked", "desc": "Reassure caller that all techs are drug-tested and background-cleared", "pre_selected": False},
                {"id": "payment_types", "label": "All Major Payment Methods Accepted", "desc": "Accept Visa, Mastercard, Amex, Discover, Check, and Apple Pay", "pre_selected": False},
            ],
        },
        "out_of_scope": {
            "title": "Guardrails & Out-of-Scope Requests",
            "desc": "Requests Riley will politely decline and escalate to human staff:",
            "topics": [
                {"id": "no_diy_gas_repair", "label": "Decline Hazardous DIY Repair Instructions", "desc": "Refuse giving DIY gas line or high-voltage capacitor wiring steps over phone", "pre_selected": True},
                {"id": "no_binding_phone_quotes", "label": "No Binding Written Quotes Without Inspection", "desc": "Politely explain technician must inspect unit to guarantee pricing", "pre_selected": True},
                {"id": "non_hvac_referral", "label": "Decline Non-HVAC Work Warmly", "desc": "Politely decline general carpentry, roofing, or unrelated trades", "pre_selected": False},
            ],
        },
    },

    "plumbing": {
        "core_services": {
            "title": "Core Services & Work Scopes",
            "desc": "Plumbing services Riley can explain and schedule:",
            "topics": [
                {"id": "drain_clearing", "label": "Emergency Drain Clearing & Snaking", "desc": "Clear backed-up sinks, showers, toilets, and main sewer lines", "pre_selected": True},
                {"id": "water_heater", "label": "Water Heater Repair & Replacement", "desc": "Service tankless, gas, and electric water heaters; fix pilot or leaks", "pre_selected": True},
                {"id": "burst_pipe", "label": "Burst Pipe & Active Leak Repair", "desc": "Locate and patch burst pipes in walls, crawlspaces, or yards", "pre_selected": True},
                {"id": "toilet_faucet", "label": "Toilet, Faucet & Fixture Repairs", "desc": "Fix running toilets, leaking faucets, shut-off valves, and garbage disposals", "pre_selected": True},
                {"id": "sewer_camera", "label": "Sewer Line Camera Inspection & Jetting", "desc": "High-definition video pipe inspection and hydro-jetting root clearing", "pre_selected": False},
                {"id": "whole_home_repiping", "label": "Whole-Home PEX & Copper Repiping", "desc": "Replace old galvanized or polybutylene pipes with modern PEX", "pre_selected": False},
                {"id": "water_filtration", "label": "Water Softeners & Filtration", "desc": "Install whole-house reverse osmosis and water softening systems", "pre_selected": False},
                {"id": "commercial_plumbing", "label": "Commercial Grease Traps & Plumbing", "desc": "Commercial kitchen plumbing, grease trap jetting, and backflow tests", "pre_selected": False},
            ],
        },
        "pricing_topics": {
            "title": "Pricing, Fees & Estimates",
            "desc": "How Riley answers plumbing pricing questions:",
            "topics": [
                {"id": "dispatch_fee", "label": "Diagnostic Service Fee Applied to Work", "desc": "Standard dispatch fee that applies directly toward the approved repair", "pre_selected": True},
                {"id": "free_repiping_estimate", "label": "Free Large-Project Estimates", "desc": "Zero-fee on-site estimates for sewer replacements and whole-home repipes", "pre_selected": False},
                {"id": "upfront_flat_rate", "label": "Upfront Menu Pricing", "desc": "Clear flat-rate quote presented before any wrench touches a pipe", "pre_selected": False},
                {"id": "financing_plumbing", "label": "Financing for Major Sewer / Heaters", "desc": "Low monthly payments available for water heaters and trenchless sewer jobs", "pre_selected": False},
            ],
        },
        "booking_topics": {
            "title": "Scheduling & Availability",
            "desc": "Dispatch timing and booking options:",
            "topics": [
                {"id": "arrival_windows", "label": "2-Hour Arrival Window with Tech Call", "desc": "Convenient arrival windows with live SMS tracking and call-ahead", "pre_selected": True},
                {"id": "same_day_plumbing", "label": "Same-Day Emergency Dispatch", "desc": "Rapid dispatch for active leaks and sewer backups", "pre_selected": True},
                {"id": "weekend_plumbing", "label": "Weekend & Holiday Availability", "desc": "Plumbers on call 7 days a week for plumbing emergencies", "pre_selected": False},
                {"id": "commercial_hours", "label": "After-Hours Commercial Dispatch", "desc": "Nighttime service for restaurants and retail spaces", "pre_selected": False},
            ],
        },
        "emergency_topics": {
            "title": "Emergency & Urgent Triage",
            "desc": "Immediate safety action for urgent plumbing disasters:",
            "topics": [
                {"id": "shutoff_guidance", "label": "Main Water Shut-Off Guidance", "desc": "Instruct caller on locating and shutting off their main water valve to stop flooding", "pre_selected": True},
                {"id": "sewage_backup", "label": "Raw Sewage Backup Triage", "desc": "Warn caller of biohazard, advise avoiding water usage, and dispatch immediately", "pre_selected": True},
                {"id": "water_heater_smoking", "label": "Water Heater Leaking or Smoking", "desc": "Instruct turning off gas/breaker and water inlet valve; dispatch tech", "pre_selected": True},
                {"id": "total_water_loss", "label": "Total Loss of Water to Building", "desc": "Priority investigation for frozen lines or municipal main breaks", "pre_selected": False},
            ],
        },
        "policies_credentials": {
            "title": "Trust, Credentials & Guarantees",
            "desc": "Qualifications Riley highlights to build customer confidence:",
            "topics": [
                {"id": "master_plumber", "label": "Licensed Master Plumber Supervised", "desc": "All technicians work under a master plumbing license and state bond", "pre_selected": True},
                {"id": "drain_guarantee", "label": "Drain Clearing Warranty", "desc": "Guaranteed clog-free period after professional clearing", "pre_selected": True},
                {"id": "clean_home_pledge", "label": "Shoe Covers & Clean Home Guarantee", "desc": "Technicians always wear floor protection and leave work areas spotless", "pre_selected": False},
                {"id": "licensed_insured", "label": "Fully Insured with $2M General Liability", "desc": "Complete property damage protection and worker's compensation", "pre_selected": False},
            ],
        },
        "out_of_scope": {
            "title": "Guardrails & Out-of-Scope Requests",
            "desc": "Boundaries Riley adheres to:",
            "topics": [
                {"id": "no_chemical_diy", "label": "Decline Harmful Chemical Drain Advice", "desc": "Advise against pouring corrosive chemicals that damage pipes or cause burns", "pre_selected": True},
                {"id": "no_phone_hidden_quotes", "label": "No Quotes for Hidden In-Wall Leaks", "desc": "Explain that drywall or slab leaks require acoustic detection on site", "pre_selected": True},
                {"id": "gas_utility_notice", "label": "Direct Severe Gas Odors to Utility / 911", "desc": "Ensure caller calls 911 or gas company first if heavy odor is reported", "pre_selected": True},
            ],
        },
    },

    "electrical": {
        "core_services": {
            "title": "Core Services & Work Scopes",
            "desc": "Electrical services Riley can explain and schedule:",
            "topics": [
                {"id": "panel_upgrades", "label": "200-Amp Electrical Panel Upgrades", "desc": "Replace old fuse boxes or 100-amp panels with modern 200-amp breakers", "pre_selected": True},
                {"id": "ev_chargers", "label": "EV Home Charger Installation (Level 2)", "desc": "Install dedicated 240V 50A circuits for Tesla, Rivian, and all EV chargers", "pre_selected": True},
                {"id": "breaker_tripping", "label": "Breaker Tripping & Fault Diagnostics", "desc": "Troubleshoot short circuits, overloaded lines, and arc faults", "pre_selected": True},
                {"id": "lighting_fans", "label": "Recessed LED Lighting & Ceiling Fans", "desc": "Install modern canless LEDs, dimmer switches, and heavy fan boxes", "pre_selected": True},
                {"id": "outlet_gfci", "label": "GFCI & Smart Outlet Repair", "desc": "Bring kitchens, bathrooms, and outdoor receptacles up to NEC code", "pre_selected": False},
                {"id": "whole_home_rewire", "label": "Whole-Home Rewiring & Knob-and-Tube", "desc": "Replace dangerous knob-and-tube or aluminum wiring with Romex", "pre_selected": False},
                {"id": "generator_install", "label": "Standby Whole-Home Generator Installs", "desc": "Install automatic Generac transfer switches and standby power", "pre_selected": False},
                {"id": "commercial_electric", "label": "Commercial 3-Phase Electrical Services", "desc": "Service commercial transformers, high-bay lights, and machinery hookups", "pre_selected": False},
            ],
        },
        "pricing_topics": {
            "title": "Pricing, Fees & Estimates",
            "desc": "How Riley answers electrical cost inquiries:",
            "topics": [
                {"id": "diagnostic_fee", "label": "Safety Diagnostic Fee Credited to Repair", "desc": "Clear inspection fee credited toward repair if customer approves work", "pre_selected": True},
                {"id": "free_panel_quote", "label": "Free Estimates on Panel & EV Installs", "desc": "Complimentary quotes for electrical panel upgrades and EV charger circuits", "pre_selected": False},
                {"id": "upfront_pricing", "label": "Upfront Guaranteed Pricing", "desc": "No hourly surprises—flat-rate quotes provided prior to starting work", "pre_selected": False},
                {"id": "financing_electric", "label": "Financing for Major Rewires & Panels", "desc": "Low-rate financing available for extensive electrical modernizations", "pre_selected": False},
            ],
        },
        "booking_topics": {
            "title": "Scheduling & Availability",
            "desc": "Scheduling and appointment parameters:",
            "topics": [
                {"id": "arrival_windows", "label": "2-4 Hour Arrival Windows", "desc": "Morning and afternoon windows with 30-minute tech notification", "pre_selected": True},
                {"id": "same_day_electric", "label": "Same-Day Emergency Power Response", "desc": "Immediate dispatch when homes or businesses lose critical power", "pre_selected": True},
                {"id": "permit_scheduling", "label": "City Permit & Inspection Handling", "desc": "Explain that all heavy work includes municipal permits and inspections", "pre_selected": False},
                {"id": "weekend_calls", "label": "Weekend Electrician Dispatch", "desc": "On-call coverage for urgent breaker and outage situations", "pre_selected": False},
            ],
        },
        "emergency_topics": {
            "title": "Emergency & Urgent Triage",
            "desc": "High-risk electrical hazards requiring priority action:",
            "topics": [
                {"id": "sparks_smoke", "label": "Sparks, Smoke or Burning Odor", "desc": "Instruct caller to shut off the main breaker if safe, evacuate, and dispatch tech", "pre_selected": True},
                {"id": "buzzing_panel", "label": "Breaker Panel Humming or Hot to Touch", "desc": "Treat as active hazard and dispatch emergency electrician immediately", "pre_selected": True},
                {"id": "downed_wire", "label": "Downed Overhead Wire Safety Warning", "desc": "Instruct caller to stay 30 feet away, call 911 and power utility immediately", "pre_selected": True},
                {"id": "total_blackout", "label": "Isolated Property Blackout", "desc": "Verify neighborhood status vs internal main breaker trip", "pre_selected": False},
            ],
        },
        "policies_credentials": {
            "title": "Trust, Credentials & Guarantees",
            "desc": "Credentials Riley highlights:",
            "topics": [
                {"id": "licensed_master", "label": "Licensed Master Electrician", "desc": "State electrical contractor license, fully bonded and insured", "pre_selected": True},
                {"id": "code_compliant", "label": "100% National Electrical Code (NEC) Guarantee", "desc": "All work strictly complies with national and municipal safety codes", "pre_selected": True},
                {"id": "parts_labor_warranty", "label": "Lifetime Workmanship Warranty", "desc": "Comprehensive guarantee on all panels, fixtures, and wiring", "pre_selected": False},
                {"id": "clean_uniformed", "label": "Background-Checked & Uniformed Electricians", "desc": "Professional, certified electricians arriving in marked company vans", "pre_selected": False},
            ],
        },
        "out_of_scope": {
            "title": "Guardrails & Out-of-Scope Requests",
            "desc": "Strict safety boundaries Riley enforces:",
            "topics": [
                {"id": "no_diy_panel_wiring", "label": "Strictly Refuse DIY High-Voltage Guidance", "desc": "Never tell a customer how to wire breakers or hot wires over the phone", "pre_selected": True},
                {"id": "no_blind_rewire_quotes", "label": "No Phone Quotes for Full Rewires", "desc": "Explain that whole-home rewires require attic and crawlspace inspection", "pre_selected": True},
                {"id": "utility_demarcation", "label": "Demarcate Utility Meter Responsibility", "desc": "Explain differences between customer equipment and utility company power lines", "pre_selected": False},
            ],
        },
    },

    "roofing": {
        "core_services": {
            "title": "Core Services & Work Scopes",
            "desc": "Roofing services Riley can explain and book:",
            "topics": [
                {"id": "leak_repair", "label": "Emergency Roof Leak Repair", "desc": "Diagnose active ceiling leaks, missing shingles, and faulty valley flashings", "pre_selected": True},
                {"id": "shingle_replacement", "label": "Architectural Shingle Replacement", "desc": "Full tear-off and installation of 30-to-50 year architectural shingles", "pre_selected": True},
                {"id": "storm_damage", "label": "Storm, Wind & Hail Damage Inspection", "desc": "Complete photo inspection for insurance hail and wind claims", "pre_selected": True},
                {"id": "gutter_installation", "label": "Seamless Gutters & Leaf Guards", "desc": "Custom-extruded seamless aluminum gutters and clog-free covers", "pre_selected": True},
                {"id": "metal_tile_roofs", "label": "Standing Seam Metal & Tile Roofing", "desc": "High-durability metal and barrel tile roof systems", "pre_selected": False},
                {"id": "flat_roof_commercial", "label": "Commercial Flat Roofs & TPO / Silicone", "desc": "TPO, EPDM, and silicone roof coatings for commercial buildings", "pre_selected": False},
                {"id": "skylight_chimney", "label": "Skylight & Chimney Flashing Repair", "desc": "Re-flash leaking skylights, chimneys, and plumbing vent pipes", "pre_selected": False},
                {"id": "attic_ventilation", "label": "Ridge Vents & Attic Ventilation", "desc": "Balance attic airflow to prevent ice dams and lower summer AC bills", "pre_selected": False},
            ],
        },
        "pricing_topics": {
            "title": "Pricing, Fees & Estimates",
            "desc": "How Riley answers roofing estimate and insurance questions:",
            "topics": [
                {"id": "free_21_point_inspection", "label": "Complimentary 21-Point Roof Inspection", "desc": "100% free inspection with drone photos and zero obligation", "pre_selected": True},
                {"id": "insurance_assistance", "label": "Insurance Claim Deductible Guidance", "desc": "Explain how we assist homeowners with adjusters and claim paperwork", "pre_selected": False},
                {"id": "financing_roofing", "label": "Low Monthly Payment Roof Financing", "desc": "Flexible financing options including 0% interest for 12 months", "pre_selected": False},
                {"id": "transparent_sq_pricing", "label": "Transparent Per-Square Pricing", "desc": "Provide clear breakdown of materials, labor, and warranty", "pre_selected": False},
            ],
        },
        "booking_topics": {
            "title": "Scheduling & Availability",
            "desc": "Inspection scheduling rules:",
            "topics": [
                {"id": "convenient_slots", "label": "Same-Day or Next-Day Inspection Slots", "desc": "Book morning or afternoon inspection times with homeowner present", "pre_selected": True},
                {"id": "drone_inspections", "label": "Contactless Drone Inspections Available", "desc": "Option for automated drone roof scan if homeowner is not on site", "pre_selected": False},
                {"id": "emergency_tarps", "label": "Emergency Storm Tarp Dispatch", "desc": "Dispatch rapid tarp crew during storms to prevent interior water damage", "pre_selected": True},
                {"id": "adjuster_coordination", "label": "Meet Insurance Adjusters On-Site", "desc": "Schedule roofing specialist to be on site when adjuster inspects", "pre_selected": False},
            ],
        },
        "emergency_topics": {
            "title": "Emergency & Urgent Triage",
            "desc": "Urgent weather emergencies requiring rapid tarping:",
            "topics": [
                {"id": "water_pouring_in", "label": "Water Actively Pouring Through Ceiling", "desc": "Instruct placing buckets, piercing blister if safe, and dispatch tarp crew", "pre_selected": True},
                {"id": "tree_limb_impact", "label": "Tree Limb Punctured Roof", "desc": "Prioritize structural inspection and waterproof tarping", "pre_selected": True},
                {"id": "flying_shingles_wind", "label": "Shingles Blown Off in High Wind", "desc": "Schedule immediate post-storm emergency patch", "pre_selected": False},
            ],
        },
        "policies_credentials": {
            "title": "Trust, Credentials & Guarantees",
            "desc": "Credentials Riley highlights:",
            "topics": [
                {"id": "gaf_owens_certified", "label": "Manufacturer Certified Installer", "desc": "Factory certified (GAF Master Elite / Owens Corning Platinum)", "pre_selected": True},
                {"id": "transferable_warranty", "label": "50-Year Non-Prorated Warranty", "desc": "Transferable warranty on lifetime shingles and labor", "pre_selected": True},
                {"id": "licensed_bonded_insured", "label": "Licensed, Bonded & Insured for Heights", "desc": "Full roofing contractor license with comprehensive workers' comp", "pre_selected": False},
            ],
        },
        "out_of_scope": {
            "title": "Guardrails & Out-of-Scope Requests",
            "desc": "Boundaries Riley adheres to:",
            "topics": [
                {"id": "no_diy_climbing_advice", "label": "Warn Callers Against Climbing Wet Roofs", "desc": "Never encourage an untrained caller to climb a slippery or steep roof", "pre_selected": True},
                {"id": "no_guaranteed_insurance_payout", "label": "Do Not Promise Guaranteed Claim Approval", "desc": "State that insurance company makes final claim determinations", "pre_selected": True},
            ],
        },
    },

    "dental_medical": {
        "core_services": {
            "title": "Core Services & Work Scopes",
            "desc": "Healthcare services Riley can schedule and discuss:",
            "topics": [
                {"id": "routine_cleaning", "label": "Routine Hygiene Exam & Cleaning", "desc": "Schedule comprehensive exams, digital x-rays, and ultrasonic cleanings", "pre_selected": True},
                {"id": "emergency_toothache", "label": "Emergency Toothache & Dental Trauma", "desc": "Fast same-day appointment for acute pain, broken tooth, or swelling", "pre_selected": True},
                {"id": "crowns_fillings", "label": "Tooth Fillings, Crowns & Bridges", "desc": "Tooth-colored composite fillings, porcelain crowns, and root canals", "pre_selected": True},
                {"id": "invisalign_cosmetic", "label": "Invisalign, Aligners & Teeth Whitening", "desc": "Clear orthodontic aligners and professional in-office whitening", "pre_selected": True},
                {"id": "dental_implants", "label": "Dental Implants & Dentures", "desc": "Permanent tooth replacement, bone grafts, and implant restorations", "pre_selected": False},
                {"id": "pediatric_dentistry", "label": "Family & Pediatric Dental Visits", "desc": "Gentle first visits for kids, fluoride treatments, and sealants", "pre_selected": False},
            ],
        },
        "pricing_topics": {
            "title": "Pricing, Insurance & Billing",
            "desc": "How Riley answers insurance and fee questions:",
            "topics": [
                {"id": "new_patient_special", "label": "New Patient Exam & X-Ray Special", "desc": "Quote fixed discounted rate for new patient exam, x-rays, and cleaning", "pre_selected": True},
                {"id": "ppo_insurance", "label": "In-Network PPO Insurance Plans", "desc": "Explain acceptance of Delta Dental, MetLife, Cigna, Guardian, Aetna, etc.", "pre_selected": False},
                {"id": "flexible_membership", "label": "In-House Dental Savings Plan", "desc": "Offer affordable annual plan for patients without dental insurance", "pre_selected": False},
                {"id": "carecredit_financing", "label": "CareCredit & Flexible Payments", "desc": "0% interest medical financing for dental procedures", "pre_selected": False},
            ],
        },
        "booking_topics": {
            "title": "Scheduling & Office Hours",
            "desc": "Appointment timing and patient intake:",
            "topics": [
                {"id": "appointment_windows", "label": "Morning & Afternoon Visit Times", "desc": "Book specific 45-60 minute chair times with doctor or hygienist", "pre_selected": True},
                {"id": "intake_forms_sms", "label": "Digital Intake Forms Sent via SMS", "desc": "Text new patient registration forms to fill out on mobile before arrival", "pre_selected": True},
                {"id": "cancellation_notice", "label": "48-Hour Reschedule Notice", "desc": "Explain policy for changing appointments without fee", "pre_selected": False},
            ],
        },
        "emergency_topics": {
            "title": "Emergency & Urgent Triage",
            "desc": "Medical/dental emergencies requiring urgent routing:",
            "topics": [
                {"id": "knocked_out_tooth", "label": "Knocked-Out Permanent Tooth", "desc": "Instruct keeping tooth in milk or saliva and transferring to doctor immediately", "pre_selected": True},
                {"id": "severe_facial_swelling", "label": "Severe Facial Swelling / Breathing Difficulty", "desc": "Instruct calling 911 or visiting nearest emergency room immediately", "pre_selected": True},
                {"id": "post_op_bleeding", "label": "Post-Extraction Excessive Bleeding", "desc": "Instruct biting on gauze and transfer immediately to on-call clinician", "pre_selected": True},
            ],
        },
        "policies_credentials": {
            "title": "Trust, Credentials & Guarantees",
            "desc": "Credentials Riley highlights:",
            "topics": [
                {"id": "board_certified", "label": "Board-Certified Doctors & ADA Members", "desc": "Highlight credentials, dental association membership, and modern tech", "pre_selected": True},
                {"id": "hipaa_compliant", "label": "Strict HIPAA Privacy & Sterilization", "desc": "Reassure complete patient confidentiality and hospital-grade sterilization", "pre_selected": True},
            ],
        },
        "out_of_scope": {
            "title": "Guardrails & Out-of-Scope Requests",
            "desc": "Medical boundaries Riley enforces:",
            "topics": [
                {"id": "no_medical_diagnosis", "label": "Strictly No Medical / Drug Diagnosis Over Phone", "desc": "Never prescribe drugs, antibiotics, or diagnose conditions without an exam", "pre_selected": True},
                {"id": "refer_severe_trauma", "label": "Direct Uncontrolled Bleeding or Trauma to 911", "desc": "Immediate referral to emergency hospital services for life-threatening trauma", "pre_selected": True},
            ],
        },
    },

    "legal": {
        "core_services": {
            "title": "Practice Areas & Intake",
            "desc": "Legal practice areas Riley can screen and schedule:",
            "topics": [
                {"id": "free_case_eval", "label": "Complimentary Initial Case Evaluation", "desc": "Screen prospective client details and schedule a 20-minute attorney review", "pre_selected": True},
                {"id": "personal_injury", "label": "Personal Injury & Auto Accidents", "desc": "Screen car crashes, slip-and-falls, and injury claims (no fee unless we win)", "pre_selected": True},
                {"id": "family_law", "label": "Family Law, Divorce & Child Custody", "desc": "Confidential intake for divorce, custody, and support consultations", "pre_selected": True},
                {"id": "estate_planning", "label": "Estate Planning, Wills & Trusts", "desc": "Schedule consultations for revocable living trusts, wills, and power of attorney", "pre_selected": True},
                {"id": "business_law", "label": "Business Formation & Contracts", "desc": "LLC creation, operating agreements, and contract reviews", "pre_selected": False},
                {"id": "criminal_defense", "label": "Criminal Defense & Traffic Citations", "desc": "Urgent intake for DUI, traffic court, and misdemeanor defense", "pre_selected": False},
                {"id": "immigration_law", "label": "Immigration & Naturalization Consultations", "desc": "Screen visa renewals, family petitions, and citizenship intake", "pre_selected": False},
                {"id": "real_estate_closing", "label": "Real Estate Closings & Title Review", "desc": "Residential and commercial real estate deed and purchase review", "pre_selected": False},
            ],
        },
        "pricing_topics": {
            "title": "Fees, Retainers & Contingency",
            "desc": "How Riley answers fee and billing questions:",
            "topics": [
                {"id": "contingency_injury", "label": "Contingency Fee Policy (Zero Upfront)", "desc": "Explain that injury cases require zero upfront retainer fee", "pre_selected": True},
                {"id": "retainer_consult", "label": "Retainer & Hourly Consultation Rates", "desc": "Explain transparent retainer policy quoted directly during consultation", "pre_selected": False},
                {"id": "flat_fee_services", "label": "Flat-Fee Estate & Business Packages", "desc": "Mention available flat packages for wills, trusts, and LLC formation", "pre_selected": False},
                {"id": "payment_plans_legal", "label": "Interest-Free Retainer Payment Plans", "desc": "Structured bi-weekly or monthly payment options for retainers", "pre_selected": False},
            ],
        },
        "booking_topics": {
            "title": "Consultation Scheduling",
            "desc": "Consultation formats and availability:",
            "topics": [
                {"id": "phone_video_inperson", "label": "Phone, Zoom or In-Person Consultations", "desc": "Offer convenient video call, phone call, or downtown office appointments", "pre_selected": True},
                {"id": "conflict_check", "label": "Advising Caller of Standard Conflict Check", "desc": "Collect opposing party name to conduct mandatory bar conflict check", "pre_selected": True},
                {"id": "weekend_consults", "label": "Saturday Morning Consultations", "desc": "Special weekend appointment slots for working clients", "pre_selected": False},
                {"id": "expedited_review", "label": "Rush Document Review (24-Hour)", "desc": "Fast-turnaround contract or agreement review appointments", "pre_selected": False},
            ],
        },
        "emergency_topics": {
            "title": "Emergency & Urgent Triage",
            "desc": "Time-sensitive legal emergencies:",
            "topics": [
                {"id": "jail_custody", "label": "Caller in Police Custody / Jail Booking", "desc": "Route immediately to on-call criminal defense attorney cell", "pre_selected": True},
                {"id": "court_deadline_24h", "label": "Court Appearance / Hearing Within 24 Hours", "desc": "Flag as urgent and alert senior paralegal or attorney", "pre_selected": True},
                {"id": "search_warrant_subpoena", "label": "Active Search Warrant or Subpoena Served", "desc": "Immediate crisis escalation to lead defense counsel", "pre_selected": True},
                {"id": "opposing_counsel", "label": "Opposing Counsel or Judge Calling", "desc": "Direct transfer to attorney handling the specific case file", "pre_selected": False},
            ],
        },
        "policies_credentials": {
            "title": "Trust, Credentials & Guarantees",
            "desc": "Credentials Riley highlights:",
            "topics": [
                {"id": "state_bar_good_standing", "label": "State Bar Association Member in Good Standing", "desc": "Confirm licensed attorneys with decades of combined courtroom experience", "pre_selected": True},
                {"id": "strict_confidentiality", "label": "Attorney-Client Privilege Confidentiality", "desc": "Assure caller that all intake information is held strictly confidential", "pre_selected": True},
                {"id": "super_lawyers_avvo", "label": "Super Lawyers & 10.0 Avvo Rating", "desc": "Recognized courtroom excellence and top peer reviews", "pre_selected": False},
            ],
        },
        "out_of_scope": {
            "title": "Guardrails & Out-of-Scope Requests",
            "desc": "Mandatory legal ethics guardrails:",
            "topics": [
                {"id": "no_legal_advice_over_phone", "label": "Strictly No Formal Legal Advice Over Phone", "desc": "Explain that receptionist cannot offer legal opinions; only attorney can advise", "pre_selected": True},
                {"id": "no_guaranteed_outcome", "label": "No Guarantee of Case Outcomes or Payouts", "desc": "Ethical prohibition against promising specific settlement numbers", "pre_selected": True},
            ],
        },
    },

    "auto": {
        "core_services": {
            "title": "Core Services & Work Scopes",
            "desc": "Automotive services Riley can explain and schedule:",
            "topics": [
                {"id": "brake_service", "label": "Brake Pad & Rotor Replacement", "desc": "Squeaking, grinding, and soft brake pedal diagnostics and replacements", "pre_selected": True},
                {"id": "oil_change_tuneup", "label": "Full Synthetic Oil Change & Filter", "desc": "Includes 30-point safety inspection, tire pressure, and fluid top-off", "pre_selected": True},
                {"id": "check_engine_light", "label": "Computer Diagnostics & Check Engine Light", "desc": "OBD-II diagnostic scan, code reading, and sensor troubleshooting", "pre_selected": True},
                {"id": "transmission_clutch", "label": "Transmission Service & Fluid Flush", "desc": "Diagnose slipping gears, fluid leaks, and clutch replacement", "pre_selected": True},
                {"id": "suspension_struts", "label": "Suspension, Struts & Shocks", "desc": "Fix bumpy rides, squeaks, worn bushings, and control arms", "pre_selected": False},
                {"id": "exhaust_muffler", "label": "Exhaust System & Catalytic Converters", "desc": "Repair noisy mufflers, catalytic theft replacements, and emissions", "pre_selected": False},
                {"id": "tires_alignment", "label": "Tires, Balancing & 4-Wheel Alignment", "desc": "Mounting, high-speed balancing, tire rotation, and alignment", "pre_selected": False},
                {"id": "auto_ac_heating", "label": "Auto AC Recharge & Climate Service", "desc": "Freon recharge, condenser leak tests, and heater core repair", "pre_selected": False},
                {"id": "battery_starter", "label": "Battery, Alternator & Starter Repair", "desc": "Testing starting charging systems and installing OEM batteries", "pre_selected": False},
            ],
        },
        "pricing_topics": {
            "title": "Pricing, Fees & Estimates",
            "desc": "How Riley answers repair cost questions:",
            "topics": [
                {"id": "diagnostic_credited", "label": "Digital Scan Fee Credited to Approved Repair", "desc": "Standard inspection fee applied directly toward approved repair order", "pre_selected": True},
                {"id": "free_brake_inspection", "label": "Free Visual Brake & Tire Inspection", "desc": "Zero-charge visual check when customer brings vehicle in", "pre_selected": False},
                {"id": "parts_labor_warranty", "label": "24-Month / 24,000-Mile Warranty", "desc": "Nationwide warranty on all OEM parts and labor", "pre_selected": False},
                {"id": "financing_auto", "label": "Snap / Synchrony Auto Repair Financing", "desc": "Easy payment plans for unexpected engine or transmission work", "pre_selected": False},
                {"id": "price_match_tires", "label": "Tire Price-Match Guarantee", "desc": "Match local competitor pricing on major brand tires", "pre_selected": False},
            ],
        },
        "booking_topics": {
            "title": "Drop-Off & Scheduling",
            "desc": "Vehicle intake options:",
            "topics": [
                {"id": "dropoff_times", "label": "Morning Drop-Off & Key Box", "desc": "Book morning drop-off slots or explain secure 24/7 night drop box", "pre_selected": True},
                {"id": "same_day_turnaround", "label": "Same-Day Turnaround on Routine Service", "desc": "Brakes, oil changes, and maintenance completed same day", "pre_selected": True},
                {"id": "night_drop", "label": "24/7 Secure Key Drop Box", "desc": "Safe after-hours vehicle drop-off with key envelope envelope", "pre_selected": False},
                {"id": "loaner_shuttle", "label": "Complimentary Local Shuttle Service", "desc": "Free rides within 5 miles while vehicle is being serviced", "pre_selected": False},
            ],
        },
        "emergency_topics": {
            "title": "Emergency & Urgent Triage",
            "desc": "Stranded driver or tow truck intake:",
            "topics": [
                {"id": "tow_truck_intake", "label": "Tow Truck Delivering Vehicle", "desc": "Instruct driver where to stage vehicle and leave keys; notify service advisor", "pre_selected": True},
                {"id": "stranded_highway", "label": "Stranded Driver Needing Tow Recommendation", "desc": "Provide trusted partner towing dispatch number", "pre_selected": True},
            ],
        },
        "policies_credentials": {
            "title": "Trust, Credentials & Guarantees",
            "desc": "Credentials Riley highlights:",
            "topics": [
                {"id": "ase_certified", "label": "ASE-Certified Master Technicians", "desc": "Highlight automotive excellence and certified training", "pre_selected": True},
                {"id": "oem_parts", "label": "OEM & Premium Quality Parts", "desc": "We use original equipment or top-tier aftermarket components", "pre_selected": False},
                {"id": "aaa_approved", "label": "AAA-Approved Auto Repair Facility", "desc": "Vetted auto facility meeting strict quality and customer service standards", "pre_selected": False},
            ],
        },
        "out_of_scope": {
            "title": "Guardrails & Out-of-Scope Requests",
            "desc": "Boundaries Riley enforces:",
            "topics": [
                {"id": "no_blind_quotes", "label": "No Phone Quotes Without Seeing Vehicle", "desc": "Explain that brake or engine repairs require inspecting exact condition first", "pre_selected": True},
                {"id": "no_diy_mechanic_advice", "label": "Decline DIY Mechanical Advice", "desc": "Refuse advising callers on how to bypass safety sensors or wire starters", "pre_selected": True},
            ],
        },
    },

    "restaurant": {
        "core_services": {
            "title": "Dining, Reservations & Catering",
            "desc": "Hospitality services Riley can handle:",
            "topics": [
                {"id": "table_reservations", "label": "Table Reservations (1-6 Guests)", "desc": "Book dinner, lunch, or brunch reservations with date, time, and party size", "pre_selected": True},
                {"id": "large_parties", "label": "Large Parties (7+ Guests) & Buyouts", "desc": "Collect details for group dining or transfer to private event coordinator", "pre_selected": True},
                {"id": "hours_kitchen_close", "label": "Operating Hours & Kitchen Closing Time", "desc": "Answer questions about daily dining hours, happy hour, and last seating", "pre_selected": True},
                {"id": "allergy_dietary", "label": "Dietary & Allergen Accommodations", "desc": "Explain gluten-free, vegan, vegetarian, and nut-allergy options", "pre_selected": True},
                {"id": "bar_cocktail_lounge", "label": "Bar & Craft Cocktail Lounge Seating", "desc": "Explain walk-in seating at the full-service cocktail bar", "pre_selected": False},
                {"id": "private_chef_table", "label": "Chef's Tasting Menu & Kitchen Table", "desc": "Book exclusive multi-course dining and wine pairings", "pre_selected": False},
                {"id": "takeout_pickup", "label": "Online Takeout & Curbside Pickup", "desc": "Provide link or instructions for direct takeout ordering", "pre_selected": False},
                {"id": "catering_events", "label": "Off-Site Catering & Corporate Trays", "desc": "Screen catering inquiries for office parties, weddings, and celebrations", "pre_selected": False},
                {"id": "corkage_cake_fee", "label": "Corkage & Cake Cutting Policy", "desc": "Explain outside wine corkage fee and celebration cake policy", "pre_selected": False},
            ],
        },
        "pricing_topics": {
            "title": "Pricing, Menus & Minimums",
            "desc": "How Riley answers dining pricing questions:",
            "topics": [
                {"id": "menu_overview", "label": "Average Entree Price Range", "desc": "Provide general entree price range and highlight signature dishes", "pre_selected": True},
                {"id": "happy_hour_specials", "label": "Happy Hour Times & Drink Specials", "desc": "Explain weekday happy hour hours, craft cocktails, and appetizer discounts", "pre_selected": False},
                {"id": "kids_menu_specials", "label": "Kids Menu & Family Night Pricing", "desc": "Explain children's dining options and family platters", "pre_selected": False},
                {"id": "private_dining_minimums", "label": "Private Dining Food & Beverage Minimums", "desc": "Explain that private rooms have seasonal food and beverage minimums", "pre_selected": False},
            ],
        },
        "booking_topics": {
            "title": "Seating & Policies",
            "desc": "Seating guidelines:",
            "topics": [
                {"id": "grace_period", "label": "15-Minute Reservation Grace Period", "desc": "Explain that tables are held for 15 minutes before being released", "pre_selected": True},
                {"id": "patio_seating", "label": "Patio / Outdoor Seating Requests", "desc": "Explain weather-permitting outdoor patio seating policy", "pre_selected": True},
                {"id": "cancellation_fee_policy", "label": "No-Show & 24-Hour Cancellation Fee", "desc": "Explain reservation hold policy for parties of 5+ or holiday seatings", "pre_selected": False},
                {"id": "parking_valet", "label": "Valet & Self-Parking Instructions", "desc": "Provide exact parking garage or valet directions", "pre_selected": False},
            ],
        },
        "emergency_topics": {
            "title": "Urgent Inquiries",
            "desc": "Time-sensitive restaurant requests:",
            "topics": [
                {"id": "running_late", "label": "Caller Running Late for Tonight's Table", "desc": "Take party name, update host stand notes, and extend grace period", "pre_selected": True},
                {"id": "severe_allergy_alert", "label": "Severe Anaphylactic Allergen Question", "desc": "Transfer to manager or chef immediately for safety verification", "pre_selected": True},
                {"id": "vendor_loading_dock", "label": "Delivery Purveyor or Vendor Calling", "desc": "Transfer to kitchen or loading dock manager", "pre_selected": False},
            ],
        },
        "policies_credentials": {
            "title": "Hospitality Policies",
            "desc": "General dining rules:",
            "topics": [
                {"id": "dress_code", "label": "Smart Casual Dress Code", "desc": "Explain attire guidelines (e.g. smart casual, no swimwear)", "pre_selected": True},
                {"id": "locally_sourced_organic", "label": "Farm-to-Table Locally Sourced Ingredients", "desc": "Highlight fresh regional produce and sustainable ingredients", "pre_selected": False},
                {"id": "pet_friendly_patio", "label": "Dog / Pet Policy on Outdoor Patio", "desc": "Confirm whether well-behaved leashed dogs are welcomed outdoors", "pre_selected": False},
            ],
        },
        "out_of_scope": {
            "title": "Guardrails & Out-of-Scope Requests",
            "desc": "Boundaries Riley enforces:",
            "topics": [
                {"id": "no_holding_tables_without_name", "label": "No Holding Tables Without Name & Phone", "desc": "Require contact details for every reservation", "pre_selected": True},
                {"id": "no_outside_alcohol", "label": "No Unlicensed Outside Liquor", "desc": "Uphold state liquor control board regulations", "pre_selected": True},
            ],
        },
    },

    "realestate": {
        "core_services": {
            "title": "Property Services & Intake",
            "desc": "Real estate services Riley can screen and book:",
            "topics": [
                {"id": "property_showings", "label": "Private Home & Condo Showings", "desc": "Schedule showings for active MLS listings with licensed agent", "pre_selected": True},
                {"id": "free_home_valuation", "label": "Complimentary Home Valuation & CMA", "desc": "Collect property address and seller timeline for a free market report", "pre_selected": True},
                {"id": "first_time_buyer", "label": "First-Time Homebuyer Consultation", "desc": "Screen budget, pre-approval status, and desired neighborhoods", "pre_selected": True},
                {"id": "property_management", "label": "Rental & Property Management Services", "desc": "Screen landlord inquiries for tenant placement and lease management", "pre_selected": True},
                {"id": "relocation_services", "label": "Corporate & Family Relocation Guidance", "desc": "School district information, neighborhood guides, and area tours", "pre_selected": False},
                {"id": "investment_1031", "label": "1031 Exchange & Multi-Family Investing", "desc": "Screen investors for duplexes, commercial, and tax-deferred exchanges", "pre_selected": False},
                {"id": "open_house_info", "label": "Open House Schedules & Directions", "desc": "Provide dates, hours, and gate codes for upcoming open houses", "pre_selected": False},
                {"id": "commercial_leasing", "label": "Commercial Leasing & Retail Spaces", "desc": "Screen square footage and lease terms for business spaces", "pre_selected": False},
            ],
        },
        "pricing_topics": {
            "title": "Commissions & Valuation",
            "desc": "How Riley answers commission and fee questions:",
            "topics": [
                {"id": "free_valuation_zero_fee", "label": "100% Free Valuation (Zero Obligation)", "desc": "Explain that property valuation and buyer consults are completely free", "pre_selected": True},
                {"id": "negotiable_commissions", "label": "Competitive & Flexible Commission Rates", "desc": "Explain that listing commissions are customized and discussed with broker", "pre_selected": False},
                {"id": "flat_fee_listing", "label": "Custom Marketing Packages & Variable Rates", "desc": "Explain tailored marketing plans suited to luxury or entry homes", "pre_selected": False},
            ],
        },
        "booking_topics": {
            "title": "Showing & Meeting Scheduling",
            "desc": "Scheduling parameters:",
            "topics": [
                {"id": "showing_windows", "label": "Daily Showing Windows (9 AM - 7 PM)", "desc": "Book morning, afternoon, or twilight showing times", "pre_selected": True},
                {"id": "proof_of_funds", "label": "Pre-Approval / Proof of Funds Inquiries", "desc": "Politely ask whether buyer has mortgage pre-approval letter ready", "pre_selected": True},
                {"id": "weekend_tours", "label": "Saturday & Sunday VIP Property Tours", "desc": "Reserved multi-home weekend showing blocks for buyers", "pre_selected": False},
            ],
        },
        "emergency_topics": {
            "title": "Urgent Real Estate Inquiries",
            "desc": "Time-sensitive buyer or seller transactions:",
            "topics": [
                {"id": "submitting_written_offer", "label": "Buyer Submitting Written Offer Today", "desc": "Immediate transfer to lead listing agent's mobile phone", "pre_selected": True},
                {"id": "title_closing_deadline", "label": "Escrow / Title Company on Closing Deadline", "desc": "Priority routing to transaction coordinator", "pre_selected": True},
                {"id": "lockbox_issue", "label": "Showing Agent Lockbox Access Issue", "desc": "Provide verified showing code or connect with listing agent", "pre_selected": False},
            ],
        },
        "policies_credentials": {
            "title": "Trust, Credentials & Guarantees",
            "desc": "Credentials Riley highlights:",
            "topics": [
                {"id": "licensed_realtor", "label": "Licensed REALTOR® & NAR Member", "desc": "Adhere strictly to National Association of Realtors Code of Ethics", "pre_selected": True},
                {"id": "local_market_expert", "label": "Top 1% Local Neighborhood Production", "desc": "Decades of localized pricing expertise and proven sales record", "pre_selected": False},
                {"id": "zillow_premier_5star", "label": "5-Star Premier Agent & Top Producer", "desc": "Over 100+ verified client reviews and five-star rating", "pre_selected": False},
            ],
        },
        "out_of_scope": {
            "title": "Guardrails & Out-of-Scope Requests",
            "desc": "Boundaries Riley enforces:",
            "topics": [
                {"id": "fair_housing_strict", "label": "Strict Adherence to Fair Housing Act", "desc": "Never comment on neighborhood demographic makeup or steer buyers", "pre_selected": True},
                {"id": "no_binding_contract_terms", "label": "No Verbal Binding Agreement on Pricing", "desc": "Clarify that all offers must be executed in writing on standard forms", "pre_selected": True},
            ],
        },
    },

    "salon_spa": {
        "core_services": {
            "title": "Beauty & Wellness Services",
            "desc": "Salon and spa treatments Riley can schedule:",
            "topics": [
                {"id": "haircut_blowout", "label": "Signature Haircut, Wash & Blowout", "desc": "Book women's, men's, and children's precision cuts and blowouts", "pre_selected": True},
                {"id": "color_balayage", "label": "Custom Balayage, Highlights & Color", "desc": "Schedule custom foil highlights, root touch-ups, and balayage", "pre_selected": True},
                {"id": "facials_skincare", "label": "Custom Facials & Hydrating Peels", "desc": "Book anti-aging facials, dermaplaning, and microdermabrasion", "pre_selected": True},
                {"id": "massage_therapy", "label": "Deep Tissue & Swedish Massage", "desc": "Book 60, 90, or 120-minute therapeutic massage sessions", "pre_selected": True},
                {"id": "waxing_hair_removal", "label": "Full Body & Facial Waxing Services", "desc": "Eyebrow, bikini, Brazilian, and gentle facial waxing", "pre_selected": False},
                {"id": "hair_extensions", "label": "Hand-Tied & Tape-In Hair Extensions", "desc": "Consultation and installation for natural volume and length", "pre_selected": False},
                {"id": "manicure_pedicure", "label": "Gel, Dip Powder & Acrylic Nails", "desc": "Schedule spa pedicures, gel manicures, and nail art", "pre_selected": False},
                {"id": "lash_brows", "label": "Eyelash Extensions & Brow Lamination", "desc": "Classic, hybrid, volume lash extensions and brow shaping", "pre_selected": False},
                {"id": "keratin_treatments", "label": "Brazilian Blowout & Keratin Smoothing", "desc": "Smooth frizz and restore hair health with deep keratin treatments", "pre_selected": False},
            ],
        },
        "pricing_topics": {
            "title": "Stylist Tiers & Pricing",
            "desc": "How Riley answers service cost questions:",
            "topics": [
                {"id": "tier_pricing", "label": "Stylist Experience Level Pricing", "desc": "Explain that prices vary based on Junior, Senior, or Master Stylist level", "pre_selected": True},
                {"id": "consultation_color", "label": "Complimentary 15-Min Color Consultations", "desc": "Offer free consultations for major transformations or color corrections", "pre_selected": False},
                {"id": "membership_blowout", "label": "Monthly Blowout & Treatment Club", "desc": "Discounted recurring membership for frequent salon guests", "pre_selected": False},
                {"id": "gift_cards", "label": "Gift Cards & Package Discounts", "desc": "Mention available holiday gift cards and multi-service packages", "pre_selected": False},
            ],
        },
        "booking_topics": {
            "title": "Appointments & Intake",
            "desc": "Scheduling and cancellation rules:",
            "topics": [
                {"id": "specific_stylist", "label": "Requesting a Specific Stylist or Therapist", "desc": "Check individual stylist calendar availability or book first available", "pre_selected": True},
                {"id": "cancellation_policy", "label": "24-Hour Cancellation Policy", "desc": "Explain 24-hour notice requirement to avoid cancellation or rebooking fee", "pre_selected": True},
                {"id": "walk_in_waitlist", "label": "Live Digital Walk-In Waitlist", "desc": "Send text notification when an immediate walk-in chair opens up", "pre_selected": False},
                {"id": "arrival_reminder", "label": "Arrive 10 Minutes Early for Spa Treatments", "desc": "Advise guests to arrive early to change into robes and relax", "pre_selected": False},
            ],
        },
        "emergency_topics": {
            "title": "Urgent Booking Inquiries",
            "desc": "Same-day changes or party adjustments:",
            "topics": [
                {"id": "running_late_salon", "label": "Guest Running Late for Booked Slot", "desc": "Notify stylist and adjust appointment sequence smoothly", "pre_selected": True},
                {"id": "bridal_party_booking", "label": "Bridal / Wedding Party Group Booking", "desc": "Transfer to bridal coordinator for contract and schedule planning", "pre_selected": True},
            ],
        },
        "policies_credentials": {
            "title": "Trust, Credentials & Guarantees",
            "desc": "Salon standards Riley highlights:",
            "topics": [
                {"id": "licensed_cosmetologists", "label": "State-Licensed Cosmetologists & Estheticians", "desc": "Highlight licensed professionals and ongoing advanced education", "pre_selected": True},
                {"id": "cruelty_free_products", "label": "Cruelty-Free & Vegan Product Lines", "desc": "Premium organic haircare and skincare brands", "pre_selected": False},
                {"id": "hospital_grade_sanitation", "label": "Autoclave & Hospital-Grade Sanitation", "desc": "Reassure clients of sanitized tools and clean environment", "pre_selected": False},
            ],
        },
        "out_of_scope": {
            "title": "Guardrails & Out-of-Scope Requests",
            "desc": "Boundaries Riley enforces:",
            "topics": [
                {"id": "no_diy_bleach_advice", "label": "Refuse Giving At-Home Bleaching Advice", "desc": "Explain that chemical lightening requires professional formulation", "pre_selected": True},
                {"id": "no_walkin_guarantee", "label": "No Guarantee of Immediate Walk-In Service", "desc": "Encourage booking in advance to ensure availability", "pre_selected": True},
            ],
        },
    },

    "general": {
        "core_services": {
            "title": "Core Services & Work Scopes",
            "desc": "Services Riley can describe, quote, and schedule:",
            "topics": [
                {"id": "general_inquiry", "label": "General Service Inquiries & Estimates", "desc": "Answer questions about business capabilities, service offerings, and scope", "pre_selected": True},
                {"id": "schedule_consult", "label": "Schedule In-Person or Phone Consultation", "desc": "Book discovery calls, on-site evaluations, or office meetings", "pre_selected": True},
                {"id": "urgent_assistance", "label": "Urgent / Same-Day Service Requests", "desc": "Prioritize urgent client inquiries and notify team immediately", "pre_selected": True},
                {"id": "emergency_dispatch_gen", "label": "Rapid Same-Day Priority Dispatch", "desc": "Direct urgent response for time-critical situations", "pre_selected": False},
                {"id": "recurring_maintenance", "label": "Scheduled Maintenance & Preventative Care", "desc": "Quarterly and annual preventative checkups and tune-ups", "pre_selected": False},
                {"id": "maintenance_contracts", "label": "Commercial Accounts & Service Contracts", "desc": "Screen commercial and recurring service agreement inquiries", "pre_selected": False},
                {"id": "billing_invoice_inquiry", "label": "Invoice & Billing Inquiries", "desc": "Collect invoice number and customer details for billing team review", "pre_selected": False},
            ],
        },
        "pricing_topics": {
            "title": "Pricing & Estimates",
            "desc": "How Riley answers cost inquiries:",
            "topics": [
                {"id": "upfront_transparent_estimates", "label": "Upfront Transparent Estimates", "desc": "Assure caller of honest, itemized quotes before any work begins", "pre_selected": True},
                {"id": "free_consultation_quote", "label": "Free Initial Consultation / Quote", "desc": "Zero-fee, zero-obligation discovery call or on-site quote", "pre_selected": False},
                {"id": "no_hidden_fees", "label": "No Hidden Travel or Fuel Surcharges", "desc": "Clear all-inclusive quotes with no unexpected surprise line items", "pre_selected": False},
                {"id": "payment_options_general", "label": "Credit Cards, ACH & Check Accepted", "desc": "Explain all accepted payment methods and invoice payment terms", "pre_selected": False},
            ],
        },
        "booking_topics": {
            "title": "Scheduling & Availability",
            "desc": "Appointment timing and dispatch:",
            "topics": [
                {"id": "convenient_windows", "label": "Flexible Appointment Windows", "desc": "Offer convenient morning or afternoon appointment slots", "pre_selected": True},
                {"id": "sms_confirmation", "label": "Instant SMS Confirmation & Reminders", "desc": "Send automated calendar invite and text reminders prior to meeting", "pre_selected": True},
                {"id": "flexible_windows", "label": "Exact 1-Hour Arrival Times", "desc": "Guaranteed narrow arrival windows with real-time text updates", "pre_selected": False},
            ],
        },
        "emergency_topics": {
            "title": "Priority Escalations",
            "desc": "Urgent situations requiring immediate transfer:",
            "topics": [
                {"id": "owner_requested", "label": "Caller Specifically Asks for Owner / Executive", "desc": "Transfer to owner mobile or send high-priority alert immediately", "pre_selected": True},
                {"id": "urgent_crisis", "label": "Time-Sensitive Client Emergency", "desc": "Initiate urgent call forwarding to on-call supervisor", "pre_selected": True},
            ],
        },
        "policies_credentials": {
            "title": "Trust, Credentials & Guarantees",
            "desc": "Credentials Riley highlights:",
            "topics": [
                {"id": "licensed_insured_general", "label": "Licensed, Bonded & Insured", "desc": "Verify professional license and general liability coverage", "pre_selected": True},
                {"id": "satisfaction_guarantee", "label": "100% Customer Satisfaction Guarantee", "desc": "Commitment to exceptional service quality and client care", "pre_selected": True},
                {"id": "bbb_accredited", "label": "BBB A+ Accredited Business", "desc": "Proven track record of consumer trust, reliability, and ethics", "pre_selected": False},
            ],
        },
        "out_of_scope": {
            "title": "Guardrails & Out-of-Scope Requests",
            "desc": "Boundaries Riley enforces:",
            "topics": [
                {"id": "no_speculation", "label": "No Speculative Guarantees Without Evaluation", "desc": "Explain that accurate assessments require examining project details first", "pre_selected": True},
                {"id": "warm_handoff_unknown", "label": "Warm Handoff for Complex Technical Questions", "desc": "Politely offer to connect caller directly with a specialist", "pre_selected": True},
            ],
        },
    },
}

def get_industry_topics(industry_id: str) -> Dict[str, Any]:
    """Returns the full categorized topic structure for an industry, falling back to general."""
    clean_id = (industry_id or "general").lower()
    return INDUSTRY_TOPIC_CATEGORIES.get(clean_id, INDUSTRY_TOPIC_CATEGORIES["general"])
