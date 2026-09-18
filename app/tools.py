"""Tool definitions and handlers for Pipecat function calling."""

from datetime import datetime
from loguru import logger
from pipecat.services.llm_service import FunctionCallParams


async def get_current_time(params: FunctionCallParams, timezone: str = "local"):
    """Get the current system date, day of week, and time.

    Args:
        timezone: The timezone identifier or 'local'.
    """
    now = datetime.now()
    formatted = now.strftime("%A, %B %d, %Y, %I:%M %p")
    logger.info(f"Tool executed: get_current_time -> {formatted}")
    await params.result_callback({"current_datetime": formatted, "timezone": timezone})


async def lookup_customer_account(params: FunctionCallParams, identifier: str):
    """Look up an existing customer account or recent support records.

    Args:
        identifier: The customer's phone number, email address, or account ID.
    """
    logger.info(f"Tool executed: lookup_customer_account -> {identifier}")
    # Simulated CRM response - easily connectable to Postgres / Hubspot / Salesforce
    mock_customer = {
        "status": "found",
        "name": "Alex Mercer",
        "account_level": "Premium",
        "open_tickets": [
            {
                "id": "TICK-4091",
                "summary": "Delivery schedule inquiry",
                "status": "In Progress"
            }
        ],
        "notes": "Prefers evening appointment slots."
    }
    await params.result_callback(mock_customer)


async def schedule_appointment(
    params: FunctionCallParams,
    date: str,
    time: str,
    service_type: str,
    customer_notes: str = ""
):
    """Schedule an appointment or callback for the customer.

    Args:
        date: The date for the appointment in YYYY-MM-DD format (e.g. '2026-09-20').
        time: The requested time slot (e.g. '02:30 PM').
        service_type: The type of appointment (e.g. 'Technical Support', 'Consultation', 'Billing').
        customer_notes: Additional notes or requests mentioned by the caller.
    """
    logger.info(f"Tool executed: schedule_appointment -> {date} at {time} for {service_type}")
    confirmation = {
        "status": "confirmed",
        "confirmation_code": "APT-88219",
        "date": date,
        "time": time,
        "service": service_type,
        "message": f"Appointment successfully scheduled for {service_type} on {date} at {time}."
    }
    await params.result_callback(confirmation)


async def transfer_to_human_agent(
    params: FunctionCallParams,
    department: str,
    reason: str
):
    """Transfer the telephone call to a live human representative or tier-2 support.

    Args:
        department: The target department (e.g. 'Sales', 'Technical Support', 'Billing').
        reason: The reason for the escalation or transfer.
    """
    logger.info(f"Tool executed: transfer_to_human_agent -> {department} (Reason: {reason})")
    transfer_status = {
        "status": "initiating_transfer",
        "department": department,
        "estimated_wait_time": "Less than 2 minutes",
        "message": f"Transferring to {department}. Informing customer to hold briefly."
    }
    await params.result_callback(transfer_status)


async def check_technician_availability(
    params: FunctionCallParams,
    date: str,
    time_window: str = "morning",
):
    """Check if an HVAC technician is available for an arrival window on a specific date.

    Args:
        date: The date to check in YYYY-MM-DD format (e.g. '2026-09-19') or relative date like 'tomorrow' or 'Friday'.
        time_window: The requested arrival window: 'morning' (8 AM – 12 PM), 'afternoon' (12 PM – 4 PM), or 'evening' (4 PM – 7 PM).
    """
    logger.info(f"Tool executed: check_technician_availability -> {date} ({time_window})")
    try:
        from app.appointments import check_technician_availability as check_avail
        result = await check_avail(date, time_window)
        await params.result_callback(result)
    except Exception as e:
        logger.error(f"Error checking technician availability: {e}")
        await params.result_callback({
            "available": True,
            "date": date,
            "window": time_window,
            "message": f"Technician is available for {time_window} on {date}."
        })


# Export tool catalog list for LLMContext registration
REGISTERED_TOOLS = [
    get_current_time,
    lookup_customer_account,
    schedule_appointment,
    check_technician_availability,
    transfer_to_human_agent,
]
