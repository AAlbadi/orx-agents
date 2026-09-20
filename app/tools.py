"""Tool definitions and handlers for voice agent function calling."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional
from loguru import logger


@dataclass
class FunctionCallParams:
    """Independent function call parameter container for LLM tool invocation callbacks."""
    result_callback: Optional[Callable[[Any], Awaitable[None]]] = None
    call_id: Optional[str] = None
    function_name: Optional[str] = None


async def _dispatch_result(params: Any, result: dict) -> dict:
    """Helper to notify result_callback if present, then return result dict."""
    if params is not None and hasattr(params, "result_callback") and callable(params.result_callback):
        cb = params.result_callback(result)
        if hasattr(cb, "__await__"):
            await cb
    return result


async def get_current_time(params: Optional[Any] = None, timezone: str = "local") -> dict:
    """Get the current system date, day of week, and time.

    Args:
        params: Optional FunctionCallParams callback container.
        timezone: The timezone identifier or 'local'.
    """
    if isinstance(params, str):
        timezone, params = params, None

    now = datetime.now()
    formatted = now.strftime("%A, %B %d, %Y, %I:%M %p")
    logger.info(f"Tool executed: get_current_time -> {formatted}")
    result = {"current_datetime": formatted, "timezone": timezone}
    return await _dispatch_result(params, result)


async def lookup_customer_account(params: Optional[Any] = None, identifier: str = "") -> dict:
    """Look up an existing customer account or recent support records.

    Args:
        params: Optional FunctionCallParams callback container.
        identifier: The customer's phone number, email address, or account ID.
    """
    if isinstance(params, str):
        identifier, params = params, None

    logger.info(f"Tool executed: lookup_customer_account -> {identifier}")
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
    return await _dispatch_result(params, mock_customer)


async def schedule_appointment(
    params: Optional[Any] = None,
    date: str = "",
    time: str = "",
    service_type: str = "",
    customer_notes: str = ""
) -> dict:
    """Schedule an appointment or callback for the customer.

    Args:
        params: Optional FunctionCallParams callback container.
        date: The date for the appointment in YYYY-MM-DD format (e.g. '2026-09-20').
        time: The requested time slot (e.g. '02:30 PM').
        service_type: The type of appointment (e.g. 'Technical Support', 'Consultation', 'Billing').
        customer_notes: Additional notes or requests mentioned by the caller.
    """
    if isinstance(params, str):
        date, time, service_type, customer_notes, params = params, date, time, service_type, None

    logger.info(f"Tool executed: schedule_appointment -> {date} at {time} for {service_type}")
    confirmation = {
        "status": "confirmed",
        "confirmation_code": "APT-88219",
        "date": date,
        "time": time,
        "service": service_type,
        "message": f"Appointment successfully scheduled for {service_type} on {date} at {time}."
    }
    return await _dispatch_result(params, confirmation)


async def transfer_to_human_agent(
    params: Optional[Any] = None,
    department: str = "",
    reason: str = ""
) -> dict:
    """Transfer the telephone call to a live human representative or tier-2 support.

    Args:
        params: Optional FunctionCallParams callback container.
        department: The target department (e.g. 'Sales', 'Technical Support', 'Billing').
        reason: The reason for the escalation or transfer.
    """
    if isinstance(params, str):
        department, reason, params = params, department, None

    logger.info(f"Tool executed: transfer_to_human_agent -> {department} (Reason: {reason})")
    transfer_status = {
        "status": "initiating_transfer",
        "department": department,
        "estimated_wait_time": "Less than 2 minutes",
        "message": f"Transferring to {department}. Informing customer to hold briefly."
    }
    return await _dispatch_result(params, transfer_status)


async def check_technician_availability(
    params: Optional[Any] = None,
    date: str = "",
    time_window: str = "morning",
) -> dict:
    """Check if an HVAC technician is available for an arrival window on a specific date.

    Args:
        params: Optional FunctionCallParams callback container.
        date: The date to check in YYYY-MM-DD format (e.g. '2026-09-19') or relative date like 'tomorrow' or 'Friday'.
        time_window: The requested arrival window: 'morning' (8 AM – 12 PM), 'afternoon' (12 PM – 4 PM), or 'evening' (4 PM – 7 PM).
    """
    if isinstance(params, str):
        date, time_window, params = params, date, None

    logger.info(f"Tool executed: check_technician_availability -> {date} ({time_window})")
    try:
        from app.appointments import check_technician_availability as check_avail
        result = await check_avail(date, time_window)
    except Exception as e:
        logger.error(f"Error checking technician availability: {e}")
        result = {
            "available": True,
            "date": date,
            "window": time_window,
            "message": f"Technician is available for {time_window} on {date}."
        }
    return await _dispatch_result(params, result)


# Export tool catalog list for LLM context registration
REGISTERED_TOOLS = [
    get_current_time,
    lookup_customer_account,
    schedule_appointment,
    check_technician_availability,
    transfer_to_human_agent,
]
