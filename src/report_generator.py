"""
Report Generator module for Accident Report Generation System.
Generates structured accident reports from AI model responses.
"""

import json
import re
import os
import logging
from datetime import datetime

from src.config import (
    REPORT_TITLE, ORGANIZATION, MODEL_NAME,
    REPORTS_DIR, generate_random_gps
)

logger = logging.getLogger(__name__)


def parse_structured_fields(model_response: str) -> dict:
    """
    Parse JSON from model response with robust error handling.

    Steps:
    1. Try direct JSON parsing
    2. If failed, extract JSON using regex pattern
    3. If still failed, use partial extraction with defaults
    4. If still empty, check for simple Yes/No responses

    Args:
        model_response: Raw text response from the model

    Returns:
        Parsed dictionary of fields
    """
    if not model_response or not model_response.strip():
        logger.warning("Empty model response received")
        return {}

    response = model_response.strip()
    logger.info(f"Raw model response: {response[:500]}")  # Log first 500 chars for debugging

    # Method 1: Direct JSON parsing
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Method 2: Extract JSON from response using regex
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    match = re.search(json_pattern, model_response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Method 3: Partial field extraction with regex
    fields = {
        'accident_detected': r'accident_detected["\s:]+([A-Za-z]+)',
        'confidence': r'confidence["\s:]+["\']?([a-z]+)["\']?',
        'reasoning': r'reasoning["\s:]+["\']?([^"\']+)["\']?',
        'accident_type': r'accident_type["\s:]+["\']?([^"\'\n,\}]+)',
        'number_of_victims': r'number_of_victims["\s:]+(\d+)',
        'vehicles_involved': r'vehicles_involved["\s:]+(\d+)',
        'accident_severity': r'accident_severity["\s:]+["\']?([^"\'\n,\}]+)',
        'injured_person_detected': r'injured_person_detected["\s:]+([A-Za-z]+)',
        'emergency_services_present': r'emergency_services_present["\s:]+([A-Za-z]+)',
        'road_blocked': r'road_blocked["\s:]+([A-Za-z]+)',
        'scene_description': r'scene_description["\s:]+["\']?(.+?)(?=["\']?\s*[, \}])'
    }

    result = {}
    for field, pattern in fields.items():
        field_match = re.search(pattern, model_response, re.IGNORECASE)
        if field_match:
            value = field_match.group(1).strip()
            if field in ['number_of_victims', 'vehicles_involved']:
                result[field] = int(value) if value.isdigit() else 0
            else:
                result[field] = value

    if result:
        logger.info(f"Extracted {len(result)} fields via regex fallback")
        return result

    # Method 4: Handle simple Yes/No responses (model returns just "Yes" or "No")
    response_lower = response.lower()
    if response_lower.startswith("yes"):
        logger.info("Detected simple 'Yes' response, assuming accident detected")
        return {
            "accident_detected": "Yes",
            "scene_description": response
        }
    elif response_lower.startswith("no"):
        logger.info("Detected simple 'No' response, assuming no accident")
        return {
            "accident_detected": "No",
            "scene_description": response
        }

    # Also try to find "yes" or "no" anywhere in the response
    yes_match = re.search(r'\byes\b', response_lower)
    no_match = re.search(r'\bno\b', response_lower)
    
    if yes_match and not no_match:
        logger.info("Found 'yes' in response, assuming accident detected")
        return {
            "accident_detected": "Yes",
            "scene_description": response
        }
    elif no_match:
        logger.info("Found 'no' in response, assuming no accident")
        return {
            "accident_detected": "No",
            "scene_description": response
        }

    logger.warning("Could not extract any structured fields from model response")
    return result


def get_current_datetime() -> dict:
    """
    Get the current date and time formatted for the report.

    Returns:
        dict with 'date' and 'time' strings
    """
    now = datetime.now()
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S")
    }


def get_next_report_number() -> int:
    """
    Get the next sequential report number based on existing files.

    Returns:
        Next report number (integer)
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    existing = [f for f in os.listdir(REPORTS_DIR) if f.startswith("accident_report_") and f.endswith(".pdf")]
    if not existing:
        return 1

    numbers = []
    for f in existing:
        try:
            num_str = f.replace("accident_report_", "").replace(".pdf", "")
            numbers.append(int(num_str))
        except ValueError:
            continue

    return max(numbers, default=0) + 1


def generate_report_filename() -> str:
    """
    Generate the next sequential report filename.

    Returns:
        Full path to the report file
    """
    num = get_next_report_number()
    filename = f"accident_report_{num:03d}.pdf"
    return os.path.join(REPORTS_DIR, filename)


def create_report_structure(detection_result: dict, analysis_result: dict) -> dict:
    """
    Create a complete report data structure combining detection, analysis, and metadata.

    Args:
        detection_result: Result from accident detection
        analysis_result: Result from accident analysis

    Returns:
        Complete report data dictionary
    """
    dt = get_current_datetime()
    lat, lon = generate_random_gps()

    report_data = {
        "report_id": f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "date": dt["date"],
        "time": dt["time"],
        "gps": {
            "latitude": lat,
            "longitude": lon
        },
        "accident_detected": detection_result.get("accident_detected", "Yes"),
        "initial_scene_description": detection_result.get("scene_description", ""),
        "accident_type": analysis_result.get("accident_type", "Unknown"),
        "number_of_victims": analysis_result.get("number_of_victims", 0),
        "vehicles_involved": analysis_result.get("vehicles_involved", 0),
        "accident_severity": analysis_result.get("accident_severity", "Unknown"),
        "injured_person_detected": analysis_result.get("injured_person_detected", "Unknown"),
        "emergency_services_present": analysis_result.get("emergency_services_present", "Unknown"),
        "road_blocked": analysis_result.get("road_blocked", "Unknown"),
        "scene_description": analysis_result.get("scene_description", "No description available"),
        "model_used": MODEL_NAME,
        "organization": ORGANIZATION
    }

    logger.info(f"Report structure created: {report_data['report_id']}")
    return report_data


def format_accident_report(report_data: dict) -> str:
    """
    Format report data into a human-readable text report.

    Args:
        report_data: Complete report data dictionary

    Returns:
        Formatted report string
    """
    report = f"""
{'=' * 80}
{REPORT_TITLE.upper().center(80)}
{'=' * 80}

Report ID:   {report_data['report_id']}
Date:        {report_data['date']}
Time:        {report_data['time']}
Location:    Latitude: {report_data['gps']['latitude']}, Longitude: {report_data['gps']['longitude']}

{'=' * 80}

ACCIDENT ANALYSIS
{'-' * 40}
  Accident Detected:         {report_data['accident_detected']}
  Accident Type:             {report_data['accident_type']}
  Number of Victims:         {report_data['number_of_victims']}
  Vehicles Involved:         {report_data['vehicles_involved']}
  Severity:                  {report_data['accident_severity']}
  Injured Persons Visible:   {report_data['injured_person_detected']}
  Emergency Services:        {report_data['emergency_services_present']}
  Road Blocked:              {report_data['road_blocked']}

{'=' * 80}

INCIDENT DESCRIPTION
{'-' * 40}
{report_data['scene_description']}

{'=' * 80}

METADATA
{'-' * 40}
  Generated by:  {report_data['organization']}
  Model Used:    {report_data['model_used']}
  GPS Coords:    ({report_data['gps']['latitude']}, {report_data['gps']['longitude']})

{'=' * 80}
"""
    return report
