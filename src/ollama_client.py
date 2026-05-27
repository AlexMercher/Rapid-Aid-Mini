"""
Ollama Client module for Accident Report Generation System.

Two-stage architecture:
  Stage 1 - bakllava (local, vision): Produces a natural-language description.
  Stage 2 - Groq cloud API (llama-3.1-8b-instant): Converts description to validated JSON.

Only bakllava runs on the local GPU (approx 4.7 GB VRAM).
The text extraction runs on Groq's cloud infrastructure, eliminating
GPU model-swap delays and VRAM pressure.

Determinism: temperature=0 on every request.
"""

import json
import time
import logging
import requests
from PIL import Image

from src.config import (
    OLLAMA_API_URL, MODEL_NAME,
    GROQ_MODEL,
    API_TIMEOUT, MAX_RETRIES,
)
import src.config as _cfg  # for GROQ_API_KEY (read dynamically at call time)
from src.image_handler import image_to_base64, preprocess_image

logger = logging.getLogger(__name__)

# ─── Determinism parameters ─────────────────────────────────────────────────────
_DETERMINISTIC_OPTIONS = {
    "temperature": 0,
    "seed": 42,
}

# ─── Stage-1 Vision Prompts (natural language — NO JSON) ────────────────────────

VISION_DETECTION_PROMPT = """\
Analyze this image step-by-step. Focus on PEOPLE and VEHICLES.

Step 1 - VEHICLES: Count every vehicle visible. Note each vehicle's type (car, truck, motorcycle, bus) and color. State whether each vehicle has visible damage: crumpled body panels, broken windshield/windows, deployed airbags, bent frame, missing parts, scratches, or dents.

Step 2 - VEHICLE POSITIONS: Are any vehicles in abnormal positions? Overturned, spun sideways, off-road, in a ditch, on a median, or pressed against another vehicle or object?

Step 3 - PEOPLE: Count every person visible. For each person: Are they standing, sitting, lying on the ground, or trapped? Do any appear injured (bleeding, unconscious, in pain)?

Step 4 - EMERGENCY INDICATORS: Is there debris on the road? Skid marks? Broken glass? Emergency vehicles (ambulance, police, fire truck)? Traffic cones or flares? Bystanders gathered around?

Step 5 - CONCLUSION: Based on steps 1-4, state clearly whether this scene shows a vehicle accident or not, and why.

Be precise and factual. Describe only what you see. Do NOT output JSON."""

VISION_ANALYSIS_PROMPT = """\
This image contains a road accident. Provide a thorough forensic description.

Step 1 - COLLISION TYPE: Identify the collision pattern. Is it a rear-end collision, head-on, side-impact (T-bone), rollover, vehicle hitting a pedestrian, vehicle hitting a fixed object (pole, wall, guardrail), or a multi-vehicle pile-up? Describe the impact point and angle.

Step 2 - VEHICLES: For EACH vehicle involved, describe:
  - Vehicle type and color
  - Specific damage: which panels are crumpled, is the windshield shattered, are doors caved in, is the engine compartment crushed, are wheels bent or detached?
  - Approximate damage severity for that vehicle (light scratch vs heavy crush)

Step 3 - PEOPLE AND INJURIES: Count every person visible. For each:
  - Location relative to the vehicles (inside, next to, on the ground nearby)
  - Apparent condition (standing normally, limping, sitting injured, lying motionless, being attended to by others)
  - Any visible injuries or signs of distress

Step 4 - EMERGENCY RESPONSE: Are any of these present: ambulance, police car, fire truck, tow truck, paramedics, officers, firefighters? Are they actively working at the scene?

Step 5 - ROAD AND ENVIRONMENT: Is the road blocked? What are the road conditions (wet, dry, icy)? Time of day (daylight, dusk, night)? Weather (clear, rain, fog)? Urban or rural setting?

Step 6 - OVERALL SEVERITY JUDGMENT: Based on the extent of vehicle damage and visible injuries, is this a Minor incident (cosmetic damage, everyone walking), Major (structural damage, possible injuries), or Critical (severe destruction, people down or trapped)?

Be detailed and precise. Describe only what is visible. Do NOT output JSON."""

# ─── Stage-2 Text-LLM Extraction Prompts ────────────────────────────────────────

TEXT_DETECTION_EXTRACTION_PROMPT = """\
You are a structured-data extraction assistant.
Based on the scene description below, output ONLY a JSON object with these fields:

{{
  "accident_detected": "Yes" or "No",
  "confidence": "high", "medium", or "low",
  "reasoning": "<one sentence>",
  "scene_description": "<summary of the scene>"
}}

Rules:
- "accident_detected" must be "Yes" ONLY if the description mentions vehicle damage, \
collision, overturned vehicles, or emergency response to a crash.
- Normal traffic, parked cars, or construction are NOT accidents -> "No".
- Output ONLY valid JSON. No markdown, no explanation.

Scene description:
{description}"""

TEXT_ANALYSIS_EXTRACTION_PROMPT = """\
You are a structured-data extraction assistant.
Based on the accident scene description below, output ONLY a JSON object with these fields:

{{
  "accident_type": "<Rear-end collision | Head-on collision | Side-impact | Rollover | Vehicle-pedestrian | Vehicle-object collision | Multi-vehicle pile-up | Other>",
  "number_of_victims": <integer>,
  "vehicles_involved": <integer>,
  "accident_severity": "<Minor | Major | Critical>",
  "injured_person_detected": "Yes" or "No",
  "emergency_services_present": "Yes" or "No",
  "road_blocked": "Yes" or "No",
  "scene_description": "<VERBOSE detailed description — at least 4-5 sentences covering: (1) collision dynamics and impact point, (2) specific vehicle damage for each vehicle, (3) victim locations and apparent conditions, (4) environmental context including road surface, weather, lighting, and surroundings, (5) road blockage and traffic impact. Be thorough and forensic in detail.>"
}}

Severity guide:
- Minor: cosmetic damage, no serious injuries
- Major: significant structural damage, potential injuries
- Critical: severe damage, serious injuries or fatalities likely

Output ONLY valid JSON. No markdown, no explanation.

Scene description:
{description}"""

# ─── JSON Schemas for validation ────────────────────────────────────────────────

ACCIDENT_DETECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "accident_detected": {"type": "string", "enum": ["Yes", "No"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reasoning": {"type": "string"},
        "scene_description": {"type": "string"},
    },
    "required": ["accident_detected", "scene_description"],
}

ACCIDENT_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "accident_type": {"type": "string"},
        "number_of_victims": {"type": "integer"},
        "vehicles_involved": {"type": "integer"},
        "accident_severity": {"type": "string", "enum": ["Minor", "Major", "Critical"]},
        "injured_person_detected": {"type": "string", "enum": ["Yes", "No"]},
        "emergency_services_present": {"type": "string", "enum": ["Yes", "No"]},
        "road_blocked": {"type": "string", "enum": ["Yes", "No"]},
        "scene_description": {"type": "string"},
    },
    "required": [
        "accident_type", "number_of_victims", "vehicles_involved",
        "accident_severity", "scene_description",
    ],
}


# ═════════════════════════════════════════════════════════════════════════════════
#  Connection & model helpers
# ═════════════════════════════════════════════════════════════════════════════════

def check_ollama_connection() -> bool:
    """Check if Ollama server is reachable."""
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            logger.info("Ollama connection successful")
            return True
        logger.error(f"Ollama returned status code: {response.status_code}")
        return False
    except requests.ConnectionError:
        logger.error(f"Cannot connect to Ollama at {OLLAMA_API_URL}")
        return False
    except requests.Timeout:
        logger.error("Ollama connection timed out")
        return False


def check_model_available(model_name: str = MODEL_NAME) -> bool:
    """Check if *model_name* is available in Ollama."""
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            available = any(model_name in m.get("name", "") for m in models)
            if available:
                logger.info(f"Model '{model_name}' is available")
            else:
                logger.warning(
                    f"Model '{model_name}' not found. "
                    f"Available: {[m['name'] for m in models]}"
                )
            return available
        return False
    except Exception as e:
        logger.error(f"Error checking model availability: {e}")
        return False


def load_model(model_name: str = MODEL_NAME) -> bool:
    """Ensure *model_name* is available; pull it if missing."""
    if check_model_available(model_name):
        return True
    try:
        logger.info(f"Attempting to pull model '{model_name}'...")
        response = requests.post(
            f"{OLLAMA_API_URL}/api/pull",
            json={"name": model_name},
            timeout=API_TIMEOUT,
        )
        if response.status_code == 200:
            logger.info(f"Model '{model_name}' pull initiated successfully")
        else:
            logger.warning(
                f"Model pull returned status {response.status_code}: {response.text}"
            )
    except Exception as exc:
        logger.error(f"Failed to pull model '{model_name}': {exc}")
        return False
    return check_model_available(model_name)


# ═════════════════════════════════════════════════════════════════════════════════
#  Low-level request helpers
# ═════════════════════════════════════════════════════════════════════════════════

def _send_vision_request(image: Image.Image, prompt: str) -> str:
    """
    Stage 1 - send an image + prompt to the local bakllava vision model.

    Returns natural-language text (no JSON).  Determinism enforced.
    """
    processed = preprocess_image(image)
    img_b64 = image_to_base64(processed)

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {**_DETERMINISTIC_OPTIONS},
    }

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                f"[Stage 1 - vision] Sending request to {MODEL_NAME} "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )
            response = requests.post(
                f"{OLLAMA_API_URL}/api/generate",
                json=payload,
                timeout=API_TIMEOUT,
            )
            if response.status_code == 200:
                result = response.json().get("response", "")
                logger.info(f"[Stage 1] Response received ({len(result)} chars)")
                return result
            last_error = f"API returned status {response.status_code}: {response.text}"
            logger.warning(last_error)
        except requests.ConnectionError as e:
            last_error = f"Connection error: {e}"
            logger.warning(f"Attempt {attempt} failed: {last_error}")
        except requests.Timeout:
            last_error = "Request timed out"
            logger.warning(f"Attempt {attempt} timed out")

        if attempt < MAX_RETRIES:
            wait = 2 ** attempt
            logger.info(f"Retrying in {wait}s...")
            time.sleep(wait)

    raise ConnectionError(
        f"Vision request failed after {MAX_RETRIES} attempts. Last error: {last_error}"
    )


def _send_text_request(prompt: str, schema: dict | None = None) -> str:
    """
    Stage 2 - send a text-only prompt to Groq cloud API (llama-3.1-8b-instant).

    Much faster than local inference and frees the GPU entirely for bakllava.
    If *schema* is provided, it is included in the system message to guide output.
    """
    from groq import Groq

    api_key = _cfg.GROQ_API_KEY  # read dynamically so sidebar updates take effect
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "and add it to your .env file."
        )

    client = Groq(api_key=api_key)

    messages = [
        {"role": "system", "content": "You are a structured-data extraction assistant. Output ONLY valid JSON. No markdown fences, no explanation."},
        {"role": "user", "content": prompt},
    ]

    # If we have a schema, add it to guide the model
    if schema:
        schema_hint = json.dumps(schema, indent=2)
        messages[0]["content"] += f"\n\nThe output must conform to this JSON schema:\n{schema_hint}"

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                f"[Stage 2 - Groq] Sending request to {GROQ_MODEL} "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            result = response.choices[0].message.content or ""
            logger.info(f"[Stage 2] Groq response received ({len(result)} chars)")
            return result
        except Exception as e:
            last_error = str(e)
            logger.warning(f"[Stage 2 - Groq] Attempt {attempt} failed: {last_error}")

        if attempt < MAX_RETRIES:
            wait = 2 ** attempt
            logger.info(f"Retrying in {wait}s...")
            time.sleep(wait)

    raise ConnectionError(
        f"Groq request failed after {MAX_RETRIES} attempts. Last error: {last_error}"
    )


# ═════════════════════════════════════════════════════════════════════════════════
#  JSON validation helper
# ═════════════════════════════════════════════════════════════════════════════════

def _validate_json(data: dict, schema: dict) -> list[str]:
    """
    Validate *data* against *schema* (lightweight — checks required keys,
    enum values, and integer types).  Returns a list of error strings
    (empty == valid).
    """
    errors: list[str] = []
    props = schema.get("properties", {})
    required = schema.get("required", [])

    for key in required:
        if key not in data:
            errors.append(f"Missing required field: '{key}'")

    for key, rules in props.items():
        if key not in data:
            continue
        val = data[key]
        expected_type = rules.get("type")
        if expected_type == "integer" and not isinstance(val, int):
            try:
                data[key] = int(val)
            except (ValueError, TypeError):
                errors.append(f"Field '{key}' must be an integer, got {type(val).__name__}")
        if expected_type == "string" and not isinstance(val, str):
            data[key] = str(val)
        enum = rules.get("enum")
        if enum and data[key] not in enum:
            errors.append(f"Field '{key}' value '{data[key]}' not in {enum}")

    return errors


def _extract_json_with_validation(
    description: str,
    extraction_prompt_template: str,
    schema: dict,
    max_attempts: int = 2,
) -> dict:
    """
    Stage 2 pipeline: send *description* to the text LLM, parse and validate
    JSON.  Re-tries up to *max_attempts* times if validation fails.
    """
    last_raw = ""
    for attempt in range(1, max_attempts + 1):
        prompt = extraction_prompt_template.format(description=description)
        raw = _send_text_request(prompt, schema=schema)
        last_raw = raw
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(
                f"[Validation] Attempt {attempt}: JSON parse failed - {raw[:200]}"
            )
            continue

        validation_errors = _validate_json(parsed, schema)
        if not validation_errors:
            logger.info(f"[Validation] JSON valid on attempt {attempt}")
            return parsed

        logger.warning(
            f"[Validation] Attempt {attempt}: schema errors - {validation_errors}"
        )

    # All attempts exhausted - try to return whatever we parsed last
    logger.error(
        f"[Validation] All {max_attempts} attempts failed. "
        f"Last raw response: {last_raw[:300]}"
    )
    try:
        return json.loads(last_raw)
    except json.JSONDecodeError:
        return {}


# ═════════════════════════════════════════════════════════════════════════════════
#  Public API
# ═════════════════════════════════════════════════════════════════════════════════

def detect_accident(image: Image.Image) -> dict:
    """
    Two-stage accident detection.

    Stage 1 (bakllava, local):  Describe the scene in natural language.
    Stage 2 (Groq cloud):      Extract accident_detected JSON.
    """
    logger.info("=== Accident Detection - Stage 1 (vision) ===")
    description = _send_vision_request(image, VISION_DETECTION_PROMPT)
    logger.info(f"Vision description: {description[:200]}")

    logger.info("=== Accident Detection - Stage 2 (Groq -> JSON) ===")
    parsed = _extract_json_with_validation(
        description,
        TEXT_DETECTION_EXTRACTION_PROMPT,
        ACCIDENT_DETECTION_SCHEMA,
    )

    # Fallback defaults
    if not isinstance(parsed, dict) or not parsed:
        parsed = {
            "accident_detected": "No",
            "scene_description": description or "Unable to parse model response",
        }

    # Normalize
    detected = parsed.get("accident_detected", "No")
    if isinstance(detected, str):
        detected = "Yes" if detected.strip().lower().startswith("yes") else "No"
    parsed["accident_detected"] = detected

    logger.info(f"Accident detection result: {detected}")
    return parsed


def analyze_accident(image: Image.Image) -> dict:
    """
    Two-stage accident analysis.

    Stage 1 (bakllava, local):  Describe the accident scene in detail.
    Stage 2 (Groq cloud):      Extract structured analysis JSON.
    """
    logger.info("=== Accident Analysis - Stage 1 (vision) ===")
    description = _send_vision_request(image, VISION_ANALYSIS_PROMPT)
    logger.info(f"Vision description: {description[:200]}")

    logger.info("=== Accident Analysis - Stage 2 (Groq -> JSON) ===")
    parsed = _extract_json_with_validation(
        description,
        TEXT_ANALYSIS_EXTRACTION_PROMPT,
        ACCIDENT_ANALYSIS_SCHEMA,
    )

    if not isinstance(parsed, dict):
        parsed = {}

    # Fallback defaults
    defaults = {
        "accident_type": "Unknown",
        "number_of_victims": 0,
        "vehicles_involved": 0,
        "accident_severity": "Unknown",
        "injured_person_detected": "Unknown",
        "emergency_services_present": "Unknown",
        "road_blocked": "Unknown",
        "scene_description": description or "No description available",
    }
    for key, default in defaults.items():
        if key not in parsed or not parsed[key]:
            parsed[key] = default

    logger.info("Accident analysis complete")
    return parsed


def generate_description(image: Image.Image, prompt: str) -> str:
    """
    Generate a free-form description for the provided image using the vision model.
    """
    logger.info("Generating free-form description via vision model")
    return _send_vision_request(image, prompt)
