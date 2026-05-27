"""
bakllava Client — Per-frame temporal event narration.

Sends 5 independent high-resolution event-state frames to bakllava
with explicit semantic role labeling and temporal ordering.

bakllava is a SEMANTIC VISUAL NARRATOR only. It does NOT:
  - produce structured JSON
  - make final verdicts
  - control dispatch decisions

It describes what it SEES across the temporal event progression.
"""
import logging
import re
import requests
import base64
import cv2
from src.config import OLLAMA_API_URL, MAX_RETRIES
from config import settings

logger = logging.getLogger(__name__)

EVENT_CONTEXT = (
    "You are analyzing a 5-frame storyboard from the same traffic event. "
    "The frames are sequential: A -> B -> C -> D -> E. "
    "Describe only what is visible. Do not output timestamps or numeric-only lines."
)

# Per-frame role descriptions for bakllava
ROLE_PROMPTS = {
    "pre_anomaly_trajectory": (
        "PRE-EVENT phase. Describe what vehicles are visible, "
        "their positions and directions. No collision has occurred yet."
    ),
    "trajectory_convergence": (
        "CONVERGENCE phase. Describe vehicles approaching each other, "
        "relative positions, any unusual movement."
    ),
    "impact_moment": (
        "IMPACT phase. Describe the collision: which vehicles are "
        "involved, visible contact, debris, deformation, or sudden "
        "displacement. This is the critical moment."
    ),
    "peak_disruption": (
        "DISRUPTION phase. Describe immediate aftermath: vehicle "
        "positions post-impact, visible damage, scattered objects."
    ),
    "stabilized_aftermath": (
        "AFTERMATH phase. Describe the stabilized scene: final vehicle "
        "positions, road condition, overall scene state."
    ),
}

SYNTHESIS_PROMPT = """\
You have just examined 5 sequential frames from the SAME traffic camera event.
The frames are chronologically ordered: A -> B -> C -> D -> E.

Now provide a UNIFIED TEMPORAL NARRATION that synthesizes what happened
across all 5 frames. Describe:

1. TRAJECTORY EVOLUTION: How did vehicle movements change from Frame A to E?
2. INTERACTION CHANGES: Did vehicles come into contact or dangerously close?
3. COLLISION EVIDENCE: Was there a visible collision? At which frame?
4. DISRUPTION PROGRESSION: How did the scene change after the critical moment?
5. AFTERMATH STATE: What is the final scene state?
6. SEVERITY ASSESSMENT: Based on visible evidence, how severe is this event?

Be precise and factual. Describe only what was visible.
Do NOT output timestamps, numeric-only lines, or JSON."""

EMPTY_MARKERS = {
    "unknown",
    "n/a",
    "none",
    "no description",
    "not sure",
    "0",
}


def preprocess_for_bakllava(frame):
    """Rotate portrait frames and normalize size for bakllava."""
    if frame is None:
        return frame

    h, w = frame.shape[:2]
    if h > w:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        h, w = frame.shape[:2]

    short_side = min(h, w)
    long_side = max(h, w)
    scale = 1.0

    if short_side < settings.BAKLLAVA_SHORT_SIDE_PX:
        scale = settings.BAKLLAVA_SHORT_SIDE_PX / short_side

    if long_side * scale > settings.BAKLLAVA_MAX_LONG_SIDE_PX:
        scale = settings.BAKLLAVA_MAX_LONG_SIDE_PX / long_side

    if scale != 1.0:
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        frame = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)

    return frame


def mask_timestamp_region(frame):
    """
    Black out CCTV overlay regions before sending to bakllava.

    bakllava reads burned-in on-screen timestamps (e.g. "2019-02-24 06:31:02"
    in the top-right corner) and produces time-code captions instead of scene
    descriptions. Masking these regions prevents that behaviour.

    Two regions are masked:
      - Top-right corner: where CCTV date/time stamps appear
      - Bottom-left corner: where camera-ID labels ("CAM 5") appear

    Ratios are read from settings so they can be adjusted per camera rig.
    """
    if frame is None or not settings.BAKLLAVA_MASK_TIMESTAMP:
        return frame

    h, w = frame.shape[:2]
    masked = frame.copy()

    # Top-right: CCTV timestamp region
    x_start = int(w * settings.BAKLLAVA_TIMESTAMP_X_START_RATIO)
    y_end   = int(h * settings.BAKLLAVA_TIMESTAMP_Y_END_RATIO)
    masked[0:y_end, x_start:w] = 0

    # Bottom-left: camera-label region
    y_start = int(h * settings.BAKLLAVA_CAM_LABEL_Y_START_RATIO)
    x_end   = int(w * settings.BAKLLAVA_CAM_LABEL_X_END_RATIO)
    masked[y_start:h, 0:x_end] = 0

    return masked


def narrate_event_frames(individual_frames, roles, custom_prompt=None):
    """
    Send 5 independent event-state frames to bakllava for temporal narration.

    Args:
        individual_frames: list of 5 BGR numpy arrays (high-res event frames)
        roles: list of role strings matching the frames
        custom_prompt: optional override for synthesis prompt

    Returns:
        str: bakllava's complete temporal narration
    """
    if not individual_frames:
        return "[No frames provided]"

    # Phase 1: Narrate each frame independently
    frame_narrations = []
    for i, (frame, role) in enumerate(zip(individual_frames, roles)):
        role_prompt = ROLE_PROMPTS.get(role, f"Frame {i+1}: Describe this frame.")
        full_prompt = f"{EVENT_CONTEXT} {role_prompt}"
        narration = _send_single_frame(frame, full_prompt)
        if _is_low_quality_narration(narration):
            narration = _retry_with_directive_prompt(frame, full_prompt)
        if _is_fallback_required(narration):
            narration = "[No visible details]"
        frame_narrations.append(
            f"### {role_prompt.split(':')[0].strip()}\n{narration}"
        )
        logger.info(f"[bakllava] Frame {i+1}/{len(individual_frames)} "
                     f"({role}): {len(narration)} chars")

    # Phase 2: Synthesis pass — combine all narrations
    combined_narrations = "\n\n".join(frame_narrations)
    synthesis = _run_synthesis(combined_narrations, custom_prompt)
    if _is_low_quality_narration(synthesis):
        synthesis = _run_synthesis(combined_narrations, custom_prompt)

    # Combine everything
    full_narration = (
        "# Event Frame Narrations\n\n"
        f"{combined_narrations}\n\n"
        "---\n\n"
        "# Temporal Synthesis\n\n"
        f"{synthesis}"
    )

    return full_narration


def narrate_storyboard(storyboard_input, custom_prompt=None):
    """
    Backward-compatible entry point.

    If storyboard_input is a dict (from new StoryboardGenerator), use
    individual frames. If it's a numpy array (old composite), fall back
    to single-image narration.
    """
    import numpy as np

    if isinstance(storyboard_input, dict):
        frames = storyboard_input.get("individual_frames", [])
        roles = storyboard_input.get("roles", [])
        return narrate_event_frames(frames, roles, custom_prompt)
    elif isinstance(storyboard_input, np.ndarray):
        # Legacy: composite storyboard image
        return _send_single_frame(storyboard_input, custom_prompt or _LEGACY_PROMPT)
    else:
        return "[Invalid storyboard input]"


def _send_single_frame(frame, prompt):
    """Send a single frame to bakllava with a prompt."""
    frame = preprocess_for_bakllava(frame)
    frame = mask_timestamp_region(frame)   # mask CCTV overlays before encoding
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_b64 = base64.b64encode(buffer).decode("utf-8")

    payload = {
        "model": "bakllava",
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {"temperature": 0, "seed": 42},
    }

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                f"{OLLAMA_API_URL}/api/generate",
                json=payload,
                timeout=settings.BAKLLAVA_TIMEOUT_SEC,
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            last_error = f"API status {response.status_code}"
        except requests.ConnectionError as e:
            last_error = f"Connection error: {e}"
        except requests.Timeout:
            last_error = "Timeout"

        if attempt < MAX_RETRIES:
            import time
            time.sleep(2 ** attempt)

    logger.error(f"[bakllava] All attempts failed: {last_error}")
    return f"[bakllava unavailable: {last_error}]"


def _run_synthesis(combined_narrations, custom_prompt=None):
    """Run text-only synthesis pass to combine frame narrations."""
    prompt = custom_prompt or SYNTHESIS_PROMPT
    full_prompt = (
        f"## Individual Frame Observations\n\n{combined_narrations}\n\n"
        f"## Task\n\n{prompt}"
    )

    payload = {
        "model": "bakllava",
        "prompt": full_prompt,
        "stream": False,
        "options": {"temperature": 0, "seed": 42},
    }

    try:
        response = requests.post(
            f"{OLLAMA_API_URL}/api/generate",
            json=payload,
            timeout=settings.BAKLLAVA_TIMEOUT_SEC,
        )
        if response.status_code == 200:
            return response.json().get("response", "")
    except Exception as e:
        logger.error(f"[bakllava] Synthesis failed: {e}")

    return "(Synthesis unavailable)"


def _retry_with_directive_prompt(frame, base_prompt):
    retry_prompt = (
        f"{base_prompt} Describe ONLY the visible vehicles and their state "
        "(moving/stopped, damaged, debris) in one clear sentence. "
        "Do not output timestamps or numeric-only lines."
    )
    return _send_single_frame(frame, retry_prompt)


def _is_empty_marker(text):
    return text.strip().lower() in EMPTY_MARKERS


def _is_fallback_required(text):
    if not text:
        return True
    cleaned = text.strip()
    if _is_empty_marker(cleaned):
        return True
    return len(cleaned) < settings.BAKLLAVA_FALLBACK_MIN_CHARS


def _is_low_quality_narration(text):
    """Detect empty or numeric-only narrations for retry."""
    if not text:
        return True
    cleaned = text.strip()
    if _is_empty_marker(cleaned):
        return True
    if len(cleaned) < settings.BAKLLAVA_MIN_NARRATION_CHARS:
        return True
    if re.fullmatch(r"[0-9\s:\-\.,]+", cleaned):
        return True
    return False


def check_bakllava_available():
    """Check if bakllava model is available in Ollama."""
    try:
        resp = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=10)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            return any("bakllava" in m.get("name", "") for m in models)
    except Exception:
        pass
    return False


# Legacy prompt for backward compatibility
_LEGACY_PROMPT = """\
You are analyzing a temporal storyboard showing 5 sequential frames from a \
traffic camera. Describe the temporal progression visible. Be precise and \
factual. Do NOT output JSON."""
