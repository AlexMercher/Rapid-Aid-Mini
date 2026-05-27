"""
Groq Reasoner — Temporal-causal synthesizer.

NOT a blind semantic veto authority.

Groq's role:
  - Synthesize RapidAid causal signals + bakllava narration + timeline
  - Produce structured JSON assessment
  - Reason about physical-semantic agreement
  - NEVER directly sees raw video or full collision sequence

Therefore Groq must NOT dominate strong physical evidence.
Its output is ADVISORY to the consensus engine.
"""
import json
import logging
import src.config as _cfg

logger = logging.getLogger(__name__)

SYNTHESIS_PROMPT = """\
You are a temporal-causal event SYNTHESIZER in a safety-critical accident
validation system. You must analyze ALL evidence and produce a structured
assessment.

IMPORTANT: You do NOT have access to the raw video. Your assessment is
based on physical sensor data from RapidAid (causal physics engine) and
semantic narration from bakllava (visual description model). The physical
engine has direct access to motion vectors, object tracking, and collision
geometry — treat its signals as PRIMARY evidence.

## EVIDENCE SOURCES

### 1. RapidAid Physical Engine (PRIMARY - causal reasoning)
{rapidaid_summary}

### 2. Impact Frame Causal Signals
{impact_signals}

### 3. Event Timeline & Confidence Progression
{timeline_summary}

### 4. Event-State Frame Labels
{event_state_labels}

### 5. Track Lifecycle
{track_lifecycle}

### 6. bakllava Visual Narration (SECONDARY - semantic description)
{bakllava_narration}

## YOUR TASK

Based on ALL evidence, output a JSON object:

{{
  "accident_detected": "Yes" or "No",
  "accident_severity": "Minor" or "Major" or "Critical",
  "accident_type": "<collision type>",
  "vehicles_involved": <integer>,
  "number_of_victims": <integer>,
  "injured_person_detected": "Yes" or "No",
  "confidence_assessment": "high" or "medium" or "low",
  "physical_semantic_agreement": "agree" or "contradict" or "partial",
  "reasoning": "<2-3 sentences explaining your assessment>",
  "scene_description": "<detailed 4-5 sentence description>",
  "temporal_progression": "<describe how the event evolved over time>"
}}

CRITICAL RULES:
1. Physical sensor data (RapidAid) is PRIMARY evidence. If RapidAid shows
   strong velocity spikes, track deaths, and geometry overlap, an accident
   very likely occurred even if bakllava's description is ambiguous.
2. bakllava only sees static frames — it cannot perceive motion, velocity,
   or temporal dynamics. Weight its evidence accordingly.
3. If RapidAid shows overwhelming evidence (conf>0.70, 4+ strong signals)
   but bakllava is uncertain, set physical_semantic_agreement to "partial"
   and still consider "Yes" with appropriate severity.
4. Only set accident_detected to "No" if BOTH physical AND semantic
   evidence clearly indicate no accident.
5. Be precise about accident_type based on the causal signals available.

Output ONLY valid JSON. No markdown, no explanation."""


def synthesize(metadata_package):
    """
    Run Groq synthesis on the full metadata package.

    Args:
        metadata_package: dict from MetadataPackager.package()

    Returns:
        dict with structured accident assessment
    """
    from groq import Groq

    api_key = _cfg.GROQ_API_KEY
    if not api_key:
        logger.error("GROQ_API_KEY not set")
        return _fallback_result(metadata_package)

    # Build rich prompt sections
    prompt = _build_prompt(metadata_package)

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=_cfg.GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a temporal-causal event synthesizer in a "
                        "safety-critical system. Physical sensor evidence is "
                        "primary. Output ONLY valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        logger.info(f"[Groq] Response received ({len(raw)} chars)")
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"[Groq] JSON parse error: {e}")
        return _fallback_result(metadata_package)
    except Exception as e:
        logger.error(f"[Groq] Error: {e}")
        return _fallback_result(metadata_package)


def _build_prompt(metadata_package):
    """Build the complete synthesis prompt with all evidence sections."""

    # ── RapidAid summary ──
    ra = metadata_package.get("rapidaid", {})
    rapidaid_summary = (
        f"- Accident detected: {ra.get('accident_detected', False)}\n"
        f"- Confidence: {ra.get('confidence', 0):.3f}\n"
        f"- Event time: {ra.get('event_time', 'N/A')}s\n"
        f"- Dominant signal: {ra.get('dominant_signal', 'N/A')}\n"
        f"- Confirmation reason: {ra.get('confirmation_reason', 'N/A')}\n"
        f"- Total tracks: {ra.get('total_tracks', 0)}\n"
        f"- Dead tracks: {ra.get('dead_tracks', 0)}"
    )

    signal_values = metadata_package.get("signal_summary", {}).get("signal_values", {})
    if signal_values:
        rapidaid_summary += "\n- Impact signals: " + ", ".join(
            f"{k}={v:.2f}" for k, v in signal_values.items()
        )

    # ── Impact frame signals ──
    impact_sigs = metadata_package.get("impact_frame_signals", {})
    if impact_sigs:
        impact_signals = "\n".join(
            f"- {k}: {v}" for k, v in impact_sigs.items()
        )
    else:
        impact_signals = "(not available)"

    # ── Timeline ──
    tl = metadata_package.get("timeline", {})
    phases = tl.get("phases", [])
    timeline_summary = f"- Total phases: {len(phases)}\n"
    for p in phases:
        timeline_summary += (
            f"  [{p.get('state', '?')}] at {p.get('start_time', 0):.1f}s "
            f"(conf={p.get('start_confidence', 0):.3f}, "
            f"causal={p.get('causal_present', False)})\n"
        )

    # Confidence progression
    conf_prog = metadata_package.get("confidence_progression", [])
    if conf_prog:
        timeline_summary += "\n- Confidence curve: " + " -> ".join(
            f"{c['t']}s:{c['conf']:.2f}[{c['state']}]" for c in conf_prog
        )

    # ── Event-state labels ──
    labels = metadata_package.get("event_state_labels", [])
    if labels:
        event_state_labels = "\n".join(
            f"- Frame {i+1}: {l['role']} at {l.get('timestamp_sec', 0):.1f}s "
            f"(conf={l.get('confidence', 0):.3f}, state={l.get('state', '?')})"
            for i, l in enumerate(labels)
        )
    else:
        event_state_labels = "(not available)"

    # ── Track lifecycle ──
    lifecycle = metadata_package.get("track_lifecycle", {})
    track_lifecycle = (
        f"- Peak tracks: {lifecycle.get('peak_tracks', 0)}\n"
        f"- Final dead tracks: {lifecycle.get('final_dead', 0)}\n"
    )
    evolution = lifecycle.get("evolution", [])
    if evolution:
        track_lifecycle += "- Evolution: " + " -> ".join(
            f"{e['t']}s:[A={e['active']},D={e['dead']},L={e['lost']}]"
            for e in evolution
        )

    # ── bakllava narration ──
    narration = metadata_package.get("bakllava_narration", "")
    if not narration:
        narration = "(bakllava narration not available)"

    return SYNTHESIS_PROMPT.format(
        rapidaid_summary=rapidaid_summary,
        impact_signals=impact_signals,
        timeline_summary=timeline_summary,
        event_state_labels=event_state_labels,
        track_lifecycle=track_lifecycle,
        bakllava_narration=narration,
    )


def _fallback_result(metadata_package):
    """Generate fallback result when Groq is unavailable."""
    ra = metadata_package.get("rapidaid", {})
    return {
        "accident_detected": "Yes" if ra.get("accident_detected") else "No",
        "accident_severity": "Major" if ra.get("confidence", 0) > 0.6 else "Minor",
        "accident_type": "Unknown",
        "vehicles_involved": ra.get("total_tracks", 0),
        "number_of_victims": 0,
        "injured_person_detected": "Unknown",
        "confidence_assessment": "low",
        "physical_semantic_agreement": "partial",
        "reasoning": "Groq unavailable - using RapidAid-only assessment",
        "scene_description": "Automated assessment based on physical signals only.",
        "temporal_progression": "Temporal analysis unavailable.",
    }
