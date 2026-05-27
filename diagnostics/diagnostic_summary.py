import json
import os
import re
from datetime import datetime

ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "events")
VETO_KEYWORDS = [
    "no collision",
    "no accident",
    "normal traffic",
    "no damage",
    "no impact",
    "parked vehicles",
]
EVENT_PATTERN = re.compile(r"^EVT_(\d{8})_(\d{6})_(.+)$")


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _parse_groq_md(text):
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}


def _latest_fullmode_events(root_dir):
    latest = {}
    for name in os.listdir(root_dir):
        match = EVENT_PATTERN.match(name)
        if not match:
            continue
        date_part, time_part, video = match.groups()
        path = os.path.join(root_dir, name)
        if not os.path.isdir(path):
            continue

        meta = os.path.join(path, "metadata.json")
        cons = os.path.join(path, "consensus.json")
        timeline = os.path.join(path, "timeline.json")
        bak = os.path.join(path, "bakllava_output.md")
        groq = os.path.join(path, "groq_reasoning.md")
        debug = os.path.join(path, "debug_reasoning.md")

        # Require full multimodal artifacts
        if not all(os.path.exists(p) for p in [meta, cons, timeline, bak, groq, debug]):
            continue

        ts = datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")
        prev = latest.get(video)
        if not prev or ts > prev["ts"]:
            latest[video] = {"ts": ts, "name": name, "path": path}
    return latest


def _check_event_integrity(path, meta, timeline):
    issues = []

    if timeline.get("video_file") == "unknown":
        issues.append("timeline_video_unknown")

    phases = timeline.get("phases", [])
    times = [p.get("start_time") for p in phases if isinstance(p.get("start_time"), (int, float))]
    if times and times != sorted(times):
        issues.append("phase_times_not_monotonic")

    if any(not isinstance(p.get("causal_present"), bool) for p in phases if "causal_present" in p):
        issues.append("phase_causal_present_nonbool")

    labels = meta.get("event_state_labels", [])
    role_ts = {l.get("role"): l.get("timestamp_sec") for l in labels}
    role_states = [l.get("state") for l in labels if l.get("state") is not None]

    impact_ts = role_ts.get("impact_moment") or meta.get("impact_frame_signals", {}).get("timestamp_sec")
    pre_ts = role_ts.get("pre_anomaly_trajectory")
    conv_ts = role_ts.get("trajectory_convergence")
    peak_ts = role_ts.get("peak_disruption")
    after_ts = role_ts.get("stabilized_aftermath")

    if impact_ts is not None and pre_ts is not None and pre_ts >= impact_ts:
        issues.append("pre_anomaly_after_impact")
    if impact_ts is not None and conv_ts is not None and conv_ts >= impact_ts:
        issues.append("convergence_after_impact")
    if impact_ts is not None and peak_ts is not None and peak_ts <= impact_ts:
        issues.append("peak_not_after_impact")
    if peak_ts is not None and after_ts is not None and after_ts <= peak_ts:
        issues.append("aftermath_not_after_peak")
    if role_states and len(set(role_states)) == 1:
        issues.append("all_event_states_same")

    frames_dir = os.path.join(path, "event_state_frames")
    missing = []
    for fname in [
        "pre_anomaly.jpg",
        "convergence.jpg",
        "impact.jpg",
        "disruption.jpg",
        "aftermath.jpg",
    ]:
        if not os.path.exists(os.path.join(frames_dir, fname)):
            missing.append(fname)
    if missing:
        issues.append("missing_event_state_frames:" + ",".join(missing))

    return issues


def build_summary():
    latest = _latest_fullmode_events(ROOT)
    summary = []

    for video in sorted(latest):
        info = latest[video]
        path = info["path"]

        meta = _read_json(os.path.join(path, "metadata.json"))
        cons = _read_json(os.path.join(path, "consensus.json"))
        timeline = _read_json(os.path.join(path, "timeline.json"))
        bak_text = _read_text(os.path.join(path, "bakllava_output.md"))
        groq_text = _read_text(os.path.join(path, "groq_reasoning.md"))
        debug_text = _read_text(os.path.join(path, "debug_reasoning.md"))
        groq_result = _parse_groq_md(groq_text)

        veto_hits = sum(1 for kw in VETO_KEYWORDS if kw in bak_text.lower())
        issues = _check_event_integrity(path, meta, timeline)

        summary.append({
            "video": video,
            "event_id": info["name"],
            "path": path,
            "tier": cons.get("tier"),
            "dispatchable": cons.get("is_dispatchable"),
            "ra_conf": cons.get("rapidaid_confidence"),
            "groq_severity": cons.get("groq_severity"),
            "groq_detected": groq_result.get("accident_detected"),
            "groq_agreement": groq_result.get("physical_semantic_agreement"),
            "physics_overwhelming": cons.get("physics_overwhelming"),
            "semantic_veto": cons.get("semantic_veto"),
            "groq_veto": cons.get("groq_veto"),
            "avg_fps": timeline.get("avg_fps"),
            "bakllava_chars": len(bak_text),
            "veto_keyword_hits": veto_hits,
            "debug_downgrade": "Downgraded" in debug_text,
            "debug_physics_safeguard": "PHYSICS SAFEGUARD" in debug_text,
            "issues": issues,
        })

    return summary


def main():
    summary = build_summary()
    out_path = os.path.join(ROOT, "diagnostic_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Summary written:", out_path)
    for row in summary:
        issues = ";".join(row["issues"])
        print(
            f"{row['video']}: tier={row['tier']} ra={row['ra_conf']} "
            f"groq={row['groq_severity']} fps={row['avg_fps']} "
            f"veto_kw={row['veto_keyword_hits']} issues={issues}"
        )


if __name__ == "__main__":
    main()