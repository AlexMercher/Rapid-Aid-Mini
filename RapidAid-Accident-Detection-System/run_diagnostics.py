"""
RapidAid Full Diagnostic Validation
Phases 2-9: YOLO, Tracker, Velocity, Disappearance, Flow, Fusion, FP, Temporal
Usage: conda activate ./venv && python run_diagnostics.py
"""
import os, sys, time, json, glob, math
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from models.tracker import TrackManager, Track
from models.velocity_analyzer import VelocityAnalyzer
from models.disappearance_analyzer import DisappearanceAnalyzer
from models.optical_flow_analyzer import OpticalFlowAnalyzer
from models.confidence_fusion import ConfidenceFusion
from config.vehicle_classes import is_vehicle_class

DIAG_DIR = os.path.join(settings.OUTPUTS_DIR, "diagnostics")
os.makedirs(DIAG_DIR, exist_ok=True)


def load_models(use_nano=True):
    from ultralytics import YOLO
    seg = os.path.join(settings.WEIGHTS_DIR, "yolov8n-seg.pt" if use_nano else "yolov8s-seg.pt")
    pose = os.path.join(settings.WEIGHTS_DIR, "yolov8n-pose.pt" if use_nano else "yolov8s-pose.pt")
    vm = YOLO(seg); pm = YOLO(pose)
    fc = None
    try:
        from models.frame_classifier import FrameClassifier
        fc = FrameClassifier()
        if not fc.is_available(): fc = None
    except: pass
    cd = None
    try:
        from models.collision_detector import CollisionDetector
        cd = CollisionDetector()
        if not cd.is_available(): cd = None
    except: pass
    return vm, pm, fc, cd

def detect_vehicles(model, frame):
    res = model(frame, verbose=False)[0]
    fh, fw = frame.shape[:2]; fa = fh * fw; vehs = []
    if res.boxes is None: return vehs
    for box in res.boxes:
        cid = int(box.cls[0]); conf = float(box.conf[0])
        if not is_vehicle_class(cid) or conf < settings.VEHICLE_CONFIDENCE_THRESHOLD: continue
        bb = [int(c) for c in box.xyxy[0].tolist()]
        ar = (bb[2]-bb[0])*(bb[3]-bb[1])/fa
        if ar < settings.MIN_VEHICLE_AREA_RATIO: continue
        vehs.append({"bbox":bb,"confidence":round(conf*100,1),"type":"vehicle","coco_class_id":cid})
    return vehs

def detect_persons(model, frame):
    res = model(frame, verbose=False)[0]; prs = []
    if res.boxes is None: return prs
    for box in res.boxes:
        cid = int(box.cls[0]); conf = float(box.conf[0])
        if cid != 0 or conf < settings.PERSON_CONFIDENCE_THRESHOLD: continue
        bb = [int(c) for c in box.xyxy[0].tolist()]
        prs.append({"bbox":bb,"confidence":round(conf*100,1),"type":"person"})
    return prs

def compute_iou(b1, b2):
    x1=max(b1[0],b2[0]);y1=max(b1[1],b2[1]);x2=min(b1[2],b2[2]);y2=min(b1[3],b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]);a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    u=a1+a2-inter
    return inter/u if u>0 else 0

def diagnose_video(video_path, vm, pm, fc, cd):
    """Run full diagnostic on one video, returning detailed per-frame data."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    interval = max(1, int(fps / settings.FRAMES_PER_SECOND_TO_ANALYZE))

    vt = TrackManager(max_lost_frames=15, iou_threshold_high=0.25)
    pt = TrackManager(max_lost_frames=10, iou_threshold_high=0.3)
    va = VelocityAnalyzer()
    da = DisappearanceAnalyzer(frame_w=fw, frame_h=fh)
    ofa = OpticalFlowAnalyzer(flow_scale=0.5)
    cf = ConfidenceFusion()

    frames_data = []
    track_registry = {}  # track_id -> full lifecycle info
    disappearance_log = []
    confidence_series = []
    signal_series = {"detector":[],"tracking":[],"velocity":[],"optical_flow":[],"disappearance":[],"geometry":[]}

    fc_count = 0; analyzed = 0
    accident_confirmed = False; confirm_win = []
    THRESH = 0.40; CCOUNT = 3; best_conf = 0; best_ts = None

    while True:
        ret, frame = cap.read()
        if not ret: break
        fc_count += 1
        if fc_count % interval != 0: continue
        analyzed += 1
        t_sec = round(fc_count / fps, 2)

        # Step 1: Detection
        vehs = detect_vehicles(vm, frame)
        pers = detect_persons(pm, frame)

        # Step 2: Tracking
        active_v = vt.update(vehs, analyzed)
        active_p = pt.update(pers, analyzed)

        # Register tracks
        for trk in active_v + vt.get_all_lost_tracks():
            if trk.track_id not in track_registry:
                track_registry[trk.track_id] = {
                    "created_frame": trk.created_frame, "type": trk.obj_type,
                    "frames_alive": 0, "occlusions_survived": 0, "id_swaps": 0,
                    "max_speed": 0, "termination_reason": None, "status": "active"
                }
            tr = track_registry[trk.track_id]
            tr["frames_alive"] = trk.total_visible_frames
            tr["max_speed"] = max(tr["max_speed"], round(trk.get_speed(), 2))
            tr["status"] = trk.status
            if trk.frames_missing > 0 and trk.status == "active":
                tr["occlusions_survived"] += 1

        # Step 3: Velocity
        vel_scores = va.analyze_tracks(active_v, vt.get_all_lost_tracks(),
                                        vt.get_recently_dead_tracks(analyzed, 20), analyzed)

        # Step 4: Optical flow
        flow = ofa.process_frame(frame)

        # Step 5: Disappearance
        dead = vt.get_recently_dead_tracks(analyzed, 20)
        dis_results = da.analyze(dead, vt.get_all_lost_tracks(), active_v, analyzed)
        for tid, dr in dis_results.items():
            if dr["disappearance_type"] == "ANOMALOUS_DISAPPEARANCE":
                disappearance_log.append({"track_id":tid,"time":t_sec,"score":dr["disappearance_score"],
                                          "reason":dr["reason"],"details":dr.get("details",{})})
            if tid in track_registry:
                track_registry[tid]["termination_reason"] = dr["disappearance_type"]

        # Step 6: M1/M4
        m1 = 0.0
        if fc: _, m1 = fc.classify(frame)
        m4 = 0.0; czones = []
        if cd:
            czones = cd.detect(frame)
            if czones: m4 = czones[0]["confidence"]

        # Step 7: Fusion
        det_s = max(np.mean([t.confidence for t in active_v]) if active_v else 0, m1*0.5, m4*0.8)
        trk_s = cf.compute_tracking_score(active_v)
        vel_s = max((max(s["velocity_collapse_score"],s["trajectory_anomaly_score"]) for s in vel_scores.values()), default=0)
        flow_s = max(flow.get("motion_burst_score",0), flow.get("optical_flow_score",0))
        dis_s = max((r["disappearance_score"] for r in dis_results.values()), default=0)
        geo_s = 0.0
        if czones: geo_s = czones[0]["confidence"]
        if len(active_v) >= 2:
            for i in range(len(active_v)):
                for j in range(i+1, len(active_v)):
                    iou = compute_iou(active_v[i].bbox, active_v[j].bbox)
                    if iou > 0.1: geo_s = max(geo_s, iou*0.8)
        geo_s = min(1.0, geo_s)

        fus = cf.compute(det_s, trk_s, vel_s, flow_s, dis_s, geo_s)
        final = fus["final_confidence"]
        confidence_series.append(final)
        for k in signal_series: signal_series[k].append(round(fus["raw_scores"].get(k,0),3))

        # Step 8: Decision
        if final >= THRESH:
            confirm_win.append(final)
            if len(confirm_win) > 8: confirm_win.pop(0)
        else:
            if confirm_win: confirm_win.pop(0)
        if len(confirm_win) >= CCOUNT and not accident_confirmed:
            accident_confirmed = True; best_conf = final; best_ts = t_sec
        if accident_confirmed and final > best_conf:
            best_conf = final; best_ts = t_sec

        # Per-frame record
        frames_data.append({
            "idx": analyzed, "t": t_sec,
            "n_det_v": len(vehs), "n_det_p": len(pers),
            "n_trk_v": len(active_v), "n_trk_p": len(active_p),
            "n_lost": len(vt.get_all_lost_tracks()), "n_dead": len(dead),
            "det": round(det_s,3), "trk": round(trk_s,3), "vel": round(vel_s,3),
            "flow": round(flow_s,3), "dis": round(dis_s,3), "geo": round(geo_s,3),
            "final": round(final,3), "dom": fus["dominant_signal"],
            "m1": round(m1,3), "m4": round(m4,3),
            "flow_mean_mag": round(flow.get("mean_magnitude",0),2),
        })

        # Stop after post-crash
        if accident_confirmed:
            if t_sec - best_ts >= settings.POST_CRASH_LOOKAHEAD_SEC: break

    cap.release()
    # Mark remaining tracks
    for trk in vt.dead_tracks:
        if trk.track_id in track_registry:
            track_registry[trk.track_id]["status"] = "dead"

    return {
        "file": os.path.basename(video_path),
        "accident": accident_confirmed, "best_conf": round(best_conf,3), "best_ts": best_ts,
        "frames_analyzed": analyzed, "fps": fps, "resolution": f"{fw}x{fh}",
        "frames_data": frames_data,
        "track_registry": {str(k):v for k,v in track_registry.items()},
        "disappearance_log": disappearance_log,
        "confidence_series": [round(c,3) for c in confidence_series],
        "signal_series": signal_series,
    }

# ===================== ANALYSIS FUNCTIONS =====================

def analyze_confidence_progression(vid):
    """Phase 7+9: Check if confidence evolves naturally or spikes."""
    cs = vid["confidence_series"]
    if len(cs) < 3: return {"verdict":"INSUFFICIENT_DATA","max_jump":0}
    jumps = [abs(cs[i]-cs[i-1]) for i in range(1,len(cs))]
    max_jump = max(jumps) if jumps else 0
    avg_jump = np.mean(jumps) if jumps else 0
    # Bad: instant spike > 0.35 in one frame
    spikes = [i for i,j in enumerate(jumps) if j > 0.35]
    # Check monotonic build near accident
    if vid["accident"]:
        ts = vid["best_ts"]
        pre = [f for f in vid["frames_data"] if f["t"] < ts]
        if len(pre) >= 3:
            last3 = [f["final"] for f in pre[-3:]]
            rising = all(last3[i] <= last3[i+1]+0.05 for i in range(len(last3)-1))
        else: rising = True
    else: rising = True
    return {
        "max_jump": round(max_jump,3), "avg_jump": round(avg_jump,3),
        "spike_frames": spikes, "n_spikes": len(spikes),
        "pre_accident_rising": rising,
        "verdict": "GOOD" if max_jump < 0.35 and len(spikes) == 0 else "WARN_SPIKES"
    }

def analyze_signal_attribution(vid):
    """Phase 10: Which signals dominate and are they correct?"""
    fd = vid["frames_data"]
    if not fd: return {}
    dom_counts = {}
    for f in fd:
        d = f["dom"]
        dom_counts[d] = dom_counts.get(d,0) + 1
    # Signal means
    sigs = {}
    for k in ["det","trk","vel","flow","dis","geo"]:
        vals = [f[k] for f in fd]
        sigs[k] = {"mean":round(np.mean(vals),3),"max":round(max(vals),3),"min":round(min(vals),3),
                    "std":round(np.std(vals),3)}
    # Check if geometry is always high (suspicious)
    geo_always_high = sigs["geo"]["min"] > 0.7
    # Check if tracking dominates without velocity/flow support
    trk_dominant = dom_counts.get("tracking",0) > len(fd)*0.6
    vel_or_flow_active = sigs["vel"]["max"] > 0.3 or sigs["flow"]["max"] > 0.3
    return {
        "dominant_signal_counts": dom_counts,
        "signal_stats": sigs,
        "geo_always_high": geo_always_high,
        "tracking_dominant_without_support": trk_dominant and not vel_or_flow_active,
    }

def analyze_tracker_health(vid):
    """Phase 3: Tracker stability."""
    tr = vid["track_registry"]
    total = len(tr)
    short_lived = sum(1 for t in tr.values() if t["frames_alive"] < 3)
    dead = sum(1 for t in tr.values() if t["status"] == "dead")
    anomalous_deaths = sum(1 for t in tr.values() if t.get("termination_reason") == "ANOMALOUS_DISAPPEARANCE")
    normal_exits = sum(1 for t in tr.values() if t.get("termination_reason") == "NORMAL_EXIT")
    max_alive = max((t["frames_alive"] for t in tr.values()), default=0)
    return {
        "total_tracks": total, "short_lived_tracks": short_lived,
        "dead_tracks": dead, "anomalous_deaths": anomalous_deaths,
        "normal_exits": normal_exits, "max_frames_alive": max_alive,
        "short_lived_ratio": round(short_lived/max(total,1),2),
        "verdict": "GOOD" if short_lived/max(total,1) < 0.5 else "WARN_UNSTABLE"
    }

def analyze_velocity_reasoning(vid):
    """Phase 4: Is velocity logic triggering correctly?"""
    fd = vid["frames_data"]
    vel_active_frames = [f for f in fd if f["vel"] > 0.3]
    vel_high_frames = [f for f in fd if f["vel"] > 0.6]
    # Velocity should only be high NEAR the accident, not from frame 1
    if vel_active_frames and vid["accident"]:
        first_vel = vel_active_frames[0]["t"]
        acc_t = vid["best_ts"]
        premature = first_vel < acc_t * 0.3 if acc_t else False
    else: premature = False
    return {
        "frames_with_velocity_signal": len(vel_active_frames),
        "frames_with_high_velocity": len(vel_high_frames),
        "premature_velocity": premature,
        "verdict": "GOOD" if not premature else "WARN_PREMATURE"
    }

def analyze_disappearance_reasoning(vid):
    """Phase 5: Are disappearances classified correctly?"""
    dl = vid["disappearance_log"]
    tr = vid["track_registry"]
    anomalous = [d for d in dl if d["score"] > 0.3]
    # Check if anomalous disappearances have supporting evidence
    supported = 0
    for d in anomalous:
        details = d.get("details",{})
        has_vel_collapse = details.get("had_velocity_collapse", False)
        not_border = not details.get("near_border", True)
        if has_vel_collapse or not_border: supported += 1
    return {
        "total_anomalous": len(anomalous),
        "supported_by_evidence": supported,
        "unsupported": len(anomalous) - supported,
        "all_disappearances": len(dl),
        "verdict": "GOOD" if len(anomalous) - supported == 0 else "WARN_UNSUPPORTED"
    }

def analyze_optical_flow(vid):
    """Phase 6: Is optical flow reliable?"""
    fd = vid["frames_data"]
    flow_frames = [f for f in fd if f["flow"] > 0.2]
    flow_high = [f for f in fd if f["flow"] > 0.5]
    mags = [f["flow_mean_mag"] for f in fd]
    # Check for sustained high flow (could be camera shake)
    sustained = 0
    for i in range(2, len(fd)):
        if fd[i]["flow"] > 0.2 and fd[i-1]["flow"] > 0.2 and fd[i-2]["flow"] > 0.2:
            sustained += 1
    return {
        "frames_with_flow": len(flow_frames),
        "frames_high_flow": len(flow_high),
        "sustained_flow_runs": sustained,
        "mean_magnitude": round(np.mean(mags),2) if mags else 0,
        "max_magnitude": round(max(mags),2) if mags else 0,
        "verdict": "GOOD" if sustained < len(fd)*0.5 else "WARN_NOISY"
    }

def analyze_early_trigger(vid):
    """Phase 8+9: Does accident confirm too early (false positive risk)?"""
    fd = vid["frames_data"]
    if not vid["accident"]: return {"verdict":"NO_ACCIDENT"}
    # Find confirmation frame
    confirm_t = None
    for f in fd:
        if f["final"] >= 0.40:
            if confirm_t is None: confirm_t = f["t"]
    # Check what signals were active at confirmation
    confirm_frame = next((f for f in fd if f["t"] == confirm_t), None) if confirm_t else None
    if not confirm_frame: return {"verdict":"NO_CONFIRM_FRAME"}
    # Was there meaningful evidence at confirmation?
    vel_at_confirm = confirm_frame["vel"]
    flow_at_confirm = confirm_frame["flow"]
    dis_at_confirm = confirm_frame["dis"]
    only_geo_trk = vel_at_confirm < 0.1 and flow_at_confirm < 0.1 and dis_at_confirm < 0.1
    return {
        "confirm_time": confirm_t,
        "signals_at_confirm": {
            "det": confirm_frame["det"], "trk": confirm_frame["trk"],
            "vel": vel_at_confirm, "flow": flow_at_confirm,
            "dis": dis_at_confirm, "geo": confirm_frame["geo"],
        },
        "only_geo_and_tracking": only_geo_trk,
        "verdict": "WARN_WEAK_EVIDENCE" if only_geo_trk else "GOOD"
    }

# ===================== REPORT GENERATION =====================

def generate_report(all_results, all_analyses):
    """Generate comprehensive markdown diagnostic report."""
    lines = ["# RapidAid Diagnostic Validation Report\n"]
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Phase 1 summary
    lines.append("## Phase 1 — Baseline Validation\n")
    lines.append(f"| Video | Accident | Conf | Time(s) | Frames |")
    lines.append(f"|-------|----------|------|---------|--------|")
    for r in all_results:
        lines.append(f"| {r['file']} | {'YES' if r['accident'] else 'No'} | {r['best_conf']} | {r['best_ts']} | {r['frames_analyzed']} |")
    det_count = sum(1 for r in all_results if r["accident"])
    lines.append(f"\n**Detection rate: {det_count}/{len(all_results)}**\n")

    for i, (r, a) in enumerate(zip(all_results, all_analyses)):
        lines.append(f"---\n## Video: {r['file']}\n")

        # Phase 3: Tracker
        th = a["tracker"]
        lines.append(f"### Phase 3 — Tracker Diagnostics [{th['verdict']}]\n")
        lines.append(f"- Total tracks: {th['total_tracks']}")
        lines.append(f"- Short-lived (<3 frames): {th['short_lived_tracks']} ({th['short_lived_ratio']*100:.0f}%)")
        lines.append(f"- Dead tracks: {th['dead_tracks']}")
        lines.append(f"- Anomalous deaths: {th['anomalous_deaths']}")
        lines.append(f"- Normal exits: {th['normal_exits']}")
        lines.append(f"- Max frames alive: {th['max_frames_alive']}\n")

        # Phase 4: Velocity
        vr = a["velocity"]
        lines.append(f"### Phase 4 — Velocity Diagnostics [{vr['verdict']}]\n")
        lines.append(f"- Frames with velocity signal (>0.3): {vr['frames_with_velocity_signal']}")
        lines.append(f"- Frames with high velocity (>0.6): {vr['frames_with_high_velocity']}")
        lines.append(f"- Premature velocity trigger: {vr['premature_velocity']}\n")

        # Phase 5: Disappearance
        dr = a["disappearance"]
        lines.append(f"### Phase 5 — Disappearance Diagnostics [{dr['verdict']}]\n")
        lines.append(f"- Total anomalous disappearances: {dr['total_anomalous']}")
        lines.append(f"- Supported by evidence: {dr['supported_by_evidence']}")
        lines.append(f"- Unsupported: {dr['unsupported']}\n")

        # Phase 6: Optical Flow
        of = a["optical_flow"]
        lines.append(f"### Phase 6 — Optical Flow Diagnostics [{of['verdict']}]\n")
        lines.append(f"- Frames with flow signal: {of['frames_with_flow']}")
        lines.append(f"- High flow frames: {of['frames_high_flow']}")
        lines.append(f"- Sustained runs (possible noise): {of['sustained_flow_runs']}")
        lines.append(f"- Mean magnitude: {of['mean_magnitude']}")
        lines.append(f"- Max magnitude: {of['max_magnitude']}\n")

        # Phase 7: Confidence Fusion
        cp = a["confidence"]
        lines.append(f"### Phase 7 — Confidence Fusion [{cp['verdict']}]\n")
        lines.append(f"- Max single-frame jump: {cp['max_jump']}")
        lines.append(f"- Avg frame-to-frame jump: {cp['avg_jump']}")
        lines.append(f"- Spike frames (>0.35 jump): {cp['n_spikes']}")
        lines.append(f"- Pre-accident rising trend: {cp['pre_accident_rising']}\n")

        # Phase 8: Early Trigger / FP Risk
        et = a["early_trigger"]
        lines.append(f"### Phase 8 — False Positive Risk [{et['verdict']}]\n")
        if "confirm_time" in et:
            lines.append(f"- Confirmation time: {et['confirm_time']}s")
            sc = et.get("signals_at_confirm",{})
            lines.append(f"- Signals at confirm: det={sc.get('det',0):.2f} trk={sc.get('trk',0):.2f} vel={sc.get('vel',0):.2f} flow={sc.get('flow',0):.2f} dis={sc.get('dis',0):.2f} geo={sc.get('geo',0):.2f}")
            lines.append(f"- Only geo+tracking (weak): {et['only_geo_and_tracking']}\n")

        # Phase 10: Signal Attribution
        sa = a["signal_attribution"]
        lines.append(f"### Phase 10 — Signal Attribution\n")
        lines.append(f"- Dominant signals: {sa.get('dominant_signal_counts',{})}")
        lines.append(f"- Geometry always high: {sa.get('geo_always_high',False)}")
        lines.append(f"- Tracking dominant without vel/flow: {sa.get('tracking_dominant_without_support',False)}\n")

        # Signal stats table
        ss = sa.get("signal_stats",{})
        if ss:
            lines.append("| Signal | Mean | Max | Min | Std |")
            lines.append("|--------|------|-----|-----|-----|")
            for k,v in ss.items():
                lines.append(f"| {k} | {v['mean']} | {v['max']} | {v['min']} | {v['std']} |")
            lines.append("")

    # ========= CROSS-VIDEO SUMMARY =========
    lines.append("---\n## Cross-Video Diagnostic Summary\n")

    # Aggregate issues
    all_verdicts = {}
    for phase in ["tracker","velocity","disappearance","optical_flow","confidence","early_trigger"]:
        verdicts = [a[phase]["verdict"] for a in all_analyses if "verdict" in a[phase]]
        warns = sum(1 for v in verdicts if v.startswith("WARN"))
        all_verdicts[phase] = {"total":len(verdicts),"warnings":warns}

    lines.append("| Phase | Videos Tested | Warnings | Status |")
    lines.append("|-------|--------------|----------|--------|")
    for phase, v in all_verdicts.items():
        status = "✅ PASS" if v["warnings"] == 0 else f"⚠️ {v['warnings']}/{v['total']} WARN"
        lines.append(f"| {phase} | {v['total']} | {v['warnings']} | {status} |")

    # Key findings
    lines.append("\n### Key Findings\n")
    # Check early trigger across all
    early_weak = sum(1 for a in all_analyses if a["early_trigger"].get("only_geo_and_tracking"))
    lines.append(f"1. **Early trigger with weak evidence**: {early_weak}/{len(all_analyses)} videos confirm accident based on geometry+tracking alone (no velocity/flow/disappearance support)")
    geo_high = sum(1 for a in all_analyses if a["signal_attribution"].get("geo_always_high"))
    lines.append(f"2. **Geometry always high**: {geo_high}/{len(all_analyses)} videos have geometry signal >0.7 from frame 1 (possible false-positive source)")
    trk_dom = sum(1 for a in all_analyses if a["signal_attribution"].get("tracking_dominant_without_support"))
    lines.append(f"3. **Tracking dominates without support**: {trk_dom}/{len(all_analyses)} videos have tracking as dominant signal without velocity/flow corroboration")

    lines.append("\n### Recommendations\n")
    if early_weak > 0:
        lines.append("- **CRITICAL**: Raise CONFIRM_THRESHOLD or require at least one causal signal (velocity/flow/disappearance) >0.2 before confirming accident")
    if geo_high > 0:
        lines.append("- **HIGH**: Geometry score starts high because vehicles with high IoU exist in normal traffic. Add temporal gating: geometry should only count after velocity or flow anomaly is detected")
    if trk_dom > 0:
        lines.append("- **MEDIUM**: Tracking score rewards track longevity, which increases naturally over time even in normal traffic. Consider reducing tracking weight or requiring causal corroboration")
    lines.append("")

    return "\n".join(lines)

# ===================== MAIN =====================

def main():
    print("="*60)
    print("  RapidAid — Full Diagnostic Validation")
    print("="*60)

    vm, pm, fc_model, cd = load_models(use_nano=True)
    videos = sorted(glob.glob("data/test_videos/*"))
    print(f"\n  Found {len(videos)} test videos\n")

    all_results = []
    all_analyses = []

    for vp in videos:
        print(f"\n{'='*60}")
        print(f"  Diagnosing: {os.path.basename(vp)}")
        print(f"{'='*60}")

        result = diagnose_video(vp, vm, pm, fc_model, cd)
        if result is None:
            print(f"  [ERROR] Could not process {vp}")
            continue

        print(f"  Accident: {result['accident']} | Conf: {result['best_conf']} | Time: {result['best_ts']}s")
        print(f"  Tracks: {len(result['track_registry'])} | Disappearances: {len(result['disappearance_log'])}")

        # Run analyses
        analysis = {
            "confidence": analyze_confidence_progression(result),
            "signal_attribution": analyze_signal_attribution(result),
            "tracker": analyze_tracker_health(result),
            "velocity": analyze_velocity_reasoning(result),
            "disappearance": analyze_disappearance_reasoning(result),
            "optical_flow": analyze_optical_flow(result),
            "early_trigger": analyze_early_trigger(result),
        }

        # Print verdicts
        for phase, a in analysis.items():
            v = a.get("verdict","N/A")
            marker = "✓" if v == "GOOD" else "!" if v.startswith("WARN") else "-"
            print(f"    [{marker}] {phase}: {v}")

        all_results.append(result)
        all_analyses.append(analysis)

        # Save per-video raw data
        raw_path = os.path.join(DIAG_DIR, f"diag_{result['file'].replace(' ','_')}.json")
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump({"result":result,"analysis":analysis}, f, indent=2, default=str)

    # Generate report
    report = generate_report(all_results, all_analyses)
    report_path = os.path.join(DIAG_DIR, "diagnostic_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Report saved: {report_path}")

    # Save summary JSON
    summary = []
    for r, a in zip(all_results, all_analyses):
        summary.append({
            "file": r["file"], "accident": r["accident"],
            "best_conf": r["best_conf"], "best_ts": r["best_ts"],
            "verdicts": {k: v.get("verdict","N/A") for k,v in a.items()},
        })
    with open(os.path.join(DIAG_DIR, "diagnostic_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Diagnostic complete: {len(all_results)} videos analyzed")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

