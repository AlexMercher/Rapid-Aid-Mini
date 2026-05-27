"""
RapidAid — Phase 2 Benchmark: YOLOv8s vs YOLOv8n

Compares small vs nano models across all test data for:
  - FPS (frames per second)
  - RAM / memory usage
  - Detection counts (vehicles, persons)
  - Detection quality (confidence distributions)
  - Missed detections

Usage:
    conda activate ./venv
    python benchmark_nano.py
"""
import sys
import os
import time
import json
import tracemalloc
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from ultralytics import YOLO
from config import settings


def benchmark_model(model_path, frames, model_type="seg"):
    """
    Benchmark a YOLO model on a list of frames.

    Args:
        model_path: path to .pt file
        frames: list of (name, numpy_array) tuples
        model_type: 'seg' or 'pose'

    Returns:
        dict with timing, memory, and detection stats
    """
    print(f"\n  Loading: {os.path.basename(model_path)}")
    model = YOLO(model_path)

    # Warmup
    _ = model(frames[0][1], verbose=False)

    results_data = []
    tracemalloc.start()
    t_start = time.perf_counter()

    for name, frame in frames:
        t0 = time.perf_counter()
        results = model(frame, verbose=False)[0]
        t1 = time.perf_counter()

        n_detections = len(results.boxes) if results.boxes is not None else 0
        confidences = []
        if results.boxes is not None:
            for box in results.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                confidences.append(conf)

        results_data.append({
            "name": name,
            "time_ms": (t1 - t0) * 1000,
            "n_detections": n_detections,
            "confidences": confidences,
        })

    t_end = time.perf_counter()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_time = t_end - t_start
    avg_fps = len(frames) / total_time if total_time > 0 else 0
    avg_time_ms = (total_time / len(frames)) * 1000

    all_confs = []
    total_dets = 0
    for r in results_data:
        total_dets += r["n_detections"]
        all_confs.extend(r["confidences"])

    return {
        "model": os.path.basename(model_path),
        "total_frames": len(frames),
        "total_time_sec": round(total_time, 2),
        "avg_fps": round(avg_fps, 1),
        "avg_time_ms": round(avg_time_ms, 1),
        "peak_memory_mb": round(peak_mem / 1024 / 1024, 1),
        "total_detections": total_dets,
        "avg_detections": round(total_dets / len(frames), 1),
        "avg_confidence": round(np.mean(all_confs), 3) if all_confs else 0,
        "per_frame": results_data,
    }


def load_test_frames():
    """Load all test frames and sample frames from test videos."""
    frames = []

    # Load test frames
    for path in sorted(glob.glob("data/test_frames/*")):
        img = cv2.imread(path)
        if img is not None:
            frames.append((os.path.basename(path), img))

    # Sample 3 frames from each test video
    for path in sorted(glob.glob("data/test_videos/*")):
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 3:
            cap.release()
            continue

        sample_indices = [total // 4, total // 2, 3 * total // 4]
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                name = f"{os.path.basename(path)}_f{idx}"
                frames.append((name, frame))
        cap.release()

    return frames


def compare_detections(small_results, nano_results):
    """Compare detection counts per frame between two models."""
    mismatches = []
    for s, n in zip(small_results["per_frame"], nano_results["per_frame"]):
        diff = s["n_detections"] - n["n_detections"]
        if diff != 0:
            mismatches.append({
                "frame": s["name"],
                "small_dets": s["n_detections"],
                "nano_dets": n["n_detections"],
                "diff": diff,
            })
    return mismatches


def main():
    print("=" * 60)
    print("  RapidAid — Phase 2: YOLOv8s vs YOLOv8n Benchmark")
    print("=" * 60)

    frames = load_test_frames()
    print(f"\n  Loaded {len(frames)} test frames")

    # Benchmark segmentation models
    print("\n" + "=" * 60)
    print("  SEGMENTATION MODEL BENCHMARK (Vehicle Detection)")
    print("=" * 60)

    seg_small = benchmark_model(
        os.path.join(settings.WEIGHTS_DIR, "yolov8s-seg.pt"), frames, "seg"
    )
    seg_nano = benchmark_model(
        os.path.join(settings.WEIGHTS_DIR, "yolov8n-seg.pt"), frames, "seg"
    )

    print("\n  {:30s} {:>10s} {:>10s}".format("Metric", "Small", "Nano"))
    print("  " + "-" * 52)
    print("  {:30s} {:>10.1f} {:>10.1f}".format(
        "FPS", seg_small["avg_fps"], seg_nano["avg_fps"]))
    print("  {:30s} {:>10.1f} {:>10.1f}".format(
        "Avg time (ms)", seg_small["avg_time_ms"], seg_nano["avg_time_ms"]))
    print("  {:30s} {:>10.1f} {:>10.1f}".format(
        "Peak memory (MB)", seg_small["peak_memory_mb"], seg_nano["peak_memory_mb"]))
    print("  {:30s} {:>10.1f} {:>10.1f}".format(
        "Avg detections/frame", seg_small["avg_detections"], seg_nano["avg_detections"]))
    print("  {:30s} {:>10.3f} {:>10.3f}".format(
        "Avg confidence", seg_small["avg_confidence"], seg_nano["avg_confidence"]))

    seg_mismatches = compare_detections(seg_small, seg_nano)
    if seg_mismatches:
        print(f"\n  Detection count differences: {len(seg_mismatches)}/{len(frames)} frames")
        for m in seg_mismatches[:5]:
            print(f"    {m['frame']}: small={m['small_dets']}, nano={m['nano_dets']} (diff={m['diff']})")

    # Benchmark pose models
    print("\n" + "=" * 60)
    print("  POSE MODEL BENCHMARK (Person Detection)")
    print("=" * 60)

    pose_small = benchmark_model(
        os.path.join(settings.WEIGHTS_DIR, "yolov8s-pose.pt"), frames, "pose"
    )
    pose_nano = benchmark_model(
        os.path.join(settings.WEIGHTS_DIR, "yolov8n-pose.pt"), frames, "pose"
    )

    print("\n  {:30s} {:>10s} {:>10s}".format("Metric", "Small", "Nano"))
    print("  " + "-" * 52)
    print("  {:30s} {:>10.1f} {:>10.1f}".format(
        "FPS", pose_small["avg_fps"], pose_nano["avg_fps"]))
    print("  {:30s} {:>10.1f} {:>10.1f}".format(
        "Avg time (ms)", pose_small["avg_time_ms"], pose_nano["avg_time_ms"]))
    print("  {:30s} {:>10.1f} {:>10.1f}".format(
        "Peak memory (MB)", pose_small["peak_memory_mb"], pose_nano["peak_memory_mb"]))
    print("  {:30s} {:>10.1f} {:>10.1f}".format(
        "Avg detections/frame", pose_small["avg_detections"], pose_nano["avg_detections"]))
    print("  {:30s} {:>10.3f} {:>10.3f}".format(
        "Avg confidence", pose_small["avg_confidence"], pose_nano["avg_confidence"]))

    pose_mismatches = compare_detections(pose_small, pose_nano)
    if pose_mismatches:
        print(f"\n  Detection count differences: {len(pose_mismatches)}/{len(frames)} frames")
        for m in pose_mismatches[:5]:
            print(f"    {m['frame']}: small={m['small_dets']}, nano={m['nano_dets']} (diff={m['diff']})")

    # Speedup summary
    print("\n" + "=" * 60)
    print("  SPEEDUP SUMMARY")
    print("=" * 60)
    seg_speedup = seg_nano["avg_fps"] / max(seg_small["avg_fps"], 0.01)
    pose_speedup = pose_nano["avg_fps"] / max(pose_small["avg_fps"], 0.01)
    print(f"  Segmentation: {seg_speedup:.2f}x faster")
    print(f"  Pose:         {pose_speedup:.2f}x faster")

    # Save results
    benchmark_results = {
        "segmentation": {
            "small": {k: v for k, v in seg_small.items() if k != "per_frame"},
            "nano": {k: v for k, v in seg_nano.items() if k != "per_frame"},
            "speedup": round(seg_speedup, 2),
            "mismatches": len(seg_mismatches),
        },
        "pose": {
            "small": {k: v for k, v in pose_small.items() if k != "per_frame"},
            "nano": {k: v for k, v in pose_nano.items() if k != "per_frame"},
            "speedup": round(pose_speedup, 2),
            "mismatches": len(pose_mismatches),
        },
    }

    out_path = os.path.join("outputs", "benchmark_nano.json")
    with open(out_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
