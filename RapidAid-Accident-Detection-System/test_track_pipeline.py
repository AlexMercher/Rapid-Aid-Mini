"""
RapidAid — Track-Centric Pipeline Test Runner (Phase 8)

Terminal-based testing of the track-centric pipeline on all test videos.
NO frontend/UI — all output goes to terminal, JSON, and annotated videos.

Usage:
    conda activate ./venv

    # Test a single video:
    python test_track_pipeline.py --video "data/test_videos/Acc Video 1.mp4"

    # Test all videos:
    python test_track_pipeline.py --all

    # Test with original (small) models:
    python test_track_pipeline.py --all --no-nano

    # Test without saving video (faster):
    python test_track_pipeline.py --all --no-video
"""
import sys
import os
import argparse
import glob
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.track_processor import TrackCentricProcessor
from config import settings


def test_single_video(processor, video_path, save_video=True):
    """Test a single video and return results."""
    print("\n" + "=" * 70)
    print(f"  Testing: {os.path.basename(video_path)}")
    print("=" * 70)

    t0 = time.time()
    result = processor.process_video(
        video_path,
        show_overlay=False,
        save_video=save_video,
        stop_on_first=True,
    )
    total_time = time.time() - t0

    return {
        "file": os.path.basename(video_path),
        "accident_detected": result["accident_detected"],
        "timestamp_sec": result.get("accident_timestamp_sec"),
        "best_confidence": result.get("best_confidence", 0),
        "avg_fps": result.get("metrics", {}).get("avg_fps", 0),
        "avg_processing_ms": result.get("metrics", {}).get(
            "avg_processing_ms", 0
        ),
        "frames_analyzed": result.get("metrics", {}).get(
            "frames_analyzed", 0
        ),
        "peak_tracks": result.get("metrics", {}).get("peak_tracks", 0),
        "total_dead_tracks": result.get("metrics", {}).get(
            "total_dead_tracks", 0
        ),
        "total_time_sec": round(total_time, 1),
        "dominant_signal": (
            result.get("best_fusion_result", {}) or {}
        ).get("dominant_signal", "N/A"),
    }


def test_all_videos(processor, save_video=True):
    """Test all videos in the test directory."""
    videos = sorted(glob.glob("data/test_videos/*"))
    if not videos:
        print("[ERROR] No test videos found in data/test_videos/")
        return []

    print(f"\n  Found {len(videos)} test videos\n")

    results = []
    for video_path in videos:
        try:
            result = test_single_video(processor, video_path, save_video)
            results.append(result)
        except Exception as e:
            print(f"\n  [ERROR] {os.path.basename(video_path)}: {e}")
            results.append({
                "file": os.path.basename(video_path),
                "accident_detected": False,
                "error": str(e),
            })

    return results


def print_summary(results):
    """Print formatted summary table."""
    print("\n" + "=" * 100)
    print("  TRACK-CENTRIC PIPELINE — TEST RESULTS SUMMARY")
    print("=" * 100)

    header = (
        f"  {'Video':<24s} {'Accident':>8s} {'Time(s)':>8s} "
        f"{'Conf':>6s} {'FPS':>6s} {'Tracks':>6s} "
        f"{'Dead':>5s} {'Dom Signal':>12s} {'RunSec':>7s}"
    )
    print(header)
    print("  " + "-" * 95)

    for r in results:
        if "error" in r:
            print(f"  {r['file']:<24s} {'ERROR':>8s} {'':>8s} "
                  f"{'':>6s} {'':>6s} {'':>6s} {'':>5s} "
                  f"{r['error'][:12]:>12s}")
            continue

        label = "YES" if r["accident_detected"] else "No"
        ts = str(r.get("timestamp_sec", "-")) if r.get("timestamp_sec") else "-"

        print(
            f"  {r['file'][:24]:<24s} {label:>8s} {ts:>8s} "
            f"{r.get('best_confidence', 0):>6.3f} "
            f"{r.get('avg_fps', 0):>6.1f} "
            f"{r.get('peak_tracks', 0):>6d} "
            f"{r.get('total_dead_tracks', 0):>5d} "
            f"{r.get('dominant_signal', 'N/A'):>12s} "
            f"{r.get('total_time_sec', 0):>7.1f}"
        )

    # Stats
    detected = sum(1 for r in results if r.get("accident_detected"))
    total = len(results)
    errors = sum(1 for r in results if "error" in r)

    print("\n  " + "-" * 50)
    print(f"  Detected: {detected}/{total} videos")
    if errors:
        print(f"  Errors:   {errors}/{total} videos")

    avg_fps = [r["avg_fps"] for r in results
               if "avg_fps" in r and r["avg_fps"] > 0]
    if avg_fps:
        print(f"  Avg FPS:  {sum(avg_fps)/len(avg_fps):.1f}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="RapidAid — Track-Centric Pipeline Tester"
    )
    parser.add_argument("--video", type=str,
                       help="Path to a single video to test")
    parser.add_argument("--all", action="store_true",
                       help="Test all videos in data/test_videos/")
    parser.add_argument("--no-nano", action="store_true",
                       help="Use original YOLOv8s models instead of nano")
    parser.add_argument("--no-video", action="store_true",
                       help="Don't save annotated output videos")

    args = parser.parse_args()

    if not args.video and not args.all:
        parser.print_help()
        print("\n[ERROR] Use --video or --all")
        sys.exit(1)

    # Initialize processor
    use_nano = not args.no_nano
    save_video = not args.no_video
    processor = TrackCentricProcessor(use_nano=use_nano)

    if args.video:
        if not os.path.exists(args.video):
            print(f"[ERROR] Video not found: {args.video}")
            sys.exit(1)
        result = test_single_video(processor, args.video, save_video)
        print_summary([result])
    else:
        results = test_all_videos(processor, save_video)
        print_summary(results)

        # Save summary JSON
        summary_path = os.path.join("outputs", "track_test_summary.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
