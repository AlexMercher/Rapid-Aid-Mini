"""
main_pipeline.py — Root entry point for the Multi-Stage Event Validation Platform.

Usage:
    python main_pipeline.py <video_path>              # Single video
    python main_pipeline.py --test-all                 # All test videos
    python main_pipeline.py --test-all --no-bakllava   # Skip bakllava
    python main_pipeline.py --test-all --no-groq       # Skip Groq
    python main_pipeline.py --test-all --offline        # No LLM (RapidAid only)
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAPIDAID_ROOT = os.path.join(PROJECT_ROOT, "RapidAid-Accident-Detection-System")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, RAPIDAID_ROOT)

from orchestration.pipeline_manager import PipelineManager
from shared.constants import EventTier, EVENTS_DIR


def find_test_videos():
    """Find all test videos in RapidAid's test_videos directory."""
    videos_dir = os.path.join(RAPIDAID_ROOT, "data", "test_videos")
    if not os.path.isdir(videos_dir):
        print(f"[ERROR] Test videos dir not found: {videos_dir}")
        return []
    videos = sorted([
        os.path.join(videos_dir, f)
        for f in os.listdir(videos_dir)
        if f.lower().endswith((".mp4", ".avi", ".mov"))
    ])
    return videos


def run_single(video_path, use_bakllava=True, use_groq=True):
    """Process a single video through the full pipeline."""
    pipeline = PipelineManager(
        use_bakllava=use_bakllava,
        use_groq=use_groq,
    )
    return pipeline.process_video(video_path)


def run_all(use_bakllava=True, use_groq=True):
    """Process all test videos and generate a summary."""
    videos = find_test_videos()
    if not videos:
        print("[ERROR] No test videos found")
        return

    print("\n" + "=" * 80)
    print("  MULTI-STAGE EVENT VALIDATION - FULL TEST SUITE")
    print(f"  Videos: {len(videos)}")
    print(f"  bakllava: {'ON' if use_bakllava else 'OFF'}")
    print(f"  Groq: {'ON' if use_groq else 'OFF'}")
    print("=" * 80)

    pipeline = PipelineManager(
        use_bakllava=use_bakllava,
        use_groq=use_groq,
    )

    results = []
    t_start = time.perf_counter()

    for i, vpath in enumerate(videos, 1):
        print(f"\n{'=' * 80}")
        print(f"  [{i}/{len(videos)}] {os.path.basename(vpath)}")
        print(f"{'=' * 80}")

        try:
            r = pipeline.process_video(vpath)
            results.append(r)
        except Exception as e:
            print(f"  [ERROR] Failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "video_name": os.path.basename(vpath),
                "error": str(e),
                "tier": "ERROR",
            })

    total_time = time.perf_counter() - t_start

    # Print summary table
    print("\n\n" + "=" * 100)
    print("  MULTI-STAGE EVENT VALIDATION - SUMMARY")
    print("=" * 100)
    print(f"  {'Video':<25} {'Tier':<18} {'RA Conf':>8} {'Time':>7} {'Groq Sev':>10} {'Dispatch':>10}")
    print(f"  {'-'*90}")

    tier_counts = {}
    for r in results:
        name = r.get("video_name", "?")[:24]
        tier = r.get("tier", "ERROR")
        ra = r.get("rapidaid", {})
        conf = ra.get("best_confidence", 0)
        t = ra.get("accident_timestamp", 0) or 0
        groq = r.get("groq_result", {}) or {}
        sev = groq.get("accident_severity", "N/A")
        consensus = r.get("consensus", {})
        disp = "YES" if consensus.get("is_dispatchable") else "no"

        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        print(f"  {name:<25} {tier:<18} {conf:>8.3f} {t:>6.1f}s {sev:>10} {disp:>10}")

    print(f"\n  {'-'*60}")
    print(f"  Total videos: {len(results)}")
    for t, c in sorted(tier_counts.items()):
        print(f"    {t}: {c}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Avg time/video: {total_time / max(len(results), 1):.1f}s")

    # Save summary
    summary_path = os.path.join(EVENTS_DIR, "test_suite_summary.json")
    summary = {
        "run_at": datetime.now().isoformat(),
        "n_videos": len(results),
        "tier_distribution": tier_counts,
        "total_time_sec": round(total_time, 1),
        "bakllava_enabled": use_bakllava,
        "groq_enabled": use_groq,
        "results": [
            {
                "video": r.get("video_name", "?"),
                "tier": r.get("tier", "ERROR"),
                "confidence": r.get("rapidaid", {}).get("best_confidence", 0),
                "timestamp": r.get("rapidaid", {}).get("accident_timestamp", 0),
                "dispatchable": r.get("consensus", {}).get("is_dispatchable", False),
            }
            for r in results
        ],
    }
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Stage Event Validation Pipeline"
    )
    parser.add_argument("video", nargs="?", help="Path to video file")
    parser.add_argument("--test-all", action="store_true",
                        help="Process all test videos")
    parser.add_argument("--no-bakllava", action="store_true",
                        help="Skip bakllava narration")
    parser.add_argument("--no-groq", action="store_true",
                        help="Skip Groq synthesis")
    parser.add_argument("--offline", action="store_true",
                        help="Offline mode (no LLMs)")

    args = parser.parse_args()

    use_bakllava = not args.no_bakllava and not args.offline
    use_groq = not args.no_groq and not args.offline

    if args.test_all:
        run_all(use_bakllava=use_bakllava, use_groq=use_groq)
    elif args.video:
        run_single(args.video, use_bakllava=use_bakllava, use_groq=use_groq)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
