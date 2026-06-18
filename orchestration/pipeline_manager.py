"""
Pipeline Manager — End-to-end event validation orchestrator.

Coordinates the full pipeline:
  1. Accept video -> 2. Run RapidAid -> 3. Extract events -> 4. Generate storyboard
  -> 5. Run bakllava -> 6. Run Groq -> 7. Consensus -> 8. Tier -> 9. Reports -> 10. Clips
"""
import os
import sys
import json
import cv2
import time
from datetime import datetime

# Add project roots to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAPIDAID_ROOT = os.path.join(PROJECT_ROOT, "RapidAid-Accident-Detection-System")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, RAPIDAID_ROOT)

from shared.constants import EventTier, TIER_DIRS, EVENTS_DIR
from config import settings
from orchestration.event_extractor import EventExtractor
from orchestration.storyboard_generator import StoryboardGenerator
from orchestration.frame_selector import FrameSelector
from orchestration.timeline_builder import TimelineBuilder
from orchestration.metadata_packager import MetadataPackager
from orchestration.consensus_engine import ConsensusEngine


class PipelineManager:
    """
    Full multi-stage event validation pipeline.

    Integrates RapidAid (physical engine), bakllava (semantic),
    Groq (structured synthesis), and consensus (tier assignment).
    """

    def __init__(self, use_bakllava=True, use_groq=True):
        self.use_bakllava = use_bakllava
        self.use_groq = use_groq

        # Orchestration components
        self.frame_selector = FrameSelector()
        self.storyboard_gen = StoryboardGenerator()
        self.timeline_builder = TimelineBuilder()
        self.packager = MetadataPackager()
        self.consensus = ConsensusEngine()

        # RapidAid processor (lazy-loaded)
        self._rapidaid = None

        # Phase 7: semantic narration client (Florence-2 or bakllava)
        self.semantic_client = None           # FlorenceClient instance, or None for bakllava
        self._semantic_client_name = "bakllava"
        if use_bakllava:
            self._init_semantic_client()

    def _init_semantic_client(self) -> None:
        """Select and initialise the semantic narration client (called once at __init__)."""
        client_setting = getattr(settings, "SEMANTIC_CLIENT", "bakllava")
        if client_setting == "florence":
            try:
                from src.florence_client import FlorenceClient
                self.semantic_client = FlorenceClient()
                if not self.semantic_client.is_available:
                    raise RuntimeError("Florence-2 model failed to load")
                self._semantic_client_name = "Florence-2-large"
                print("  Semantic client: Florence-2-large")
            except Exception as exc:
                print(f"  [WARN] Florence-2 unavailable ({exc}), falling back to bakllava")
                self.semantic_client = None
                self._semantic_client_name = "bakllava (fallback)"
        else:
            self._semantic_client_name = "bakllava (configured)"
            print(f"  Semantic client: bakllava (configured)")

    def _get_rapidaid(self):
        """Lazy-load RapidAid processor."""
        if self._rapidaid is None:
            from pipeline.track_processor import TrackCentricProcessor
            self._rapidaid = TrackCentricProcessor(use_nano=False)  # Phase 1: use settings.VEHICLE_MODEL (yolo11s)
        return self._rapidaid

    def process_video(self, video_path, save_clips=True):
        """
        Process a video through the full multi-stage pipeline.

        Returns:
            dict with complete event validation results
        """
        print("\n" + "=" * 70)
        print("  MULTI-STAGE EVENT VALIDATION PIPELINE")
        print("=" * 70)
        print(f"  Video: {video_path}")
        print(f"  Time: {datetime.now().isoformat()}")
        print("=" * 70)

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        event_id = f"EVT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{video_name}"
        event_dir = os.path.join(EVENTS_DIR, event_id)
        os.makedirs(event_dir, exist_ok=True)

        result = {
            "event_id": event_id,
            "video_file": video_path,
            "video_name": video_name,
            "started_at": datetime.now().isoformat(),
        }

        # === STAGE 1: RapidAid Processing ===
        print("\n[STAGE 1] Running RapidAid causal engine...")
        t0 = time.perf_counter()

        processor = self._get_rapidaid()
        ra_result = processor.process_video(
            video_path, show_overlay=False,
            save_video=False, stop_on_first=True,
        )
        ra_result["video_file"] = video_path

        ra_time = time.perf_counter() - t0
        result["rapidaid_time_sec"] = round(ra_time, 1)

        # Map RapidAid keys to unified naming
        metrics = ra_result.get("metrics", {})
        best_fusion = ra_result.get("best_fusion_result", {}) or {}
        result["rapidaid"] = {
            "accident_detected": ra_result.get("accident_detected", False),
            "best_confidence": ra_result.get("best_confidence", 0),
            "accident_timestamp": ra_result.get("accident_timestamp_sec", None),
            "dominant_signal": best_fusion.get("dominant_signal", "none"),
            "total_tracks_created": metrics.get("peak_tracks", 0),
            "dead_tracks": metrics.get("total_dead_tracks", 0),
            "avg_fps": metrics.get("avg_fps", 0),
            "confirmation_reason": best_fusion.get("confirmation_reason", ""),
        }

        per_frame = ra_result.get("per_frame_data", [])
        event_time = ra_result.get("accident_timestamp_sec", 0) or 0
        # Phase 6: event_time = best_candidate.first_confirmed_timestamp (Phase 4 multi-event).
        # This IS the first_confirmed time of the selected event and the correct impact zone anchor.
        # ra_result["first_confirmed_time"] is the GLOBAL earliest CONFIRMED across all candidates
        # (e.g. V1: 3.33s bus-pass trigger) -- do NOT use it as anchor.
        anchor_time = event_time
        anchor_source = "first_confirmed"  # event_time = first_confirmed of selected candidate

        print(f"  RapidAid: detected={ra_result.get('accident_detected')}, "
              f"conf={ra_result.get('best_confidence', 0):.3f}, "
              f"time={ra_result.get('accident_timestamp_sec')}s "
              f"({ra_time:.1f}s processing)")

        # If no accident detected, still generate event report
        if not ra_result.get("accident_detected", False):
            print("  [INFO] No accident detected by RapidAid")
            result["tier"] = EventTier.LOW_CONFIDENCE
            result["consensus"] = {
                "tier": EventTier.LOW_CONFIDENCE,
                "reasoning": ["RapidAid detected no accident"],
                "is_dispatchable": False,
            }
            self._save_event(event_dir, result)
            return result

        # === STAGE 2: Event Clip Extraction ===
        print("\n[STAGE 2] Extracting event clip...")
        event_clip_meta = None
        if save_clips:
            event_clip_meta = self._extract_clip(
                video_path, ra_result, event_dir, event_id
            )
            result["event_clip"] = event_clip_meta

        # === STAGE 3: Event-State Frame Selection + Individual Frames ===
        print("\n[STAGE 3] Selecting event-state frames...")
        storyboard_path = None
        storyboard_result = None
        frame_selection = None

        frame_store = self._collect_key_frames(
            video_path, per_frame, ra_result, anchor_time, anchor_source, video_name
        )
        frame_selection = self.frame_selector.select_storyboard_frames(
            per_frame,
            frame_store,
            anchor_time=anchor_time,
            anchor_source=anchor_source,
            video_name=video_name,
        )

        if frame_selection:
            roles = frame_selection.get("roles", [])
            print(f"  Event states: {', '.join(roles)}")
            print(f"  Impact frame: idx={frame_selection['impact_idx']}, "
                  f"disruption scores computed")

            sb_path = os.path.join(event_dir, "storyboard.jpg")
            storyboard_result = self.storyboard_gen.generate(
                frame_selection["frames"],
                frame_selection["timestamps"],
                event_data=frame_selection["frame_data"],
                output_path=sb_path,
                roles=roles,
                annotated_frames=frame_selection.get("annotated_frames"),
            )
            storyboard_path = sb_path
            result["storyboard_path"] = storyboard_path

            frame_paths = storyboard_result.get("frame_paths", [])
            debug_paths = storyboard_result.get("debug_frame_paths", [])
            print(f"  Saved {len(frame_paths)} clean event-state frames")
            print(f"  Saved {len(debug_paths)} debug overlay frames")
            print(f"  Composite saved: {sb_path}")
        else:
            print("  [WARN] Could not generate event-state frames")

        # === STAGE 4: Timeline ===
        print("\n[STAGE 4] Building event timeline...")
        timeline = self.timeline_builder.build_timeline(
            ra_result, per_frame, video_file=os.path.basename(video_path)
        )
        tl_path = os.path.join(event_dir, "timeline.json")
        self.timeline_builder.save_timeline(timeline, tl_path)
        result["timeline"] = timeline

        # === STAGE 5: Semantic Narration (Florence-2 or bakllava) ===
        bakllava_output = None
        if self.use_bakllava and storyboard_result is not None:
            print(f"\n[STAGE 5] Running semantic narration ({self._semantic_client_name})...")
            t0 = time.perf_counter()
            try:
                if self.semantic_client is not None:
                    # ── Florence-2 path ────────────────────────────────────────
                    frames = storyboard_result.get("individual_frames", [])
                    roles  = storyboard_result.get("roles", [])
                    narrations = self.semantic_client.narrate_event(frames, roles)
                    parts = [
                        f"### {role}\n{narrations.get(role, '[no narration]')}"
                        for role in roles
                    ]
                    bakllava_output = "\n\n".join(parts)
                    florence_path = os.path.join(event_dir, "florence_output.md")
                    with open(florence_path, "w", encoding="utf-8") as f:
                        f.write(f"# Florence-2 Narration\n\n{bakllava_output}")
                    # Also write bakllava_output.md for backward-compat (app.py, reports)
                    bk_path = os.path.join(event_dir, "bakllava_output.md")
                    with open(bk_path, "w", encoding="utf-8") as f:
                        f.write(f"# Florence-2 Narration (via florence_client)\n\n{bakllava_output}")
                    print(f"  Florence: {len(bakllava_output)} chars "
                          f"({time.perf_counter() - t0:.1f}s)")
                else:
                    # ── bakllava path (primary or auto-fallback) ───────────────
                    from src.bakllava_client import (
                        narrate_storyboard, check_bakllava_available
                    )
                    if check_bakllava_available():
                        bakllava_output = narrate_storyboard(storyboard_result)
                        bk_path = os.path.join(event_dir, "bakllava_output.md")
                        with open(bk_path, "w", encoding="utf-8") as f:
                            f.write(f"# bakllava Narration\n\n{bakllava_output}")
                        print(f"  bakllava: {len(bakllava_output)} chars "
                              f"({time.perf_counter() - t0:.1f}s)")
                    else:
                        print("  [SKIP] bakllava not available")
            except Exception as exc:
                print(f"  [ERROR] Semantic narration failed: {exc}")
                import traceback
                traceback.print_exc()
        else:
            print("\n[STAGE 5] Semantic narration skipped")
        result["bakllava_narration"] = bakllava_output

        # === STAGE 6: Metadata Packaging ===
        print("\n[STAGE 6] Packaging metadata...")
        package = self.packager.package(
            timeline=timeline,
            rapidaid_result=ra_result,
            storyboard_path=storyboard_path,
            bakllava_output=bakllava_output,
            event_clip_meta=event_clip_meta,
            frame_selection=frame_selection,
        )
        pkg_path = os.path.join(event_dir, "metadata.json")
        self.packager.save_package(package, pkg_path)

        # === STAGE 7: Groq Synthesis ===
        groq_result = None
        if self.use_groq:
            print("\n[STAGE 7] Running Groq synthesis...")
            t0 = time.perf_counter()
            try:
                from src.groq_reasoner import synthesize
                groq_result = synthesize(package)
                groq_path = os.path.join(event_dir, "groq_reasoning.md")
                with open(groq_path, "w", encoding="utf-8") as f:
                    f.write(f"# Groq Reasoning\n\n```json\n"
                            f"{json.dumps(groq_result, indent=2)}\n```")
                print(f"  Groq: severity={groq_result.get('accident_severity', '?')} "
                      f"agreement={groq_result.get('physical_semantic_agreement', '?')} "
                      f"({time.perf_counter() - t0:.1f}s)")
            except Exception as e:
                print(f"  [ERROR] Groq failed: {e}")
        else:
            print("\n[STAGE 7] Groq skipped")
        result["groq_result"] = groq_result

        # === STAGE 8: Consensus + Tier Assignment ===
        print("\n[STAGE 8] Running consensus engine...")
        consensus = self.consensus.evaluate(package, groq_result)
        cs_path = os.path.join(event_dir, "consensus.json")
        self.consensus.save_consensus(consensus, cs_path)
        result["consensus"] = consensus
        result["tier"] = consensus["tier"]

        print(f"  Tier: {consensus['tier']}")
        print(f"  Dispatchable: {consensus['is_dispatchable']}")
        print(f"  Physics overwhelming: {consensus.get('physics_overwhelming', False)}")
        for r in consensus.get("reasoning", []):
            print(f"    -> {r}")

        # === STAGE 9: Save to Tier Directory ===
        tier_dir = TIER_DIRS.get(consensus["tier"], TIER_DIRS["LOW_CONFIDENCE"])
        self._copy_to_tier(event_dir, tier_dir, event_id)

        # === STAGE 10: Final Report + Debug Reasoning ===
        result["completed_at"] = datetime.now().isoformat()
        self._save_event(event_dir, result)
        self._save_debug_reasoning(event_dir, result, consensus, frame_selection)

        print("\n" + "=" * 70)
        print("  EVENT VALIDATION COMPLETE")
        print(f"  Event ID:    {event_id}")
        print(f"  Tier:        {consensus['tier']}")
        print(f"  Confidence:  {ra_result.get('best_confidence', 0):.3f}")
        print(f"  Dispatchable: {consensus['is_dispatchable']}")
        print(f"  Event Dir:   {event_dir}")
        print("=" * 70)

        return result

    def _extract_clip(self, video_path, ra_result, event_dir, event_id):
        """Extract event clip from video using event extractor."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        extractor = EventExtractor(fps=fps)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t = frame_count / fps
            extractor.add_frame(frame, timestamp_sec=t)
            frame_count += 1

        cap.release()

        return extractor.extract_event_clip(ra_result, event_id=event_id)

    def _collect_key_frames(
        self,
        video_path,
        per_frame_data,
        ra_result,
        anchor_time,
        anchor_source,
        video_name,
    ):
        """
        Re-read video to collect frames matching per_frame indices.

        Uses FrameSelector.get_needed_frame_indices() to know which
        frames are needed for the causal event-state extraction.
        """
        if not per_frame_data:
            return {}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        from config.settings import FRAMES_PER_SECOND_TO_ANALYZE
        frame_interval = max(1, int(fps / FRAMES_PER_SECOND_TO_ANALYZE))

        # Use FrameSelector to determine needed indices
        needed_analyzed_indices = self.frame_selector.get_needed_frame_indices(
            per_frame_data,
            anchor_time=anchor_time,
            anchor_source=anchor_source,
            video_name=video_name,
        )

        frame_store = {}
        frame_count = 0
        analyzed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_interval != 0:
                continue
            analyzed += 1

            if analyzed in needed_analyzed_indices:
                frame_store[analyzed] = {
                    "clean": frame.copy(),
                    "annotated": frame.copy(),
                }

            if len(frame_store) >= len(needed_analyzed_indices):
                break

        cap.release()
        return frame_store

    def _copy_to_tier(self, event_dir, tier_dir, event_id):
        """Copy key files to the appropriate tier directory."""
        import shutil
        dest = os.path.join(tier_dir, event_id)
        os.makedirs(dest, exist_ok=True)

        for fname in ["consensus.json", "metadata.json", "timeline.json",
                       "storyboard.jpg", "bakllava_output.md",
                       "groq_reasoning.md", "debug_reasoning.md"]:
            src = os.path.join(event_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dest, fname))

        # Copy frame directories
        for frames_dir_name in [
            "clean_event_frames",
            "overlay_debug_frames",
            "event_state_frames",
        ]:
            frames_dir = os.path.join(event_dir, frames_dir_name)
            if os.path.isdir(frames_dir):
                dest_frames = os.path.join(dest, frames_dir_name)
                if os.path.exists(dest_frames):
                    shutil.rmtree(dest_frames)
                shutil.copytree(frames_dir, dest_frames)

    def _save_event(self, event_dir, result):
        """Save the full event result."""
        # Remove non-serializable items
        clean = {}
        for k, v in result.items():
            if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                clean[k] = v

        path = os.path.join(event_dir, "final_report.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2, default=str)

    def _save_debug_reasoning(self, event_dir, result, consensus, frame_selection):
        """Save debug_reasoning.md with full diagnostic information."""
        lines = [
            "# Debug Reasoning\n",
            f"## Event: {result.get('event_id', '?')}\n",
            f"## Tier: {consensus.get('tier', '?')}\n",
            "",
            "### Consensus Reasoning",
        ]
        for r in consensus.get("reasoning", []):
            lines.append(f"- {r}")

        lines.append("")
        lines.append("### Physical Evidence Summary")
        lines.append(f"- Confidence: {consensus.get('rapidaid_confidence', 0)}")
        lines.append(f"- Causal evidence: {consensus.get('causal_evidence', False)}")
        lines.append(f"- Strong signals: {consensus.get('strong_signal_count', 0)}")
        lines.append(f"- Physics overwhelming: {consensus.get('physics_overwhelming', False)}")
        lines.append(f"- Physics strong: {consensus.get('physics_strong', False)}")

        lines.append("")
        lines.append("### Semantic Evidence")
        lines.append(f"- Semantic veto: {consensus.get('semantic_veto', False)}")
        lines.append(f"- Semantic disagreement: {consensus.get('semantic_disagreement', False)}")
        lines.append(f"- Groq veto: {consensus.get('groq_veto', False)}")
        lines.append(f"- Groq severity: {consensus.get('groq_severity', '?')}")

        if frame_selection:
            lines.append("")
            lines.append("### Event-State Frame Selection")
            roles = frame_selection.get("roles", [])
            timestamps = frame_selection.get("timestamps", [])
            for i, role in enumerate(roles):
                ts = timestamps[i] if i < len(timestamps) else "?"
                lines.append(f"- {role}: t={ts}s")

        lines.append("")
        lines.append("### Signal Values at Impact")
        for k, v in consensus.get("signal_values", {}).items():
            lines.append(f"- {k}: {v}")

        path = os.path.join(event_dir, "debug_reasoning.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
