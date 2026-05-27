"""
Consensus Engine — Weighted fusion tier assignment.

ARCHITECTURE:
  RapidAid  -> primary truth engine (physics + causal reasoning)
  bakllava  -> semantic narrator (advisory)
  Groq      -> temporal-causal synthesizer (advisory)
  Consensus -> weighted fusion (NOT semantic veto dominance)

CRITICAL RULE:
  Overwhelming physical evidence CANNOT be fully suppressed by
  semantic disagreement alone. Semantic contradiction is LOGGED
  but does NOT blindly override strong causal physics.
"""
import json
import os
from datetime import datetime
from shared.constants import EventTier, CONSENSUS_VETO_KEYWORDS


class ConsensusEngine:
    """
    Multi-signal weighted consensus engine.

    Philosophy:
      - Physics primary
      - Semantics advisory
      - Fusion weighted
    """

    def __init__(self,
                 major_min_confidence=0.60,
                 minor_min_confidence=0.35,
                 veto_keyword_threshold=2):
        self.major_min = major_min_confidence
        self.minor_min = minor_min_confidence
        self.veto_threshold = veto_keyword_threshold

    def evaluate(self, metadata_package, groq_result=None):
        """
        Evaluate all evidence and assign a tier using weighted fusion.

        Returns:
            dict with tier, reasoning, veto flags, physical safeguard details
        """
        rapidaid = metadata_package.get("rapidaid", {})
        narration = metadata_package.get("bakllava_narration", "") or ""
        ra_conf = rapidaid.get("confidence", 0)
        ra_detected = rapidaid.get("accident_detected", False)

        signal_summary = metadata_package.get("signal_summary", {})
        causal = signal_summary.get("causal_signals_active", False)
        signal_values = signal_summary.get("signal_values", {})

        # Groq data
        groq_severity = "Unknown"
        groq_detected = None
        if groq_result:
            groq_severity = groq_result.get("accident_severity", "Unknown")
            groq_detected = groq_result.get("accident_detected", None)
            if isinstance(groq_detected, str):
                groq_detected = groq_detected.strip().lower().startswith("yes")

        # Semantic veto check
        veto_count = sum(
            1 for kw in CONSENSUS_VETO_KEYWORDS
            if kw.lower() in narration.lower()
        )
        semantic_veto = veto_count >= self.veto_threshold
        groq_veto = groq_detected is False

        # ── Physical evidence strength assessment ──
        strong_signals = self._count_strong_signals(signal_values)
        vel_signal = signal_values.get("velocity", 0)
        physics_overwhelming = (
            ra_conf >= 0.70
            and strong_signals >= 4
            and vel_signal >= 0.50
        )
        physics_strong = (
            ra_conf >= 0.60
            and strong_signals >= 3
            and causal
        )

        # ── Build reasoning chain ──
        reasons = []
        tier = EventTier.LOW_CONFIDENCE

        # Step 1: RapidAid physical evidence (PRIMARY)
        if not ra_detected:
            reasons.append(f"RapidAid: No accident detected (conf={ra_conf:.3f})")
            tier = EventTier.LOW_CONFIDENCE
        elif ra_conf >= self.major_min and causal:
            reasons.append(
                f"RapidAid: Strong detection (conf={ra_conf:.3f}, causal=True, "
                f"strong_signals={strong_signals})"
            )
            tier = EventTier.VERIFIED_MAJOR
        elif ra_conf >= self.minor_min:
            reasons.append(f"RapidAid: Moderate detection (conf={ra_conf:.3f})")
            tier = EventTier.VERIFIED_MINOR
        else:
            reasons.append(f"RapidAid: Weak detection (conf={ra_conf:.3f})")
            tier = EventTier.LOW_CONFIDENCE

        # Step 2: Semantic validation (ADVISORY, not authority)
        semantic_disagreement = False

        if semantic_veto:
            reasons.append(
                f"SEMANTIC NOTE: bakllava narration contains {veto_count} "
                f"contradiction keywords"
            )
            semantic_disagreement = True

            if physics_overwhelming:
                # Physical evidence safeguard: LOG but do NOT suppress
                reasons.append(
                    "PHYSICS SAFEGUARD: Overwhelming physical evidence "
                    f"(conf={ra_conf:.3f}, strong_signals={strong_signals}, "
                    f"vel={vel_signal:.2f}) — semantic contradiction logged "
                    "but tier PRESERVED"
                )
            elif physics_strong:
                # Strong physics: downgrade one level at most
                if tier == EventTier.VERIFIED_MAJOR:
                    tier = EventTier.VERIFIED_MINOR
                    reasons.append(
                        "Downgraded VERIFIED_MAJOR -> VERIFIED_MINOR due to "
                        "semantic contradiction (physics strong but not overwhelming)"
                    )
            else:
                # Weak physics + semantic disagreement: full downgrade
                if tier == EventTier.VERIFIED_MAJOR:
                    tier = EventTier.AMBIGUOUS
                    reasons.append(
                        "Downgraded to AMBIGUOUS: semantic contradiction + "
                        "insufficient physical evidence"
                    )
                elif tier == EventTier.VERIFIED_MINOR:
                    tier = EventTier.LOW_CONFIDENCE
                    reasons.append(
                        "Downgraded to LOW_CONFIDENCE: semantic contradiction + "
                        "weak physical evidence"
                    )

        # Step 3: Groq veto (ADVISORY, not authority)
        if groq_veto:
            reasons.append("GROQ NOTE: Groq structured analysis says no accident")
            semantic_disagreement = True

            if physics_overwhelming:
                reasons.append(
                    "PHYSICS SAFEGUARD: Groq veto overridden by overwhelming "
                    f"physical evidence (conf={ra_conf:.3f}, "
                    f"strong_signals={strong_signals})"
                )
            elif physics_strong:
                if tier == EventTier.VERIFIED_MAJOR:
                    tier = EventTier.VERIFIED_MINOR
                    reasons.append(
                        "Downgraded VERIFIED_MAJOR -> VERIFIED_MINOR: "
                        "Groq disagrees but physics are strong"
                    )
            else:
                if tier in (EventTier.VERIFIED_MAJOR, EventTier.VERIFIED_MINOR):
                    tier = EventTier.AMBIGUOUS
                    reasons.append(
                        "Downgraded to AMBIGUOUS: Groq disagrees + weak physics"
                    )

        # Step 4: Groq severity alignment (advisory refinement)
        if groq_severity == "Minor" and tier == EventTier.VERIFIED_MAJOR:
            if not physics_overwhelming:
                tier = EventTier.VERIFIED_MINOR
                reasons.append(
                    "Downgraded to VERIFIED_MINOR: Groq rates Minor + "
                    "physics not overwhelming"
                )
            else:
                reasons.append(
                    "PHYSICS SAFEGUARD: Groq rates Minor but overwhelming "
                    "physical evidence preserves VERIFIED_MAJOR"
                )
        elif groq_severity == "Critical" and tier == EventTier.VERIFIED_MINOR:
            if ra_conf >= 0.50:
                tier = EventTier.VERIFIED_MAJOR
                reasons.append(
                    "Upgraded to VERIFIED_MAJOR: Groq rates Critical + "
                    "adequate RA confidence"
                )

        # Step 5: No causal evidence guard
        if not causal and tier == EventTier.VERIFIED_MAJOR:
            tier = EventTier.AMBIGUOUS
            reasons.append(
                "Downgraded: No causal signal "
                "(velocity/flow/disappearance) present"
            )

        is_dispatchable = tier in EventTier.DISPATCHABLE

        return {
            "tier": tier,
            "is_dispatchable": is_dispatchable,
            "rapidaid_confidence": round(ra_conf, 3),
            "causal_evidence": bool(causal),
            "strong_signal_count": strong_signals,
            "physics_overwhelming": bool(physics_overwhelming),
            "physics_strong": bool(physics_strong),
            "semantic_veto": semantic_veto,
            "semantic_disagreement": semantic_disagreement,
            "groq_veto": groq_veto,
            "groq_severity": groq_severity,
            "reasoning": reasons,
            "signal_values": {k: round(v, 3) for k, v in signal_values.items()},
            "timestamp": datetime.now().isoformat(),
        }

    def _count_strong_signals(self, signal_values, threshold=0.25):
        """
        Count how many causal signal families exceed the threshold.

        Families: velocity, optical_flow, geometry, disappearance,
                  tracking, detector
        """
        families = [
            "velocity", "optical_flow", "geometry",
            "disappearance", "tracking", "detector",
        ]
        return sum(
            1 for f in families
            if signal_values.get(f, 0) > threshold
        )

    def save_consensus(self, consensus, output_path):
        """Save consensus result to JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(consensus, f, indent=2, default=str)
        return output_path
