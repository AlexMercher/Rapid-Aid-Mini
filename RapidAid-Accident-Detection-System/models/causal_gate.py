"""
RapidAid — Causal Intelligence Gate

Implements causal reasoning constraints for accident confirmation:
  1. Causal Signal Gating — require physical evidence, not just proximity
  4. Evidence Diversity — require 2+ independent evidence families
  8. Non-Causal Confirmation Detection — flag weak confirmations
  9. Track Death Density — clustered deaths increase suspicion
  12. Suspicious State Machine — CLEAR → SUSPICIOUS → CONFIRMED
"""


class AccidentState:
    CLEAR = "CLEAR"
    SUSPICIOUS = "SUSPICIOUS"
    CONFIRMED = "CONFIRMED"
    AFTERMATH = "AFTERMATH"


class CausalGate:
    """
    Enforces causal reasoning before accident confirmation.

    Prevents confirmation based solely on traffic-density signals
    (detector, tracking, geometry) by requiring at least one
    causal signal (velocity, flow, disappearance).
    """

    # Evidence families for diversity constraint
    FAMILIES = {
        "spatial":       ["detector", "geometry"],
        "temporal":      ["tracking"],
        "motion":        ["velocity", "optical_flow"],
        "disappearance": ["disappearance"],
    }

    def __init__(self,
                 causal_vel_threshold=0.1,
                 causal_flow_threshold=0.2,
                 causal_dis_threshold=0.1,
                 min_evidence_families=2,
                 suspicious_threshold=0.30,
                 confirm_threshold=0.45,
                 confirm_frames_required=3,
                 aftermath_duration_frames=15):
        self.causal_vel_threshold = causal_vel_threshold
        self.causal_flow_threshold = causal_flow_threshold
        self.causal_dis_threshold = causal_dis_threshold
        self.min_evidence_families = min_evidence_families
        self.suspicious_threshold = suspicious_threshold
        self.confirm_threshold = confirm_threshold
        self.confirm_frames_required = confirm_frames_required
        self.aftermath_duration_frames = aftermath_duration_frames

        # State machine
        self.state = AccidentState.CLEAR
        self.suspicious_count = 0
        self.confirmed_frame = None
        self.confirmation_reason = None

        # History for cross-frame consistency
        self.history = []

    def reset(self):
        self.state = AccidentState.CLEAR
        self.suspicious_count = 0
        self.confirmed_frame = None
        self.confirmation_reason = None
        self.history = []

    def evaluate(self, signals, frame_idx):
        """
        Evaluate whether to advance the accident state machine.

        Args:
            signals: dict with keys: velocity, optical_flow, disappearance,
                     detector, tracking, geometry, final_confidence
            frame_idx: current analyzed frame index

        Returns:
            dict with state, causal_present, families_active, flags
        """
        vel = signals.get("velocity", 0)
        flow = signals.get("optical_flow", 0)
        dis = signals.get("disappearance", 0)
        final = signals.get("final_confidence", 0)

        # 1. Causal signal gating
        causal_present = (
            vel > self.causal_vel_threshold or
            flow > self.causal_flow_threshold or
            dis > self.causal_dis_threshold
        )

        # 4. Evidence diversity
        active_families = self._count_active_families(signals)
        diversity_met = len(active_families) >= self.min_evidence_families

        # 9. Track death density (passed in as a signal)
        death_density = signals.get("death_density", 0)

        # Store frame record
        record = {
            "frame_idx": frame_idx,
            "final": final,
            "causal": causal_present,
            "families": len(active_families),
            "vel": vel, "flow": flow, "dis": dis,
            "death_density": death_density,
        }
        self.history.append(record)
        if len(self.history) > 30:
            self.history.pop(0)

        # 10. Cross-frame consistency
        cross_frame_consistent = self._check_cross_frame(final)

        # 12. State machine transitions
        prev_state = self.state
        flags = []

        if self.state == AccidentState.CLEAR:
            if final >= self.suspicious_threshold and causal_present:
                self.state = AccidentState.SUSPICIOUS
                self.suspicious_count = 1
                flags.append("TRANSITION_TO_SUSPICIOUS")
            elif final >= self.suspicious_threshold and death_density > 0.3:
                self.state = AccidentState.SUSPICIOUS
                self.suspicious_count = 1
                flags.append("SUSPICIOUS_VIA_DEATH_DENSITY")

        elif self.state == AccidentState.SUSPICIOUS:
            if final >= self.confirm_threshold and causal_present:
                self.suspicious_count += 1
            elif final >= self.suspicious_threshold:
                # Don't increment but don't reset either
                pass
            else:
                self.suspicious_count = max(0, self.suspicious_count - 1)
                if self.suspicious_count == 0:
                    self.state = AccidentState.CLEAR
                    flags.append("RETURNED_TO_CLEAR")

            # 3. Confirmation persistence
            if (self.suspicious_count >= self.confirm_frames_required
                    and causal_present and diversity_met
                    and cross_frame_consistent):
                self.state = AccidentState.CONFIRMED
                self.confirmed_frame = frame_idx
                self.confirmation_reason = self._build_reason(signals, active_families)
                flags.append("CONFIRMED_WITH_CAUSAL_EVIDENCE")

        elif self.state == AccidentState.CONFIRMED:
            if frame_idx - self.confirmed_frame > self.aftermath_duration_frames:
                self.state = AccidentState.AFTERMATH
                flags.append("ENTERED_AFTERMATH")

        # 8. Non-causal confirmation detection
        non_causal_flag = False
        if self.state == AccidentState.CONFIRMED and not causal_present:
            non_causal_flag = True
            flags.append("NON_CAUSAL_CONFIRMATION")

        return {
            "state": self.state,
            "prev_state": prev_state,
            "causal_present": causal_present,
            "active_families": list(active_families),
            "n_families": len(active_families),
            "diversity_met": diversity_met,
            "suspicious_count": self.suspicious_count,
            "cross_frame_consistent": cross_frame_consistent,
            "non_causal_flag": non_causal_flag,
            "flags": flags,
            "confirmation_reason": self.confirmation_reason,
        }

    def _count_active_families(self, signals):
        """Count how many evidence families have active signals."""
        active = set()
        thresholds = {
            "detector": 0.3, "geometry": 0.3,
            "tracking": 0.3,
            "velocity": 0.1, "optical_flow": 0.15,
            "disappearance": 0.1,
        }
        for family, members in self.FAMILIES.items():
            for member in members:
                val = signals.get(member, 0)
                if val > thresholds.get(member, 0.1):
                    active.add(family)
                    break
        return active

    def _check_cross_frame(self, current_conf):
        """Check if confidence is consistent across recent frames."""
        if len(self.history) < 3:
            return True
        recent = [h["final"] for h in self.history[-3:]]
        # All recent frames must be above suspicious threshold
        return all(c >= self.suspicious_threshold * 0.8 for c in recent)

    def _build_reason(self, signals, families):
        """Build human-readable confirmation reason."""
        parts = []
        if signals.get("velocity", 0) > self.causal_vel_threshold:
            parts.append(f"velocity_anomaly={signals['velocity']:.2f}")
        if signals.get("optical_flow", 0) > self.causal_flow_threshold:
            parts.append(f"flow_burst={signals['optical_flow']:.2f}")
        if signals.get("disappearance", 0) > self.causal_dis_threshold:
            parts.append(f"disappearance={signals['disappearance']:.2f}")
        parts.append(f"families={list(families)}")
        return "; ".join(parts)
