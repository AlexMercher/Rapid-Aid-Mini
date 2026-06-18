"""
Florence-2-large semantic narration client.
Replaces bakllava as the primary frame narration engine.
Interface identical to bakllava_client.py: frame in -> narration str out.

Model:  microsoft/Florence-2-large (0.77B parameters, MIT license)
Weights: HuggingFace cache only — NOT stored in project directory.
Cache:  ~/.cache/huggingface/hub/  (Windows: C:\\Users\\{user}\\.cache\\huggingface\\hub\\)

DO NOT copy weights into the project directory.
DO NOT add weight files to .gitignore — they are not in the project at all.
"""

import os
import time
import logging

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from config import settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Florence-2 task prompts per event-state role.
# <DETAILED_CAPTION> gives the richest natural-language paragraph — best for
# Groq reasoning about collision severity, vehicle positions, and damage.
# <CAPTION> is too brief; <DENSE_REGION_CAPTION> emits bbox coordinates that
# contaminate Groq's text prompt with numeric noise.
# ──────────────────────────────────────────────────────────────────────────────
ROLE_TASK_PROMPTS = {
    "pre_anomaly_trajectory": "<DETAILED_CAPTION>",
    "trajectory_convergence": "<DETAILED_CAPTION>",
    "impact_moment":          "<DETAILED_CAPTION>",   # most important frame
    "peak_disruption":        "<DETAILED_CAPTION>",
    "stabilized_aftermath":   "<DETAILED_CAPTION>",
}

# Role context prefix injected before the task token so Florence-2 knows the
# temporal position of each frame within the 5-frame event storyboard.
ROLE_CONTEXT = {
    "pre_anomaly_trajectory": "PRE-EVENT traffic scene before any collision:",
    "trajectory_convergence": "CONVERGENCE phase, vehicles approaching each other:",
    "impact_moment":          "IMPACT MOMENT, collision occurring or just occurred:",
    "peak_disruption":        "PEAK DISRUPTION, immediate aftermath of collision:",
    "stabilized_aftermath":   "STABILIZED AFTERMATH, scene settling post-event:",
}


class FlorenceClient:
    """
    Florence-2-large semantic narration client.

    Model is loaded ONCE at instantiation via __init__ and reused across all
    narrate_frame() calls within a pipeline run.  Loading per-frame would add
    30+ seconds of overhead per frame; module-level loading fires on every
    import even when the client is not used.  __init__ gives controlled,
    single-load semantics.
    """

    MODEL_ID = settings.FLORENCE_MODEL_ID   # local: weights/florence2/ (set in settings.py)

    def __init__(self) -> None:
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype     = torch.float16 if self.device == "cuda" else torch.float32
        self._model     = None
        self._processor = None
        self._load_model()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load Florence-2 processor and model.  Called once at __init__."""
        logger.info(f"[Florence-2] Loading {self.MODEL_ID} on {self.device} ...")
        t0 = time.time()
        try:
            local_only = os.path.isdir(self.MODEL_ID)   # True for local weights path
            self._processor = AutoProcessor.from_pretrained(
                self.MODEL_ID,
                trust_remote_code=True,
                local_files_only=local_only,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_ID,
                torch_dtype=self.dtype,
                trust_remote_code=True,
                local_files_only=local_only,
            ).to(self.device)          # explicit device placement (device_map unsupported)
            self._model.eval()
            elapsed = time.time() - t0
            logger.info(f"[Florence-2] Loaded in {elapsed:.1f}s on {self.device}")
        except Exception as exc:
            logger.error(f"[Florence-2] Load failed: {exc}")
            self._model     = None
            self._processor = None

    @staticmethod
    def _bgr_to_pil(frame: np.ndarray) -> Image.Image:
        """Convert OpenCV BGR ndarray to RGB PIL Image."""
        frame_rgb = frame[:, :, ::-1].copy()               # BGR → RGB channel reversal
        return Image.fromarray(frame_rgb.astype(np.uint8))

    def _extract_narration(self,
                            parsed: dict,
                            task_prompt: str,
                            role_label: str) -> str:
        """Extract a clean narration string from Florence-2 parsed output dict."""
        # Primary key: exact task token
        text = parsed.get(task_prompt, "")

        # Fallback: iterate parsed values for any non-empty string
        if not text:
            for val in parsed.values():
                if isinstance(val, str) and len(val.strip()) > 10:
                    text = val
                    break

        text = text.strip()
        if not text or len(text) < 10:
            return f"[florence: no narration for {role_label}]"
        return text

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """True if model and processor loaded successfully."""
        return self._model is not None and self._processor is not None

    def narrate_frame(self,
                      frame: np.ndarray,
                      role_label: str,
                      n_frames: int = 5,
                      event_context: str = "") -> str:
        """
        Generate semantic narration for a single event-state frame.

        Args:
            frame:         Clean (no overlay) BGR np.ndarray.
            role_label:    Event-state role string ('impact_moment', etc.).
            n_frames:      Total frames in event — included for context logging.
            event_context: Brief event description (optional, not currently
                           injected into the prompt but available for subclasses).

        Returns:
            Narration string.  Never empty, never "0".
        """
        if not self.is_available:
            return f"[florence: model unavailable for {role_label}]"

        try:
            image = self._bgr_to_pil(frame)

            # Florence-2 requires the task token to be the ONLY input text —
            # any prefix causes "Task token should be the only token" error.
            task_prompt = ROLE_TASK_PROMPTS.get(role_label, "<DETAILED_CAPTION>")
            role_ctx    = ROLE_CONTEXT.get(role_label, "")

            inputs = self._processor(
                text=task_prompt,        # task token only — no prefix
                images=image,
                return_tensors="pt",
            )
            # Move inputs to model device; cast pixel_values to model dtype
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

            t0 = time.time()
            with torch.no_grad():
                generated_ids = self._model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=settings.FLORENCE_MAX_NEW_TOKENS,
                    num_beams=settings.FLORENCE_NUM_BEAMS,
                    early_stopping=True,
                )
            elapsed = time.time() - t0

            generated_text = self._processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]
            parsed = self._processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(image.width, image.height),
            )

            narration = self._extract_narration(parsed, task_prompt, role_label)
            # Prepend role context so Groq receives temporal framing (prompt cannot include it)
            if role_ctx:
                narration = f"{role_ctx} {narration}"
            logger.info(
                f"[Florence-2] [{role_label}] {elapsed:.2f}s → {narration[:80]}"
            )
            return narration

        except Exception as exc:
            logger.warning(f"[Florence-2] Inference failed for {role_label}: {exc}")
            return f"[florence: inference error for {role_label}]"

    def narrate_event(self,
                      frames: list,
                      role_labels: list,
                      event_context: str = "") -> dict:
        """
        Narrate all event-state frames.

        Args:
            frames:        List of clean BGR np.ndarray frames.
            role_labels:   List of role strings matching frames.
            event_context: Brief accident description (optional).

        Returns:
            Dict mapping role_label → narration string.
        """
        results = {}
        for frame, role in zip(frames, role_labels):
            results[role] = self.narrate_frame(
                frame, role, len(frames), event_context
            )
        return results
