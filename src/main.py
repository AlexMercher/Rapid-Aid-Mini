"""
Main orchestration module for Accident Report Generation System.
Coordinates the full pipeline: image → detection → analysis → report → PDF.
"""

import os
import sys
import logging
import argparse
from PIL import Image

from src.config import setup_logging, REPORTS_DIR
from src.image_handler import load_image, preprocess_image, prepare_for_pdf
from src.ollama_client import check_ollama_connection, check_model_available, detect_accident, analyze_accident
from src.report_generator import create_report_structure, format_accident_report, generate_report_filename
from src.pdf_generator import create_pdf_report

logger = logging.getLogger(__name__)


def generate_accident_report(image_path: str = None, image: Image.Image = None,
                             progress_callback=None) -> dict:
    """
    Full pipeline: process an image and generate an accident report PDF.

    Args:
        image_path: Path to the accident image (used when called from CLI)
        image: PIL Image object (used when called from Streamlit)
        progress_callback: Optional callback function for progress updates (msg, pct)

    Returns:
        dict with keys:
          - 'status': 'no_accident' | 'report_generated' | 'error'
          - 'message': Human-readable message
          - 'report_path': Path to generated PDF (if applicable)
          - 'report_data': Structured report data (if applicable)
          - 'detection_result': Detection result dict
          - 'analysis_result': Analysis result dict (if applicable)
    """

    def _progress(msg, pct=None):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg, pct)

    result = {
        "status": "error",
        "message": "",
        "report_path": None,
        "report_data": None,
        "detection_result": None,
        "analysis_result": None,
    }

    try:
        # ── Step 1: Load image ───────────────────────────────────────────────
        _progress("Loading and validating image...", 10)
        if image is None and image_path:
            image = load_image(image_path)
        elif image is None:
            raise ValueError("No image provided. Provide either image_path or image.")

        # ── Step 2: Check Ollama connectivity ────────────────────────────────
        _progress("Checking Ollama API connectivity...", 20)
        if not check_ollama_connection():
            result["message"] = "Cannot connect to Ollama server. Make sure Ollama is running."
            return result

        _progress("Verifying bakllava model availability...", 25)
        if not check_model_available():
            result["message"] = "bakllava model is not available. Run: ollama pull bakllava"
            return result

        # ── Step 3: Accident detection ───────────────────────────────────────
        _progress("Analyzing image for accident detection...", 35)
        detection = detect_accident(image)
        result["detection_result"] = detection

        if detection.get("accident_detected", "No") != "Yes":
            result["status"] = "no_accident"
            result["message"] = "No accident detected"
            _progress("No accident detected. Stopping execution.", 100)
            return result

        # ── Step 4: Detailed analysis ────────────────────────────────────────
        _progress("Accident detected! Running detailed analysis...", 50)
        analysis = analyze_accident(image)
        result["analysis_result"] = analysis

        # ── Step 5: Build report structure ───────────────────────────────────
        _progress("Generating structured accident report...", 70)
        report_data = create_report_structure(detection, analysis)
        result["report_data"] = report_data

        # ── Step 6: Generate PDF ─────────────────────────────────────────────
        _progress("Creating PDF report with embedded image...", 85)
        output_path = generate_report_filename()
        pdf_image = prepare_for_pdf(image)
        pdf_path = create_pdf_report(report_data, output_path, pdf_image)

        result["status"] = "report_generated"
        result["message"] = f"Report generated successfully: {pdf_path}"
        result["report_path"] = pdf_path
        _progress(f"Report saved: {pdf_path}", 100)

        return result

    except FileNotFoundError as e:
        result["message"] = f"Image not found: {e}"
        logger.error(result["message"])
    except ValueError as e:
        result["message"] = f"Invalid input: {e}"
        logger.error(result["message"])
    except ConnectionError as e:
        result["message"] = f"Connection error: {e}"
        logger.error(result["message"])
    except Exception as e:
        result["message"] = f"Unexpected error: {e}"
        logger.error(result["message"], exc_info=True)

    return result


def main():
    """CLI entry point."""
    setup_logging()

    parser = argparse.ArgumentParser(description="Accident Report Generation System")
    parser.add_argument("--image", required=True, help="Path to accident image")
    args = parser.parse_args()

    print("=" * 60)
    print("  Intelligent Accident Detection & Emergency Response System")
    print("=" * 60)

    result = generate_accident_report(image_path=args.image)

    if result["status"] == "no_accident":
        print(f"\n  Result: {result['message']}")
    elif result["status"] == "report_generated":
        print(f"\n  ✓ {result['message']}")
    else:
        print(f"\n  ✗ Error: {result['message']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
