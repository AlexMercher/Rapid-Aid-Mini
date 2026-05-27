#!/usr/bin/env python
"""
Terminal-first pipeline for Accident Detection & Report Generation.

Usage:
    python process_image.py <image_path>
    python process_image.py accidents/accident.png
    python process_image.py --batch accidents/

Processes an image (or all images in a folder) through the two-stage pipeline
and prints structured JSON results to stdout.
"""

import os
import sys
import json
import argparse
import logging

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import setup_logging, SUPPORTED_IMAGE_FORMATS, GROQ_MODEL
from src.image_handler import load_image, validate_image
from src.ollama_client import (
    check_ollama_connection,
    check_model_available,
    detect_accident,
    analyze_accident,
    MODEL_NAME,
)
from src.report_generator import create_report_structure, format_accident_report, generate_report_filename
from src.pdf_generator import create_pdf_report


def process_single_image(image_path: str, save_pdf: bool = True) -> dict:
    """
    Run the full accident detection + analysis pipeline on a single image.

    Returns a result dict printed as JSON.
    """
    abs_path = os.path.abspath(image_path)
    result = {"image": abs_path, "status": "unknown"}

    # --- Validate & Load image ---
    try:
        validate_image(abs_path)
    except (FileNotFoundError, ValueError) as e:
        result["status"] = "error"
        result["error"] = str(e)
        return result

    image = load_image(abs_path)
    if image is None:
        result["status"] = "error"
        result["error"] = "Failed to load image"
        return result

    # --- Stage 1+2: Accident detection ---
    detection = detect_accident(image)
    result["detection"] = detection

    accident_detected = detection.get("accident_detected", "No")
    if accident_detected != "Yes":
        result["status"] = "no_accident"
        result["message"] = "No accident detected in image"
        return result

    # --- Stage 1+2: Accident analysis ---
    analysis = analyze_accident(image)
    result["analysis"] = analysis

    # --- Build report ---
    report_data = create_report_structure(detection, analysis)
    formatted = format_accident_report(report_data)
    result["formatted_report"] = formatted

    # --- Generate PDF ---
    if save_pdf:
        try:
            from src.image_handler import prepare_for_pdf
            output_path = generate_report_filename()
            pdf_image = prepare_for_pdf(image)
            pdf_path = create_pdf_report(report_data, output_path, pdf_image)
            result["pdf_path"] = pdf_path
        except Exception as exc:
            result["pdf_error"] = str(exc)

    result["status"] = "accident_reported"
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Accident Detection & Report Generation — Terminal Pipeline"
    )
    parser.add_argument(
        "path",
        help="Path to a single image or a directory of images",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Treat 'path' as a directory and process all supported images",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF report generation",
    )
    args = parser.parse_args()

    # --- Setup ---
    logger = setup_logging()

    # --- Connectivity checks ---
    if not check_ollama_connection():
        print(json.dumps({"error": "Cannot connect to Ollama server"}, indent=2))
        sys.exit(1)

    if not check_model_available(MODEL_NAME):
        print(json.dumps({"error": f"Vision model '{MODEL_NAME}' not available"}, indent=2))
        sys.exit(1)

    print(f"Vision model: {MODEL_NAME} (local GPU)", file=sys.stderr)
    print(f"Text model:   {GROQ_MODEL} (Groq cloud)", file=sys.stderr)

    # --- Collect image paths ---
    target = os.path.abspath(args.path)
    if args.batch or os.path.isdir(target):
        if not os.path.isdir(target):
            print(json.dumps({"error": f"Not a directory: {target}"}, indent=2))
            sys.exit(1)
        image_paths = sorted(
            os.path.join(target, f)
            for f in os.listdir(target)
            if f.lower().endswith(SUPPORTED_IMAGE_FORMATS)
        )
        if not image_paths:
            print(json.dumps({"error": f"No supported images in {target}"}, indent=2))
            sys.exit(1)
    else:
        image_paths = [target]

    # --- Process ---
    results = []
    for img_path in image_paths:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Processing: {img_path}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        result = process_single_image(img_path, save_pdf=not args.no_pdf)
        results.append(result)

        # Print individual result immediately
        print(json.dumps(result, indent=2, default=str))
        print()  # blank line separator

    # --- Summary ---
    total = len(results)
    accidents = sum(1 for r in results if r["status"] == "accident_reported")
    no_accidents = sum(1 for r in results if r["status"] == "no_accident")
    errors = sum(1 for r in results if r["status"] == "error")

    summary = {
        "summary": {
            "total_images": total,
            "accidents_detected": accidents,
            "no_accident": no_accidents,
            "errors": errors,
        }
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
