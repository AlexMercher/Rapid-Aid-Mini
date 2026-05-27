"""
Test 10: No Accident Detection Handling
Verifies system stops execution when no accident is detected.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from unittest.mock import patch
from src.main import generate_accident_report


class TestNoAccidentHandling:
    """Test suite for no-accident detection handling."""

    def test_no_accident_returns_correct_status(self):
        """Test that the system returns 'no_accident' status."""
        img = Image.new('RGB', (400, 300), color=(50, 200, 50))

        with patch('src.main.check_ollama_connection', return_value=True), \
             patch('src.main.check_model_available', return_value=True), \
             patch('src.main.detect_accident', return_value={
                 "accident_detected": "No",
                 "scene_description": "A normal city street"
             }):
            result = generate_accident_report(image=img)
            assert result["status"] == "no_accident"
            assert result["message"] == "No accident detected"

    def test_no_pdf_generated_when_no_accident(self):
        """Test that no PDF is generated when no accident is detected."""
        img = Image.new('RGB', (400, 300), color=(50, 200, 50))

        with patch('src.main.check_ollama_connection', return_value=True), \
             patch('src.main.check_model_available', return_value=True), \
             patch('src.main.detect_accident', return_value={
                 "accident_detected": "No",
                 "scene_description": "A park"
             }):
            result = generate_accident_report(image=img)
            assert result["report_path"] is None

    def test_execution_terminates_cleanly(self):
        """Test that execution terminates cleanly with no errors."""
        img = Image.new('RGB', (400, 300), color=(50, 200, 50))

        with patch('src.main.check_ollama_connection', return_value=True), \
             patch('src.main.check_model_available', return_value=True), \
             patch('src.main.detect_accident', return_value={
                 "accident_detected": "No",
                 "scene_description": "A building"
             }):
            result = generate_accident_report(image=img)
            assert result["status"] == "no_accident"
            assert result["analysis_result"] is None
