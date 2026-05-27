"""
Test 13: Full Pipeline with Accident Detection
Verifies complete pipeline including accident detection validation.
Tests both accident and non-accident image paths.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from unittest.mock import patch
from src.main import generate_accident_report


@pytest.fixture
def accident_image():
    return Image.new('RGB', (400, 300), color=(200, 50, 50))


@pytest.fixture
def non_accident_image():
    return Image.new('RGB', (400, 300), color=(50, 200, 50))


class TestFullPipelineAccidentDetection:
    """Test suite for full pipeline with accident detection validation."""

    def test_accident_image_generates_report(self, accident_image):
        """Test that an accident image produces a full report."""
        detection = {"accident_detected": "Yes", "scene_description": "Car crash"}
        analysis = {
            "accident_type": "Side collision",
            "number_of_victims": 1,
            "vehicles_involved": 2,
            "accident_severity": "Minor",
            "injured_person_detected": "No",
            "emergency_services_present": "No",
            "road_blocked": "No",
            "scene_description": "Minor side collision at parking lot"
        }

        with patch('src.main.check_ollama_connection', return_value=True), \
             patch('src.main.check_model_available', return_value=True), \
             patch('src.main.detect_accident', return_value=detection), \
             patch('src.main.analyze_accident', return_value=analysis):

            result = generate_accident_report(image=accident_image)
            assert result["status"] == "report_generated"
            assert result["report_path"] is not None
            assert os.path.exists(result["report_path"])

            # Verify PDF starts with valid header
            with open(result["report_path"], 'rb') as f:
                assert f.read(5) == b'%PDF-'

            # Cleanup
            os.unlink(result["report_path"])

    def test_non_accident_image_no_report(self, non_accident_image):
        """Test that a non-accident image produces no report."""
        detection = {"accident_detected": "No", "scene_description": "A peaceful park"}

        with patch('src.main.check_ollama_connection', return_value=True), \
             patch('src.main.check_model_available', return_value=True), \
             patch('src.main.detect_accident', return_value=detection):

            result = generate_accident_report(image=non_accident_image)
            assert result["status"] == "no_accident"
            assert result["message"] == "No accident detected"
            assert result["report_path"] is None
            assert result["analysis_result"] is None

    def test_all_fields_in_generated_report(self, accident_image):
        """Test that the generated report contains all required fields."""
        detection = {"accident_detected": "Yes", "scene_description": "Crash"}
        analysis = {
            "accident_type": "Head-on",
            "number_of_victims": 3,
            "vehicles_involved": 2,
            "accident_severity": "Critical",
            "injured_person_detected": "Yes",
            "emergency_services_present": "Yes",
            "road_blocked": "Yes",
            "scene_description": "Severe head-on collision"
        }

        with patch('src.main.check_ollama_connection', return_value=True), \
             patch('src.main.check_model_available', return_value=True), \
             patch('src.main.detect_accident', return_value=detection), \
             patch('src.main.analyze_accident', return_value=analysis):

            result = generate_accident_report(image=accident_image)
            report = result["report_data"]

            assert report["accident_type"] == "Head-on"
            assert report["number_of_victims"] == 3
            assert report["vehicles_involved"] == 2
            assert report["accident_severity"] == "Critical"
            assert report["injured_person_detected"] == "Yes"
            assert report["emergency_services_present"] == "Yes"
            assert report["road_blocked"] == "Yes"
            assert report["date"] is not None
            assert report["time"] is not None
            assert report["gps"]["latitude"] is not None
            assert report["gps"]["longitude"] is not None

            # Cleanup
            if result["report_path"] and os.path.exists(result["report_path"]):
                os.unlink(result["report_path"])

    def test_image_embedded_in_pdf(self, accident_image):
        """Test that the accident image is embedded in the generated PDF."""
        detection = {"accident_detected": "Yes", "scene_description": "Crash"}
        analysis = {
            "accident_type": "Rollover",
            "number_of_victims": 1,
            "vehicles_involved": 1,
            "accident_severity": "Major",
            "injured_person_detected": "Yes",
            "emergency_services_present": "No",
            "road_blocked": "Yes",
            "scene_description": "Vehicle rollover on highway"
        }

        with patch('src.main.check_ollama_connection', return_value=True), \
             patch('src.main.check_model_available', return_value=True), \
             patch('src.main.detect_accident', return_value=detection), \
             patch('src.main.analyze_accident', return_value=analysis):

            result = generate_accident_report(image=accident_image)
            assert os.path.exists(result["report_path"])
            # PDF with embedded image should be larger than bare minimum
            assert os.path.getsize(result["report_path"]) > 1000

            # Cleanup
            os.unlink(result["report_path"])
