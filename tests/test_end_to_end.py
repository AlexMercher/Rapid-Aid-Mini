"""
Test 8: End-to-End Pipeline
Verifies complete workflow from image to PDF report.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from unittest.mock import patch
from src.main import generate_accident_report
from src.config import REPORTS_DIR


@pytest.fixture
def sample_image():
    return Image.new('RGB', (400, 300), color=(200, 50, 50))


class TestEndToEnd:
    """Test suite for end-to-end pipeline validation."""

    def test_full_pipeline_with_mocked_model(self, sample_image):
        """Test the full pipeline with mocked Ollama responses."""
        detection = {
            "accident_detected": "Yes",
            "scene_description": "Car crash scene"
        }
        analysis = {
            "accident_type": "Rear-end collision",
            "number_of_victims": 2,
            "vehicles_involved": 3,
            "accident_severity": "Major",
            "injured_person_detected": "Yes",
            "emergency_services_present": "No",
            "road_blocked": "Yes",
            "scene_description": "Multiple vehicle collision on highway"
        }

        with patch('src.main.check_ollama_connection', return_value=True), \
             patch('src.main.check_model_available', return_value=True), \
             patch('src.main.detect_accident', return_value=detection), \
             patch('src.main.analyze_accident', return_value=analysis):

            result = generate_accident_report(image=sample_image)
            assert result["status"] == "report_generated"
            assert result["report_path"] is not None
            assert os.path.exists(result["report_path"])
            assert result["report_data"] is not None

            # Verify metadata
            assert result["report_data"]["accident_type"] == "Rear-end collision"
            assert result["report_data"]["date"] is not None
            assert result["report_data"]["time"] is not None
            assert "latitude" in result["report_data"]["gps"]

            # Cleanup
            if os.path.exists(result["report_path"]):
                os.unlink(result["report_path"])

    def test_pipeline_no_image_provided(self):
        """Test pipeline behavior when no image is provided."""
        result = generate_accident_report()
        assert result["status"] == "error"

    def test_pipeline_connection_failure(self, sample_image):
        """Test pipeline behavior when Ollama is unreachable."""
        with patch('src.main.check_ollama_connection', return_value=False):
            result = generate_accident_report(image=sample_image)
            assert result["status"] == "error"
            assert "Cannot connect" in result["message"]
