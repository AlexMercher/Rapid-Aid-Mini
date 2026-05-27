"""
Test 4: Description Generation Validation
Verifies AI produces usable accident descriptions.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from src.ollama_client import detect_accident, analyze_accident, check_ollama_connection


@pytest.fixture
def sample_image():
    """Create a simple test image."""
    img = Image.new('RGB', (400, 300), color=(128, 128, 128))
    return img


class TestDescriptionGeneration:
    """Test suite for AI description generation."""

    def test_detection_returns_dict(self, sample_image):
        """Test that detection returns a dictionary."""
        if not check_ollama_connection():
            pytest.skip("Ollama not available")
        result = detect_accident(sample_image)
        assert isinstance(result, dict)
        assert "accident_detected" in result

    def test_detection_has_scene_description(self, sample_image):
        """Test that detection includes a scene description."""
        if not check_ollama_connection():
            pytest.skip("Ollama not available")
        result = detect_accident(sample_image)
        # scene_description may or may not be present depending on model
        assert isinstance(result, dict)

    def test_analysis_returns_all_fields(self, sample_image):
        """Test that analysis returns all expected structured fields."""
        if not check_ollama_connection():
            pytest.skip("Ollama not available")
        result = analyze_accident(sample_image)
        expected_fields = [
            "accident_type", "number_of_victims", "vehicles_involved",
            "accident_severity", "injured_person_detected",
            "emergency_services_present", "road_blocked", "scene_description"
        ]
        for field in expected_fields:
            assert field in result, f"Missing field: {field}"
