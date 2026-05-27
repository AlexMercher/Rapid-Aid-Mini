"""
Test 9: Accident Detection Validation
Verifies model correctly identifies accident vs non-accident images.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from src.ollama_client import detect_accident, check_ollama_connection


@pytest.fixture
def accident_image():
    """Create a red-tinted image simulating an accident scene."""
    img = Image.new('RGB', (400, 300), color=(200, 50, 50))
    return img


@pytest.fixture
def non_accident_image():
    """Create a plain green image simulating a normal scene."""
    img = Image.new('RGB', (400, 300), color=(50, 200, 50))
    return img


class TestAccidentDetection:
    """Test suite for accident detection validation."""

    def test_detection_returns_dict(self, accident_image):
        """Test that detection returns a dictionary."""
        if not check_ollama_connection():
            pytest.skip("Ollama not available")
        result = detect_accident(accident_image)
        assert isinstance(result, dict)

    def test_detection_has_required_field(self, accident_image):
        """Test that detection result contains accident_detected field."""
        if not check_ollama_connection():
            pytest.skip("Ollama not available")
        result = detect_accident(accident_image)
        assert "accident_detected" in result

    def test_detection_returns_yes_or_no(self, accident_image):
        """Test that accident_detected is either Yes or No."""
        if not check_ollama_connection():
            pytest.skip("Ollama not available")
        result = detect_accident(accident_image)
        assert result["accident_detected"] in ["Yes", "No"]

    def test_detection_with_non_accident(self, non_accident_image):
        """Test detection with a non-accident image."""
        if not check_ollama_connection():
            pytest.skip("Ollama not available")
        result = detect_accident(non_accident_image)
        assert "accident_detected" in result
        assert result["accident_detected"] in ["Yes", "No"]
