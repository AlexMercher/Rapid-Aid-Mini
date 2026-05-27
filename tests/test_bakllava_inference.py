"""
Test 2: bakllava Inference
Verifies the vision model can process images and return responses.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from src.ollama_client import _send_vision_request, check_ollama_connection


@pytest.fixture
def sample_image():
    """Create a simple test image."""
    img = Image.new('RGB', (200, 200), color=(255, 0, 0))
    return img


class TestBakllavaInference:
    """Test suite for bakllava model inference."""

    def test_model_responds_to_image(self, sample_image):
        """Test that the model returns a response for an image."""
        if not check_ollama_connection():
            pytest.skip("Ollama not available")
        response = _send_vision_request(sample_image, "Describe this image briefly.")
        assert response is not None
        assert len(response) > 0

    def test_response_is_string(self, sample_image):
        """Test that the model response is a valid string."""
        if not check_ollama_connection():
            pytest.skip("Ollama not available")
        response = _send_vision_request(sample_image, "What do you see?")
        assert isinstance(response, str)
        assert len(response.strip()) > 0
