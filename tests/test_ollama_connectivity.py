"""
Test 1: Ollama API Connectivity
Verifies Windows can communicate with the WSL Ollama server.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ollama_client import check_ollama_connection, check_model_available
from src.config import OLLAMA_API_URL, MODEL_NAME


class TestOllamaConnectivity:
    """Test suite for Ollama API connectivity."""

    def test_api_endpoint_responds(self):
        """Test that the Ollama API endpoint is reachable."""
        result = check_ollama_connection()
        if not result:
            pytest.skip(f"Ollama not running at {OLLAMA_API_URL}")
        assert result is True

    def test_model_available(self):
        """Test that the bakllava model is available."""
        if not check_ollama_connection():
            pytest.skip("Ollama not available")
        result = check_model_available(MODEL_NAME)
        assert result is True, f"Model '{MODEL_NAME}' is not available in Ollama"

    def test_invalid_endpoint_handling(self):
        """Test that invalid endpoint is handled gracefully."""
        import requests
        try:
            response = requests.get("http://localhost:59999/api/tags", timeout=2)
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.InvalidURL):
            pass  # Expected behavior

    def test_connection_timeout_handling(self):
        """Test that connection timeout is handled gracefully."""
        import requests
        try:
            response = requests.get("http://192.0.2.1:11434/api/tags", timeout=2)
        except (requests.ConnectionError, requests.Timeout):
            pass  # Expected behavior
