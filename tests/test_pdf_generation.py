"""
Test 6: PDF Generation
Verifies PDF creation works correctly.
"""

import pytest
import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from src.pdf_generator import create_pdf_report


@pytest.fixture
def sample_report_data():
    return {
        "report_id": "RPT-TEST001",
        "date": "2026-03-04",
        "time": "15:30:00",
        "gps": {"latitude": 12.9716, "longitude": 77.5946},
        "accident_detected": "Yes",
        "accident_type": "Rear-end collision",
        "number_of_victims": 2,
        "vehicles_involved": 3,
        "accident_severity": "Major",
        "injured_person_detected": "Yes",
        "emergency_services_present": "No",
        "road_blocked": "Yes",
        "scene_description": "A rear-end collision involving three vehicles on a highway.",
        "model_used": "bakllava",
        "organization": "Intelligent Accident Detection System"
    }


@pytest.fixture
def sample_image():
    return Image.new('RGB', (400, 300), color=(255, 100, 100))


class TestPdfGeneration:
    """Test suite for PDF generation."""

    def test_pdf_file_created(self, sample_report_data, sample_image):
        """Test that a PDF file is created."""
        tmp = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        tmp_path = tmp.name
        tmp.close()
        path = create_pdf_report(sample_report_data, tmp_path, sample_image)
        assert os.path.exists(path)
        os.unlink(path)

    def test_pdf_not_empty(self, sample_report_data, sample_image):
        """Test that the generated PDF is not empty."""
        tmp = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        tmp_path = tmp.name
        tmp.close()
        path = create_pdf_report(sample_report_data, tmp_path, sample_image)
        assert os.path.getsize(path) > 0
        os.unlink(path)

    def test_pdf_without_image(self, sample_report_data):
        """Test that PDF can be created without an image."""
        tmp = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        tmp_path = tmp.name
        tmp.close()
        path = create_pdf_report(sample_report_data, tmp_path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
        os.unlink(path)

    def test_pdf_is_valid(self, sample_report_data, sample_image):
        """Test that the PDF starts with the correct header."""
        tmp = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        tmp_path = tmp.name
        tmp.close()
        path = create_pdf_report(sample_report_data, tmp_path, sample_image)
        with open(path, 'rb') as f:
            header = f.read(5)
        assert header == b'%PDF-'
        os.unlink(path)
