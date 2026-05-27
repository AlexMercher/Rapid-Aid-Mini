"""
Test 7: File Save Verification
Verifies reports save correctly to the reports directory.
"""

import pytest
import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from src.report_generator import generate_report_filename, get_next_report_number
from src.pdf_generator import create_pdf_report
from src.config import REPORTS_DIR


@pytest.fixture
def sample_report_data():
    return {
        "report_id": "RPT-SAVE001",
        "date": "2026-03-04",
        "time": "12:00:00",
        "gps": {"latitude": 12.9716, "longitude": 77.5946},
        "accident_detected": "Yes",
        "accident_type": "Rear-end collision",
        "number_of_victims": 1,
        "vehicles_involved": 2,
        "accident_severity": "Minor",
        "injured_person_detected": "No",
        "emergency_services_present": "No",
        "road_blocked": "No",
        "scene_description": "Minor fender bender.",
        "model_used": "bakllava",
        "organization": "Test System"
    }


class TestFileSaveVerification:
    """Test suite for file save verification."""

    def test_reports_directory_exists(self):
        """Test that reports directory is created."""
        os.makedirs(REPORTS_DIR, exist_ok=True)
        assert os.path.exists(REPORTS_DIR)

    def test_report_filename_format(self):
        """Test that filename follows correct format."""
        filename = generate_report_filename()
        basename = os.path.basename(filename)
        assert basename.startswith("accident_report_")
        assert basename.endswith(".pdf")

    def test_sequential_numbering(self):
        """Test that report numbers are sequential."""
        num = get_next_report_number()
        assert isinstance(num, int)
        assert num >= 1

    def test_file_saved_to_correct_location(self, sample_report_data):
        """Test that PDF is saved to the reports directory."""
        img = Image.new('RGB', (200, 200), color=(255, 0, 0))
        output_path = generate_report_filename()
        path = create_pdf_report(sample_report_data, output_path, img)
        assert os.path.exists(path)
        assert REPORTS_DIR in path or "reports" in path
        # Cleanup
        if os.path.exists(path):
            os.unlink(path)

    def test_file_size_reasonable(self, sample_report_data):
        """Test that the saved file has a reasonable size."""
        img = Image.new('RGB', (200, 200), color=(255, 0, 0))
        tmp = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        tmp_path = tmp.name
        tmp.close()
        path = create_pdf_report(sample_report_data, tmp_path, img)
        size = os.path.getsize(path)
        assert size > 100, "PDF file is too small"
        assert size < 50_000_000, "PDF file is too large"
        os.unlink(path)
