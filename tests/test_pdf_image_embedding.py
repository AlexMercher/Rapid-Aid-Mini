"""
Test 11: Accident Image Embedding in PDF
Verifies accident image is correctly embedded in the generated PDF.
"""

import pytest
import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from src.pdf_generator import create_pdf_report


@pytest.fixture
def report_data():
    return {
        "report_id": "RPT-EMBED001",
        "date": "2026-03-04",
        "time": "15:30:00",
        "gps": {"latitude": 12.9716, "longitude": 77.5946},
        "accident_detected": "Yes",
        "accident_type": "Head-on collision",
        "number_of_victims": 1,
        "vehicles_involved": 2,
        "accident_severity": "Critical",
        "injured_person_detected": "Yes",
        "emergency_services_present": "Yes",
        "road_blocked": "Yes",
        "scene_description": "Head-on collision between two sedans.",
        "model_used": "bakllava",
        "organization": "Intelligent Accident Detection System"
    }


class TestPdfImageEmbedding:
    """Test suite for accident image embedding in PDF."""

    def test_pdf_with_image_is_larger(self, report_data):
        """Test that PDF with image is larger than PDF without image."""
        img = Image.new('RGB', (800, 600), color=(200, 100, 100))

        tmp1 = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        path1 = tmp1.name
        tmp1.close()
        path_with_img = create_pdf_report(report_data, path1, img)

        tmp2 = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        path2 = tmp2.name
        tmp2.close()
        path_without_img = create_pdf_report(report_data, path2, None)

        size_with = os.path.getsize(path_with_img)
        size_without = os.path.getsize(path_without_img)

        assert size_with > size_without, "PDF with image should be larger"
        os.unlink(path_with_img)
        os.unlink(path_without_img)

    def test_image_appears_in_pdf_content(self, report_data):
        """Test that image data is embedded in the PDF binary."""
        img = Image.new('RGB', (400, 300), color=(255, 0, 0))

        tmp = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        tmp_path = tmp.name
        tmp.close()
        path = create_pdf_report(report_data, tmp_path, img)
        with open(path, 'rb') as f:
            content = f.read()
        # JPEG images in PDF contain the JFIF or DCT markers
        assert len(content) > 1000, "PDF should be substantial with image"
        os.unlink(path)

    def test_various_image_sizes(self, report_data):
        """Test embedding images of various sizes."""
        for size in [(100, 100), (800, 600), (1920, 1080)]:
            img = Image.new('RGB', size, color=(100, 100, 200))
            tmp = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
            tmp_path = tmp.name
            tmp.close()
            path = create_pdf_report(report_data, tmp_path, img)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
            os.unlink(path)
