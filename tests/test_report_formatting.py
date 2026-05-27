"""
Test 5: Report Formatting Validation
Verifies report structure is correct and contains all required fields.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.report_generator import (
    create_report_structure, format_accident_report,
    get_current_datetime, parse_structured_fields
)


@pytest.fixture
def sample_detection():
    return {"accident_detected": "Yes", "scene_description": "A car crash at an intersection."}


@pytest.fixture
def sample_analysis():
    return {
        "accident_type": "Rear-end collision",
        "number_of_victims": 2,
        "vehicles_involved": 3,
        "accident_severity": "Major",
        "injured_person_detected": "Yes",
        "emergency_services_present": "No",
        "road_blocked": "Yes",
        "scene_description": "Two cars collided at an intersection causing significant damage."
    }


class TestReportFormatting:
    """Test suite for report formatting validation."""

    def test_datetime_format(self):
        """Test that date and time are properly formatted."""
        dt = get_current_datetime()
        assert "date" in dt
        assert "time" in dt
        assert len(dt["date"]) == 10  # YYYY-MM-DD
        assert len(dt["time"]) == 8   # HH:MM:SS

    def test_report_structure_creation(self, sample_detection, sample_analysis):
        """Test that report structure contains all required fields."""
        report = create_report_structure(sample_detection, sample_analysis)
        required_fields = [
            "report_id", "date", "time", "gps", "accident_detected",
            "accident_type", "number_of_victims", "vehicles_involved",
            "accident_severity", "injured_person_detected",
            "emergency_services_present", "road_blocked", "scene_description",
            "model_used", "organization"
        ]
        for field in required_fields:
            assert field in report, f"Missing field: {field}"

    def test_gps_coordinates_present(self, sample_detection, sample_analysis):
        """Test that GPS coordinates are included."""
        report = create_report_structure(sample_detection, sample_analysis)
        assert "latitude" in report["gps"]
        assert "longitude" in report["gps"]
        assert isinstance(report["gps"]["latitude"], float)
        assert isinstance(report["gps"]["longitude"], float)

    def test_formatted_report_contains_sections(self, sample_detection, sample_analysis):
        """Test that formatted report string has all sections."""
        report = create_report_structure(sample_detection, sample_analysis)
        formatted = format_accident_report(report)
        assert "ACCIDENT REPORT" in formatted
        assert "ACCIDENT ANALYSIS" in formatted
        assert "INCIDENT DESCRIPTION" in formatted
        assert "METADATA" in formatted

    def test_formatted_report_contains_data(self, sample_detection, sample_analysis):
        """Test that formatted report contains actual data values."""
        report = create_report_structure(sample_detection, sample_analysis)
        formatted = format_accident_report(report)
        assert "Rear-end collision" in formatted
        assert "Major" in formatted
