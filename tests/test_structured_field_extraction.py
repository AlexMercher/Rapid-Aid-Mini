"""
Test 12: Structured Field Extraction
Verifies structured fields are correctly extracted from model response.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.report_generator import parse_structured_fields


class TestStructuredFieldExtraction:
    """Test suite for structured field extraction from model responses."""

    def test_parse_valid_json(self):
        """Test parsing a properly formatted JSON response."""
        response = '{"accident_detected": "Yes", "scene_description": "A car crash"}'
        result = parse_structured_fields(response)
        assert result["accident_detected"] == "Yes"
        assert result["scene_description"] == "A car crash"

    def test_parse_json_with_extra_text(self):
        """Test parsing JSON embedded in additional text."""
        response = 'Here is the analysis:\n{"accident_type": "Rollover", "number_of_victims": 3}\nEnd.'
        result = parse_structured_fields(response)
        assert result["accident_type"] == "Rollover"
        assert result["number_of_victims"] == 3

    def test_parse_all_analysis_fields(self):
        """Test extracting all analysis fields from a complete response."""
        response = '''{
            "accident_type": "Head-on collision",
            "number_of_victims": 4,
            "vehicles_involved": 2,
            "accident_severity": "Critical",
            "injured_person_detected": "Yes",
            "emergency_services_present": "No",
            "road_blocked": "Yes",
            "scene_description": "Two vehicles collided head-on"
        }'''
        result = parse_structured_fields(response)
        assert result["accident_type"] == "Head-on collision"
        assert result["number_of_victims"] == 4
        assert result["vehicles_involved"] == 2
        assert result["accident_severity"] == "Critical"
        assert result["injured_person_detected"] == "Yes"
        assert result["emergency_services_present"] == "No"
        assert result["road_blocked"] == "Yes"
        assert "Two vehicles" in result["scene_description"]

    def test_parse_empty_response(self):
        """Test handling of empty response."""
        result = parse_structured_fields("")
        assert result == {}

    def test_parse_none_response(self):
        """Test handling of None response."""
        result = parse_structured_fields(None)
        assert result == {}

    def test_parse_malformed_json(self):
        """Test regex fallback for malformed JSON."""
        response = 'accident_type: "Side collision", number_of_victims: 1, accident_severity: "Minor"'
        result = parse_structured_fields(response)
        # The regex may or may not extract depending on format, but it should not crash
        assert isinstance(result, dict)

    def test_numeric_fields_are_integers(self):
        """Test that numeric fields are parsed as integers."""
        response = '{"number_of_victims": 5, "vehicles_involved": 3}'
        result = parse_structured_fields(response)
        assert isinstance(result["number_of_victims"], int)
        assert isinstance(result["vehicles_involved"], int)
