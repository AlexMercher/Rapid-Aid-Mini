#!/usr/bin/env python
"""
Terminal Test: Police FIR Database System

Run:  python tests/test_police_database.py
Tests the police database independently before Streamlit integration.
"""

import os
import sys
import io
import tempfile

# Force UTF-8 stdout on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.police_database import PoliceDatabase


def print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_pass(test_name: str, detail: str = ""):
    print(f"  [PASS] {test_name}" + (f" -- {detail}" if detail else ""))


def print_fail(test_name: str, detail: str = ""):
    print(f"  [FAIL] {test_name}" + (f" -- {detail}" if detail else ""))


# Use a temporary DB for testing so we don't pollute production
TEST_DB_PATH = os.path.join(PROJECT_ROOT, "data", "test_accident_reports.db")


def cleanup():
    """Remove test database if it exists."""
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


def test_database_creation():
    """Test that the database and table are created successfully."""
    print_header("Test 1: Database Creation")
    passed = True

    cleanup()
    db = PoliceDatabase(db_path=TEST_DB_PATH)

    if os.path.exists(TEST_DB_PATH):
        print_pass("Database file created", TEST_DB_PATH)
    else:
        print_fail("Database file", "File not created")
        passed = False

    count = db.get_report_count()
    if count == 0:
        print_pass("Empty database", f"{count} records")
    else:
        print_fail("Initial count", f"Expected 0, got {count}")
        passed = False

    return passed


def test_seed_fake_data():
    """Test seeding with 25 fake reports."""
    print_header("Test 2: Seed 25 Fake Accident Reports")
    passed = True

    cleanup()
    db = PoliceDatabase(db_path=TEST_DB_PATH)
    inserted = db.seed_fake_data(n=25)

    if inserted == 25:
        print_pass("Inserted count", f"{inserted} reports seeded")
    else:
        print_fail("Inserted count", f"Expected 25, got {inserted}")
        passed = False

    total = db.get_report_count()
    if total == 25:
        print_pass("Total count", f"{total} reports in database")
    else:
        print_fail("Total count", f"Expected 25, got {total}")
        passed = False

    # Check that re-seeding doesn't duplicate
    inserted_again = db.seed_fake_data(n=25)
    if inserted_again == 0:
        print_pass("No duplicates on re-seed", f"Inserted {inserted_again} (correct)")
    else:
        print_fail("Re-seed", f"Should insert 0, got {inserted_again}")
        passed = False

    # Print a sample report
    all_reports = db.get_all_reports(limit=3)
    print("\n  Sample seeded reports:")
    for r in all_reports:
        print(f"    {r['report_id']} | {r['date']} | {r['area']} | {r['accident_type']} | {r['severity']}")

    return passed


def test_search_by_date():
    """Test searching reports by date."""
    print_header("Test 3: Search by Date")
    passed = True

    db = PoliceDatabase(db_path=TEST_DB_PATH)

    # Get available dates
    dates = db.get_unique_dates()
    if len(dates) > 0:
        print_pass("Dates available", f"{len(dates)} unique dates")
    else:
        print_fail("No dates", "Database has no dates")
        return False

    # Search for a known date
    target_date = dates[0]
    results = db.search_by_date(target_date)

    if len(results) > 0:
        print_pass(f"Search date '{target_date}'", f"Found {len(results)} report(s)")
        for r in results:
            print(f"    → {r['report_id']} | {r['location_name']} | {r['severity']}")
    else:
        print_fail(f"Search date '{target_date}'", "No results found")
        passed = False

    # Search for a date that shouldn't exist
    results_none = db.search_by_date("1999-01-01")
    if len(results_none) == 0:
        print_pass("Search non-existent date", "Correctly returned 0 results")
    else:
        print_fail("Search non-existent date", f"Expected 0, got {len(results_none)}")
        passed = False

    return passed


def test_search_by_location():
    """Test searching reports by location name."""
    print_header("Test 4: Search by Location")
    passed = True

    db = PoliceDatabase(db_path=TEST_DB_PATH)

    # Get available areas
    areas = db.get_unique_areas()
    if len(areas) > 0:
        print_pass("Areas available", f"{len(areas)} unique areas: {', '.join(areas[:5])}...")
    else:
        print_fail("No areas", "Database has no areas")
        return False

    # Search for a known area
    target_area = areas[0]
    results = db.search_by_location(target_area)

    if len(results) > 0:
        print_pass(f"Search location '{target_area}'", f"Found {len(results)} report(s)")
        for r in results[:3]:
            print(f"    → {r['report_id']} | {r['date']} | {r['location_name']}")
    else:
        print_fail(f"Search location '{target_area}'", "No results found")
        passed = False

    # Search partial match
    results_partial = db.search_by_location("Bangalore")
    if len(results_partial) > 0:
        print_pass("Partial match 'Bangalore'", f"Found {len(results_partial)} report(s)")
    else:
        # City is stored as Bangalore, but location_name may have different format
        print_fail("Partial match 'Bangalore'", "No results (city field should match)")
        passed = False

    # Search no match
    results_no = db.search_by_location("Timbuktu")
    if len(results_no) == 0:
        print_pass("Search non-existent location", "Correctly returned 0 results")
    else:
        print_fail("Search non-existent location", f"Expected 0, got {len(results_no)}")
        passed = False

    return passed


def test_combined_search():
    """Test combined date + location search."""
    print_header("Test 5: Combined Date + Location Search")
    passed = True

    db = PoliceDatabase(db_path=TEST_DB_PATH)

    # Get a report to derive search params from
    all_rpts = db.get_all_reports(limit=5)
    if not all_rpts:
        print_fail("No reports", "Database empty")
        return False

    target = all_rpts[0]
    target_date = target["date"]
    target_area = target["area"]

    results = db.search_by_date_and_location(target_date, target_area)
    if len(results) > 0:
        print_pass(
            f"Search date='{target_date}' + location='{target_area}'",
            f"Found {len(results)} report(s)"
        )
        for r in results:
            print(f"    → {r['report_id']} | {r['location_name']} | {r['accident_type']}")
    else:
        print_fail("Combined search", "No results found for known date+location")
        passed = False

    # Non-matching combination
    results_none = db.search_by_date_and_location("1999-01-01", "Timbuktu")
    if len(results_none) == 0:
        print_pass("Non-matching combined search", "Correctly returned 0 results")
    else:
        print_fail("Non-matching combined", f"Expected 0, got {len(results_none)}")
        passed = False

    return passed


def test_insert_real_report():
    """Test inserting a real pipeline report into the database."""
    print_header("Test 6: Insert Real Pipeline Report")
    passed = True

    db = PoliceDatabase(db_path=TEST_DB_PATH)

    # Simulate a report from the accident detection pipeline
    pipeline_report = {
        "report_id": "RPT-20260416153000",
        "date": "2026-04-16",
        "time": "15:30:00",
        "gps": {"latitude": 12.9352, "longitude": 77.6245},
        "accident_type": "Head-on collision",
        "accident_severity": "Critical",
        "vehicles_involved": 2,
        "number_of_victims": 3,
        "injured_person_detected": "Yes",
        "road_blocked": "Yes",
        "scene_description": (
            "A critical head-on collision occurred at Koramangala 5th Block. "
            "Two vehicles, a white Maruti Suzuki Swift and a blue Tata Nexon, "
            "collided resulting in severe structural damage. Three victims were "
            "found — Rajesh Kumar, Priya Sharma, and Suresh Babu. Emergency "
            "services were dispatched immediately. The road was completely blocked."
        ),
        "location_name": "Koramangala 5th Block, 80 Feet Road",
        "area": "Koramangala",
    }

    report_id = db.insert_report(pipeline_report)
    if report_id == "RPT-20260416153000":
        print_pass("Insert returned correct ID", report_id)
    else:
        print_fail("Insert ID", f"Expected RPT-20260416153000, got {report_id}")
        passed = False

    # Verify it's searchable
    fetched = db.get_report_by_id("RPT-20260416153000")
    if fetched is not None:
        print_pass("Fetch by ID", f"Found report: {fetched['report_id']}")
        print(f"    Location: {fetched['location_name']}")
        print(f"    Severity: {fetched['severity']}")
        print(f"    Description: {fetched['description'][:100]}...")
    else:
        print_fail("Fetch by ID", "Report not found after insert")
        passed = False

    # Search by its date
    by_date = db.search_by_date("2026-04-16")
    found = any(r["report_id"] == "RPT-20260416153000" for r in by_date)
    if found:
        print_pass("Searchable by date", f"Found in date search for 2026-04-16")
    else:
        print_fail("Date search", "Inserted report not found by date")
        passed = False

    # Search by its location
    by_loc = db.search_by_location("Koramangala")
    found_loc = any(r["report_id"] == "RPT-20260416153000" for r in by_loc)
    if found_loc:
        print_pass("Searchable by location", f"Found in location search for 'Koramangala'")
    else:
        print_fail("Location search", "Inserted report not found by location")
        passed = False

    return passed


def test_report_structure():
    """Test that report records have the correct structure."""
    print_header("Test 7: Report Record Structure")
    passed = True

    db = PoliceDatabase(db_path=TEST_DB_PATH)
    all_rpts = db.get_all_reports(limit=1)

    if not all_rpts:
        print_fail("No reports", "Database empty")
        return False

    report = all_rpts[0]

    required_fields = [
        "id", "report_id", "date", "time", "location_name", "area",
        "city", "state", "accident_type", "severity", "vehicles_involved",
        "number_of_victims", "description", "created_at"
    ]

    for field in required_fields:
        if field in report:
            print_pass(f"Field '{field}'", f"Value: {str(report[field])[:60]}")
        else:
            print_fail(f"Field '{field}'", "Missing from report")
            passed = False

    # Check description is verbose (at least 100 chars)
    desc = report.get("description", "")
    if len(desc) >= 100:
        print_pass("Description verbosity", f"{len(desc)} chars")
    else:
        print_fail("Description verbosity", f"Only {len(desc)} chars (expected >= 100)")
        passed = False

    return passed


def test_unique_queries():
    """Test unique dates and areas helper functions."""
    print_header("Test 8: Unique Dates and Areas")
    passed = True

    db = PoliceDatabase(db_path=TEST_DB_PATH)

    dates = db.get_unique_dates()
    if len(dates) > 0:
        print_pass("Unique dates", f"{len(dates)} dates: {', '.join(dates[:5])}...")
    else:
        print_fail("Unique dates", "No dates returned")
        passed = False

    areas = db.get_unique_areas()
    if len(areas) > 0:
        print_pass("Unique areas", f"{len(areas)} areas: {', '.join(areas[:5])}...")
    else:
        print_fail("Unique areas", "No areas returned")
        passed = False

    return passed


def main():
    print("\n" + "=" * 70)
    print("  POLICE FIR DATABASE — TERMINAL TESTS")
    print("=" * 70)

    # Clean start
    cleanup()

    tests = [
        ("Database Creation", test_database_creation),
        ("Seed Fake Data", test_seed_fake_data),
        ("Search by Date", test_search_by_date),
        ("Search by Location", test_search_by_location),
        ("Combined Search", test_combined_search),
        ("Insert Real Report", test_insert_real_report),
        ("Report Structure", test_report_structure),
        ("Unique Queries", test_unique_queries),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print_fail(name, f"Exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print_header("TEST SUMMARY")
    total = len(results)
    passed = sum(1 for _, r in results if r)
    failed = total - passed

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status}: {name}")

    print(f"\n  Total: {total} | Passed: {passed} | Failed: {failed}")

    if failed == 0:
        print("\n  ALL TESTS PASSED!\n")
    else:
        print(f"\n  {failed} test(s) failed.\n")

    # Cleanup test database
    cleanup()

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
