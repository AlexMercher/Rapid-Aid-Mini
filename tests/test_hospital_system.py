#!/usr/bin/env python
"""
Terminal Test: Hospital Notification System

Run:  python tests/test_hospital_system.py
Tests the hospital system independently before Streamlit integration.
"""

import os
import sys
import io
import time

# Force UTF-8 stdout on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.hospital_system import HospitalNotificationManager, HOSPITAL_REGISTRY, _haversine_km


def print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_pass(test_name: str, detail: str = ""):
    print(f"  [PASS] {test_name}" + (f" -- {detail}" if detail else ""))


def print_fail(test_name: str, detail: str = ""):
    print(f"  [FAIL] {test_name}" + (f" -- {detail}" if detail else ""))


def test_hospital_registry():
    """Test that the hospital registry has valid data."""
    print_header("Test 1: Hospital Registry Validation")
    passed = True

    if len(HOSPITAL_REGISTRY) < 5:
        print_fail("Registry size", f"Expected >= 5 hospitals, got {len(HOSPITAL_REGISTRY)}")
        passed = False
    else:
        print_pass("Registry size", f"{len(HOSPITAL_REGISTRY)} hospitals registered")

    # Check all required fields
    required_fields = ["id", "name", "address", "latitude", "longitude", "capacity", "phone", "specialties"]
    for h in HOSPITAL_REGISTRY:
        for field in required_fields:
            if field not in h:
                print_fail(f"Hospital {h.get('id', '?')}", f"Missing field: {field}")
                passed = False

    if passed:
        print_pass("All hospitals have required fields")

    # Print hospital list
    print("\n  Registered Hospitals:")
    for h in HOSPITAL_REGISTRY:
        print(f"    {h['id']}: {h['name']} ({h['address']})")

    return passed


def test_haversine_distance():
    """Test the distance calculation function."""
    print_header("Test 2: Haversine Distance Calculation")
    passed = True

    # Bangalore center to MG Road (~0.5km)
    dist = _haversine_km(12.9716, 77.5946, 12.9758, 77.6033)
    if 0 < dist < 5:
        print_pass("Distance: Center to MG Road", f"{dist:.2f} km")
    else:
        print_fail("Distance: Center to MG Road", f"Got {dist:.2f} km, expected 0-5 km")
        passed = False

    # Same point should be ~0
    dist_same = _haversine_km(12.9716, 77.5946, 12.9716, 77.5946)
    if dist_same < 0.01:
        print_pass("Distance: Same point", f"{dist_same:.6f} km (≈ 0)")
    else:
        print_fail("Distance: Same point", f"Got {dist_same:.6f} km, expected ≈ 0")
        passed = False

    return passed


def test_nearby_hospitals():
    """Test finding 3 nearest hospitals."""
    print_header("Test 3: Find 3 Nearest Hospitals (within 10km)")
    passed = True

    manager = HospitalNotificationManager()
    # Use Bangalore center coordinates
    hospitals = manager.get_nearby_hospitals(12.9716, 77.5946, radius_km=10, count=3)

    if len(hospitals) == 3:
        print_pass("Found 3 hospitals", f"Got {len(hospitals)}")
    else:
        print_fail("Hospital count", f"Expected 3, got {len(hospitals)}")
        passed = False

    # Check distances are sorted
    distances = [h["distance_km"] for h in hospitals]
    if distances == sorted(distances):
        print_pass("Sorted by distance", f"Distances: {distances}")
    else:
        print_fail("Sort order", f"Distances not sorted: {distances}")
        passed = False

    # Check all within 10km
    for h in hospitals:
        if h["distance_km"] <= 10:
            print_pass(f"  {h['name']}", f"{h['distance_km']} km away")
        else:
            print_fail(f"  {h['name']}", f"{h['distance_km']} km (>10km!)")
            passed = False

    return passed


def test_forward_report():
    """Test forwarding a report to hospitals."""
    print_header("Test 4: Forward Report to Hospitals")
    passed = True

    manager = HospitalNotificationManager()

    # Simulate a report
    fake_report = {
        "report_id": "RPT-TEST-001",
        "accident_type": "Rear-end collision",
        "accident_severity": "Major",
        "number_of_victims": 2,
        "vehicles_involved": 3,
        "gps": {"latitude": 12.9716, "longitude": 77.5946},
        "scene_description": "A major rear-end collision at MG Road junction involving 3 vehicles.",
    }

    notification = manager.forward_report(fake_report)

    if notification["report_id"] == "RPT-TEST-001":
        print_pass("Report ID preserved", notification["report_id"])
    else:
        print_fail("Report ID", f"Expected RPT-TEST-001, got {notification['report_id']}")
        passed = False

    if len(notification["hospitals"]) == 3:
        print_pass("Forwarded to 3 hospitals")
    else:
        print_fail("Hospital count", f"Expected 3, got {len(notification['hospitals'])}")
        passed = False

    for h in notification["hospitals"]:
        if h["status"] == "pending":
            print_pass(f"  {h['hospital_name']}", f"Status: {h['status']}, Distance: {h['distance_km']}km")
        else:
            print_fail(f"  {h['hospital_name']}", f"Expected pending, got {h['status']}")
            passed = False

    if notification["overall_status"] == "pending":
        print_pass("Overall status: pending")
    else:
        print_fail("Overall status", f"Expected pending, got {notification['overall_status']}")
        passed = False

    # Check timer exists
    if notification.get("forwarded_at") and notification.get("deadline"):
        print_pass("Timer fields present", f"Forwarded: {notification['forwarded_at']}, Deadline: {notification['deadline']}")
    else:
        print_fail("Timer fields", "Missing forwarded_at or deadline")
        passed = False

    return passed


def test_countdown_timer():
    """Test that 30-minute countdown works correctly."""
    print_header("Test 5: 30-Minute Countdown Timer")
    passed = True

    manager = HospitalNotificationManager()
    fake_report = {
        "report_id": "RPT-TEST-TIMER",
        "accident_type": "Head-on collision",
        "accident_severity": "Critical",
        "number_of_victims": 3,
        "vehicles_involved": 2,
        "gps": {"latitude": 12.9716, "longitude": 77.5946},
        "scene_description": "Critical head-on collision.",
    }

    manager.forward_report(fake_report)
    status = manager.get_report_status("RPT-TEST-TIMER")

    if status is None:
        print_fail("Status retrieval", "Got None")
        return False

    for h in status["hospitals"]:
        remaining = h.get("remaining_seconds", 0)
        display = h.get("remaining_display", "??:??")

        # Should be close to 30 minutes (1800 seconds), within a few seconds tolerance
        if 1790 <= remaining <= 1800:
            print_pass(f"  {h['hospital_name']}", f"Timer: {display} ({remaining}s remaining)")
        else:
            print_fail(f"  {h['hospital_name']}", f"Expected ~1800s, got {remaining}s ({display})")
            passed = False

    return passed


def test_accept_report():
    """Test that accepting a report marks the hospital as occupied."""
    print_header("Test 6: Hospital Accepts Report → Status: Occupied")
    passed = True

    manager = HospitalNotificationManager()
    fake_report = {
        "report_id": "RPT-TEST-ACCEPT",
        "accident_type": "Side-impact collision",
        "accident_severity": "Major",
        "number_of_victims": 1,
        "vehicles_involved": 2,
        "gps": {"latitude": 12.9716, "longitude": 77.5946},
        "scene_description": "Side-impact collision at junction.",
    }

    notification = manager.forward_report(fake_report)
    accepting_hospital = notification["hospitals"][0]
    hospital_id = accepting_hospital["hospital_id"]
    hospital_name = accepting_hospital["hospital_name"]

    # Before accepting
    status_before = manager.get_hospital_status(hospital_id)
    if status_before == "available":
        print_pass("Before accept", f"{hospital_name} status: {status_before}")
    else:
        print_fail("Before accept", f"Expected 'available', got '{status_before}'")
        passed = False

    # Accept the report
    updated = manager.accept_report(hospital_id, "RPT-TEST-ACCEPT")

    # Check hospital is now occupied
    status_after = manager.get_hospital_status(hospital_id)
    if status_after == "occupied":
        print_pass("After accept", f"{hospital_name} status: {status_after} ✓")
    else:
        print_fail("After accept", f"Expected 'occupied', got '{status_after}'")
        passed = False

    # Check notification updated
    for h in updated["hospitals"]:
        if h["hospital_id"] == hospital_id:
            if h["status"] == "accepted":
                print_pass("Notification status", f"{hospital_name}: {h['status']}")
            else:
                print_fail("Notification status", f"Expected 'accepted', got '{h['status']}'")
                passed = False
            break

    if updated["overall_status"] == "accepted":
        print_pass("Overall status: accepted")
    else:
        print_fail("Overall status", f"Expected 'accepted', got '{updated['overall_status']}'")
        passed = False

    return passed


def test_reject_report():
    """Test that rejecting a report keeps the hospital available."""
    print_header("Test 7: Hospital Rejects Report")
    passed = True

    manager = HospitalNotificationManager()
    fake_report = {
        "report_id": "RPT-TEST-REJECT",
        "accident_type": "Two-wheeler collision",
        "accident_severity": "Minor",
        "number_of_victims": 1,
        "vehicles_involved": 2,
        "gps": {"latitude": 12.9716, "longitude": 77.5946},
        "scene_description": "Minor two-wheeler collision.",
    }

    notification = manager.forward_report(fake_report)
    rejecting_hospital = notification["hospitals"][1]
    hospital_id = rejecting_hospital["hospital_id"]
    hospital_name = rejecting_hospital["hospital_name"]

    # Reject the report
    updated = manager.reject_report(hospital_id, "RPT-TEST-REJECT")

    # Hospital should still be available
    status = manager.get_hospital_status(hospital_id)
    if status == "available":
        print_pass("After reject", f"{hospital_name} still available")
    else:
        print_fail("After reject", f"Expected 'available', got '{status}'")
        passed = False

    # Notification should show rejected
    for h in updated["hospitals"]:
        if h["hospital_id"] == hospital_id:
            if h["status"] == "rejected":
                print_pass("Notification status", f"{hospital_name}: {h['status']}")
            else:
                print_fail("Notification status", f"Expected 'rejected', got '{h['status']}'")
                passed = False
            break

    return passed


def test_get_all_notifications():
    """Test retrieving all active notifications."""
    print_header("Test 8: Get All Notifications")
    passed = True

    manager = HospitalNotificationManager()

    # Forward two reports
    for i in range(2):
        manager.forward_report({
            "report_id": f"RPT-ALL-{i+1}",
            "accident_type": "Rear-end collision",
            "accident_severity": "Minor",
            "number_of_victims": 1,
            "vehicles_involved": 2,
            "gps": {"latitude": 12.9716, "longitude": 77.5946},
            "scene_description": f"Test report {i+1}.",
        })

    all_notifs = manager.get_all_notifications()
    if len(all_notifs) == 2:
        print_pass("Notification count", f"Got {len(all_notifs)} notifications")
    else:
        print_fail("Notification count", f"Expected 2, got {len(all_notifs)}")
        passed = False

    for n in all_notifs:
        print_pass(f"  Report {n['report_id']}", f"{len(n['hospitals'])} hospitals, status: {n['overall_status']}")

    return passed


def main():
    print("\n" + "=" * 70)
    print("  HOSPITAL NOTIFICATION SYSTEM -- TERMINAL TESTS")
    print("=" * 70)

    tests = [
        ("Hospital Registry", test_hospital_registry),
        ("Haversine Distance", test_haversine_distance),
        ("Nearby Hospitals", test_nearby_hospitals),
        ("Forward Report", test_forward_report),
        ("Countdown Timer", test_countdown_timer),
        ("Accept Report", test_accept_report),
        ("Reject Report", test_reject_report),
        ("All Notifications", test_get_all_notifications),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print_fail(name, f"Exception: {e}")
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

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
