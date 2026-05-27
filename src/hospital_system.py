"""
Hospital Notification System for Accident Report Forwarding.

When an accident is detected, the system identifies the 3 nearest hospitals
within a 10km radius of the camera location and forwards the accident report.
Each hospital has a 30-minute countdown to accept/reject the report.
If a hospital accepts, its ambulance slot is marked as "occupied".
"""

import math
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Fake Hospital Registry (Bangalore area, within ~10km of base coords) ────
# Base location: 12.9716, 77.5946 (Bangalore city center)

HOSPITAL_REGISTRY = [
    {
        "id": "HOSP-001",
        "name": "Rajiv Gandhi Institute of Public Health",
        "address": "Jayanagar 4th Block, Bangalore",
        "latitude": 12.9250,
        "longitude": 77.5938,
        "distance_km": 0.0,  # will be computed dynamically
        "capacity": 250,
        "available_ambulances": 4,
        "phone": "+91-80-2653-4100",
        "specialties": ["Trauma Care", "Emergency Medicine", "Orthopaedics"],
        "status": "available",
    },
    {
        "id": "HOSP-002",
        "name": "Smt. Lakshmi Devi Memorial Hospital",
        "address": "Koramangala 5th Block, Bangalore",
        "latitude": 12.9352,
        "longitude": 77.6245,
        "distance_km": 0.0,
        "capacity": 180,
        "available_ambulances": 3,
        "phone": "+91-80-2553-7800",
        "specialties": ["Trauma Care", "Neurosurgery", "General Surgery"],
        "status": "available",
    },
    {
        "id": "HOSP-003",
        "name": "Shri Venkateshwara Emergency Hospital",
        "address": "Indiranagar 100 Feet Road, Bangalore",
        "latitude": 12.9784,
        "longitude": 77.6408,
        "distance_km": 0.0,
        "capacity": 200,
        "available_ambulances": 5,
        "phone": "+91-80-2527-1200",
        "specialties": ["Emergency Medicine", "Burns Unit", "ICU"],
        "status": "available",
    },
    {
        "id": "HOSP-004",
        "name": "Dr. B.R. Ambedkar Multispeciality Hospital",
        "address": "MG Road, Bangalore",
        "latitude": 12.9758,
        "longitude": 77.6033,
        "distance_km": 0.0,
        "capacity": 320,
        "available_ambulances": 6,
        "phone": "+91-80-2559-3000",
        "specialties": ["Trauma Care", "Cardiology", "Orthopaedics", "Neurosurgery"],
        "status": "available",
    },
    {
        "id": "HOSP-005",
        "name": "Sanjay Nagar Government General Hospital",
        "address": "Rajajinagar, Bangalore",
        "latitude": 12.9900,
        "longitude": 77.5520,
        "distance_km": 0.0,
        "capacity": 150,
        "available_ambulances": 2,
        "phone": "+91-80-2332-1500",
        "specialties": ["General Surgery", "Emergency Medicine"],
        "status": "available",
    },
    {
        "id": "HOSP-006",
        "name": "Padmashree Dr. Mohan Rao Trauma Centre",
        "address": "Electronic City Phase 1, Bangalore",
        "latitude": 12.8456,
        "longitude": 77.6603,
        "distance_km": 0.0,
        "capacity": 120,
        "available_ambulances": 3,
        "phone": "+91-80-2852-9600",
        "specialties": ["Trauma Care", "Intensive Care", "Burn Care"],
        "status": "available",
    },
    {
        "id": "HOSP-007",
        "name": "Kalyani Srinivasa Rao District Hospital",
        "address": "Whitefield Main Road, Bangalore",
        "latitude": 12.9698,
        "longitude": 77.7500,
        "distance_km": 0.0,
        "capacity": 200,
        "available_ambulances": 4,
        "phone": "+91-80-2845-2200",
        "specialties": ["Emergency Medicine", "Orthopaedics", "Paediatrics"],
        "status": "available",
    },
    {
        "id": "HOSP-008",
        "name": "Smt. Parvathi Devi Charitable Hospital",
        "address": "Basavanagudi, Bangalore",
        "latitude": 12.9430,
        "longitude": 77.5730,
        "distance_km": 0.0,
        "capacity": 100,
        "available_ambulances": 2,
        "phone": "+91-80-2661-4400",
        "specialties": ["General Surgery", "Emergency Medicine", "Paediatrics"],
        "status": "available",
    },
]

# Countdown duration for hospital response
RESPONSE_TIMEOUT_MINUTES = 30


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two GPS points in km."""
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


class HospitalNotificationManager:
    """Manages accident report forwarding to nearby hospitals."""

    def __init__(self):
        # Active notifications: report_id -> notification_data
        self._notifications: dict = {}
        # Hospital status tracking: hospital_id -> status
        self._hospital_status: dict = {
            h["id"]: h["status"] for h in HOSPITAL_REGISTRY
        }

    def get_nearby_hospitals(self, lat: float, lon: float,
                             radius_km: float = 10.0, count: int = 3) -> list[dict]:
        """
        Find the nearest hospitals within a radius of the accident location.

        Args:
            lat: Accident latitude
            lon: Accident longitude
            radius_km: Search radius in km
            count: Number of hospitals to return

        Returns:
            List of hospital dicts sorted by distance, limited to `count`
        """
        hospitals_with_dist = []
        for hospital in HOSPITAL_REGISTRY:
            dist = _haversine_km(lat, lon, hospital["latitude"], hospital["longitude"])
            if dist <= radius_km:
                h_copy = dict(hospital)
                h_copy["distance_km"] = round(dist, 2)
                h_copy["status"] = self._hospital_status.get(h_copy["id"], "available")
                hospitals_with_dist.append(h_copy)

        hospitals_with_dist.sort(key=lambda h: h["distance_km"])
        result = hospitals_with_dist[:count]

        logger.info(
            f"Found {len(result)} hospitals within {radius_km}km of "
            f"({lat}, {lon}): {[h['name'] for h in result]}"
        )
        return result

    def forward_report(self, report_data: dict,
                       hospitals: list[dict] | None = None) -> dict:
        """
        Forward an accident report to the nearest hospitals.

        Args:
            report_data: The accident report data dict (from create_report_structure)
            hospitals: Optional list of hospitals to forward to. If None,
                      auto-discovers 3 nearest hospitals from the report GPS.

        Returns:
            Notification dict with report_id, hospitals, timers, and status
        """
        report_id = report_data.get("report_id", f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}")

        # Auto-discover hospitals if not provided
        if hospitals is None:
            gps = report_data.get("gps", {})
            lat = gps.get("latitude", 12.9716)
            lon = gps.get("longitude", 77.5946)
            hospitals = self.get_nearby_hospitals(lat, lon)

        forwarded_at = datetime.now()
        deadline = forwarded_at + timedelta(minutes=RESPONSE_TIMEOUT_MINUTES)

        # Build notification
        hospital_notifications = []
        for h in hospitals:
            hospital_notifications.append({
                "hospital_id": h["id"],
                "hospital_name": h["name"],
                "hospital_address": h["address"],
                "hospital_phone": h["phone"],
                "distance_km": h.get("distance_km", 0),
                "status": "pending",  # pending | accepted | rejected | expired
                "forwarded_at": forwarded_at.isoformat(),
                "deadline": deadline.isoformat(),
            })

        notification = {
            "report_id": report_id,
            "accident_type": report_data.get("accident_type", "Unknown"),
            "severity": report_data.get("accident_severity", "Unknown"),
            "victims": report_data.get("number_of_victims", 0),
            "vehicles": report_data.get("vehicles_involved", 0),
            "location": report_data.get("gps", {}),
            "scene_description": report_data.get("scene_description", ""),
            "report_pdf_path": report_data.get("report_pdf_path", ""),
            "forwarded_at": forwarded_at.isoformat(),
            "deadline": deadline.isoformat(),
            "hospitals": hospital_notifications,
            "overall_status": "pending",  # pending | accepted | expired
        }

        self._notifications[report_id] = notification

        logger.info(
            f"Report {report_id} forwarded to {len(hospital_notifications)} hospitals: "
            f"{[h['hospital_name'] for h in hospital_notifications]}"
        )
        return notification

    def accept_report(self, hospital_id: str, report_id: str) -> dict:
        """
        A hospital accepts the accident report. Marks the hospital as occupied.

        Args:
            hospital_id: ID of the accepting hospital
            report_id: ID of the accident report

        Returns:
            Updated notification dict

        Raises:
            ValueError: If report_id or hospital_id not found
        """
        if report_id not in self._notifications:
            raise ValueError(f"Report '{report_id}' not found in notifications")

        notification = self._notifications[report_id]
        hospital_found = False

        for h in notification["hospitals"]:
            if h["hospital_id"] == hospital_id:
                h["status"] = "accepted"
                h["accepted_at"] = datetime.now().isoformat()
                hospital_found = True

                # Mark hospital as occupied
                self._hospital_status[hospital_id] = "occupied"
                logger.info(
                    f"Hospital {h['hospital_name']} ({hospital_id}) "
                    f"accepted report {report_id} — status: OCCUPIED"
                )
                break

        if not hospital_found:
            raise ValueError(
                f"Hospital '{hospital_id}' not found in notification for report '{report_id}'"
            )

        # Update overall status
        notification["overall_status"] = "accepted"
        return notification

    def reject_report(self, hospital_id: str, report_id: str) -> dict:
        """
        A hospital rejects the accident report.

        Args:
            hospital_id: ID of the rejecting hospital
            report_id: ID of the accident report

        Returns:
            Updated notification dict
        """
        if report_id not in self._notifications:
            raise ValueError(f"Report '{report_id}' not found in notifications")

        notification = self._notifications[report_id]

        for h in notification["hospitals"]:
            if h["hospital_id"] == hospital_id:
                h["status"] = "rejected"
                h["rejected_at"] = datetime.now().isoformat()
                logger.info(
                    f"Hospital {h['hospital_name']} ({hospital_id}) "
                    f"rejected report {report_id}"
                )
                break

        return notification

    def get_report_status(self, report_id: str) -> Optional[dict]:
        """Get the current notification status for a report."""
        notification = self._notifications.get(report_id)
        if notification is None:
            return None

        # Compute remaining time for each hospital
        now = datetime.now()
        for h in notification["hospitals"]:
            if h["status"] == "pending":
                deadline = datetime.fromisoformat(h["deadline"])
                remaining = deadline - now
                if remaining.total_seconds() <= 0:
                    h["status"] = "expired"
                    h["remaining_seconds"] = 0
                    h["remaining_display"] = "00:00"
                else:
                    h["remaining_seconds"] = int(remaining.total_seconds())
                    mins = int(remaining.total_seconds() // 60)
                    secs = int(remaining.total_seconds() % 60)
                    h["remaining_display"] = f"{mins:02d}:{secs:02d}"
            elif h["status"] == "accepted":
                h["remaining_seconds"] = 0
                h["remaining_display"] = "ACCEPTED"
            elif h["status"] == "rejected":
                h["remaining_seconds"] = 0
                h["remaining_display"] = "REJECTED"
            else:
                h["remaining_seconds"] = 0
                h["remaining_display"] = "EXPIRED"

        # Check if all expired
        all_expired = all(
            h["status"] in ("expired", "rejected")
            for h in notification["hospitals"]
        )
        if all_expired and notification["overall_status"] == "pending":
            notification["overall_status"] = "expired"

        return notification

    def get_all_notifications(self) -> list[dict]:
        """Get all active notifications with computed timers."""
        result = []
        for report_id in self._notifications:
            status = self.get_report_status(report_id)
            if status:
                result.append(status)
        return result

    def get_hospital_status(self, hospital_id: str) -> str:
        """Get the current status of a hospital (available/occupied)."""
        return self._hospital_status.get(hospital_id, "unknown")

    def get_all_hospitals(self) -> list[dict]:
        """Get all hospitals with their current status."""
        result = []
        for h in HOSPITAL_REGISTRY:
            h_copy = dict(h)
            h_copy["status"] = self._hospital_status.get(h["id"], "available")
            result.append(h_copy)
        return result

    def reset_hospital(self, hospital_id: str) -> None:
        """Reset a hospital status back to available."""
        self._hospital_status[hospital_id] = "available"
        logger.info(f"Hospital {hospital_id} reset to available")
