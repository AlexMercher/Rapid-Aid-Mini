"""
Police FIR Database for Accident Report Storage and Search.

Provides a SQLite-based database for storing accident reports that can be
searched by date and location name (not GPS coordinates). Seeded with
realistic fake data using Indian/Bangalore-area locations and names.

This system allows police to search accident records for FIR filing
or for victims to retrieve their accident report.
"""

import os
import sqlite3
import logging
import random
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Database Path ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "accident_reports.db")

# ─── Fake Data Templates (All Indian / Bangalore area) ──────────────────────────

BANGALORE_LOCATIONS = [
    {"location_name": "MG Road Junction near Brigade Road", "area": "MG Road", "city": "Bangalore", "state": "Karnataka", "lat": 12.9758, "lon": 77.6033},
    {"location_name": "Silk Board Junction, Hosur Road", "area": "Silk Board", "city": "Bangalore", "state": "Karnataka", "lat": 12.9172, "lon": 77.6227},
    {"location_name": "Koramangala 5th Block, 80 Feet Road", "area": "Koramangala", "city": "Bangalore", "state": "Karnataka", "lat": 12.9352, "lon": 77.6245},
    {"location_name": "Indiranagar 100 Feet Road near Signal", "area": "Indiranagar", "city": "Bangalore", "state": "Karnataka", "lat": 12.9784, "lon": 77.6408},
    {"location_name": "Whitefield Main Road near ITPL", "area": "Whitefield", "city": "Bangalore", "state": "Karnataka", "lat": 12.9698, "lon": 77.7500},
    {"location_name": "Electronic City Phase 1, Hosur Road", "area": "Electronic City", "city": "Bangalore", "state": "Karnataka", "lat": 12.8456, "lon": 77.6603},
    {"location_name": "Hebbal Flyover Junction", "area": "Hebbal", "city": "Bangalore", "state": "Karnataka", "lat": 13.0358, "lon": 77.5970},
    {"location_name": "Jayanagar 4th Block, 11th Cross", "area": "Jayanagar", "city": "Bangalore", "state": "Karnataka", "lat": 12.9250, "lon": 77.5938},
    {"location_name": "Marathahalli Bridge, Outer Ring Road", "area": "Marathahalli", "city": "Bangalore", "state": "Karnataka", "lat": 12.9591, "lon": 77.6974},
    {"location_name": "Rajajinagar 1st Block, West of Chord Road", "area": "Rajajinagar", "city": "Bangalore", "state": "Karnataka", "lat": 12.9900, "lon": 77.5520},
    {"location_name": "KR Puram Railway Bridge", "area": "KR Puram", "city": "Bangalore", "state": "Karnataka", "lat": 12.9988, "lon": 77.7069},
    {"location_name": "Basavanagudi, Bull Temple Road", "area": "Basavanagudi", "city": "Bangalore", "state": "Karnataka", "lat": 12.9430, "lon": 77.5730},
    {"location_name": "Yeshwanthpur Circle, Tumkur Road", "area": "Yeshwanthpur", "city": "Bangalore", "state": "Karnataka", "lat": 13.0220, "lon": 77.5440},
    {"location_name": "Bannerghatta Road near Arekere Gate", "area": "Bannerghatta", "city": "Bangalore", "state": "Karnataka", "lat": 12.8860, "lon": 77.5970},
    {"location_name": "Yelahanka New Town, NH 44", "area": "Yelahanka", "city": "Bangalore", "state": "Karnataka", "lat": 13.1007, "lon": 77.5963},
    {"location_name": "Peenya Industrial Area, Ring Road", "area": "Peenya", "city": "Bangalore", "state": "Karnataka", "lat": 13.0300, "lon": 77.5180},
    {"location_name": "Nagarbhavi Circle, Mysore Road", "area": "Nagarbhavi", "city": "Bangalore", "state": "Karnataka", "lat": 12.9600, "lon": 77.5090},
    {"location_name": "HSR Layout Sector 2, 27th Main", "area": "HSR Layout", "city": "Bangalore", "state": "Karnataka", "lat": 12.9116, "lon": 77.6474},
    {"location_name": "BTM Layout 2nd Stage, Madiwala", "area": "BTM Layout", "city": "Bangalore", "state": "Karnataka", "lat": 12.9166, "lon": 77.6101},
    {"location_name": "Domlur Flyover, Old Airport Road", "area": "Domlur", "city": "Bangalore", "state": "Karnataka", "lat": 12.9610, "lon": 77.6387},
    {"location_name": "Malleshwaram 8th Cross, Sampige Road", "area": "Malleshwaram", "city": "Bangalore", "state": "Karnataka", "lat": 12.9969, "lon": 77.5700},
    {"location_name": "JP Nagar 6th Phase, NICE Road Junction", "area": "JP Nagar", "city": "Bangalore", "state": "Karnataka", "lat": 12.8950, "lon": 77.5850},
    {"location_name": "Sadashivanagar, Palace Road", "area": "Sadashivanagar", "city": "Bangalore", "state": "Karnataka", "lat": 12.9980, "lon": 77.5830},
    {"location_name": "Ulsoor Lake Road, near Halasuru", "area": "Ulsoor", "city": "Bangalore", "state": "Karnataka", "lat": 12.9810, "lon": 77.6200},
    {"location_name": "Kengeri Satellite Town, Mysore Road", "area": "Kengeri", "city": "Bangalore", "state": "Karnataka", "lat": 12.9097, "lon": 77.4878},
]

ACCIDENT_TYPES = [
    "Rear-end collision",
    "Head-on collision",
    "Side-impact collision",
    "Vehicle-pedestrian accident",
    "Vehicle-object collision",
    "Rollover accident",
    "Multi-vehicle pile-up",
    "Two-wheeler collision",
    "Auto-rickshaw collision",
    "Bus-car collision",
]

SEVERITIES = ["Minor", "Major", "Critical"]

INDIAN_VEHICLE_DESCRIPTIONS = [
    "a white Maruti Suzuki Swift",
    "a red Honda Activa scooter",
    "a black Hyundai Creta SUV",
    "a blue Tata Nexon compact SUV",
    "a silver Toyota Innova Crysta",
    "a grey Mahindra Thar",
    "a green BMTC city bus",
    "a yellow auto-rickshaw",
    "a dark grey Royal Enfield motorcycle",
    "a white Ashok Leyland truck",
    "a red Bajaj Pulsar motorcycle",
    "a maroon Kia Seltos",
    "a blue TVS Apache motorcycle",
    "a white Ola electric scooter",
]

INDIAN_NAMES = [
    "Rajesh Kumar", "Priya Sharma", "Suresh Babu", "Anitha Devi",
    "Venkatesh Murthy", "Lakshmi Narasimhan", "Ramesh Gowda", "Deepa Hegde",
    "Srinivas Rao", "Meena Kumari", "Arjun Reddy", "Kavitha Shetty",
    "Prakash Nair", "Sunitha Rani", "Manoj Kumar", "Padmini Bhat",
    "Ganesh Prasad", "Asha Devi", "Karthik Rajan", "Revathi Sundaram",
    "Mohan Das", "Vijaya Lakshmi", "Ravi Shankar", "Girija Pai",
    "Naveen Kumar", "Shobha Rao", "Anil Kumar", "Sarita Deshpande",
]


def _generate_verbose_description(
    location: dict,
    accident_type: str,
    severity: str,
    vehicles_involved: int,
    num_victims: int,
) -> str:
    """Generate a realistic verbose accident description for seeded data."""
    loc_name = location["location_name"]
    area = location["area"]

    # Pick random vehicles
    vehicles = random.sample(INDIAN_VEHICLE_DESCRIPTIONS, min(vehicles_involved, len(INDIAN_VEHICLE_DESCRIPTIONS)))
    vehicle_desc = ", ".join(vehicles[:vehicles_involved])

    # Pick random names for victims/witnesses
    names = random.sample(INDIAN_NAMES, min(num_victims + 2, len(INDIAN_NAMES)))
    victim_names = names[:num_victims]
    witness_names = names[num_victims:num_victims + 2]

    # Time of day
    time_descs = ["early morning around 7:30 AM", "during the morning rush hour at 9:15 AM",
                  "at approximately 11:45 AM", "during the afternoon at 2:30 PM",
                  "during evening peak traffic at 6:00 PM", "late at night around 10:45 PM",
                  "in the late evening at 8:30 PM", "during the lunch hour at 1:15 PM"]
    time_desc = random.choice(time_descs)

    # Weather
    weather_descs = ["clear and dry conditions", "light drizzle making the roads slippery",
                     "heavy monsoon rain with reduced visibility", "foggy morning conditions",
                     "overcast but dry weather"]
    weather = random.choice(weather_descs)

    # Build description
    desc_parts = [
        f"A {accident_type.lower()} occurred at {loc_name} in the {area} area of {location['city']}, "
        f"{time_desc}. The weather at the time was {weather}.",

        f"The accident involved {vehicles_involved} vehicle(s): {vehicle_desc}. "
        f"The collision resulted in significant damage to the vehicles involved, "
        f"with {'severe structural deformation and airbag deployment' if severity == 'Critical' else 'moderate body damage and cracked windshields' if severity == 'Major' else 'minor dents, scratches, and broken tail lights'}.",

        f"A total of {num_victims} person(s) were affected — "
        + (f"{', '.join(victim_names[:3])}. " if victim_names else "names not immediately available. ")
        + (f"{'The victims were found in critical condition with multiple injuries and were immediately rushed to the nearest hospital by ambulance' if severity == 'Critical' else 'Some victims sustained minor injuries including bruises and cuts, and were attended to by bystanders before the ambulance arrived' if severity == 'Major' else 'No serious injuries were reported; victims were shaken but able to walk away from the scene'}. "),

        f"{'Emergency services including police and an ambulance from the nearby government hospital arrived at the scene within 12 minutes' if severity in ('Critical', 'Major') else 'Local traffic police arrived and managed the situation'}. "
        f"The road was {'completely blocked for over 45 minutes causing severe traffic congestion on both sides' if severity == 'Critical' else 'partially blocked for about 20 minutes while debris was cleared' if severity == 'Major' else 'briefly obstructed but cleared within 10 minutes'}.",

        f"Eyewitnesses {', '.join(witness_names)} reported that the incident appeared to be caused by "
        + random.choice([
            "overspeeding on the wet road surface",
            "the vehicle jumping the red signal at the junction",
            "sudden lane change without signaling",
            "a stray animal crossing the road causing the driver to swerve",
            "the driver being distracted by a mobile phone",
            "brake failure in the larger vehicle",
            "poor visibility due to the weather conditions",
            "reckless overtaking in a no-passing zone",
        ]) + ". "
        f"{'An FIR has been registered at the local police station under relevant sections of the Motor Vehicles Act.' if severity in ('Critical', 'Major') else 'A traffic violation report has been filed.'}",
    ]

    return " ".join(desc_parts)


class PoliceDatabase:
    """SQLite-based accident report database for police FIR search."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create the database table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS accident_reports (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id       TEXT UNIQUE NOT NULL,
                date            TEXT NOT NULL,
                time            TEXT NOT NULL,
                location_name   TEXT NOT NULL,
                area            TEXT NOT NULL,
                city            TEXT NOT NULL DEFAULT 'Bangalore',
                state           TEXT NOT NULL DEFAULT 'Karnataka',
                latitude        REAL,
                longitude       REAL,
                accident_type   TEXT NOT NULL,
                severity        TEXT NOT NULL,
                vehicles_involved INTEGER DEFAULT 0,
                number_of_victims INTEGER DEFAULT 0,
                injured_persons TEXT DEFAULT 'No',
                road_blocked    TEXT DEFAULT 'No',
                description     TEXT,
                report_pdf_path TEXT,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def seed_fake_data(self, n: int = 25) -> int:
        """
        Seed the database with n realistic fake accident reports.

        Args:
            n: Number of fake reports to seed

        Returns:
            Number of reports actually inserted
        """
        conn = self._get_connection()
        inserted = 0

        # Check if already seeded
        count = conn.execute("SELECT COUNT(*) FROM accident_reports").fetchone()[0]
        if count >= n:
            logger.info(f"Database already has {count} records, skipping seed")
            conn.close()
            return 0

        for i in range(n):
            # Random date in the last 6 months
            days_ago = random.randint(1, 180)
            report_date = datetime.now() - timedelta(days=days_ago)
            date_str = report_date.strftime("%Y-%m-%d")
            time_str = f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}"

            # Random location
            location = random.choice(BANGALORE_LOCATIONS)

            # Random accident details
            accident_type = random.choice(ACCIDENT_TYPES)
            severity = random.choices(SEVERITIES, weights=[50, 35, 15])[0]
            vehicles = random.randint(1, 4)
            victims = random.randint(0, 5) if severity != "Minor" else random.randint(0, 2)
            injured = "Yes" if victims > 0 and severity != "Minor" else "No"
            road_blocked = "Yes" if severity in ("Major", "Critical") else random.choice(["Yes", "No"])

            # Verbose description
            description = _generate_verbose_description(
                location, accident_type, severity, vehicles, victims
            )

            report_id = f"FIR-{report_date.strftime('%Y%m%d')}-{i + 1:03d}"

            try:
                conn.execute("""
                    INSERT INTO accident_reports
                    (report_id, date, time, location_name, area, city, state,
                     latitude, longitude, accident_type, severity,
                     vehicles_involved, number_of_victims, injured_persons,
                     road_blocked, description, report_pdf_path, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    report_id, date_str, time_str,
                    location["location_name"], location["area"],
                    location["city"], location["state"],
                    location["lat"], location["lon"],
                    accident_type, severity,
                    vehicles, victims, injured, road_blocked,
                    description, None,
                    report_date.strftime("%Y-%m-%d %H:%M:%S"),
                ))
                inserted += 1
            except sqlite3.IntegrityError:
                logger.warning(f"Duplicate report_id: {report_id}, skipping")
                continue

        conn.commit()
        conn.close()
        logger.info(f"Seeded {inserted} fake accident reports into database")
        return inserted

    def insert_report(self, report_data: dict) -> str:
        """
        Insert a real accident report (from the pipeline) into the database.

        Args:
            report_data: Report data dict from create_report_structure

        Returns:
            The report_id of the inserted record
        """
        conn = self._get_connection()
        report_id = report_data.get("report_id", f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}")
        gps = report_data.get("gps", {})

        # Try to derive location name from GPS (for real-time reports we use a
        # default location description since reverse geocoding is out of scope)
        location_name = report_data.get("location_name", f"Near coordinates ({gps.get('latitude', 'N/A')}, {gps.get('longitude', 'N/A')})")
        area = report_data.get("area", "Bangalore Urban")

        try:
            conn.execute("""
                INSERT INTO accident_reports
                (report_id, date, time, location_name, area, city, state,
                 latitude, longitude, accident_type, severity,
                 vehicles_involved, number_of_victims, injured_persons,
                 road_blocked, description, report_pdf_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report_id,
                report_data.get("date", datetime.now().strftime("%Y-%m-%d")),
                report_data.get("time", datetime.now().strftime("%H:%M:%S")),
                location_name, area,
                "Bangalore", "Karnataka",
                gps.get("latitude"), gps.get("longitude"),
                report_data.get("accident_type", "Unknown"),
                report_data.get("accident_severity", "Unknown"),
                report_data.get("vehicles_involved", 0),
                report_data.get("number_of_victims", 0),
                report_data.get("injured_person_detected", "Unknown"),
                report_data.get("road_blocked", "Unknown"),
                report_data.get("scene_description", ""),
                report_data.get("report_pdf_path", ""),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ))
            conn.commit()
            logger.info(f"Inserted real report into police DB: {report_id}")
        except sqlite3.IntegrityError:
            logger.warning(f"Report {report_id} already exists in police DB")
        finally:
            conn.close()

        return report_id

    def search_by_date(self, date_str: str) -> list[dict]:
        """
        Search accident reports by date.

        Args:
            date_str: Date string in YYYY-MM-DD format

        Returns:
            List of matching report dicts
        """
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT * FROM accident_reports WHERE date = ? ORDER BY time DESC",
            (date_str,)
        ).fetchall()
        conn.close()

        results = [dict(row) for row in rows]
        logger.info(f"Search by date '{date_str}': found {len(results)} results")
        return results

    def search_by_location(self, location_query: str) -> list[dict]:
        """
        Search accident reports by location name (fuzzy partial match).
        Searches across location_name, area, and city fields.

        Args:
            location_query: Location search string (e.g., "Koramangala", "MG Road")

        Returns:
            List of matching report dicts
        """
        conn = self._get_connection()
        query_lower = f"%{location_query}%"
        rows = conn.execute("""
            SELECT * FROM accident_reports
            WHERE location_name LIKE ? COLLATE NOCASE
               OR area LIKE ? COLLATE NOCASE
               OR city LIKE ? COLLATE NOCASE
            ORDER BY date DESC, time DESC
        """, (query_lower, query_lower, query_lower)).fetchall()
        conn.close()

        results = [dict(row) for row in rows]
        logger.info(f"Search by location '{location_query}': found {len(results)} results")
        return results

    def search_by_date_and_location(self, date_str: str, location_query: str) -> list[dict]:
        """
        Search accident reports by both date and location.

        Args:
            date_str: Date string in YYYY-MM-DD format
            location_query: Location search string

        Returns:
            List of matching report dicts
        """
        conn = self._get_connection()
        query_lower = f"%{location_query}%"
        rows = conn.execute("""
            SELECT * FROM accident_reports
            WHERE date = ?
              AND (location_name LIKE ? COLLATE NOCASE
                   OR area LIKE ? COLLATE NOCASE
                   OR city LIKE ? COLLATE NOCASE)
            ORDER BY time DESC
        """, (date_str, query_lower, query_lower, query_lower)).fetchall()
        conn.close()

        results = [dict(row) for row in rows]
        logger.info(
            f"Search by date '{date_str}' + location '{location_query}': "
            f"found {len(results)} results"
        )
        return results

    def get_report_by_id(self, report_id: str) -> Optional[dict]:
        """Fetch a single report by its report_id."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM accident_reports WHERE report_id = ?",
            (report_id,)
        ).fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    def get_all_reports(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """Get all reports, paginated."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT * FROM accident_reports ORDER BY date DESC, time DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_report_count(self) -> int:
        """Get total number of reports in the database."""
        conn = self._get_connection()
        count = conn.execute("SELECT COUNT(*) FROM accident_reports").fetchone()[0]
        conn.close()
        return count

    def get_unique_dates(self) -> list[str]:
        """Get all unique dates in the database."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT DISTINCT date FROM accident_reports ORDER BY date DESC"
        ).fetchall()
        conn.close()
        return [row["date"] for row in rows]

    def get_unique_areas(self) -> list[str]:
        """Get all unique areas/locations in the database."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT DISTINCT area FROM accident_reports ORDER BY area"
        ).fetchall()
        conn.close()
        return [row["area"] for row in rows]
