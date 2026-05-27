"""
Streamlit page component for the Police FIR Database System.

Displays:
- Search interface with date picker + location text input
- Results table with expandable report details
- Database statistics and overview
"""

import streamlit as st
from datetime import datetime, date

from src.police_database import PoliceDatabase, DB_PATH


def _get_db() -> PoliceDatabase:
    """Get or create the singleton PoliceDatabase in session state."""
    if "police_db" not in st.session_state:
        db = PoliceDatabase()
        # Seed if empty
        if db.get_report_count() == 0:
            db.seed_fake_data(n=25)
        st.session_state["police_db"] = db
    return st.session_state["police_db"]


def render_police_page():
    """Render the Police FIR Database page."""

    st.markdown("""
    <div class="main-header">
        <h1>Police FIR Database</h1>
        <p>Accident Report Search & Retrieval for FIR Filing</p>
    </div>
    """, unsafe_allow_html=True)

    db = _get_db()

    # ── Sidebar: DB Stats ─────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Database Stats")
        total_reports = db.get_report_count()
        unique_dates = db.get_unique_dates()
        unique_areas = db.get_unique_areas()
        st.markdown(f"**Total Reports:** {total_reports}")
        st.markdown(f"**Unique Dates:** {len(unique_dates)}")
        st.markdown(f"**Unique Areas:** {len(unique_areas)}")
        st.markdown("---")

    # ── Top Metrics ───────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Reports", db.get_report_count())
    with col2:
        st.metric("Coverage Areas", len(db.get_unique_areas()))
    with col3:
        st.metric("Date Range", f"{len(db.get_unique_dates())} days")

    st.markdown("---")

    # ── Search Interface ──────────────────────────────────────────────────────
    st.markdown("### Search Accident Reports")
    st.markdown("""
    <div class="status-card">
        <strong>Search Options</strong><br>
        Search by date, location, or both. Location search supports partial matching
        (e.g., searching "Koramangala" will find all reports in the Koramangala area).
    </div>
    """, unsafe_allow_html=True)

    search_col1, search_col2, search_col3 = st.columns([1, 1, 1])

    with search_col1:
        search_mode = st.selectbox(
            "Search Mode",
            ["By Date", "By Location", "By Date + Location", "All Reports"],
            help="Choose how to search the database"
        )

    with search_col2:
        if search_mode in ("By Date", "By Date + Location"):
            search_date = st.date_input(
                "Select Date",
                value=None,
                help="Pick the date of the accident"
            )
        else:
            search_date = None

    with search_col3:
        if search_mode in ("By Location", "By Date + Location"):
            # Show available areas as hint
            areas = db.get_unique_areas()
            area_hint = ", ".join(areas[:5]) + "..." if len(areas) > 5 else ", ".join(areas)
            search_location = st.text_input(
                "Location / Area",
                placeholder=f"e.g., {areas[0] if areas else 'Koramangala'}",
                help=f"Available areas: {area_hint}"
            )
        else:
            search_location = None

    # ── Search Button ─────────────────────────────────────────────────────────
    search_btn = st.button("Search", type="primary", use_container_width=True)

    if search_btn:
        results = _perform_search(db, search_mode, search_date, search_location)
        st.session_state["search_results"] = results
        st.session_state["search_performed"] = True

    # ── Display Results ───────────────────────────────────────────────────────
    st.markdown("---")

    if st.session_state.get("search_performed"):
        results = st.session_state.get("search_results", [])
        _render_search_results(results)
    else:
        st.markdown("""
        <div class="status-card">
            <strong>Ready to Search</strong><br>
            Select a search mode and criteria above, then click Search to find accident reports.
        </div>
        """, unsafe_allow_html=True)

        # Show recent reports preview
        st.markdown("### Recent Reports")
        recent = db.get_all_reports(limit=5)
        if recent:
            _render_results_table(recent)


def _perform_search(db: PoliceDatabase, mode: str,
                    search_date, search_location: str) -> list:
    """Execute search based on selected mode."""
    if mode == "By Date":
        if search_date is None:
            st.warning("Please select a date to search.")
            return []
        date_str = search_date.strftime("%Y-%m-%d")
        return db.search_by_date(date_str)

    elif mode == "By Location":
        if not search_location or not search_location.strip():
            st.warning("Please enter a location to search.")
            return []
        return db.search_by_location(search_location.strip())

    elif mode == "By Date + Location":
        if search_date is None:
            st.warning("Please select a date.")
            return []
        if not search_location or not search_location.strip():
            st.warning("Please enter a location.")
            return []
        date_str = search_date.strftime("%Y-%m-%d")
        return db.search_by_date_and_location(date_str, search_location.strip())

    elif mode == "All Reports":
        return db.get_all_reports(limit=50)

    return []


def _render_search_results(results: list):
    """Render search results with summary and expandable details."""
    st.markdown(f"### Search Results ({len(results)} found)")

    if not results:
        st.markdown("""
        <div class="status-card warning">
            <strong>No Results Found</strong><br>
            Try a different date or location query. Location search supports partial matching.
        </div>
        """, unsafe_allow_html=True)
        return

    _render_results_table(results)

    # Expandable details for each
    st.markdown("### Report Details")
    for r in results:
        severity = r.get("severity", "Unknown")
        sev_lower = severity.lower() if severity.lower() in ("minor", "major", "critical") else "unknown"

        with st.expander(
            f"{r['report_id']} | {r['date']} | {r['area']} | {r['accident_type']} | {severity}",
            expanded=False
        ):
            _render_report_detail(r)


def _render_results_table(results: list):
    """Render results as a data table."""
    table_data = []
    for r in results:
        table_data.append({
            "Report ID": r["report_id"],
            "Date": r["date"],
            "Time": r["time"],
            "Location": r["location_name"],
            "Area": r["area"],
            "Type": r["accident_type"],
            "Severity": r["severity"],
            "Victims": r["number_of_victims"],
            "Vehicles": r["vehicles_involved"],
        })

    st.dataframe(table_data, use_container_width=True, hide_index=True)


def _render_report_detail(r: dict):
    """Render a single report's full details."""
    detail_cols = st.columns(2)

    with detail_cols[0]:
        st.markdown(f"""
        <div class="result-card">
            <h3>Report Information</h3>
            <p>
                <b>Report ID:</b> {r.get('report_id', 'N/A')}<br>
                <b>Date:</b> {r.get('date', 'N/A')}<br>
                <b>Time:</b> {r.get('time', 'N/A')}<br>
                <b>Location:</b> {r.get('location_name', 'N/A')}<br>
                <b>Area:</b> {r.get('area', 'N/A')}<br>
                <b>City:</b> {r.get('city', 'N/A')}, {r.get('state', 'N/A')}
            </p>
        </div>
        """, unsafe_allow_html=True)

    with detail_cols[1]:
        severity = r.get("severity", "Unknown")
        sev_lower = severity.lower() if severity.lower() in ("minor", "major", "critical") else "unknown"
        st.markdown(f"""
        <div class="result-card">
            <h3>Accident Details</h3>
            <p>
                <b>Type:</b> {r.get('accident_type', 'N/A')}<br>
                <b>Severity:</b> <span class="severity-badge severity-{sev_lower}">{severity}</span><br>
                <b>Vehicles:</b> {r.get('vehicles_involved', 'N/A')}<br>
                <b>Victims:</b> {r.get('number_of_victims', 'N/A')}<br>
                <b>Injured:</b> {r.get('injured_persons', 'N/A')}<br>
                <b>Road Blocked:</b> {r.get('road_blocked', 'N/A')}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Full description
    description = r.get("description", "No description available")
    st.markdown(f"""
    <div class="result-card">
        <h3>Incident Description</h3>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)

    # GPS coordinates
    lat = r.get("latitude", "N/A")
    lon = r.get("longitude", "N/A")
    if lat != "N/A" and lon != "N/A" and lat is not None and lon is not None:
        st.markdown(f"""
        <div class="result-card">
            <h3>GPS Coordinates</h3>
            <p><b>Latitude:</b> {lat} | <b>Longitude:</b> {lon}</p>
        </div>
        """, unsafe_allow_html=True)
