"""
Streamlit page component for the Hospital Notification System.

Displays:
- Active accident notifications forwarded to hospitals
- 30-minute countdown timers per hospital
- Accept/Reject buttons for each hospital
- Hospital status cards (available/occupied)
"""

import streamlit as st
from datetime import datetime

from src.hospital_system import HospitalNotificationManager, HOSPITAL_REGISTRY


def _get_manager() -> HospitalNotificationManager:
    """Get or create the singleton HospitalNotificationManager in session state."""
    if "hospital_manager" not in st.session_state:
        st.session_state["hospital_manager"] = HospitalNotificationManager()
    return st.session_state["hospital_manager"]


def render_hospital_page():
    """Render the Hospital Notification System page."""

    st.markdown("""
    <div class="main-header">
        <h1>🏥 Hospital Notification System</h1>
        <p>Real-Time Accident Report Forwarding & Emergency Response Tracking</p>
    </div>
    """, unsafe_allow_html=True)

    manager = _get_manager()

    # ── Sidebar: Hospital Registry ────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🏥 Hospital Registry")
        all_hospitals = manager.get_all_hospitals()
        available = sum(1 for h in all_hospitals if h["status"] == "available")
        occupied = sum(1 for h in all_hospitals if h["status"] == "occupied")
        st.markdown(f"**Available:** {available} | **Occupied:** {occupied}")
        st.markdown("---")

    # ── Top Metrics Row ───────────────────────────────────────────────────────
    notifications = manager.get_all_notifications()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Notifications", len(notifications))
    with col2:
        pending = sum(1 for n in notifications if n["overall_status"] == "pending")
        st.metric("Pending", pending)
    with col3:
        accepted_count = sum(1 for n in notifications if n["overall_status"] == "accepted")
        st.metric("Accepted", accepted_count)
    with col4:
        expired_count = sum(1 for n in notifications if n["overall_status"] == "expired")
        st.metric("Expired", expired_count)

    st.markdown("---")

    # ── Active Notifications ──────────────────────────────────────────────────
    st.markdown("### Active Accident Notifications")

    if not notifications:
        st.markdown("""
        <div class="status-card">
            <strong>No Active Notifications</strong><br>
            When an accident is detected in the main detection page, the report will be
            automatically forwarded to the 3 nearest hospitals. Notifications will appear here.
        </div>
        """, unsafe_allow_html=True)
    else:
        for notif in reversed(notifications):
            _render_notification_card(notif, manager)

    st.markdown("---")

    # ── Hospital Registry Table ───────────────────────────────────────────────
    st.markdown("### Hospital Registry")
    _render_hospital_registry(manager)


def _render_notification_card(notif: dict, manager: HospitalNotificationManager):
    """Render a single notification card with hospital details and timers."""
    report_id = notif["report_id"]
    severity = notif.get("severity", "Unknown")
    sev_lower = severity.lower() if severity.lower() in ("minor", "major", "critical") else "unknown"

    # Card header
    overall = notif["overall_status"]
    if overall == "accepted":
        border_color = "#4caf50"
        status_label = "ACCEPTED"
    elif overall == "expired":
        border_color = "#9e9e9e"
        status_label = "EXPIRED"
    else:
        border_color = "#ff9800"
        status_label = "PENDING"

    st.markdown(f"""
    <div style="border: 2px solid {border_color}; border-radius: 12px; padding: 1rem; margin-bottom: 1rem; background: var(--card-bg, #fff);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-weight: 700; font-size: 1.1rem; color: var(--card-heading, #1a237e);">
                Report: {report_id}
            </span>
            <span class="severity-badge severity-{sev_lower}" style="display: inline-block; padding: 0.25rem 0.75rem; border-radius: 20px; font-weight: 600; font-size: 0.85rem; color: #fff;">
                {severity}
            </span>
        </div>
        <p style="margin: 0; font-size: 0.9rem; color: var(--card-text, #333);">
            <b>Type:</b> {notif.get('accident_type', 'N/A')} |
            <b>Victims:</b> {notif.get('victims', 0)} |
            <b>Vehicles:</b> {notif.get('vehicles', 0)} |
            <b>Status:</b> {status_label}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Hospital response cards
    cols = st.columns(len(notif["hospitals"]))
    for idx, h in enumerate(notif["hospitals"]):
        with cols[idx]:
            _render_hospital_response_card(h, report_id, manager)


def _render_hospital_response_card(h: dict, report_id: str,
                                    manager: HospitalNotificationManager):
    """Render a single hospital's response card with timer and buttons."""
    status = h["status"]
    hospital_id = h["hospital_id"]
    hospital_name = h["hospital_name"]

    # Color coding
    if status == "accepted":
        border = "#4caf50"
        bg = "rgba(76, 175, 80, 0.08)"
        timer_text = "ACCEPTED"
    elif status == "rejected":
        border = "#f44336"
        bg = "rgba(244, 67, 54, 0.08)"
        timer_text = "REJECTED"
    elif status == "expired":
        border = "#9e9e9e"
        bg = "rgba(158, 158, 158, 0.08)"
        timer_text = "EXPIRED"
    else:
        remaining = h.get("remaining_display", "30:00")
        remaining_secs = h.get("remaining_seconds", 1800)
        border = "#ff9800" if remaining_secs > 300 else "#f44336"
        bg = "rgba(255, 152, 0, 0.08)" if remaining_secs > 300 else "rgba(244, 67, 54, 0.08)"
        timer_text = remaining

    st.markdown(f"""
    <div style="border: 1px solid {border}; border-radius: 8px; padding: 0.8rem; background: {bg}; text-align: center;">
        <p style="font-weight: 600; font-size: 0.9rem; margin: 0 0 0.3rem 0; color: var(--card-heading, #1a237e);">{hospital_name}</p>
        <p style="font-size: 0.75rem; margin: 0 0 0.3rem 0; color: var(--card-text, #666);">{h.get('hospital_address', '')}</p>
        <p style="font-size: 0.75rem; margin: 0 0 0.3rem 0; color: var(--card-text, #666);">Distance: {h.get('distance_km', '?')} km</p>
        <p style="font-size: 1.5rem; font-weight: 800; margin: 0.5rem 0; color: {border};">{timer_text}</p>
    </div>
    """, unsafe_allow_html=True)

    # Buttons only for pending
    if status == "pending":
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Accept", key=f"accept_{report_id}_{hospital_id}",
                         type="primary", use_container_width=True):
                manager.accept_report(hospital_id, report_id)
                st.rerun()
        with btn_col2:
            if st.button("Reject", key=f"reject_{report_id}_{hospital_id}",
                         use_container_width=True):
                manager.reject_report(hospital_id, report_id)
                st.rerun()


def _render_hospital_registry(manager: HospitalNotificationManager):
    """Render the complete hospital registry as a table."""
    hospitals = manager.get_all_hospitals()

    # Build table data
    table_data = []
    for h in hospitals:
        status = h["status"]
        status_display = f"Available" if status == "available" else "OCCUPIED"
        table_data.append({
            "ID": h["id"],
            "Hospital Name": h["name"],
            "Address": h["address"],
            "Capacity": h["capacity"],
            "Ambulances": h["available_ambulances"],
            "Phone": h["phone"],
            "Specialties": ", ".join(h["specialties"]),
            "Status": status_display,
        })

    st.dataframe(
        table_data,
        use_container_width=True,
        column_config={
            "Status": st.column_config.TextColumn(
                "Status",
                help="Current hospital status",
            ),
        },
        hide_index=True,
    )
