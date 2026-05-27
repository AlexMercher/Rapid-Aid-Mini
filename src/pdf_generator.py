"""
PDF Generator module for Accident Report Generation System.
Converts accident reports to PDF format with embedded images.
"""

import os
import io
import logging
from datetime import datetime
from PIL import Image as PILImage

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

from src.config import REPORTS_DIR, REPORT_TITLE, ORGANIZATION

logger = logging.getLogger(__name__)

# ─── Color Palette ───────────────────────────────────────────────────────────────
PRIMARY = HexColor("#1a237e")      # Deep indigo
SECONDARY = HexColor("#c62828")    # Deep red (emergency)
ACCENT = HexColor("#f57f17")       # Amber
BG_LIGHT = HexColor("#f5f5f5")
TEXT_DARK = HexColor("#212121")
TEXT_MEDIUM = HexColor("#616161")
BORDER = HexColor("#bdbdbd")
SEVERITY_COLORS = {
    "Minor": HexColor("#4caf50"),
    "Major": HexColor("#ff9800"),
    "Critical": HexColor("#f44336"),
    "Unknown": HexColor("#9e9e9e"),
}


def _get_styles():
    """Create custom paragraph styles for the PDF."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        'ReportTitle',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=20,
        textColor=PRIMARY,
        spaceAfter=6,
        alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        'ReportSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        textColor=TEXT_MEDIUM,
        alignment=TA_CENTER,
        spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=PRIMARY,
        spaceBefore=16,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        'FieldLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=TEXT_DARK,
    ))
    styles.add(ParagraphStyle(
        'FieldValue',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=TEXT_DARK,
    ))
    styles.add(ParagraphStyle(
        'DescriptionText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=TEXT_DARK,
        leading=14,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        textColor=TEXT_MEDIUM,
        alignment=TA_CENTER,
    ))

    return styles


def _build_image_element(image: PILImage.Image, max_width=16 * cm, max_height=10 * cm):
    """
    Convert a PIL Image to a ReportLab Image flowable.

    Args:
        image: PIL Image object
        max_width: Maximum width in points
        max_height: Maximum height in points

    Returns:
        ReportLab Image flowable
    """
    buf = io.BytesIO()
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(buf, format='JPEG', quality=90)
    buf.seek(0)

    img_w, img_h = image.size
    aspect = img_w / img_h

    if img_w / max_width > img_h / max_height:
        w = max_width
        h = w / aspect
    else:
        h = max_height
        w = h * aspect

    return Image(buf, width=w, height=h)


def create_pdf_report(report_data: dict, output_path: str,
                      accident_image: PILImage.Image = None) -> str:
    """
    Create a professional PDF accident report.

    Args:
        report_data: Complete report data dictionary
        output_path: Path to save the PDF
        accident_image: PIL Image of the accident scene

    Returns:
        Path to the generated PDF file
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else REPORTS_DIR, exist_ok=True)
    styles = _get_styles()
    elements = []

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    # ── Title Block ──────────────────────────────────────────────────────────
    elements.append(Paragraph(REPORT_TITLE, styles['ReportTitle']))
    elements.append(Paragraph(ORGANIZATION, styles['ReportSubtitle']))
    elements.append(HRFlowable(width="100%", thickness=2, color=PRIMARY))
    elements.append(Spacer(1, 8))

    # ── Report Metadata Table ────────────────────────────────────────────────
    meta_data = [
        [Paragraph("<b>Report ID:</b>", styles['FieldLabel']),
         Paragraph(report_data.get('report_id', 'N/A'), styles['FieldValue']),
         Paragraph("<b>Date:</b>", styles['FieldLabel']),
         Paragraph(report_data.get('date', 'N/A'), styles['FieldValue'])],
        [Paragraph("<b>Time:</b>", styles['FieldLabel']),
         Paragraph(report_data.get('time', 'N/A'), styles['FieldValue']),
         Paragraph("<b>GPS:</b>", styles['FieldLabel']),
         Paragraph(f"Lat: {report_data.get('gps', {}).get('latitude', 'N/A')}, "
                   f"Lon: {report_data.get('gps', {}).get('longitude', 'N/A')}",
                   styles['FieldValue'])],
    ]
    meta_table = Table(meta_data, colWidths=[3 * cm, 5 * cm, 3 * cm, 5 * cm])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BG_LIGHT),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.25, BORDER),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 12))

    # ── Accident Image ───────────────────────────────────────────────────────
    if accident_image:
        elements.append(Paragraph("ACCIDENT IMAGE", styles['SectionHeader']))
        elements.append(HRFlowable(width="100%", thickness=1, color=BORDER))
        elements.append(Spacer(1, 6))
        try:
            img_element = _build_image_element(accident_image)
            elements.append(img_element)
        except Exception as e:
            logger.error(f"Failed to embed image in PDF: {e}")
            elements.append(Paragraph(f"[Image could not be embedded: {e}]", styles['FieldValue']))
        elements.append(Spacer(1, 12))

    # ── Accident Analysis ────────────────────────────────────────────────────
    elements.append(Paragraph("ACCIDENT ANALYSIS", styles['SectionHeader']))
    elements.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    elements.append(Spacer(1, 6))

    severity = report_data.get('accident_severity', 'Unknown')
    sev_color = SEVERITY_COLORS.get(severity, SEVERITY_COLORS["Unknown"])

    analysis_fields = [
        ["Accident Detected", report_data.get('accident_detected', 'N/A')],
        ["Accident Type", report_data.get('accident_type', 'N/A')],
        ["Number of Victims", str(report_data.get('number_of_victims', 'N/A'))],
        ["Vehicles Involved", str(report_data.get('vehicles_involved', 'N/A'))],
        ["Severity", severity],
        ["Injured Persons Visible", report_data.get('injured_person_detected', 'N/A')],
        ["Emergency Services Present", report_data.get('emergency_services_present', 'N/A')],
        ["Road Blocked", report_data.get('road_blocked', 'N/A')],
    ]

    analysis_table_data = []
    for label, value in analysis_fields:
        analysis_table_data.append([
            Paragraph(f"<b>{label}:</b>", styles['FieldLabel']),
            Paragraph(str(value), styles['FieldValue']),
        ])

    analysis_table = Table(analysis_table_data, colWidths=[6 * cm, 10 * cm])
    analysis_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), BG_LIGHT),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.25, BORDER),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(analysis_table)
    elements.append(Spacer(1, 12))

    # ── Scene Description ────────────────────────────────────────────────────
    elements.append(Paragraph("INCIDENT DESCRIPTION", styles['SectionHeader']))
    elements.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    elements.append(Spacer(1, 6))
    description = report_data.get('scene_description', 'No description available')
    # Wrap long descriptions safely
    for para_text in str(description).split('\n'):
        if para_text.strip():
            elements.append(Paragraph(para_text.strip(), styles['DescriptionText']))
    elements.append(Spacer(1, 12))

    # ── Footer / Metadata ────────────────────────────────────────────────────
    elements.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        f"Generated by: {ORGANIZATION}  |  Model: {report_data.get('model_used', 'N/A')}  |  "
        f"GPS: ({report_data.get('gps', {}).get('latitude', '')}, "
        f"{report_data.get('gps', {}).get('longitude', '')})",
        styles['Footer']
    ))

    # ── Build PDF ────────────────────────────────────────────────────────────
    try:
        doc.build(elements)
        logger.info(f"PDF report created: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to create PDF: {e}")
        raise IOError(f"Failed to create PDF: {e}")
