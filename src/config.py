"""
Configuration module for Accident Report Generation System.
Centralized configuration management with environment variable support.
"""

import os
import random
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ─── Ollama Configuration ───────────────────────────────────────────────────────
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "bakllava")

# ─── Groq Cloud Configuration (Stage 2 - text JSON extraction) ──────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# ─── GPS Coordinates ────────────────────────────────────────────────────────────
BASE_LATITUDE = 12.9716
BASE_LONGITUDE = 77.5946


def generate_random_gps(base_lat: float = BASE_LATITUDE,
                        base_lon: float = BASE_LONGITUDE,
                        radius_km: float = 0.5) -> tuple:
    """
    Generate random GPS coordinates within a radius of the base location.

    Args:
        base_lat: Base latitude (default: Bangalore)
        base_lon: Base longitude (default: Bangalore)
        radius_km: Radius in kilometers for random offset

    Returns:
        (latitude, longitude) tuple
    """
    lat_offset = random.uniform(-radius_km / 111, radius_km / 111)
    lon_offset = random.uniform(-radius_km / 111, radius_km / 111)
    return (round(base_lat + lat_offset, 6),
            round(base_lon + lon_offset, 6))


# ─── File Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
INPUT_IMAGES_DIR = os.path.join(PROJECT_ROOT, "input_images")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# ─── Report Template ────────────────────────────────────────────────────────────
REPORT_TITLE = "Accident Report"
ORGANIZATION = "Intelligent Accident Detection System"

# ─── Supported Image Formats ────────────────────────────────────────────────────
SUPPORTED_IMAGE_FORMATS = (".jpg", ".jpeg", ".png")

# ─── API Configuration ──────────────────────────────────────────────────────────
API_TIMEOUT = 120  # seconds (bakllava only; Groq is cloud-fast)
MAX_RETRIES = 3

# ─── Logging Setup ──────────────────────────────────────────────────────────────
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(INPUT_IMAGES_DIR, exist_ok=True)

# Track whether logging has been configured
_logging_configured = False


def setup_logging():
    """Configure application logging. Only configures handlers once."""
    global _logging_configured
    if _logging_configured:
        return logging.getLogger(__name__)

    from datetime import datetime
    log_file = os.path.join(LOG_DIR, f"accident_report_{datetime.now().strftime('%Y%m%d')}.log")

    root_logger = logging.getLogger()
    # Clear any existing handlers to prevent duplicates
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    root_logger.addHandler(file_handler)

    # Console handler (force UTF-8 on Windows)
    import sys
    console_handler = logging.StreamHandler(
        open(sys.stdout.fileno(), 'w', encoding='utf-8', closefd=False)
    )
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    root_logger.addHandler(console_handler)

    _logging_configured = True
    return logging.getLogger(__name__)
