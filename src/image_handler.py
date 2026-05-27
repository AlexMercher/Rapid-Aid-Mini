"""
Image Handler module for Accident Report Generation System.
Handles image loading, validation, preprocessing, and preparation for API/PDF.
"""

import os
import io
import base64
import logging
from PIL import Image

from src.config import SUPPORTED_IMAGE_FORMATS

logger = logging.getLogger(__name__)


def validate_image(image_path: str) -> bool:
    """
    Validate that the image file exists and is a supported format.

    Args:
        image_path: Path to the image file

    Returns:
        True if image is valid

    Raises:
        FileNotFoundError: If image file does not exist
        ValueError: If image format is not supported
    """
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        raise FileNotFoundError(f"Image file not found: {image_path}")

    ext = os.path.splitext(image_path)[1].lower()
    if ext not in SUPPORTED_IMAGE_FORMATS:
        logger.error(f"Unsupported image format: {ext}")
        raise ValueError(f"Unsupported image format: {ext}. Supported: {SUPPORTED_IMAGE_FORMATS}")

    return True


def load_image(image_path: str) -> Image.Image:
    """
    Load an image from file path.

    Args:
        image_path: Path to the image file

    Returns:
        PIL Image object
    """
    validate_image(image_path)
    try:
        image = Image.open(image_path)
        image.load()  # Force load to detect corrupted images
        logger.info(f"Image loaded successfully: {image_path} ({image.size})")
        return image
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        raise ValueError(f"Failed to load image: {e}")


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    Load an image from raw bytes (for Streamlit file uploader).

    Args:
        image_bytes: Raw image bytes

    Returns:
        PIL Image object
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
        logger.info(f"Image loaded from bytes: {image.size}")
        return image
    except Exception as e:
        logger.error(f"Failed to load image from bytes: {e}")
        raise ValueError(f"Failed to load image from bytes: {e}")


def preprocess_image(image: Image.Image, max_size: tuple = (1024, 1024)) -> Image.Image:
    """
    Preprocess image for API submission.

    Args:
        image: PIL Image object
        max_size: Maximum dimensions (width, height)

    Returns:
        Preprocessed PIL Image object
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')

    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        logger.info(f"Image resized to: {image.size}")

    # Log warnings for low quality
    if image.size[0] < 300 or image.size[1] < 300:
        logger.warning("Low resolution image detected")

    return image


def prepare_for_pdf(image: Image.Image, max_dim: int = 800) -> Image.Image:
    """
    Prepare image for embedding in PDF.

    Args:
        image: PIL Image object
        max_dim: Maximum dimension for PDF embedding

    Returns:
        Prepared PIL Image object
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')

    if max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        logger.info(f"Image prepared for PDF: {image.size}")

    return image


def image_to_base64(image: Image.Image) -> str:
    """
    Convert PIL Image to base64 string for Ollama API.

    Args:
        image: PIL Image object

    Returns:
        Base64 encoded string
    """
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
