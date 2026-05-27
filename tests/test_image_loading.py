"""
Test 3: Image Loading
Verifies image handling works correctly for all scenarios.
"""

import pytest
import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from src.image_handler import load_image, validate_image, preprocess_image, image_to_base64


@pytest.fixture
def valid_image_path():
    """Create a temporary valid JPEG image."""
    img = Image.new('RGB', (500, 500), color=(100, 150, 200))
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    tmp_path = tmp.name
    tmp.close()
    img.save(tmp_path)
    yield tmp_path
    try:
        os.unlink(tmp_path)
    except PermissionError:
        pass


@pytest.fixture
def valid_png_path():
    """Create a temporary valid PNG image."""
    img = Image.new('RGBA', (300, 300), color=(100, 150, 200, 255))
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp_path = tmp.name
    tmp.close()
    img.save(tmp_path)
    yield tmp_path
    try:
        os.unlink(tmp_path)
    except PermissionError:
        pass


class TestImageLoading:
    """Test suite for image loading and validation."""

    def test_load_valid_jpeg(self, valid_image_path):
        """Test loading a valid JPEG image."""
        image = load_image(valid_image_path)
        assert isinstance(image, Image.Image)
        assert image.size == (500, 500)

    def test_load_valid_png(self, valid_png_path):
        """Test loading a valid PNG image."""
        image = load_image(valid_png_path)
        assert isinstance(image, Image.Image)

    def test_missing_file(self):
        """Test handling of missing file."""
        with pytest.raises(FileNotFoundError):
            load_image("nonexistent_image.jpg")

    def test_invalid_format(self):
        """Test handling of invalid format."""
        tmp = tempfile.NamedTemporaryFile(suffix='.bmp', delete=False)
        tmp.close()
        with pytest.raises(ValueError):
            validate_image(tmp.name)
        os.unlink(tmp.name)

    def test_preprocess_converts_to_rgb(self):
        """Test that preprocessing converts RGBA to RGB."""
        img = Image.new('RGBA', (200, 200), color=(100, 150, 200, 255))
        result = preprocess_image(img)
        assert result.mode == 'RGB'

    def test_preprocess_resizes_large_image(self):
        """Test that preprocessing resizes oversized images."""
        img = Image.new('RGB', (2000, 2000), color=(100, 150, 200))
        result = preprocess_image(img, max_size=(1024, 1024))
        assert result.size[0] <= 1024
        assert result.size[1] <= 1024

    def test_image_to_base64(self):
        """Test base64 encoding of image."""
        img = Image.new('RGB', (100, 100), color=(255, 0, 0))
        b64 = image_to_base64(img)
        assert isinstance(b64, str)
        assert len(b64) > 0
