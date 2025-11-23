"""
Core package for authentication and utilities
"""

from .auth import face_auth, FaceAuthenticator
from .decorators import login_required
from .utils import (
    decode_base64_image,
    encode_image_to_base64,
    convert_mm_to_inch,
    allowed_file
)

__all__ = [
    'face_auth',
    'FaceAuthenticator',
    'login_required',
    'decode_base64_image',
    'encode_image_to_base64',
    'convert_mm_to_inch',
    'allowed_file'
]
