"""
Module 4: Image Stitching and SIFT Feature Extraction
Routes and blueprint definition
"""

from flask import Blueprint, render_template, request, jsonify
from core.decorators import login_required
from .handlers import (
    stitch_images_with_reference_handler
)

# Create blueprint
module4_bp = Blueprint(
    'module4',
    __name__,
    template_folder='templates',
    url_prefix='/module4'
)

@module4_bp.route('/')
@login_required
def home():
    """Module 4 - Image Stitching with SIFT Feature Extraction"""
    return render_template('module4/problem1.html')

# API Routes
@module4_bp.route('/api/stitch_images', methods=['POST'])
@login_required
def api_stitch_images():
    """API endpoint for image stitching with optional reference comparison"""
    try:
        data = request.json
        images = data.get('images', [])
        reference_image = data.get('reference_image')  # Optional reference image
        if not images:
            return jsonify({'success': False, 'error': 'No images provided'})
        result = stitch_images_with_reference_handler(images, reference_image)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


