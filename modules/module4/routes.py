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
@module4_bp.route('/api/stitch_image', methods=['POST'])  # Support both singular and plural
@login_required
def api_stitch_images():
    """API endpoint for image stitching with optional reference comparison"""
    import traceback
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        images = data.get('images', [])
        reference_image = data.get('reference_image')  # Optional reference image
        
        if not images:
            return jsonify({'success': False, 'error': 'No images provided'}), 400
        
        if len(images) > 10:
            return jsonify({'success': False, 'error': 'Too many images. Maximum 10 images allowed.'}), 400
        
        logger.info("Processing %d images for stitching", len(images))
        result = stitch_images_with_reference_handler(images, reference_image)
        
        if not result.get('success'):
            logger.warning("Stitching failed: %s", result.get('error', 'Unknown error'))
        
        return jsonify(result)
    except MemoryError:
        logger.error("Out of memory during image stitching")
        return jsonify({'success': False, 'error': 'Out of memory. Try using fewer or smaller images.'}), 500
    except Exception as e:
        logger.error("Error in stitch_images: %s\n%s", str(e), traceback.format_exc())
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500


