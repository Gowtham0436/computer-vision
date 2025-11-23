"""
Module 4: Image Stitching and SIFT Feature Extraction
Routes and blueprint definition
"""

from flask import Blueprint, render_template, request, jsonify
from core.decorators import login_required
from .handlers import (
    stitch_images_handler,
    extract_sift_features_handler,
    match_sift_features_handler
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
    """Module 4 home page - problem selection"""
    return render_template('module4/module4_home.html')

@module4_bp.route('/problem1')
@login_required
def problem1():
    """Problem 1: Image Stitching"""
    return render_template('module4/problem1.html')

@module4_bp.route('/problem2')
@login_required
def problem2():
    """Problem 2: SIFT Feature Extraction"""
    return render_template('module4/problem2.html')

# API Routes
@module4_bp.route('/api/stitch_images', methods=['POST'])
@login_required
def api_stitch_images():
    """API endpoint for image stitching"""
    try:
        data = request.json
        images = data.get('images', [])
        if not images:
            return jsonify({'success': False, 'error': 'No images provided'})
        result = stitch_images_handler(images)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@module4_bp.route('/api/extract_sift', methods=['POST'])
@login_required
def api_extract_sift():
    """API endpoint for SIFT feature extraction"""
    try:
        data = request.json
        result = extract_sift_features_handler(
            image_data=data.get('image'),
            compare_with_opencv=data.get('compare_with_opencv', True)
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@module4_bp.route('/api/match_sift', methods=['POST'])
@login_required
def api_match_sift():
    """API endpoint for SIFT feature matching with RANSAC"""
    try:
        data = request.json
        result = match_sift_features_handler(
            image1_data=data.get('image1'),
            image2_data=data.get('image2'),
            use_ransac=data.get('use_ransac', True)
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

