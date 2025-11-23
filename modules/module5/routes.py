"""
Module 5: Motion Tracking and Object Tracking
Routes and blueprint definition
"""

from flask import Blueprint, render_template, request, jsonify
from core.decorators import login_required
from .handlers import (
    track_with_marker_handler,
    track_markerless_handler,
    track_with_sam2_handler,
    compute_motion_estimate_handler
)

# Create blueprint
module5_bp = Blueprint(
    'module5',
    __name__,
    template_folder='templates',
    url_prefix='/module5'
)

@module5_bp.route('/')
@login_required
def home():
    """Module 5 home page - problem selection"""
    return render_template('module5/module5_home.html')

@module5_bp.route('/problem1')
@login_required
def problem1():
    """Problem 1: Theoretical derivations"""
    return render_template('module5/problem1.html')

@module5_bp.route('/problem2')
@login_required
def problem2():
    """Problem 2: Real-time Object Tracking"""
    return render_template('module5/problem2.html')

# API Routes
@module5_bp.route('/api/track_marker', methods=['POST'])
@login_required
def api_track_marker():
    """API endpoint for marker-based tracking"""
    try:
        data = request.json
        result = track_with_marker_handler(
            image_data=data.get('image'),
            marker_type=data.get('marker_type', 'aruco')
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@module5_bp.route('/api/track_markerless', methods=['POST'])
@login_required
def api_track_markerless():
    """API endpoint for markerless tracking"""
    try:
        data = request.json
        result = track_markerless_handler(
            image_data=data.get('image'),
            bbox=data.get('bbox'),
            method=data.get('method', 'kcf')
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@module5_bp.route('/api/track_sam2', methods=['POST'])
@login_required
def api_track_sam2():
    """API endpoint for SAM2-based tracking"""
    try:
        data = request.json
        result = track_with_sam2_handler(
            image_data=data.get('image'),
            sam2_mask_data=data.get('sam2_mask')
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@module5_bp.route('/api/compute_motion', methods=['POST'])
@login_required
def api_compute_motion():
    """API endpoint for motion estimation"""
    try:
        data = request.json
        result = compute_motion_estimate_handler(
            image1_data=data.get('image1'),
            image2_data=data.get('image2')
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

