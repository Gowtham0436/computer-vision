"""
Module 7: Stereo Vision and Pose Estimation
Routes and blueprint definition - Exact implementation from Assignment 7
"""

from flask import Blueprint, render_template, request, jsonify
from core.decorators import login_required
from .handlers import (
    calibrate_stereo_handler,
    triangulate_points_handler,
    measure_object_size_handler,
    detect_chessboard_handler,
    debug_chessboard_handler
)

# Create blueprint
module7_bp = Blueprint(
    'module7',
    __name__,
    template_folder='templates',
    url_prefix='/module7'
)

@module7_bp.route('/')
@login_required
def home():
    """Module 7 home page - problem selection"""
    return render_template('module7/module7_home.html')

@module7_bp.route('/problem1')
@login_required
def problem1():
    """Problem 1: Stereo Calibration & Measurement"""
    return render_template('module7/problem1.html')

@module7_bp.route('/problem2')
@login_required
def problem2():
    """Problem 2: Uncalibrated Stereo Derivation"""
    return render_template('module7/problem2.html')

@module7_bp.route('/problem3')
@login_required
def problem3():
    """Problem 3: Pose Estimation and Hand Tracking"""
    return render_template('module7/problem3.html')

# API Routes - Matching Assignment 7 endpoints
@module7_bp.route('/api/calibrate', methods=['POST'])
@login_required
def api_calibrate():
    """API endpoint for stereo calibration"""
    try:
        data = request.json
        result = calibrate_stereo_handler(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@module7_bp.route('/api/triangulate', methods=['POST'])
@login_required
def api_triangulate():
    """API endpoint for 3D triangulation"""
    try:
        data = request.json
        result = triangulate_points_handler(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@module7_bp.route('/api/measure_size', methods=['POST'])
@login_required
def api_measure_size():
    """API endpoint for object size measurement"""
    try:
        data = request.json
        result = measure_object_size_handler(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@module7_bp.route('/api/detect_chessboard', methods=['POST'])
@login_required
def api_detect_chessboard():
    """API endpoint for chessboard detection"""
    try:
        data = request.json
        result = detect_chessboard_handler(data)
        if not result.get('success'):
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@module7_bp.route('/api/debug_chessboard', methods=['POST'])
@login_required
def api_debug_chessboard():
    """API endpoint for chessboard debugging"""
    try:
        data = request.json
        result = debug_chessboard_handler(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@module7_bp.route('/api/health', methods=['GET'])
@login_required
def api_health():
    """Health check endpoint"""
    from .handlers import calibration_storage
    return jsonify({'status': 'ok', 'calibrations': len(calibration_storage)})
