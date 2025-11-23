"""
Module 3: Image Processing and Segmentation
Routes and blueprint definition
"""

from flask import Blueprint, render_template, request, jsonify
from core.decorators import login_required
from .handlers import (
    compute_gradient_and_log_handler,
    detect_edges_handler,
    detect_corners_handler,
    detect_boundary_handler,
    segment_with_aruco_handler,
    compare_with_sam2_handler
)

# Create blueprint
module3_bp = Blueprint(
    'module3',
    __name__,
    template_folder='templates',
    url_prefix='/module3'
)

@module3_bp.route('/')
@login_required
def home():
    """Module 3 home page - problem selection"""
    return render_template('module3/module3_home.html')

@module3_bp.route('/problem1')
@login_required
def problem1():
    """Problem 1: Gradient and Laplacian of Gaussian"""
    return render_template('module3/problem1.html')

@module3_bp.route('/problem2')
@login_required
def problem2():
    """Problem 2: Edge and Corner Detection"""
    return render_template('module3/problem2.html')

@module3_bp.route('/problem3')
@login_required
def problem3():
    """Problem 3: Object Boundary Detection"""
    return render_template('module3/problem3.html')

@module3_bp.route('/problem4')
@login_required
def problem4():
    """Problem 4: ArUco Marker Segmentation"""
    return render_template('module3/problem4.html')

@module3_bp.route('/problem5')
@login_required
def problem5():
    """Problem 5: SAM2 Comparison"""
    return render_template('module3/problem5.html')

# API Routes
@module3_bp.route('/api/gradient_log', methods=['POST'])
@login_required
def api_gradient_log():
    """API endpoint for gradient and LoG computation"""
    try:
        data = request.json
        result = compute_gradient_and_log_handler(
            image_data=data.get('image')
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@module3_bp.route('/api/detect_edges', methods=['POST'])
@login_required
def api_detect_edges():
    """API endpoint for edge detection"""
    try:
        data = request.json
        result = detect_edges_handler(
            image_data=data.get('image'),
            threshold1=data.get('threshold1', 50),
            threshold2=data.get('threshold2', 150)
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@module3_bp.route('/api/detect_corners', methods=['POST'])
@login_required
def api_detect_corners():
    """API endpoint for corner detection"""
    try:
        data = request.json
        result = detect_corners_handler(
            image_data=data.get('image'),
            max_corners=data.get('max_corners', 100),
            quality=data.get('quality', 0.01),
            min_distance=data.get('min_distance', 10)
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@module3_bp.route('/api/detect_boundary', methods=['POST'])
@login_required
def api_detect_boundary():
    """API endpoint for boundary detection"""
    try:
        data = request.json
        result = detect_boundary_handler(
            image_data=data.get('image'),
            method=data.get('method', 'contour')
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@module3_bp.route('/api/segment_aruco', methods=['POST'])
@login_required
def api_segment_aruco():
    """API endpoint for ArUco segmentation"""
    try:
        data = request.json
        result = segment_with_aruco_handler(
            image_data=data.get('image')
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@module3_bp.route('/api/compare_sam2', methods=['POST'])
@login_required
def api_compare_sam2():
    """API endpoint for SAM2 comparison"""
    try:
        data = request.json
        result = compare_with_sam2_handler(
            image_data=data.get('image'),
            sam2_result_data=data.get('sam2_result')
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

