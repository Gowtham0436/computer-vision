"""
Module 1: Object Dimension Measurement
Routes and blueprint definition
"""

from flask import Blueprint, render_template, request, jsonify
from core.decorators import login_required
from .handlers import calculate_roi_dimensions, calculate_points_dimensions, calculate_evaluation_metrics, calculate_evaluation_metrics

# Create blueprint
module1_bp = Blueprint(
    'module1',
    __name__,
    template_folder='templates',
    url_prefix='/module1'
)

@module1_bp.route('/')
@login_required
def home():
    """Module 1 home page - problem selection"""
    return render_template('module1/module1_home.html')

@module1_bp.route('/problem1')
@login_required
def problem1():
    """Problem 1: ROI Selection Method"""
    return render_template('module1/problem1.html')

@module1_bp.route('/problem2')
@login_required
def problem2():
    """Problem 2: Two-Point Click Method"""
    return render_template('module1/problem2.html')

# API Routes
@module1_bp.route('/api/calculate_roi', methods=['POST'])
@login_required
def api_calculate_roi():
    """
    API endpoint for ROI selection calculation
    Expects JSON with: image, roi, params
    """
    try:
        data = request.json
        result = calculate_roi_dimensions(
            image_data=data.get('image'),
            roi=data.get('roi'),
            params=data.get('params')
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@module1_bp.route('/api/calculate_points', methods=['POST'])
@login_required
def api_calculate_points():
    """
    API endpoint for two-point calculation
    Expects JSON with: image, point1, point2, params
    """
    try:
        data = request.json
        result = calculate_points_dimensions(
            image_data=data.get('image'),
            point1=data.get('point1'),
            point2=data.get('point2'),
            params=data.get('params')
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@module1_bp.route('/api/evaluation', methods=['GET'])
@login_required
def api_evaluation():
    """
    API endpoint to get evaluation metrics for all objects
    """
    try:
        results = calculate_evaluation_metrics()
        return jsonify({'success': True, 'data': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
