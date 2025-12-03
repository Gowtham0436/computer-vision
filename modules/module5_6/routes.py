"""
Module 5-6: Motion Tracking and Real-time Object Tracking
Routes and blueprint definition
"""

from flask import Blueprint, render_template, request, jsonify
from core.decorators import login_required
from .handlers import (
    compute_motion_estimate_handler
)

# Create blueprint
module5_6_bp = Blueprint(
    'module5_6',
    __name__,
    template_folder='templates',
    url_prefix='/module5_6'
)

@module5_6_bp.route('/')
@login_required
def home():
    """Module 5-6 home page - problem selection"""
    return render_template('module5_6/module5_6_home.html')

@module5_6_bp.route('/problem1')
@login_required
def problem1():
    """Problem 1: Theoretical derivations and motion estimation"""
    return render_template('module5_6/problem1.html')

@module5_6_bp.route('/problem2')
@login_required
def problem2():
    """Problem 2: Real-time Object Tracking (Marker-based, Markerless, SAM2)"""
    return render_template('module5_6/problem2.html')

# API Routes
@module5_6_bp.route('/api/compute_motion', methods=['POST'])
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

