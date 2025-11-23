"""
Module 7: Stereo Vision and Pose Estimation
Routes and blueprint definition
"""

from flask import Blueprint, render_template, request, jsonify, send_file
from core.decorators import login_required
from .handlers import (
    estimate_size_calibrated_stereo_handler,
    estimate_pose_hand_tracking_handler,
    save_pose_csv_to_file
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
    """Problem 1: Calibrated Stereo Size Estimation"""
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

# API Routes
@module7_bp.route('/api/calibrated_stereo', methods=['POST'])
@login_required
def api_calibrated_stereo():
    """API endpoint for calibrated stereo size estimation"""
    try:
        data = request.json
        result = estimate_size_calibrated_stereo_handler(
            left_image_data=data.get('left_image'),
            right_image_data=data.get('right_image'),
            camera_params=data.get('camera_params', {}),
            object_type=data.get('object_type', 'rectangular')
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@module7_bp.route('/api/pose_tracking', methods=['POST'])
@login_required
def api_pose_tracking():
    """API endpoint for pose and hand tracking"""
    try:
        data = request.json
        result = estimate_pose_hand_tracking_handler(
            image_data=data.get('image'),
            use_mediapipe=data.get('use_mediapipe', True)
        )
        
        # Save CSV if successful
        if result.get('success') and result.get('csv_data'):
            csv_filename = save_pose_csv_to_file(result['csv_data'])
            result['csv_filename'] = csv_filename
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@module7_bp.route('/api/download_csv/<filename>')
@login_required
def download_csv(filename):
    """Download CSV file"""
    try:
        return send_file(f'static/outputs/{filename}', as_attachment=True, mimetype='text/csv')
    except Exception as e:
        return jsonify({'error': str(e)}), 404

