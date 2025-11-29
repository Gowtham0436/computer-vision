"""
Module 2: Advanced Computer Vision
Routes and blueprint definition
"""

from flask import Blueprint, render_template, request, jsonify
from core.decorators import login_required
from .handlers import (
    match_template_handler,
    restore_image_handler,
    detect_and_blur_handler,
    save_template,
    list_templates,
    delete_template
)

# Create blueprint
module2_bp = Blueprint(
    'module2',
    __name__,
    template_folder='templates',
    url_prefix='/module2'
)

@module2_bp.route('/')
@login_required
def home():
    """Module 2 home page - problem selection"""
    return render_template('module2/module2_home.html')

@module2_bp.route('/problem1')
@login_required
def problem1():
    """Problem 1: Template Matching"""
    return render_template('module2/problem1.html')

@module2_bp.route('/problem2')
@login_required
def problem2():
    """Problem 2: Fourier Image Restoration"""
    return render_template('module2/problem2.html')

@module2_bp.route('/problem3')
@login_required
def problem3():
    """Problem 3: Multi-object Detection and Blurring"""
    return render_template('module2/problem3.html')

# API Routes
@module2_bp.route('/api/match_template', methods=['POST'])
@login_required
def api_match_template():
    """
    API endpoint for template matching
    Expects JSON with: template, target, threshold (optional, default 0.60)
    """
    try:
        data = request.json
        result = match_template_handler(
            template_data=data.get('template'),
            target_data=data.get('target'),
            threshold=float(data.get('threshold', 0.60))
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@module2_bp.route('/api/restore_image', methods=['POST'])
@login_required
def api_restore_image():
    """
    API endpoint for Fourier image restoration
    Expects JSON with: image
    """
    try:
        data = request.json
        result = restore_image_handler(
            image_data=data.get('image')
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@module2_bp.route('/api/detect_blur', methods=['POST'])
@login_required
def api_detect_blur():
    """
    API endpoint for object detection and blurring
    Expects JSON with: image
    """
    try:
        data = request.json
        result = detect_and_blur_handler(
            image_data=data.get('image')
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@module2_bp.route('/api/upload_template', methods=['POST'])
@login_required
def api_upload_template():
    """
    API endpoint for uploading template images
    Expects JSON with: image, name (optional)
    """
    try:
        data = request.json
        result = save_template(
            image_data=data.get('image'),
            template_name=data.get('name')
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@module2_bp.route('/api/list_templates', methods=['GET'])
@login_required
def api_list_templates():
    """
    API endpoint for listing all uploaded templates
    """
    try:
        result = list_templates()
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@module2_bp.route('/api/delete_template', methods=['POST'])
@login_required
def api_delete_template():
    """
    API endpoint for deleting a template
    Expects JSON with: filename
    """
    try:
        data = request.json
        result = delete_template(filename=data.get('filename'))
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
