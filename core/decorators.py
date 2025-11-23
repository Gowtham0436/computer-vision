"""
Decorators for route protection and authentication
"""

from functools import wraps
from flask import session, jsonify, redirect, url_for

def login_required(f):
    """
    Decorator to require authentication for routes
    Returns JSON error for API calls, redirects for page views
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('authenticated'):
            # Check if this is an API call (JSON request)
            from flask import request
            if request.is_json or '/api/' in request.path:
                return jsonify({'error': 'Not authenticated'}), 403
            # Otherwise redirect to login
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function
