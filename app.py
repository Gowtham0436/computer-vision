"""
CSc 8830 Assignment - Refactored Flask Application
Main application file - handles initialization and blueprint registration
"""

from flask import Flask, render_template, session
from datetime import timedelta
import secrets
import os

def create_app():
    """Application factory pattern"""
    app = Flask(__name__)
    
    # Configuration
    # SECRET_KEY must be set as environment variable (required for production)
    app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
    if app.secret_key == 'dev-key-change-in-production' and os.environ.get('FLASK_ENV') == 'production':
        raise ValueError("SECRET_KEY environment variable must be set for production")
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
    app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
    
    # Session configuration - persist across page refreshes and server restarts
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)  # 24 hours
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    # Set to True in production with HTTPS (Render, Railway provide HTTPS)
    app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FLASK_ENV') == 'production'
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    app.config['SESSION_COOKIE_NAME'] = 'cv_session'
    # Make session cookie persist across browser sessions
    app.config['SESSION_COOKIE_MAX_AGE'] = 86400  # 24 hours in seconds
    
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('static/outputs', exist_ok=True)
    
    # Register core routes (authentication, module selection)
    register_core_routes(app)
    
    # Register module blueprints
    register_modules(app)
    
    return app

def register_core_routes(app):
    """Register authentication and main navigation routes"""
    from core.auth import face_auth
    from core.decorators import login_required
    from flask import jsonify, request
    
    @app.route('/')
    def index():
        """Landing page with face authentication"""
        return render_template('index.html')
    
    @app.route('/check_auth')
    def check_auth():
        """Check if user is authenticated"""
        return jsonify({'authenticated': session.get('authenticated', False)})
    
    @app.route('/register_face', methods=['POST'])
    def register_face():
        """Register a new face"""
        data = request.json
        image_data = data.get('image')
        name = data.get('name', 'User')
        
        success, message = face_auth.register_face_from_image(image_data, name)
        return jsonify({'success': success, 'message': message})
    
    @app.route('/authenticate', methods=['POST'])
    def authenticate():
        """Authenticate user via face recognition"""
        from datetime import datetime
        data = request.json
        image_data = data.get('image')
        
        success, name, confidence = face_auth.authenticate_from_image(image_data)
        
        if success:
            # Make session permanent so it persists across page refreshes
            session.permanent = True
            session['authenticated'] = True
            session['user_name'] = name
            session['auth_time'] = datetime.now().isoformat()
            # Mark session as modified to ensure it's saved
            session.modified = True
            return jsonify({
                'success': True, 
                'message': f'Welcome, {name}!',
                'confidence': f'{confidence*100:.1f}%'
            })
        else:
            return jsonify({'success': False, 'message': name})
    
    @app.route('/logout')
    def logout():
        """Logout user"""
        session.clear()
        return jsonify({'success': True})
    
    @app.route('/modules')
    @login_required
    def modules():
        """Module selection page after authentication"""
        return render_template('modules.html')

def register_modules(app):
    """Register all module blueprints"""
    # Import and register module blueprints
    from modules.module1 import module1_bp
    from modules.module2 import module2_bp
    from modules.module3 import module3_bp
    from modules.module4 import module4_bp
    from modules.module5 import module5_bp
    from modules.module7 import module7_bp
    
    app.register_blueprint(module1_bp)
    app.register_blueprint(module2_bp)
    app.register_blueprint(module3_bp)
    app.register_blueprint(module4_bp)
    app.register_blueprint(module5_bp)
    app.register_blueprint(module7_bp)
    
    # Future modules will be registered here:
    # from modules.module6 import module6_bp
    # app.register_blueprint(module6_bp)

# Create app instance for Gunicorn
app = create_app()

if __name__ == "__main__":
    # In production, use environment variable for port (Render, Railway, etc. set PORT)
    port = int(os.environ.get('PORT', 5000))
    # Only enable debug in development
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host="0.0.0.0", port=port, debug=debug)
