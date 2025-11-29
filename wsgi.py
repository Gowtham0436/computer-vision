"""
WSGI entry point for PythonAnywhere deployment
This file is used by PythonAnywhere to serve your Flask application

INSTRUCTIONS FOR PYTHONANYWHERE:
1. Copy this file content to your WSGI configuration file in PythonAnywhere
2. Replace 'yourusername' with your actual PythonAnywhere username
3. Update the project_home path if your repository is in a different location
"""

import sys
import os

# IMPORTANT: Replace 'yourusername' with your PythonAnywhere username
# Example: '/home/johndoe/computer-vision'
project_home = '/home/yourusername/computer-vision'

# Add your project directory to the Python path
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Change to project directory (important for relative paths)
os.chdir(project_home)

# Set environment variables
os.environ['FLASK_ENV'] = 'production'
# Generate a secret key: python3 -c "import secrets; print(secrets.token_hex(32))"
os.environ['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-this-to-a-random-secret-key')

# Import the Flask app
# Note: app.py creates 'app' instance at module level for WSGI compatibility
from app import app

# PythonAnywhere looks for 'application' variable
application = app

# Alternative if using create_app() pattern:
# from app import create_app
# application = create_app()

