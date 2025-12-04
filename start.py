#!/usr/bin/env python3
"""Startup script for Railway - reads PORT from environment"""
import os
import sys

port = os.environ.get('PORT', '8080')
try:
    port_int = int(port)
except ValueError:
    print(f"Error: Invalid PORT value: {port}", file=sys.stderr)
    sys.exit(1)

# Start gunicorn
os.execvp('gunicorn', [
    'gunicorn',
    'app:app',
    '--bind', f'0.0.0.0:{port_int}',
    '--workers', '2',
    '--threads', '2',
    '--timeout', '120'
])

