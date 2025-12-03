#!/bin/bash
# Startup script for Railway deployment
# Uses PORT environment variable set by Railway

PORT=${PORT:-8080}
exec gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 120

