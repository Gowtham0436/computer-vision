# Use Python 3.11 slim image with Debian Bookworm (stable) for better package availability
FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV and MediaPipe
# Using packages that work with both Bookworm and newer Debian versions
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Railway will set PORT env var)
EXPOSE 8080

# Set environment variable for OpenCV to use headless backend
ENV OPENCV_IO_ENABLE_OPENEXR=0
ENV QT_QPA_PLATFORM=offscreen

# Run the application (Railway sets PORT env var)
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-8080} --workers 2 --threads 2 --timeout 120

