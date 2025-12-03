# Computer Vision Assignment 3 - Run Commands

This project contains multiple web-based computer vision applications using OpenCV.js. All applications need to be served via a local web server (cannot be opened directly as file:// due to CORS restrictions).

## Project Structure

```
Assignment 3/
├── webapp/              # Main comprehensive computer vision demo application
├── Edge&Corner/         # Edges and corners detection application
├── objectBoundary/      # Object boundary detection application
├── dataset/             # Image datasets for testing
└── outputs/             # Generated output results
```

## Prerequisites

- A modern web browser (Chrome, Firefox, Edge, Safari)
- Python 3.x (for Python HTTP server) OR Node.js (for http-server) OR any local web server

## Running the Applications

### Option 1: Using Python HTTP Server (Recommended - Cross-platform)

#### Main Web Application (`webapp/`)

**Windows (PowerShell):**
```powershell
cd webapp
python -m http.server 8000
```

**Windows (Command Prompt):**
```cmd
cd webapp
python -m http.server 8000
```

**Linux/Mac:**
```bash
cd webapp
python3 -m http.server 8000
```

Then open in browser: `http://localhost:8000`

---

#### Edges & Corners Application (`Edge&Corner/`)

**Windows (PowerShell):**
```powershell
cd "Edge&Corner"
python -m http.server 8001
```

**Windows (Command Prompt):**
```cmd
cd "Edge&Corner"
python -m http.server 8001
```

**Linux/Mac:**
```bash
cd "Edge&Corner"
python3 -m http.server 8001
```

Then open in browser: `http://localhost:8001/edges_corners_app.html`

---

#### Object Boundary Application (`objectBoundary/`)

**Windows (PowerShell):**
```powershell
cd objectBoundary
python -m http.server 8002
```

**Windows (Command Prompt):**
```cmd
cd objectBoundary
python -m http.server 8002
```

**Linux/Mac:**
```bash
cd objectBoundary
python3 -m http.server 8002
```

Then open in browser: `http://localhost:8002/object_boundary_app.html`

---

### Option 2: Using Node.js http-server

First, install http-server globally (if not already installed):
```bash
npm install -g http-server
```

#### Main Web Application (`webapp/`)
```bash
cd webapp
http-server -p 8000
```

#### Edges & Corners Application (`Edge&Corner/`)
```bash
cd "Edge&Corner"
http-server -p 8001
```

#### Object Boundary Application (`objectBoundary/`)
```bash
cd objectBoundary
http-server -p 8002
```

---

### Option 3: Using PHP Built-in Server

**Windows (PowerShell):**
```powershell
cd webapp
php -S localhost:8000
```

**Linux/Mac:**
```bash
cd webapp
php -S localhost:8000
```

---

### Option 4: Using VS Code Live Server Extension

1. Install the "Live Server" extension in VS Code
2. Right-click on `index.html` in the respective folder
3. Select "Open with Live Server"

---

## Quick Start - Run All Applications Simultaneously

### Windows (PowerShell) - Run in Separate Terminals

**Terminal 1 - Main Web App:**
```powershell
cd "C:\MS-CS25\Fall 25\Computer Vison\Assignment 3\webapp"
python -m http.server 8000
```

**Terminal 2 - Edges & Corners:**
```powershell
cd "C:\MS-CS25\Fall 25\Computer Vison\Assignment 3\Edge&Corner"
python -m http.server 8001
```

**Terminal 3 - Object Boundary:**
```powershell
cd "C:\MS-CS25\Fall 25\Computer Vison\Assignment 3\objectBoundary"
python -m http.server 8002
```

### Windows (PowerShell) - Run in Background

```powershell
# Main Web App
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\MS-CS25\Fall 25\Computer Vison\Assignment 3\webapp'; python -m http.server 8000"

# Edges & Corners
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\MS-CS25\Fall 25\Computer Vison\Assignment 3\Edge&Corner'; python -m http.server 8001"

# Object Boundary
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\MS-CS25\Fall 25\Computer Vison\Assignment 3\objectBoundary'; python -m http.server 8002"
```

---

## Application URLs

Once the servers are running, access the applications at:

- **Main Web Application:** http://localhost:8000
- **Edges & Corners App:** http://localhost:8001/edges_corners_app.html
- **Object Boundary App:** http://localhost:8002/object_boundary_app.html

---

## Features

### Main Web Application (`webapp/`)
- Gradient magnitude and angle visualization
- Laplacian of Gaussian (LoG) edge detection
- Canny edge detection with adjustable parameters
- Harris corner detection
- Object boundary detection
- ArUco marker detection
- SAM2 (Segment Anything Model 2) integration
- Webcam support
- Image upload and processing

### Edges & Corners Application (`Edge&Corner/`)
- Sobel edge detection with NMS and hysteresis
- Harris corner detection
- Real-time parameter adjustment

### Object Boundary Application (`objectBoundary/`)
- Classic OpenCV object boundary detection
- Interactive boundary selection
- No machine learning dependencies

---

## Troubleshooting

### Port Already in Use
If a port is already in use, choose a different port:
```bash
python -m http.server 8080  # Use 8080 instead of 8000
```

### CORS Errors
If you see CORS errors, make sure you're accessing the files through a web server (http://localhost) and not directly as file:// URLs.

### OpenCV.js Not Loading
Ensure that the `webapp/build_wasm/bin/` folder contains the OpenCV.js files:
- `opencv.js`
- `opencv_js.js`
- `loader.js`

### Webcam Not Working
- Ensure you grant camera permissions when prompted
- Use HTTPS or localhost (some browsers require secure context for camera access)

---

## Notes

- All applications require a local web server due to browser security restrictions (CORS)
- The main webapp uses OpenCV.js with WebAssembly for performance
- Image datasets are located in the `dataset/` folder
- Generated outputs are saved in the `outputs/` folder

