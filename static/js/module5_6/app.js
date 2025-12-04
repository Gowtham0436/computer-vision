// Main application logic

let video = null;
let canvas = null;
let ctx = null;
let stream = null;
let tracker = null;
let isRunning = false;
let isSelectingRegion = false;
let isSAM2Selecting = false;
let selectionStart = null;
let selectionRect = null;

// Initialize when OpenCV is ready
function onOpenCvReady() {
    console.log('OpenCV.js is ready');
    tracker = new ObjectTracker();
    initializeUI();
}

// Check if OpenCV is already loaded
if (typeof cv !== 'undefined') {
    onOpenCvReady();
} else {
    cv['onRuntimeInitialized'] = onOpenCvReady;
}

function initializeUI() {
    video = document.getElementById('video');
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');
    
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const selectRegionBtn = document.getElementById('selectRegionBtn');
    const trackingMode = document.getElementById('trackingMode');
    const sam2File = document.getElementById('sam2File');
    const loadSam2Btn = document.getElementById('loadSam2Btn');
    const selectObjectBtn = document.getElementById('selectObjectBtn');
    const sam2Controls = document.getElementById('sam2Controls');
    
    startBtn.addEventListener('click', startCamera);
    stopBtn.addEventListener('click', stopCamera);
    selectRegionBtn.addEventListener('click', toggleRegionSelection);
    trackingMode.addEventListener('change', onModeChange);
    loadSam2Btn.addEventListener('click', loadSAM2File);
    if (selectObjectBtn) {
        selectObjectBtn.addEventListener('click', startSAM2Selection);
    }
    
    // Canvas click handler for region selection
    canvas.addEventListener('mousedown', onMouseDown);
    canvas.addEventListener('mousemove', onMouseMove);
    canvas.addEventListener('mouseup', onMouseUp);
}

function onModeChange() {
    const mode = document.getElementById('trackingMode').value;
    tracker.setMode(mode);
    
    const sam2Controls = document.getElementById('sam2Controls');
    if (mode === 'sam2') {
        sam2Controls.style.display = 'block';
    } else {
        sam2Controls.style.display = 'none';
    }
    
    updateStatus(`Mode changed to: ${mode}`);
}

function startCamera() {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(mediaStream => {
            stream = mediaStream;
            video.srcObject = stream;
            // Mirror the video element for natural left/right movement
            video.style.transform = 'scaleX(-1)';
            video.play();
            
            video.addEventListener('loadedmetadata', () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                isRunning = true;
                processVideo();
            });
            
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            document.getElementById('selectRegionBtn').disabled = false;
            updateStatus('Camera started');
        })
        .catch(err => {
            console.error('Error accessing camera:', err);
            updateStatus('Error: Could not access camera');
            alert('Could not access camera. Please check permissions.');
        });
}

function stopCamera() {
    isRunning = false;
    
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    video.srcObject = null;
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
    document.getElementById('selectRegionBtn').disabled = true;
    updateStatus('Camera stopped');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function toggleRegionSelection() {
    isSelectingRegion = !isSelectingRegion;
    const btn = document.getElementById('selectRegionBtn');
    
    if (isSelectingRegion) {
        btn.textContent = 'Cancel Selection';
        btn.style.background = '#f44336';
        updateStatus('Click and drag on video to select region');
    } else {
        btn.textContent = 'Select Region';
        btn.style.background = '#2196F3';
        selectionRect = null;
        updateStatus('Selection cancelled');
    }
}

function onMouseDown(e) {
    if (!isSelectingRegion || !isRunning) return;
    
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);
    
    selectionStart = { x, y };
    selectionRect = null; // Reset selection
}

function onMouseMove(e) {
    if (!isSelectingRegion || !selectionStart || !isRunning) return;
    
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);
    
    selectionRect = {
        x: Math.min(selectionStart.x, x),
        y: Math.min(selectionStart.y, y),
        width: Math.abs(x - selectionStart.x),
        height: Math.abs(y - selectionStart.y)
    };
}

function onMouseUp(e) {
    if (!isSelectingRegion || !selectionStart || !isRunning) return;
    
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);
    
    selectionRect = {
        x: Math.min(selectionStart.x, x),
        y: Math.min(selectionStart.y, y),
        width: Math.abs(x - selectionStart.x),
        height: Math.abs(y - selectionStart.y)
    };
    
    if (selectionRect.width > 10 && selectionRect.height > 10) {
        const mode = document.getElementById('trackingMode').value;
        
        if (mode === 'sam2' && isSAM2Selecting) {
            // For SAM2 mode, create mask from selected region and start tracking
            createSAM2FromSelection(selectionRect);
            isSAM2Selecting = false;
            const selectObjectBtn = document.getElementById('selectObjectBtn');
            if (selectObjectBtn) {
                selectObjectBtn.textContent = 'Select Object to Track';
                selectObjectBtn.style.background = '#9c27b0';
            }
        } else {
            // Capture template for markerless tracking
        captureTemplate(selectionRect);
            updateStatus('Region selected. Tracking started.');
        }
        
        isSelectingRegion = false;
        document.getElementById('selectRegionBtn').textContent = 'Select Region';
        document.getElementById('selectRegionBtn').style.background = '#2196F3';
    } else {
        // Selection too small, reset
        selectionRect = null;
        updateStatus('Selection too small. Try again.');
    }
    
    selectionStart = null;
}

function captureTemplate(rect) {
    // Create a temporary canvas to capture the video frame
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(video, 0, 0);
    
    // Get image data
    const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
    
    // Convert to OpenCV Mat
    const src = cv.matFromImageData(imageData);
    
    // Convert selection coordinates from mirrored canvas to original video coordinates
    // Canvas is mirrored, so x coordinate needs to be flipped
    const mirroredX = canvas.width - rect.x - rect.width;
    const videoRect = {
        x: Math.max(0, Math.round(mirroredX)),
        y: Math.max(0, Math.round(rect.y)),
        width: Math.min(Math.round(rect.width), tempCanvas.width - Math.round(mirroredX)),
        height: Math.min(Math.round(rect.height), tempCanvas.height - Math.round(rect.y))
    };
    
    // Set template in tracker using original video coordinates
    tracker.setTemplate(videoRect, src);
    
    src.delete();
}

async function loadSAM2File() {
    const fileInput = document.getElementById('sam2File');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select an NPZ file');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = async function(e) {
        try {
            updateStatus('Loading SAM2 file...');
            await tracker.loadSAM2Data(e.target.result);
            
            const maskCount = tracker.sam2Masks ? tracker.sam2Masks.length : 0;
            updateStatus(`SAM2 file loaded: ${maskCount} mask(s) ready for tracking`);
        } catch (err) {
            console.error('Error loading SAM2 file:', err);
            updateStatus('Error loading SAM2 file: ' + err.message);
        }
    };
    reader.readAsArrayBuffer(file);
}

function startSAM2Selection() {
    if (!isRunning) {
        alert('Please start the camera first');
        return;
    }
    
    isSAM2Selecting = true;
    isSelectingRegion = true;
    selectionRect = null;
    
    const selectObjectBtn = document.getElementById('selectObjectBtn');
    if (selectObjectBtn) {
        selectObjectBtn.textContent = 'Draw box around object...';
        selectObjectBtn.style.background = '#f44336';
    }
    
    updateStatus('Draw a box around the object you want to track');
}

function createSAM2FromSelection(rect) {
    try {
        // Get current frame from video (not canvas, since canvas is mirrored)
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = video.videoWidth;
        tempCanvas.height = video.videoHeight;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(video, 0, 0);
        const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
        const src = cv.matFromImageData(imageData);
        
        // Convert selection coordinates from mirrored canvas to original video coordinates
        // Canvas is mirrored, so x coordinate needs to be flipped
        const mirroredX = canvas.width - rect.x - rect.width;
        const x = Math.max(0, Math.round(mirroredX));
        const y = Math.max(0, Math.round(rect.y));
        const w = Math.min(Math.round(rect.width), tempCanvas.width - x);
        const h = Math.min(Math.round(rect.height), tempCanvas.height - y);
        
        if (w < 10 || h < 10) {
            updateStatus('Selection too small. Try again.');
            src.delete();
            return;
        }
        
        // Create a simple mask for the selected region (use video dimensions, not canvas)
        const mask = new cv.Mat.zeros(tempCanvas.height, tempCanvas.width, cv.CV_8UC1);
        cv.rectangle(mask, new cv.Point(x, y), new cv.Point(x + w, y + h), new cv.Scalar(255), -1);
        
        // Clear any existing masks
        if (tracker.sam2Masks) {
            tracker.sam2Masks.forEach(m => { if (m && !m.isDeleted()) m.delete(); });
        }
        if (tracker.sam2Template && !tracker.sam2Template.isDeleted()) {
            tracker.sam2Template.delete();
        }
        
        // Set up tracker with this mask
        tracker.sam2Data = true; // Mark as having data
        tracker.sam2Masks = [mask];
        tracker.sam2Centroids = [{
            x: x + w / 2,
            y: y + h / 2
        }];
        
        // Extract template from selected region
        tracker.sam2Template = src.roi(new cv.Rect(x, y, w, h)).clone();
        tracker.sam2TemplateRect = { x: x, y: y, width: w, height: h };
        
        src.delete();
        tempCanvas.width = 0; // Clean up
        
        updateStatus('✓ Object selected! Tracking started. Move the object around.');
        console.log('SAM2 tracking initialized from selection:', w, 'x', h, 'at', x, y);
    } catch (err) {
        console.error('Error creating SAM2 from selection:', err);
        updateStatus('Error: ' + err.message);
    }
}

function processVideo() {
    if (!isRunning) return;
    
    try {
        // Mirror the video when drawing to canvas (horizontal flip)
        ctx.save();
        ctx.scale(-1, 1);
        ctx.translate(-canvas.width, 0);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.restore();
        
        // Get image data with willReadFrequently for better performance
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        
        // Convert to OpenCV Mat
        const src = cv.matFromImageData(imageData);
        const dst = src.clone();
        
        // Process with tracker (only if not selecting region or if template is set)
        let tracked = false;
        if (!isSelectingRegion || tracker.template) {
            tracked = tracker.processFrame(src, dst);
        }
        
        // Convert back to image data and draw
        cv.imshow(canvas, dst);
        
        // Draw selection rectangle on top (after OpenCV processing)
        if (isSelectingRegion && selectionRect) {
            ctx.strokeStyle = 'yellow';
            ctx.fillStyle = 'rgba(255, 255, 0, 0.2)';
            ctx.lineWidth = 3;
            ctx.setLineDash([5, 5]);
            ctx.fillRect(
                selectionRect.x,
                selectionRect.y,
                selectionRect.width,
                selectionRect.height
            );
            ctx.strokeRect(
                selectionRect.x,
                selectionRect.y,
                selectionRect.width,
                selectionRect.height
            );
            ctx.setLineDash([]);
        }
        
        // Clean up
        src.delete();
        dst.delete();
        
        // Update status
        if (isSelectingRegion) {
            if (selectionRect) {
                updateStatus('Drag to adjust selection, release to confirm');
            } else {
                updateStatus('Click and drag on video to select region');
            }
        } else if (tracked) {
            updateStatus('Tracking: Object detected');
        } else if (tracker.template) {
            updateStatus('Tracking: Searching...');
        } else {
            updateStatus('Ready');
        }
    } catch (err) {
        console.error('Processing error:', err);
    }
    
    // Continue processing
    requestAnimationFrame(processVideo);
}

function updateStatus(message) {
    const statusEl = document.getElementById('status');
    statusEl.textContent = `Status: ${message}`;
}

