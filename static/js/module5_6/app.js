// Main application logic

let video = null;
let canvas = null;
let ctx = null;
let stream = null;
let tracker = null;
let isRunning = false;
let isSelectingRegion = false;
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
    const captureFrameBtn = document.getElementById('captureFrameBtn');
    const sam2Controls = document.getElementById('sam2Controls');
    
    startBtn.addEventListener('click', startCamera);
    stopBtn.addEventListener('click', stopCamera);
    selectRegionBtn.addEventListener('click', toggleRegionSelection);
    trackingMode.addEventListener('change', onModeChange);
    loadSam2Btn.addEventListener('click', loadSAM2File);
    if (captureFrameBtn) {
        captureFrameBtn.addEventListener('click', captureFrameAndCreateNPZ);
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
        
        if (mode === 'sam2') {
            // For SAM2 mode, create NPZ from selected region
            createNPZFromRegion(selectionRect);
        } else {
            // For markerless mode, capture template
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
    
    // Set template in tracker
    tracker.setTemplate(rect, src);
    
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
            
            // Note: For manually loaded NPZ files, we don't have the original template
            // The tracking will use the mask position (static overlay)
            // To enable dynamic tracking, use "Capture Frame & Create NPZ" button
            
            const maskCount = tracker.sam2Masks ? tracker.sam2Masks.length : 0;
            updateStatus(`SAM2 file loaded: ${maskCount} mask(s) ready (static tracking - use "Capture Frame" for dynamic tracking)`);
        } catch (err) {
            console.error('Error loading SAM2 file:', err);
            updateStatus('Error loading SAM2 file: ' + err.message);
        }
    };
    reader.readAsArrayBuffer(file);
}

async function createNPZFromRegion(rect) {
    if (!video || !video.videoWidth) {
        alert('Please start the camera first');
        return;
    }
    
    try {
        updateStatus('Creating NPZ from selected region...');
        
        // Capture current frame from canvas
        const imageData = canvas.toDataURL('image/jpeg', 0.9);
        const base64Data = imageData.split(',')[1];
        
        // Send to backend to create NPZ from region
        const response = await fetch('/module5_6/api/create_sam2_npz_region', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                image: base64Data,
                x: rect.x,
                y: rect.y,
                width: rect.width,
                height: rect.height
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Convert base64 NPZ to blob
            const npzData = Uint8Array.from(atob(result.npz_file), c => c.charCodeAt(0));
            
            // Capture the current frame as template for tracking
            const frameImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const templateMat = cv.matFromImageData(frameImageData);
            
            // Load the NPZ directly into tracker
            await tracker.loadSAM2Data(npzData.buffer);
            
            // Store the template image for template matching (from the selected region)
            const x = Math.max(0, Math.round(rect.x));
            const y = Math.max(0, Math.round(rect.y));
            const w = Math.min(Math.round(rect.width), templateMat.cols - x);
            const h = Math.min(Math.round(rect.height), templateMat.rows - y);
            
            if (w > 0 && h > 0) {
                tracker.sam2TemplateRect = new cv.Rect(x, y, w, h);
                tracker.sam2Template = templateMat.roi(new cv.Rect(x, y, w, h));
                console.log('Template stored for SAM2 tracking from region:', w, 'x', h);
            }
            
            templateMat.delete();
            
            const maskCount = tracker.sam2Masks ? tracker.sam2Masks.length : 0;
            updateStatus(`✓ NPZ created from selected region: ${maskCount} mask(s) ready for tracking`);
            
            // Optionally download the file
            const blob = new Blob([npzData], { type: 'application/zip' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'sam2_selected_region.npz';
            a.click();
            URL.revokeObjectURL(url);
        } else {
            alert('Error: ' + (result.error || 'Failed to create NPZ from region'));
            updateStatus('Error: ' + (result.error || 'Failed to create NPZ from region'));
        }
    } catch (err) {
        console.error('Error creating NPZ from region:', err);
        updateStatus('Error: ' + err.message);
        alert('Error creating NPZ: ' + err.message);
    }
}

async function captureFrameAndCreateNPZ() {
    if (!video || !video.videoWidth) {
        alert('Please start the camera first');
        return;
    }
    
    try {
        updateStatus('Capturing frame and creating NPZ...');
        
        // Capture current frame from canvas
        const imageData = canvas.toDataURL('image/jpeg', 0.9);
        const base64Data = imageData.split(',')[1];
        
        // Send to backend to create NPZ
        const response = await fetch('/module5_6/api/create_sam2_npz', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ image: base64Data })
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Convert base64 NPZ to blob
            const npzData = Uint8Array.from(atob(result.npz_file), c => c.charCodeAt(0));
            
            // Capture the current frame as template for tracking
            const frameImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const templateMat = cv.matFromImageData(frameImageData);
            
            // Load the NPZ directly into tracker
            await tracker.loadSAM2Data(npzData.buffer);
            
            // Store the template image for template matching
            if (tracker.sam2Masks && tracker.sam2Masks.length > 0) {
                const mask = tracker.sam2Masks[0];
                const rect = cv.boundingRect(mask);
                
                // Scale rect to match captured frame size
                const scaleX = templateMat.cols / mask.cols;
                const scaleY = templateMat.rows / mask.rows;
                
                const scaledRect = new cv.Rect(
                    Math.round(rect.x * scaleX),
                    Math.round(rect.y * scaleY),
                    Math.round(rect.width * scaleX),
                    Math.round(rect.height * scaleY)
                );
                
                tracker.sam2TemplateRect = scaledRect;
                
                // Extract template region from captured frame
                const x = Math.max(0, scaledRect.x);
                const y = Math.max(0, scaledRect.y);
                const w = Math.min(scaledRect.width, templateMat.cols - x);
                const h = Math.min(scaledRect.height, templateMat.rows - y);
                
                if (w > 0 && h > 0) {
                    tracker.sam2Template = templateMat.roi(new cv.Rect(x, y, w, h));
                    console.log('Template stored for SAM2 tracking:', w, 'x', h);
                }
            }
            
            templateMat.delete();
            
            const maskCount = tracker.sam2Masks ? tracker.sam2Masks.length : 0;
            updateStatus(`✓ NPZ created and loaded: ${maskCount} mask(s) ready for tracking`);
            
            // Optionally download the file
            const blob = new Blob([npzData], { type: 'application/zip' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'sam2_captured_frame.npz';
            a.click();
            URL.revokeObjectURL(url);
        } else {
            alert('Error: ' + (result.error || 'Failed to create NPZ'));
            updateStatus('Error: ' + (result.error || 'Failed to create NPZ'));
        }
    } catch (err) {
        console.error('Error creating NPZ:', err);
        updateStatus('Error: ' + err.message);
        alert('Error creating NPZ: ' + err.message);
    }
}

function processVideo() {
    if (!isRunning) return;
    
    try {
        // Draw video frame to canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
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

