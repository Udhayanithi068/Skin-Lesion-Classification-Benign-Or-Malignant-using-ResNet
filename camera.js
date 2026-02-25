/**
 * Camera functionality for Skin Lesion Classification
 */

class CameraHandler {
    constructor() {
        this.video = null;
        this.canvas = null;
        this.stream = null;
        this.capturedImageBlob = null;
        
        this.init();
    }
    
    init() {
        this.video = document.getElementById('camera-video');
        this.canvas = document.getElementById('camera-canvas');
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        // Camera controls
        const startButton = document.getElementById('start-camera');
        const captureButton = document.getElementById('capture-image');
        const retakeButton = document.getElementById('retake-image');
        const analyzeButton = document.getElementById('analyze-capture');
        
        if (startButton) {
            startButton.addEventListener('click', () => this.startCamera());
        }
        
        if (captureButton) {
            captureButton.addEventListener('click', () => this.captureImage());
        }
        
        if (retakeButton) {
            retakeButton.addEventListener('click', () => this.retakeImage());
        }
        
        if (analyzeButton) {
            analyzeButton.addEventListener('click', () => this.analyzeCapture());
        }
    }
    
    async startCamera() {
        try {
            // Request camera permission
            const constraints = {
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'environment' // Use back camera on mobile
                }
            };
            
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            
            // Show video element
            this.video.srcObject = this.stream;
            this.video.style.display = 'block';
            this.canvas.style.display = 'none';
            
            // Hide placeholder and start button
            const placeholder = document.getElementById('camera-placeholder');
            const startButton = document.getElementById('start-camera');
            const captureButton = document.getElementById('capture-image');
            
            if (placeholder) placeholder.style.display = 'none';
            if (startButton) startButton.style.display = 'none';
            if (captureButton) captureButton.style.display = 'block';
            
            // Show camera tips
            this.showToast('Camera started successfully! Position the lesion in view and click Capture.', 'success');
            
        } catch (error) {
            console.error('Camera access error:', error);
            this.handleCameraError(error);
        }
    }
    
    captureImage() {
        if (!this.video || !this.canvas) {
            this.showToast('Camera not properly initialized', 'error');
            return;
        }
        
        try {
            // Set canvas dimensions to match video
            const context = this.canvas.getContext('2d');
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;
            
            // Draw current video frame to canvas
            context.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            
            // Convert canvas to blob
            this.canvas.toBlob((blob) => {
                this.capturedImageBlob = blob;
                
                // Show canvas, hide video
                this.video.style.display = 'none';
                this.canvas.style.display = 'block';
                
                // Update button states
                const captureButton = document.getElementById('capture-image');
                const retakeButton = document.getElementById('retake-image');
                const analyzeButton = document.getElementById('analyze-capture');
                
                if (captureButton) captureButton.style.display = 'none';
                if (retakeButton) retakeButton.style.display = 'block';
                if (analyzeButton) analyzeButton.style.display = 'block';
                
                this.showToast('Image captured! Review and click Analyze when ready.', 'success');
                
            }, 'image/jpeg', 0.9);
            
        } catch (error) {
            console.error('Capture error:', error);
            this.showToast('Failed to capture image', 'error');
        }
    }
    
    retakeImage() {
        // Show video, hide canvas
        this.video.style.display = 'block';
        this.canvas.style.display = 'none';
        
        // Reset button states
        const captureButton = document.getElementById('capture-image');
        const retakeButton = document.getElementById('retake-image');
        const analyzeButton = document.getElementById('analyze-capture');
        
        if (captureButton) captureButton.style.display = 'block';
        if (retakeButton) retakeButton.style.display = 'none';
        if (analyzeButton) analyzeButton.style.display = 'none';
        
        // Clear captured image
        this.capturedImageBlob = null;
    }
    
    async analyzeCapture() {
        if (!this.capturedImageBlob) {
            this.showToast('No image captured', 'error');
            return;
        }
        
        try {
            // Show loading state
            this.showAnalysisLoading(true);
            
            // Prepare form data
            const formData = new FormData();
            formData.append('file', this.capturedImageBlob, 'camera-capture.jpg');
            
            // Generate session ID for temp capture
            const sessionId = this.generateSessionId();
            formData.append('session_id', sessionId);
            
            // Make API request
            const response = await fetch(`${window.location.origin}/predict/temp`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Analysis failed');
            }
            
            const result = await response.json();
            
            // Display results (use the main app's display function)
            if (window.app) {
                window.app.displayResults(result);
            }
            
            // Stop camera after successful analysis
            this.stopCamera();
            
            this.showToast('Analysis completed successfully!', 'success');
            
        } catch (error) {
            console.error('Analysis error:', error);
            this.showToast(`Analysis failed: ${error.message}`, 'error');
        } finally {
            this.showAnalysisLoading(false);
        }
    }
    
    stopCamera() {
        // Stop video stream
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        // Reset UI
        this.video.style.display = 'none';
        this.canvas.style.display = 'none';
        
        const placeholder = document.getElementById('camera-placeholder');
        const startButton = document.getElementById('start-camera');
        const captureButton = document.getElementById('capture-image');
        const retakeButton = document.getElementById('retake-image');
        const analyzeButton = document.getElementById('analyze-capture');
        
        if (placeholder) placeholder.style.display = 'flex';
        if (startButton) startButton.style.display = 'block';
        if (captureButton) captureButton.style.display = 'none';
        if (retakeButton) retakeButton.style.display = 'none';
        if (analyzeButton) analyzeButton.style.display = 'none';
        
        // Clear captured image
        this.capturedImageBlob = null;
    }
    
    handleCameraError(error) {
        let message = 'Camera access failed';
        
        if (error.name === 'NotAllowedError') {
            message = 'Camera permission denied. Please allow camera access and try again.';
        } else if (error.name === 'NotFoundError') {
            message = 'No camera found. Please ensure your device has a camera.';
        } else if (error.name === 'NotSupportedError') {
            message = 'Camera not supported by your browser. Please use a modern browser.';
        } else if (error.name === 'NotReadableError') {
            message = 'Camera is being used by another application. Please close other apps and try again.';
        }
        
        this.showToast(message, 'error');
        
        // Show fallback message
        const placeholder = document.getElementById('camera-placeholder');
        if (placeholder) {
            placeholder.innerHTML = `
                <div class="text-center">
                    <i class="fas fa-exclamation-triangle fa-3x text-warning mb-3"></i>
                    <p class="text-muted">${message}</p>
                    <p class="small text-muted">You can still use the image upload feature above.</p>
                </div>
            `;
        }
    }
    
    showAnalysisLoading(show) {
        const analyzeButton = document.getElementById('analyze-capture');
        
        if (show && analyzeButton) {
            analyzeButton.innerHTML = `
                <span class="spinner-border spinner-border-sm me-2" role="status"></span>
                Analyzing...
            `;
            analyzeButton.disabled = true;
        } else if (analyzeButton) {
            analyzeButton.innerHTML = `
                <i class="fas fa-brain me-2"></i>Analyze
            `;
            analyzeButton.disabled = false;
        }
    }
    
    generateSessionId() {
        return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
    
    showToast(message, type = 'info') {
        // Use the main app's toast function if available
        if (window.app && window.app.showToast) {
            window.app.showToast(message, type);
        } else {
            // Fallback to console log
            console.log(`${type.toUpperCase()}: ${message}`);
        }
    }
    
    // Check if camera is supported
    static isCameraSupported() {
        return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    }
    
    // Get available camera devices
    async getAvailableCameras() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            return devices.filter(device => device.kind === 'videoinput');
        } catch (error) {
            console.error('Error getting camera devices:', error);
            return [];
        }
    }
}

// Global functions for HTML onclick handlers
function startCamera() {
    if (window.cameraHandler) {
        window.cameraHandler.startCamera();
    }
}

function captureImage() {
    if (window.cameraHandler) {
        window.cameraHandler.captureImage();
    }
}

function retakeImage() {
    if (window.cameraHandler) {
        window.cameraHandler.retakeImage();
    }
}

function analyzeCapture() {
    if (window.cameraHandler) {
        window.cameraHandler.analyzeCapture();
    }
}

// Initialize camera handler when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Check camera support
    if (!CameraHandler.isCameraSupported()) {
        console.warn('Camera not supported in this browser');
        
        const placeholder = document.getElementById('camera-placeholder');
        if (placeholder) {
            placeholder.innerHTML = `
                <div class="text-center">
                    <i class="fas fa-exclamation-triangle fa-3x text-warning mb-3"></i>
                    <p class="text-muted">Camera not supported in this browser</p>
                    <p class="small text-muted">Please use the image upload feature instead.</p>
                </div>
            `;
        }
        
        // Hide camera controls
        const startButton = document.getElementById('start-camera');
        if (startButton) {
            startButton.style.display = 'none';
        }
        
        return;
    }
    
    // Initialize camera handler
    window.cameraHandler = new CameraHandler();
});

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (window.cameraHandler) {
        window.cameraHandler.stopCamera();
    }
});