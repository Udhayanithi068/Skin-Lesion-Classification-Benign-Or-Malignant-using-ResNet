/**
 * Main Application JavaScript for Skin Lesion Classification
 */

class SkinLesionApp {
    constructor() {
        this.apiBaseUrl = window.location.origin;
        this.authToken = null;
        this.currentUser = null;
        
        // Don't use localStorage in Claude.ai artifacts - use memory instead
        this.init();
    }
    
    async init() {
        // Initialize UI
        this.initializeUI();
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Load user history if authenticated
        if (this.currentUser) {
            this.loadUserHistory();
        }
    }
    
    async checkAuthStatus() {
        try {
            if (!this.authToken) return false;
            
            const response = await fetch(`${this.apiBaseUrl}/system/info`, {
                headers: {
                    'Authorization': `Bearer ${this.authToken}`
                }
            });
            
            if (response.ok) {
                // Token is valid, show user section
                this.showUserSection();
                await this.loadUserHistory();
                return true;
            } else {
                // Token is invalid, clear it
                this.logout();
                return false;
            }
        } catch (error) {
            console.error('Auth check failed:', error);
            this.logout();
            return false;
        }
    }
    
    initializeUI() {
        // Update auth UI based on login status
        if (this.authToken && this.currentUser) {
            this.showUserSection();
        } else {
            this.showAuthSection();
        }
    }
    
    setupEventListeners() {
        // File input change listener
        const fileInput = document.getElementById('imageFile');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.previewImage(e.target));
        }
        
        // Navigation smooth scrolling
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    }
    
    showAuthSection() {
        const authSection = document.getElementById('auth-section');
        const userSection = document.getElementById('user-section');
        
        if (authSection) authSection.style.display = 'block';
        if (userSection) userSection.style.display = 'none';
    }
    
    showUserSection() {
        const authSection = document.getElementById('auth-section');
        const userSection = document.getElementById('user-section');
        const usernameDisplay = document.getElementById('username-display');
        
        if (authSection) authSection.style.display = 'none';
        if (userSection) userSection.style.display = 'block';
        
        if (usernameDisplay && this.currentUser) {
            usernameDisplay.textContent = this.currentUser.username;
        }
    }
    
    previewImage(input) {
        const preview = document.getElementById('image-preview');
        const previewImg = document.getElementById('preview-img');
        
        if (input.files && input.files[0]) {
            const file = input.files[0];
            
            // Validate file type
            if (!file.type.startsWith('image/')) {
                this.showToast('Please select a valid image file', 'error');
                return;
            }
            
            // Validate file size (10MB max)
            if (file.size > 10 * 1024 * 1024) {
                this.showToast('File size must be less than 10MB', 'error');
                return;
            }
            
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImg.src = e.target.result;
                preview.style.display = 'block';
                preview.classList.add('fade-in');
            };
            reader.readAsDataURL(file);
        }
    }
    
    async uploadImage() {
        const fileInput = document.getElementById('imageFile');
        const file = fileInput.files[0];
        
        if (!file) {
            this.showToast('Please select an image first', 'error');
            return;
        }
        
        // Show loading state
        this.showLoadingState(true);
        
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            // Choose endpoint based on authentication
            const endpoint = this.authToken ? '/predict' : '/predict/temp';
            const headers = {};
            
            if (this.authToken) {
                headers['Authorization'] = `Bearer ${this.authToken}`;
            }
            
            const response = await fetch(`${this.apiBaseUrl}${endpoint}`, {
                method: 'POST',
                headers: headers,
                body: formData
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Prediction failed');
            }
            
            const result = await response.json();
            this.displayResults(result);
            
            // Refresh history if user is authenticated
            if (this.authToken) {
                await this.loadUserHistory();
            }
            
        } catch (error) {
            console.error('Upload error:', error);
            this.showToast(`Error: ${error.message}`, 'error');
        } finally {
            this.showLoadingState(false);
        }
    }
    
    displayResults(result) {
        const resultsSection = document.getElementById('results');
        const resultsContent = document.getElementById('results-content');
        
        if (!resultsSection || !resultsContent) return;
        
        // Handle different prediction types from HF
        let isPredictionMalignant = false;
        let displayPrediction = result.prediction;
        
        if (result.prediction === 'potentially_concerning') {
            isPredictionMalignant = true;
            displayPrediction = 'Potentially Concerning';
        } else if (result.prediction === 'lesion_detected') {
            isPredictionMalignant = false;
            displayPrediction = 'Lesion Detected';
        } else if (result.prediction === 'malignant') {
            isPredictionMalignant = true;
            displayPrediction = 'Malignant';
        } else if (result.prediction === 'benign') {
            isPredictionMalignant = false;
            displayPrediction = 'Benign';
        } else {
            isPredictionMalignant = false;
            displayPrediction = 'Uncertain';
        }
        
        const confidenceLevel = this.getConfidenceLevel(result.confidence);
        const riskInfo = this.calculateRiskLevel(result.prediction, result.confidence);
        
        const resultsHTML = `
            <div class="result-card card ${isPredictionMalignant ? 'result-malignant' : 'result-benign'} fade-in">
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-6">
                            <h4 class="mb-3">
                                <i class="fas ${isPredictionMalignant ? 'fa-exclamation-triangle text-danger' : 'fa-check-circle text-success'} me-2"></i>
                                Analysis: ${displayPrediction}
                            </h4>
                            
                            <div class="mb-3">
                                <h6>Confidence Level</h6>
                                <div class="confidence-bar">
                                    <div class="confidence-fill confidence-${confidenceLevel.class}" 
                                         style="width: ${result.confidence * 100}%">
                                        ${(result.confidence * 100).toFixed(1)}%
                                    </div>
                                </div>
                                <small class="text-muted">${confidenceLevel.text}</small>
                            </div>
                            
                            <div class="mb-3">
                                <h6>Risk Assessment</h6>
                                <div class="alert alert-${riskInfo.alertClass} py-2">
                                    <strong>${riskInfo.level}</strong><br>
                                    <small>${riskInfo.recommendation}</small>
                                </div>
                            </div>
                            
                            <div class="row text-center">
                                <div class="col-6">
                                    <div class="border rounded p-3">
                                        <h6 class="text-success mb-1">Benign/Normal</h6>
                                        <strong>${(result.probabilities.benign * 100).toFixed(1)}%</strong>
                                    </div>
                                </div>
                                <div class="col-6">
                                    <div class="border rounded p-3">
                                        <h6 class="text-danger mb-1">Concerning</h6>
                                        <strong>${(result.probabilities.malignant * 100).toFixed(1)}%</strong>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="col-md-6">
                            ${result.hf_analysis ? `
                                <div class="card bg-light">
                                    <div class="card-header">
                                        <h6 class="mb-0">
                                            <i class="fas fa-robot me-2"></i>AI Analysis
                                        </h6>
                                    </div>
                                    <div class="card-body">
                                        <p class="small mb-0">${result.hf_analysis}</p>
                                    </div>
                                </div>
                            ` : ''}
                            
                            <div class="mt-3">
                                <h6>Technical Details</h6>
                                <ul class="list-unstyled small text-muted">
                                    <li><strong>Service:</strong> Hugging Face Vision API</li>
                                    <li><strong>Model:</strong> ${result.model_type || 'Vision Transformer'}</li>
                                    <li><strong>Processing Time:</strong> ${result.processing_time ? result.processing_time.toFixed(2) + 's' : 'N/A'}</li>
                                    <li><strong>Analysis Date:</strong> ${new Date(result.timestamp || Date.now()).toLocaleString()}</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="alert alert-warning fade-in">
                <div class="d-flex align-items-center">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    <div>
                        <strong>Important Disclaimer:</strong> This AI analysis is for educational and screening purposes only. 
                        Always consult with a qualified healthcare professional for proper medical diagnosis and treatment.
                        This tool should not replace professional medical advice.
                    </div>
                </div>
            </div>
        `;
        
        resultsContent.innerHTML = resultsHTML;
        resultsSection.style.display = 'block';
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    getConfidenceLevel(confidence) {
        if (confidence >= 0.9) {
            return { class: 'high', text: 'Very High Confidence' };
        } else if (confidence >= 0.8) {
            return { class: 'high', text: 'High Confidence' };
        } else if (confidence >= 0.6) {
            return { class: 'moderate', text: 'Moderate Confidence' };
        } else {
            return { class: 'low', text: 'Low Confidence' };
        }
    }
    
    calculateRiskLevel(prediction, confidence) {
        const isConcerning = prediction === 'malignant' || prediction === 'potentially_concerning';
        
        if (isConcerning) {
            if (confidence >= 0.8) {
                return {
                    level: 'High Priority - Professional Evaluation Recommended',
                    recommendation: 'Please schedule an appointment with a dermatologist promptly',
                    alertClass: 'danger'
                };
            } else if (confidence >= 0.6) {
                return {
                    level: 'Moderate Priority - Monitor Closely',
                    recommendation: 'Consider professional evaluation and monitor for changes',
                    alertClass: 'warning'
                };
            } else {
                return {
                    level: 'Uncertain - Caution Advised',
                    recommendation: 'Professional evaluation recommended for proper diagnosis',
                    alertClass: 'warning'
                };
            }
        } else {
            if (confidence >= 0.8) {
                return {
                    level: 'Low Concern',
                    recommendation: 'Continue regular skin monitoring and routine check-ups',
                    alertClass: 'success'
                };
            } else if (confidence >= 0.6) {
                return {
                    level: 'Monitor and Track',
                    recommendation: 'Keep watching for changes and maintain regular check-ups',
                    alertClass: 'info'
                };
            } else {
                return {
                    level: 'Uncertain Analysis',
                    recommendation: 'Professional evaluation recommended for accurate diagnosis',
                    alertClass: 'secondary'
                };
            }
        }
    }
    
    async loadUserHistory() {
        if (!this.authToken) return;
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/predictions`, {
                headers: {
                    'Authorization': `Bearer ${this.authToken}`
                }
            });
            
            if (response.ok) {
                const predictions = await response.json();
                this.displayHistory(predictions);
            }
        } catch (error) {
            console.error('Failed to load history:', error);
        }
    }
    
    displayHistory(predictions) {
        const historyContent = document.getElementById('history-content');
        if (!historyContent) return;
        
        if (predictions.length === 0) {
            historyContent.innerHTML = `
                <div class="text-center py-4">
                    <i class="fas fa-clipboard-list fa-3x text-muted mb-3"></i>
                    <p class="text-muted">No analysis history yet. Upload an image to get started!</p>
                </div>
            `;
            return;
        }
        
        const historyHTML = `
            <div class="table-responsive">
                <table class="table history-table">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Image</th>
                            <th>Prediction</th>
                            <th>Confidence</th>
                            <th>Service</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${predictions.map(pred => {
                            let badgeClass = 'secondary';
                            let displayPred = pred.prediction;
                            
                            if (pred.prediction === 'malignant' || pred.prediction === 'potentially_concerning') {
                                badgeClass = 'danger';
                            } else if (pred.prediction === 'benign' || pred.prediction === 'lesion_detected') {
                                badgeClass = 'success';
                            }
                            
                            return `
                                <tr>
                                    <td>${new Date(pred.timestamp).toLocaleDateString()}</td>
                                    <td><span class="small text-muted">${pred.image_filename}</span></td>
                                    <td>
                                        <span class="badge bg-${badgeClass}">
                                            ${displayPred.charAt(0).toUpperCase() + displayPred.slice(1)}
                                        </span>
                                    </td>
                                    <td>
                                        <div class="progress" style="height: 20px;">
                                            <div class="progress-bar ${pred.confidence > 0.8 ? 'bg-success' : pred.confidence > 0.6 ? 'bg-warning' : 'bg-danger'}" 
                                                 style="width: ${pred.confidence * 100}%">
                                                ${(pred.confidence * 100).toFixed(1)}%
                                            </div>
                                        </div>
                                    </td>
                                    <td><span class="small text-muted">Hugging Face</span></td>
                                    <td>
                                        <button class="btn btn-sm btn-outline-primary" onclick="app.viewPredictionDetails('${pred.id}')">
                                            <i class="fas fa-eye"></i>
                                        </button>
                                    </td>
                                </tr>
                            `;
                        }).join('')}
                    </tbody>
                </table>
            </div>
        `;
        
        historyContent.innerHTML = historyHTML;
    }
    
    showLoadingState(show) {
        const uploadForm = document.getElementById('upload-form');
        const uploadLoading = document.getElementById('upload-loading');
        
        if (show) {
            if (uploadForm) uploadForm.style.display = 'none';
            if (uploadLoading) uploadLoading.style.display = 'block';
        } else {
            if (uploadForm) uploadForm.style.display = 'block';
            if (uploadLoading) uploadLoading.style.display = 'none';
        }
    }
    
    showToast(message, type = 'info') {
        // Create toast container if it doesn't exist
        let toastContainer = document.querySelector('.toast-container');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
            toastContainer.style.zIndex = '9999';
            document.body.appendChild(toastContainer);
        }
        
        // Create toast element
        const toastId = 'toast-' + Date.now();
        const toastHTML = `
            <div id="${toastId}" class="toast" role="alert" aria-live="assertive" aria-atomic="true" data-bs-autohide="true">
                <div class="toast-header">
                    <i class="fas fa-${this.getToastIcon(type)} me-2 text-${type === 'error' ? 'danger' : type}"></i>
                    <strong class="me-auto">SkinLesion AI</strong>
                    <button type="button" class="btn-close" data-bs-dismiss="toast"></button>
                </div>
                <div class="toast-body">
                    ${message}
                </div>
            </div>
        `;
        
        toastContainer.insertAdjacentHTML('beforeend', toastHTML);
        
        // Initialize and show toast using Bootstrap 5
        const toastElement = document.getElementById(toastId);
        if (window.bootstrap && bootstrap.Toast) {
            const toast = new bootstrap.Toast(toastElement);
            toast.show();
            
            // Remove toast element after it's hidden
            toastElement.addEventListener('hidden.bs.toast', () => {
                toastElement.remove();
            });
        } else {
            // Fallback for when Bootstrap isn't loaded
            setTimeout(() => {
                toastElement.remove();
            }, 5000);
        }
    }
    
    getToastIcon(type) {
        switch (type) {
            case 'success': return 'check-circle';
            case 'error': return 'exclamation-circle';
            case 'warning': return 'exclamation-triangle';
            default: return 'info-circle';
        }
    }
    
    logout() {
        this.authToken = null;
        this.currentUser = null;
        this.showAuthSection();
        this.loadUserHistory(); // This will show the login prompt
        this.showToast('Logged out successfully', 'success');
    }
    
    scrollToSection(sectionId) {
        const section = document.getElementById(sectionId);
        if (section) {
            section.scrollIntoView({ behavior: 'smooth' });
        }
    }
    
    viewPredictionDetails(predictionId) {
        // TODO: Implement prediction details modal
        this.showToast('Prediction details feature coming soon!', 'info');
    }
}

// Global functions for HTML onclick handlers
function previewImage(input) {
    app.previewImage(input);
}

function uploadImage() {
    app.uploadImage();
}

function scrollToSection(sectionId) {
    app.scrollToSection(sectionId);
}

function showLoginModal() {
    const loginModal = new bootstrap.Modal(document.getElementById('loginModal'));
    loginModal.show();
}

function showRegisterModal() {
    const registerModal = new bootstrap.Modal(document.getElementById('registerModal'));
    registerModal.show();
}

function showProfile() {
    // TODO: Implement profile modal
    app.showToast('Profile feature coming soon!', 'info');
}

function logout() {
    app.logout();
}

// Initialize app when DOM is loaded
let app;
document.addEventListener('DOMContentLoaded', function() {
    app = new SkinLesionApp();
});