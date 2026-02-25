/**
 * Authentication functionality for Skin Lesion Classification
 */

class AuthManager {
    constructor() {
        this.apiBaseUrl = window.location.origin;
        this.init();
    }
    
    init() {
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        // Login form submission
        const loginForm = document.getElementById('login-form');
        if (loginForm) {
            loginForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.login();
            });
        }
        
        // Register form submission
        const registerForm = document.getElementById('register-form');
        if (registerForm) {
            registerForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.register();
            });
        }
        
        // Enter key handling for modals
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                const loginModal = document.getElementById('loginModal');
                const registerModal = document.getElementById('registerModal');
                
                if (loginModal && loginModal.classList.contains('show')) {
                    this.login();
                } else if (registerModal && registerModal.classList.contains('show')) {
                    this.register();
                }
            }
        });
    }
    
    async login() {
        const username = document.getElementById('login-username').value.trim();
        const password = document.getElementById('login-password').value;
        
        // Validation
        if (!username || !password) {
            this.showToast('Please fill in all fields', 'error');
            return;
        }
        
        try {
            // Show loading state
            this.showLoginLoading(true);
            
            const response = await fetch(`${this.apiBaseUrl}/auth/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    username: username,
                    password: password
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                // Store auth token
                localStorage.setItem('authToken', data.access_token);
                
                // Update app state
                if (window.app) {
                    window.app.authToken = data.access_token;
                    window.app.currentUser = data.user;
                    window.app.showUserSection();
                    await window.app.loadUserHistory();
                }
                
                // Close modal
                const loginModal = bootstrap.Modal.getInstance(document.getElementById('loginModal'));
                if (loginModal) {
                    loginModal.hide();
                }
                
                // Clear form
                this.clearLoginForm();
                
                this.showToast(`Welcome back, ${data.user.username}!`, 'success');
                
            } else {
                this.showToast(data.detail || 'Login failed', 'error');
            }
            
        } catch (error) {
            console.error('Login error:', error);
            this.showToast('Login failed. Please try again.', 'error');
        } finally {
            this.showLoginLoading(false);
        }
    }
    
    async register() {
        const username = document.getElementById('reg-username').value.trim();
        const email = document.getElementById('reg-email').value.trim();
        const fullname = document.getElementById('reg-fullname').value.trim();
        const password = document.getElementById('reg-password').value;
        
        // Validation
        if (!username || !email || !password) {
            this.showToast('Please fill in all required fields', 'error');
            return;
        }
        
        if (!this.isValidEmail(email)) {
            this.showToast('Please enter a valid email address', 'error');
            return;
        }
        
        if (password.length < 6) {
            this.showToast('Password must be at least 6 characters long', 'error');
            return;
        }
        
        try {
            // Show loading state
            this.showRegisterLoading(true);
            
            const response = await fetch(`${this.apiBaseUrl}/auth/register`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    username: username,
                    email: email,
                    full_name: fullname || null,
                    password: password
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                // Store auth token
                localStorage.setItem('authToken', data.access_token);
                
                // Update app state
                if (window.app) {
                    window.app.authToken = data.access_token;
                    window.app.currentUser = data.user;
                    window.app.showUserSection();
                    await window.app.loadUserHistory();
                }
                
                // Close modal
                const registerModal = bootstrap.Modal.getInstance(document.getElementById('registerModal'));
                if (registerModal) {
                    registerModal.hide();
                }
                
                // Clear form
                this.clearRegisterForm();
                
                this.showToast(`Welcome, ${data.user.username}! Account created successfully.`, 'success');
                
            } else {
                this.showToast(data.detail || 'Registration failed', 'error');
            }
            
        } catch (error) {
            console.error('Registration error:', error);
            this.showToast('Registration failed. Please try again.', 'error');
        } finally {
            this.showRegisterLoading(false);
        }
    }
    
    showLoginLoading(show) {
        const modal = document.getElementById('loginModal');
        if (!modal) return;
        
        const submitButton = modal.querySelector('.modal-footer .btn-primary');
        const formInputs = modal.querySelectorAll('input');
        
        if (show) {
            submitButton.innerHTML = `
                <span class="spinner-border spinner-border-sm me-2" role="status"></span>
                Logging in...
            `;
            submitButton.disabled = true;
            formInputs.forEach(input => input.disabled = true);
        } else {
            submitButton.innerHTML = 'Login';
            submitButton.disabled = false;
            formInputs.forEach(input => input.disabled = false);
        }
    }
    
    showRegisterLoading(show) {
        const modal = document.getElementById('registerModal');
        if (!modal) return;
        
        const submitButton = modal.querySelector('.modal-footer .btn-primary');
        const formInputs = modal.querySelectorAll('input');
        
        if (show) {
            submitButton.innerHTML = `
                <span class="spinner-border spinner-border-sm me-2" role="status"></span>
                Creating account...
            `;
            submitButton.disabled = true;
            formInputs.forEach(input => input.disabled = true);
        } else {
            submitButton.innerHTML = 'Register';
            submitButton.disabled = false;
            formInputs.forEach(input => input.disabled = false);
        }
    }
    
    clearLoginForm() {
        const usernameField = document.getElementById('login-username');
        const passwordField = document.getElementById('login-password');
        
        if (usernameField) usernameField.value = '';
        if (passwordField) passwordField.value = '';
    }
    
    clearRegisterForm() {
        const usernameField = document.getElementById('reg-username');
        const emailField = document.getElementById('reg-email');
        const fullnameField = document.getElementById('reg-fullname');
        const passwordField = document.getElementById('reg-password');
        
        if (usernameField) usernameField.value = '';
        if (emailField) emailField.value = '';
        if (fullnameField) fullnameField.value = '';
        if (passwordField) passwordField.value = '';
    }
    
    isValidEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }
    
    showToast(message, type = 'info') {
        // Use the main app's toast function if available
        if (window.app && window.app.showToast) {
            window.app.showToast(message, type);
        } else {
            // Fallback implementation
            console.log(`${type.toUpperCase()}: ${message}`);
            
            // Simple alert fallback
            if (type === 'error') {
                alert(`Error: ${message}`);
            } else if (type === 'success') {
                alert(`Success: ${message}`);
            }
        }
    }
    
    // Password strength checker
    checkPasswordStrength(password) {
        let strength = 0;
        let feedback = [];
        
        if (password.length >= 8) {
            strength += 1;
        } else {
            feedback.push('At least 8 characters');
        }
        
        if (/[a-z]/.test(password)) {
            strength += 1;
        } else {
            feedback.push('One lowercase letter');
        }
        
        if (/[A-Z]/.test(password)) {
            strength += 1;
        } else {
            feedback.push('One uppercase letter');
        }
        
        if (/[0-9]/.test(password)) {
            strength += 1;
        } else {
            feedback.push('One number');
        }
        
        if (/[^A-Za-z0-9]/.test(password)) {
            strength += 1;
        } else {
            feedback.push('One special character');
        }
        
        return {
            strength: strength,
            feedback: feedback,
            level: this.getStrengthLevel(strength)
        };
    }
    
    getStrengthLevel(strength) {
        if (strength <= 2) return { level: 'weak', class: 'danger', text: 'Weak' };
        if (strength <= 3) return { level: 'fair', class: 'warning', text: 'Fair' };
        if (strength <= 4) return { level: 'good', class: 'info', text: 'Good' };
        return { level: 'strong', class: 'success', text: 'Strong' };
    }
    
    // Real-time password strength feedback
    setupPasswordStrengthIndicator() {
        const passwordField = document.getElementById('reg-password');
        if (!passwordField) return;
        
        // Create strength indicator
        const strengthIndicator = document.createElement('div');
        strengthIndicator.id = 'password-strength';
        strengthIndicator.className = 'mt-2';
        strengthIndicator.style.display = 'none';
        
        passwordField.parentNode.insertBefore(strengthIndicator, passwordField.nextSibling);
        
        passwordField.addEventListener('input', (e) => {
            const password = e.target.value;
            
            if (password.length === 0) {
                strengthIndicator.style.display = 'none';
                return;
            }
            
            const result = this.checkPasswordStrength(password);
            const strengthLevel = result.level;
            
            strengthIndicator.style.display = 'block';
            strengthIndicator.innerHTML = `
                <div class="d-flex align-items-center">
                    <small class="me-2">Password strength:</small>
                    <span class="badge bg-${strengthLevel.class}">${strengthLevel.text}</span>
                </div>
                ${result.feedback.length > 0 ? `
                    <small class="text-muted">
                        Missing: ${result.feedback.join(', ')}
                    </small>
                ` : ''}
            `;
        });
    }
    
    // Form validation styling
    setupFormValidation() {
        // Login form validation
        const loginUsername = document.getElementById('login-username');
        const loginPassword = document.getElementById('login-password');
        
        if (loginUsername) {
            loginUsername.addEventListener('blur', () => {
                this.validateField(loginUsername, loginUsername.value.trim().length >= 3, 'Username must be at least 3 characters');
            });
        }
        
        if (loginPassword) {
            loginPassword.addEventListener('blur', () => {
                this.validateField(loginPassword, loginPassword.value.length >= 1, 'Password is required');
            });
        }
        
        // Register form validation
        const regUsername = document.getElementById('reg-username');
        const regEmail = document.getElementById('reg-email');
        const regPassword = document.getElementById('reg-password');
        
        if (regUsername) {
            regUsername.addEventListener('blur', () => {
                const isValid = regUsername.value.trim().length >= 3;
                this.validateField(regUsername, isValid, 'Username must be at least 3 characters');
            });
        }
        
        if (regEmail) {
            regEmail.addEventListener('blur', () => {
                const isValid = this.isValidEmail(regEmail.value.trim());
                this.validateField(regEmail, isValid, 'Please enter a valid email address');
            });
        }
        
        if (regPassword) {
            regPassword.addEventListener('blur', () => {
                const isValid = regPassword.value.length >= 6;
                this.validateField(regPassword, isValid, 'Password must be at least 6 characters');
            });
        }
    }
    
    validateField(field, isValid, errorMessage) {
        const feedbackElement = field.parentNode.querySelector('.invalid-feedback');
        
        if (isValid) {
            field.classList.remove('is-invalid');
            field.classList.add('is-valid');
            if (feedbackElement) {
                feedbackElement.style.display = 'none';
            }
        } else {
            field.classList.remove('is-valid');
            field.classList.add('is-invalid');
            
            if (!feedbackElement) {
                const feedback = document.createElement('div');
                feedback.className = 'invalid-feedback';
                field.parentNode.appendChild(feedback);
            }
            
            const feedback = field.parentNode.querySelector('.invalid-feedback');
            feedback.textContent = errorMessage;
            feedback.style.display = 'block';
        }
    }
}

// Global functions for HTML onclick handlers
function login() {
    if (window.authManager) {
        window.authManager.login();
    }
}

function register() {
    if (window.authManager) {
        window.authManager.register();
    }
}

// Initialize auth manager when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.authManager = new AuthManager();
    
    // Setup additional features
    window.authManager.setupPasswordStrengthIndicator();
    window.authManager.setupFormValidation();
    
    // Clear forms when modals are hidden
    const loginModal = document.getElementById('loginModal');
    const registerModal = document.getElementById('registerModal');
    
    if (loginModal) {
        loginModal.addEventListener('hidden.bs.modal', () => {
            window.authManager.clearLoginForm();
            // Remove validation classes
            loginModal.querySelectorAll('.is-valid, .is-invalid').forEach(el => {
                el.classList.remove('is-valid', 'is-invalid');
            });
        });
    }
    
    if (registerModal) {
        registerModal.addEventListener('hidden.bs.modal', () => {
            window.authManager.clearRegisterForm();
            // Remove validation classes and hide password strength
            registerModal.querySelectorAll('.is-valid, .is-invalid').forEach(el => {
                el.classList.remove('is-valid', 'is-invalid');
            });
            const strengthIndicator = document.getElementById('password-strength');
            if (strengthIndicator) {
                strengthIndicator.style.display = 'none';
            }
        });
    }
});