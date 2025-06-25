"""
Security middleware and utilities
"""
import os
import time
import logging
from typing import Optional, Dict, Any, Callable
from fastapi import HTTPException, status, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import re
from datetime import datetime
from .auth import verify_token, user_store, UserInDB, UserPublic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting configuration
limiter = Limiter(key_func=get_remote_address)

# Security configuration
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_FILE_TYPES = {
    'application/json', 'text/plain', 'text/csv', 
    'application/jsonl', 'application/x-jsonlines'
}

# Security headers
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
    "Referrer-Policy": "strict-origin-when-cross-origin"
}

class SecurityManager:
    """Central security management class"""
    
    def __init__(self):
        self.failed_login_attempts: Dict[str, Dict[str, Any]] = {}
        self.blocked_ips: Dict[str, datetime] = {}
        self.audit_log: list = []
    
    def log_security_event(self, event_type: str, details: Dict[str, Any], request: Request = None):
        """Log security events for audit purposes"""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "details": details,
            "ip_address": get_remote_address(request) if request else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown") if request else "unknown"
        }
        
        self.audit_log.append(event)
        logger.info(f"Security Event: {event_type} - {details}")
        
        # Keep only last 1000 events
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]
    
    def track_failed_login(self, email: str, ip_address: str) -> bool:
        """Track failed login attempts and return True if should block"""
        now = datetime.utcnow()
        key = f"{email}:{ip_address}"
        
        if key not in self.failed_login_attempts:
            self.failed_login_attempts[key] = {"count": 0, "last_attempt": now}
        
        self.failed_login_attempts[key]["count"] += 1
        self.failed_login_attempts[key]["last_attempt"] = now
        
        # Block after 5 failed attempts within 15 minutes
        if self.failed_login_attempts[key]["count"] >= 5:
            time_diff = (now - self.failed_login_attempts[key]["last_attempt"]).total_seconds()
            if time_diff < 900:  # 15 minutes
                return True
        
        return False
    
    def clear_failed_login(self, email: str, ip_address: str):
        """Clear failed login attempts after successful login"""
        key = f"{email}:{ip_address}"
        if key in self.failed_login_attempts:
            del self.failed_login_attempts[key]

# Global security manager
security_manager = SecurityManager()

# Authentication dependency
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserPublic:
    """Get current authenticated user"""
    try:
        payload = verify_token(credentials.credentials)
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        
        user = user_store.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Inactive user"
            )
        
        return UserPublic(
            id=user.id,
            email=user.email,
            name=user.name,
            is_admin=user.is_admin,
            subscription_status=user.subscription_status,
            subscription_plan=user.subscription_plan,
            training_credits=user.training_credits,
            created_at=user.created_at,
            last_login=user.last_login
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

async def get_admin_user(current_user: UserPublic = Depends(get_current_user)) -> UserPublic:
    """Get current user and ensure they have admin privileges"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user

# Input validation and sanitization
def sanitize_input(text: str) -> str:
    """Sanitize text input to prevent XSS and injection attacks"""
    if not isinstance(text, str):
        return str(text)
    
    # Remove potentially dangerous characters
    dangerous_chars = ['<', '>', '"', "'", '&', '\\', '/', '\x00', '\n', '\r', '\t']
    for char in dangerous_chars:
        text = text.replace(char, '')
    
    # Limited length
    return text[:1000] if len(text) > 1000 else text

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_filename(filename: str) -> bool:
    """Validate uploaded filename"""
    if not filename:
        return False
    
    # Check for directory traversal
    if '..' in filename or '/' in filename or '\\' in filename:
        return False
    
    # Check length
    if len(filename) > 255:
        return False
    
    # Check for dangerous extensions
    dangerous_extensions = ['.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js', '.jar', '.app']
    filename_lower = filename.lower()
    
    for ext in dangerous_extensions:
        if filename_lower.endswith(ext):
            return False
    
    return True

def validate_file_type(content_type: str) -> bool:
    """Validate file content type"""
    return content_type in ALLOWED_FILE_TYPES

def validate_file_size(size: int) -> bool:
    """Validate file size"""
    return size <= MAX_FILE_SIZE

# Security middleware
async def security_headers_middleware(request: Request, call_next):
    """Add security headers to all responses"""
    response = await call_next(request)
    
    for header, value in SECURITY_HEADERS.items():
        response.headers[header] = value
    
    return response

async def audit_middleware(request: Request, call_next):
    """Log requests for audit purposes"""
    start_time = time.time()
    
    # Log request
    security_manager.log_security_event(
        "request",
        {
            "method": request.method,
            "url": str(request.url),
            "client_ip": get_remote_address(request)
        },
        request
    )
    
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    security_manager.log_security_event(
        "response",
        {
            "status_code": response.status_code,
            "process_time": round(process_time, 4)
        },
        request
    )
    
    return response

# Rate limiting decorators
def rate_limit_auth(limit: str = "5/minute"):
    """Rate limit decorator for authentication endpoints"""
    def decorator(func):
        return limiter.limit(limit)(func)
    return decorator

def rate_limit_api(limit: str = "100/minute"):
    """Rate limit decorator for API endpoints"""
    def decorator(func):
        return limiter.limit(limit)(func)
    return decorator

def rate_limit_upload(limit: str = "10/minute"):
    """Rate limit decorator for file upload endpoints"""
    def decorator(func):
        return limiter.limit(limit)(func)
    return decorator

# CORS configuration for production
def get_cors_config():
    """Get CORS configuration based on environment"""
    environment = os.getenv("ENVIRONMENT", "development")
    
    if environment == "production":
        return {
            "allow_origins": [
                "https://yourdomain.com",
                "https://www.yourdomain.com"
            ],
            "allow_credentials": True,
            "allow_methods": ["GET", "POST", "PUT", "DELETE"],
            "allow_headers": ["*"],
        }
    else:
        return {
            "allow_origins": ["*"],
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }

# Environment variable protection
def get_safe_env_vars() -> Dict[str, str]:
    """Get environment variables safe for client exposure"""
    safe_vars = {
        "ENVIRONMENT": os.getenv("ENVIRONMENT", "development"),
        "API_VERSION": os.getenv("API_VERSION", "1.0.0"),
        "FRONTEND_URL": os.getenv("FRONTEND_URL", "http://localhost:3000"),
    }
    
    return safe_vars

# API key management for Modal.com
def get_modal_config() -> Dict[str, Any]:
    """Get Modal configuration with security checks"""
    token_id = os.getenv("MODAL_TOKEN_ID")
    token_secret = os.getenv("MODAL_TOKEN_SECRET")
    
    if not token_id or not token_secret:
        logger.warning("Modal credentials not configured")
        return {"configured": False}
    
    return {
        "configured": True,
        "profile": "ai-tool-pool",
        "app_name": "llm-finetuner"
    }