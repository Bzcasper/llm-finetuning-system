"""
Authentication routes and endpoints
"""
from fastapi import APIRouter, HTTPException, status, Depends, Request
from fastapi.security import HTTPAuthorizationCredentials
from slowapi.util import get_remote_address

from .auth import (
    UserCreate, UserLogin, TokenResponse, UserPublic, user_store, 
    generate_token_pair, verify_token
)
from .security import (
    get_current_user, security_manager, rate_limit_auth,
    sanitize_input, validate_email
)

router = APIRouter()

@router.post("/auth/register", response_model=TokenResponse)
@rate_limit_auth("3/minute")
async def register(user_data: UserCreate, request: Request):
    """Register a new user"""
    try:
        # Sanitize inputs
        user_data.email = sanitize_input(user_data.email.lower())
        if user_data.name:
            user_data.name = sanitize_input(user_data.name)
        
        # Validate email format
        if not validate_email(user_data.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid email format"
            )
        
        # Create user
        user = user_store.create_user(user_data)
        
        # Log security event
        security_manager.log_security_event(
            "user_registration",
            {"email": user.email, "user_id": user.id},
            request
        )
        
        # Generate tokens
        token_response = generate_token_pair({
            "id": user.id,
            "email": user.email
        })
        
        # Store refresh token
        user_store.store_refresh_token(token_response.refresh_token, user.id)
        
        return token_response
        
    except HTTPException:
        raise
    except Exception as e:
        security_manager.log_security_event(
            "registration_error",
            {"error": str(e), "email": user_data.email},
            request
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@router.post("/auth/login", response_model=TokenResponse)
@rate_limit_auth("5/minute")
async def login(credentials: UserLogin, request: Request):
    """Authenticate user and return tokens"""
    try:
        ip_address = get_remote_address(request)
        
        # Check if IP is temporarily blocked due to failed attempts
        if security_manager.track_failed_login(credentials.email, ip_address):
            security_manager.log_security_event(
                "login_blocked",
                {"email": credentials.email, "reason": "too_many_failed_attempts"},
                request
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many failed login attempts. Please try again later."
            )
        
        # Authenticate user
        user = user_store.authenticate_user(credentials.email, credentials.password)
        
        if not user:
            security_manager.log_security_event(
                "login_failed",
                {"email": credentials.email, "reason": "invalid_credentials"},
                request
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Clear failed login attempts on successful login
        security_manager.clear_failed_login(credentials.email, ip_address)
        
        # Log successful login
        security_manager.log_security_event(
            "login_success",
            {"email": user.email, "user_id": user.id},
            request
        )
        
        # Generate tokens
        token_response = generate_token_pair({
            "id": user.id,
            "email": user.email
        })
        
        # Store refresh token
        user_store.store_refresh_token(token_response.refresh_token, user.id)
        
        return token_response
        
    except HTTPException:
        raise
    except Exception as e:
        security_manager.log_security_event(
            "login_error",
            {"error": str(e), "email": credentials.email},
            request
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@router.post("/auth/refresh", response_model=TokenResponse)
@rate_limit_auth("10/minute")
async def refresh_token(request: Request, credentials: HTTPAuthorizationCredentials = Depends()):
    """Refresh access token using refresh token"""
    try:
        # Verify refresh token
        payload = verify_token(credentials.credentials, token_type="refresh")
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        
        # Validate stored refresh token
        if not user_store.validate_refresh_token(credentials.credentials):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Get user
        user = user_store.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        # Generate new token pair
        token_response = generate_token_pair({
            "id": user.id,
            "email": user.email
        })
        
        # Replace old refresh token with new one
        user_store.revoke_refresh_token(credentials.credentials)
        user_store.store_refresh_token(token_response.refresh_token, user.id)
        
        security_manager.log_security_event(
            "token_refresh",
            {"user_id": user.id},
            request
        )
        
        return token_response
        
    except HTTPException:
        raise
    except Exception as e:
        security_manager.log_security_event(
            "token_refresh_error",
            {"error": str(e)},
            request
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token refresh failed"
        )

@router.post("/auth/logout")
async def logout(request: Request, current_user: UserPublic = Depends(get_current_user)):
    """Logout user and revoke tokens"""
    try:
        # In a real implementation, you'd revoke the specific token
        # For now, we'll just log the logout event
        security_manager.log_security_event(
            "logout",
            {"user_id": current_user.id},
            request
        )
        
        return {"message": "Logged out successfully"}
        
    except Exception as e:
        security_manager.log_security_event(
            "logout_error",
            {"error": str(e), "user_id": current_user.id},
            request
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )

@router.get("/auth/me", response_model=UserPublic)
async def get_current_user_info(current_user: UserPublic = Depends(get_current_user)):
    """Get current user information"""
    return current_user

@router.put("/auth/profile", response_model=UserPublic)
async def update_profile(
    profile_data: dict,
    request: Request,
    current_user: UserPublic = Depends(get_current_user)
):
    """Update user profile"""
    try:
        # Get the full user record
        user = user_store.get_user_by_id(current_user.id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Update allowed fields
        if "name" in profile_data and profile_data["name"]:
            user.name = sanitize_input(profile_data["name"])
        
        security_manager.log_security_event(
            "profile_update",
            {"user_id": current_user.id, "fields": list(profile_data.keys())},
            request
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
        security_manager.log_security_event(
            "profile_update_error",
            {"error": str(e), "user_id": current_user.id},
            request
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Profile update failed"
        )

@router.get("/auth/audit-logs")
async def get_audit_logs(current_user: UserPublic = Depends(get_current_user)):
    """Get security audit logs (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    
    # Return last 100 audit log entries
    return {
        "logs": security_manager.audit_log[-100:],
        "total_events": len(security_manager.audit_log)
    }