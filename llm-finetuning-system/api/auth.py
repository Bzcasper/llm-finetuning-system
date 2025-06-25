"""
Authentication utilities and JWT token management
"""
import os
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from passlib.context import CryptContext
from fastapi import HTTPException, status
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv

load_dotenv()

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Pydantic models
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRE_MINUTES * 60

class UserInDB(BaseModel):
    id: str
    email: str
    name: Optional[str] = None
    password_hash: str
    is_active: bool = True
    is_admin: bool = False
    subscription_status: str = "FREE"
    subscription_plan: str = "free"
    training_credits: int = 3
    created_at: datetime
    last_login: Optional[datetime] = None

class UserPublic(BaseModel):
    id: str
    email: str
    name: Optional[str] = None
    is_admin: bool = False
    subscription_status: str = "FREE"
    subscription_plan: str = "free"
    training_credits: int = 3
    created_at: datetime
    last_login: Optional[datetime] = None

# Password utilities
def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def validate_password_strength(password: str) -> bool:
    """Validate password strength"""
    if len(password) < 8:
        return False
    
    # Check for at least one uppercase letter, one lowercase letter, and one number
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    
    return has_upper and has_lower and has_digit

# JWT token utilities
def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create a JWT refresh token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str, token_type: str = "access") -> Dict[str, Any]:
    """Verify and decode a JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Check token type
        if payload.get("type") != token_type:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        # Check expiration
        exp = payload.get("exp")
        if exp and datetime.utcnow() > datetime.fromtimestamp(exp):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        
        return payload
        
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

def generate_token_pair(user_data: Dict[str, Any]) -> TokenResponse:
    """Generate both access and refresh tokens"""
    access_token = create_access_token(data={"sub": user_data["id"], "email": user_data["email"]})
    refresh_token = create_refresh_token(data={"sub": user_data["id"], "email": user_data["email"]})
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

# In-memory user storage (replace with database in production)
class UserStore:
    def __init__(self):
        self.users: Dict[str, UserInDB] = {}
        self.email_to_id: Dict[str, str] = {}
        self.refresh_tokens: Dict[str, str] = {}  # token -> user_id
    
    def create_user(self, user_data: UserCreate) -> UserInDB:
        """Create a new user"""
        if user_data.email in self.email_to_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        if not validate_password_strength(user_data.password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 8 characters with uppercase, lowercase, and number"
            )
        
        user_id = f"user_{len(self.users) + 1}"
        password_hash = hash_password(user_data.password)
        
        user = UserInDB(
            id=user_id,
            email=user_data.email,
            name=user_data.name,
            password_hash=password_hash,
            created_at=datetime.utcnow()
        )
        
        self.users[user_id] = user
        self.email_to_id[user_data.email] = user_id
        
        return user
    
    def get_user_by_email(self, email: str) -> Optional[UserInDB]:
        """Get user by email"""
        user_id = self.email_to_id.get(email)
        if user_id:
            return self.users.get(user_id)
        return None
    
    def get_user_by_id(self, user_id: str) -> Optional[UserInDB]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    def authenticate_user(self, email: str, password: str) -> Optional[UserInDB]:
        """Authenticate user with email and password"""
        user = self.get_user_by_email(email)
        if not user:
            return None
        
        if not verify_password(password, user.password_hash):
            return None
        
        # Update last login
        user.last_login = datetime.utcnow()
        return user
    
    def store_refresh_token(self, token: str, user_id: str):
        """Store refresh token"""
        self.refresh_tokens[token] = user_id
    
    def validate_refresh_token(self, token: str) -> Optional[str]:
        """Validate refresh token and return user ID"""
        return self.refresh_tokens.get(token)
    
    def revoke_refresh_token(self, token: str):
        """Revoke refresh token"""
        if token in self.refresh_tokens:
            del self.refresh_tokens[token]

# Global user store instance
user_store = UserStore()