"""
Authentication endpoints.

Handles:
- register (email/password -> create user + return JWT)
- login (email/password -> verify + return JWT)
- reset_password (dev convenience -> overwrite password + return JWT)
- logout (stateless; frontend just discards token)

All requests/returns are JSON, not form-encoded.
"""

from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
import logging
from .. import models, auth
from ..database import get_db

logger = logging.getLogger("docuchat.router.auth")

# We enforce a max password length to avoid pathological input
# and to keep hashing stable in constrained environments.
MAX_PW_LEN = 72

router = APIRouter(prefix="/auth", tags=["auth"])


# ---------- Pydantic request models ----------

class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class ResetPasswordRequest(BaseModel):
    email: str
    new_password: str


# ---------- /auth/register ----------

@router.post("/register", status_code=status.HTTP_201_CREATED)
def register_user(
    body: RegisterRequest,
    db: Session = Depends(get_db),
):
    """
    Create a new user, hash their password, store them,
    and immediately return a JWT so the client is "logged in".

    We also enforce a max password length before hashing to avoid
    runtime explosions and to keep behavior predictable.
    """

    email_clean = body.email.strip().lower()
    if not email_clean or not body.password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email and password are required",
        )

    # Reject dupes
    existing = (
        db.query(models.User)
        .filter(models.User.email == email_clean)
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Enforce max password length
    if len(body.password.encode("utf-8")) > MAX_PW_LEN:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Password too long. Maximum {MAX_PW_LEN} characters allowed.",
        )

    # Hash password (auth.py now uses pbkdf2_sha256)
    hashed_pw = auth.get_password_hash(body.password)

    # Create user in DB
    new_user = models.User(
        email=email_clean,
        password_hash=hashed_pw,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Mint JWT for immediate login
    access_token = auth.create_access_token(
        data={"sub": new_user.email},
        expires_delta=timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    logger.info(f"register success for {new_user.email} (id={new_user.id})")

    # Respond in the shape Streamlit expects
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": new_user.id,
            "email": new_user.email,
        },
    }


# ---------- /auth/login ----------

@router.post("/login", status_code=status.HTTP_200_OK)
def login_user(
    creds: LoginRequest,
    db: Session = Depends(get_db),
):
    print("DEBUG login_user creds=", creds)  # dev trace

    """
    Authenticate a user from {email, password} and return JWT.
    We normalize email the same way we do in /register.
    """

    email_clean = creds.email.strip().lower()
    password_raw = creds.password

    # Lookup user
    user = (
        db.query(models.User)
        .filter(models.User.email == email_clean)
        .first()
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # Verify hash
    if not auth.verify_password(password_raw, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # Password ok -> issue access token
    access_token = auth.create_access_token(
        data={"sub": user.email},
        expires_delta=timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
        },
    }


# ---------- /auth/reset_password ----------

@router.post("/reset_password", status_code=status.HTTP_200_OK)
def reset_password(
    body: ResetPasswordRequest,
    db: Session = Depends(get_db),
):
    """
    DEV-MODE convenience:
    Overwrite the password for an existing user (no email verification).
    Return a new JWT so the frontend can treat the user as logged in.

    We enforce the same max-length rule on new_password that we enforce
    on /register.
    """

    email_clean = body.email.strip().lower()
    new_password_raw = body.new_password

    # Enforce max password length
    if len(new_password_raw.encode("utf-8")) > MAX_PW_LEN:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Password too long. Maximum {MAX_PW_LEN} characters allowed.",
        )

    # Find user
    user = (
        db.query(models.User)
        .filter(models.User.email == email_clean)
        .first()
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Update stored hash
    user.password_hash = auth.get_password_hash(new_password_raw)
    db.add(user)
    db.commit()
    db.refresh(user)

    # Return fresh token
    access_token = auth.create_access_token(
        data={"sub": user.email},
        expires_delta=timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
        },
        "message": "Password reset successful",
    }


# ---------- /auth/logout ----------

@router.post("/logout", status_code=status.HTTP_200_OK)
def logout_dummy():
    """
    Stateless logout.
    The frontend should just delete the stored JWT.
    """
    return {"detail": "Logged out"}
