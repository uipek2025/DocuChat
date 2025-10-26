# docuchat_backend/app/auth.py
from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from .database import get_db  # use the canonical get_db from database.py
from . import models

logger = logging.getLogger("docuchat.auth")
logger.setLevel(logging.INFO)

# -----------------------------
# Password hashing
# -----------------------------
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Check a plaintext password against a stored bcrypt hash.
    """
    return _pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a plaintext password using bcrypt.
    """
    return _pwd_context.hash(password)


# -----------------------------
# JWT config
# -----------------------------
SECRET_KEY = os.getenv("JWT_SECRET")
if not SECRET_KEY:
    # Hard fail if no secret is provided. This is intentional.
    raise RuntimeError(
        "FATAL: JWT_SECRET is not set. "
        "Set the JWT_SECRET environment variable before starting the API."
    )

ALGORITHM = os.getenv("JWT_ALG", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # 1 hour


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a signed JWT for `data` with an 'exp' claim.
    The caller is responsible for passing {"sub": user.email} etc.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta if expires_delta
        else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def authenticate_user(db: Session, email: str, password: str) -> Optional[models.User]:
    """
    Legacy helper. Kept for compatibility, but we now normalize email,
    and routers/auth.py actually does the logic inline.
    """
    email_clean = email.strip().lower()
    user = db.query(models.User).filter(models.User.email == email_clean).first()
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


# -----------------------------
# Bearer token dependency for protected routes
# -----------------------------
_bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    db: Session = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> models.User:
    """
    - Extracts "Bearer <token>" from Authorization header
    - Verifies the JWT
    - Loads that user from DB
    - Raises 401 if any check fails
    """
    if credentials is None:
        logger.warning("get_current_user: no credentials provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )

    token = credentials.credentials
    logger.info(f"get_current_user: got token (len={len(token)})")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        logger.info(f"get_current_user: decoded payload {payload}")
    except JWTError as e:
        logger.warning(f"get_current_user: JWT decode failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )

    email: Optional[str] = payload.get("sub")
    if email is None:
        logger.warning("get_current_user: 'sub' missing in JWT payload")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )

    # Important: we do NOT lowercase again here.
    # We stored 'sub' = user.email at login/reset/register time,
    # and that email is already normalized once on ingest.
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user:
        logger.warning(f"get_current_user: user {email} not found in DB")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    return user
