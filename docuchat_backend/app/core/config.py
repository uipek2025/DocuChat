"""Application configuration settings.

Values can be overridden via environment variables.  For example,
``SECRET_KEY`` controls JWT token signing, and ``ACCESS_TOKEN_EXPIRE_MINUTES``
defines how long login tokens remain valid.
"""
import os

class Settings:
    # Cryptographic key used to sign JWT tokens.  In production, set this to a
    # long, random string via an environment variable.
    SECRET_KEY: str = os.getenv("SECRET_KEY", "CHANGEME_SUPER_SECRET")
    ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

settings = Settings()
