"""Database configuration and session management for DocuChat.

This module defines the SQLAlchemy engine and session maker.  It reads
configuration from environment variables with sensible defaults, and
exports a dependency that yields a database session per request.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

def get_database_url() -> str:
    """
    Always use a stable absolute path for the SQLite DB:
    <repo_root>/docuchat_backend/docuchat.db

    This prevents accidental creation of multiple docuchat.db files
    when uvicorn is started from different working directories.
    """
    # __file__ = .../DocuChat/docuchat_backend/app/database.py
    app_dir = os.path.dirname(os.path.abspath(__file__))            # .../docuchat_backend/app
    backend_dir = os.path.abspath(os.path.join(app_dir, ".."))      # .../docuchat_backend
    db_path = os.path.join(backend_dir, "docuchat.db")              # .../docuchat_backend/docuchat.db

    return os.getenv("DATABASE_URL", f"sqlite:///{db_path}")

# Create the SQLAlchemy engine.  For SQLite, we need check_same_thread=False
# because FastAPI runs each request in a separate thread.
DATABASE_URL = get_database_url()
connect_args = {}  # extra arguments for create_engine
if DATABASE_URL.startswith("sqlite"):  # pragma: no cover
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, connect_args=connect_args)

class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """FastAPI dependency that provides a database session per request.

    Yields a SQLAlchemy session and ensures it is closed after the request
    is complete, even if an exception occurs.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
