from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field


# =========================
# Auth / Tokens
# =========================
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    # Optional: seconds until expiry if you choose to return it
    expires_in: Optional[int] = None


# =========================
# Users
# =========================
class UserBase(BaseModel):
    email: EmailStr


class UserCreate(UserBase):
    password: str


class UserRead(UserBase):
    id: int

    class Config:
        # Pydantic v2 replacement for orm_mode
        from_attributes = True


# =========================
# Documents
# =========================
class DocumentOut(BaseModel):
    id: int
    filename: str

    file_size: Optional[int] = None
    file_type: Optional[str] = None

    status: str
    upload_date: datetime

    # AI enrichment
    summary: Optional[str] = None
    topics: Optional[str] = None  # comma-separated tags, e.g. "pricing, renewal, SLA"

    collection_id: Optional[int] = None

    class Config:
        from_attributes = True  # Pydantic v2 equivalent of orm_mode


# =========================
# Query (RAG)
# =========================
class QueryRequest(BaseModel):
    question: str
    document_ids: Optional[List[int]] = None
    # how many chunks to retrieve from the vector store
    top_k: int = Field(default=4, ge=1, le=20)


class Citation(BaseModel):
    doc_id: int
    chunk_index: int
    filename: Optional[str] = None
    distance: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    citations: Optional[List[Citation]] = None
