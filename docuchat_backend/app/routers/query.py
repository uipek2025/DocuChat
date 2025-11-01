# docuchat_backend/app/routers/query.py
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .. import auth, models, rag
from ..database import get_db

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question to ask your docs")
    document_ids: Optional[List[int]] = Field(None, description="List of document IDs to restrict search")
    top_k: int = Field(4, description="How many chunks to retrieve internally")


class QueryResponse(BaseModel):
    answer: str
    citations: Optional[List[dict]]


@router.post("/", response_model=QueryResponse)
def ask_question(
    req: QueryRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    try:
        result = rag.answer_question_with_memory(
            db=db,
            user_id=current_user.id,
            question_text=req.question,
            selected_doc_ids=req.document_ids,
            top_k=req.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(answer=result["answer"], citations=result["citations"])
