from __future__ import annotations

import logging
from pathlib import Path as _Path
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status, Path, Response
from sqlalchemy.orm import Session

from .. import models, schemas, auth, rag
from ..database import get_db

logger = logging.getLogger("docuchat.documents")

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("/", response_model=List[schemas.DocumentOut])
def list_documents(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    docs = (
        db.query(models.Document)
        .filter(models.Document.user_id == current_user.id)
        .order_by(models.Document.upload_date.desc())
        .all()
    )
    return docs


@router.post("/upload")
def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    # 1) save to disk
    upload_dir = _Path(__file__).parent / "uploads"
    upload_dir.mkdir(exist_ok=True, parents=True)
    saved_path = upload_dir / file.filename
    with open(saved_path, "wb") as out_f:
        out_f.write(file.file.read())

    # 2) DB row
    doc = models.Document(
        user_id=current_user.id,
        filename=file.filename,
        file_path=str(saved_path),
        file_size=saved_path.stat().st_size,
        file_type=file.content_type,
        status="processing",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # 3) index + insights
    try:
        rag.store_document(doc.id, str(saved_path))
        insights = rag.generate_doc_insights(str(saved_path))
        doc.summary = insights.get("summary")
        topics_list = insights.get("topics") or []
        doc.topics = ", ".join(topics_list) if topics_list else None
        doc.status = "ready"
        db.add(doc)
        db.commit()
        db.refresh(doc)
    except Exception as e:
        doc.status = "error"
        db.add(doc)
        db.commit()
        db.refresh(doc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index document: {e}",
        )

    # 4) response
    return {
        "id": doc.id,
        "filename": doc.filename,
        "status": doc.status,
        "file_size": doc.file_size,
        "summary": doc.summary,
        "topics": doc.topics,
    }


@router.delete(
    "/{doc_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_document(
    doc_id: int = Path(..., description="Document ID to delete"),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    doc = (
        db.query(models.Document)
        .filter(models.Document.id == doc_id, models.Document.user_id == current_user.id)
        .first()
    )
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    try:
        rag.delete_document_embeddings(doc.id)
    except Exception:
        pass

    try:
        p = _Path(doc.file_path)
        if p.exists():
            p.unlink()
    except Exception:
        pass

    db.delete(doc)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
