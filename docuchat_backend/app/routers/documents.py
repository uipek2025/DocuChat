from __future__ import annotations

import logging
import shutil
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
    """
    Save upload -> DB row (processing) -> index (Chroma+OCR) -> insights -> ready.
    Hardened save (seek + copyfileobj) and better error logs.
    """
    # 1) Ensure upload dir
    upload_dir = _Path(__file__).parent / "uploads"
    upload_dir.mkdir(exist_ok=True, parents=True)

    # 2) Save to disk (robust copy)
    saved_path = upload_dir / file.filename
    try:
        file.file.seek(0)
        with open(saved_path, "wb") as out_f:
            shutil.copyfileobj(file.file, out_f)
    except Exception as e:
        logger.exception("Failed to write uploaded file to disk: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {e}",
        )

    # 3) Create DB row
    doc = models.Document(
        user_id=current_user.id,
        filename=file.filename,
        file_path=str(saved_path),
        file_size=saved_path.stat().st_size if saved_path.exists() else None,
        file_type=file.content_type,
        status="processing",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # 4) Index + insights
    try:
        rag.store_document(doc.id, str(saved_path))
        insights = rag.generate_doc_insights(str(saved_path))
        doc.summary = insights.get("summary") or None

        topics_list = insights.get("topics") or []
        doc.topics = ", ".join([t for t in topics_list if t]) if topics_list else None

        doc.status = "ready"
        db.add(doc)
        db.commit()
        db.refresh(doc)

    except Exception as e:
        logger.exception("Indexing pipeline failed for doc_id=%s: %s", doc.id, e)
        # Mark failure but *return metadata* so client can show error state
        doc.status = "error"
        db.add(doc)
        db.commit()
        db.refresh(doc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index document: {e}",
        )

    # 5) Response
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
    # Verify ownership
    doc = (
        db.query(models.Document)
        .filter(
            models.Document.id == doc_id,
            models.Document.user_id == current_user.id,
        )
        .first()
    )
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    # Vector cleanup (best-effort)
    try:
        rag.delete_document_embeddings(doc.id)
    except Exception:
        pass

    # File cleanup (best-effort)
    try:
        p = _Path(doc.file_path)
        if p.exists():
            p.unlink()
    except Exception:
        pass

    db.delete(doc)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
