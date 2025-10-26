from __future__ import annotations

import logging
from fastapi import Response
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
    """
    Return all documents for the logged-in user.
    """
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
    Receive an uploaded file from the client, save it, create a DB row,
    index it (OCR + embeddings), and return metadata.
    """

    # 1. Ensure upload dir exists
    upload_dir = _Path(__file__).parent / "uploads"
    upload_dir.mkdir(exist_ok=True, parents=True)

    # 2. Save file to disk
    saved_path = upload_dir / file.filename
    with open(saved_path, "wb") as out_f:
        out_f.write(file.file.read())

    # 3. Create DB row with status "processing"
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


    # 4. Run indexing pipeline + generate AI insights
    try:
        # 4a. Embed into Chroma
        rag.store_document(doc.id, str(saved_path))

        # 4b. Summarize + extract topics
        insights = rag.generate_doc_insights(str(saved_path))
        doc.summary = insights.get("summary", None)

        topics_list = insights.get("topics", [])
        if topics_list:
            # store "a, b, c" in DB for now
            doc.topics = ", ".join(topics_list)
        else:
            doc.topics = None

        # 4c. Mark success
        doc.status = "ready"
        db.add(doc)
        db.commit()
        db.refresh(doc)

    except Exception as e:
        # mark failure (don't hide the fact that indexing blew up)
        doc.status = "error"
        db.add(doc)
        db.commit()
        db.refresh(doc)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index document: {e}",
        )


    # 5. Respond to client
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
    """
    Delete a document owned by the current user:
    - remove its embeddings from Chroma
    - delete the file from disk
    - remove the DB row
    """

    # 1. Fetch and verify ownership
    doc = (
        db.query(models.Document)
        .filter(
            models.Document.id == doc_id,
            models.Document.user_id == current_user.id,
        )
        .first()
    )

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    # 2. Remove embeddings for this doc from Chroma
    try:
        rag.delete_document_embeddings(doc.id)
    except Exception:
        # don't block delete if vector cleanup fails
        pass

    # 3. Remove file from disk
    try:
        p = _Path(doc.file_path)
        if p.exists():
            p.unlink()
    except Exception:
        pass

    # 4. Delete DB row
    db.delete(doc)
    db.commit()

    # 5. Return an explicit 204
    return Response(status_code=status.HTTP_204_NO_CONTENT)

