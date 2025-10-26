"""Background job definitions.

In a production deployment you would use Celery, RQ or another task
queue to run these functions asynchronously.  For demonstration
purposes, they are synchronous functions called directly from the API
handlers.
"""
import logging
from sqlalchemy.orm import Session
from . import models, rag

logger = logging.getLogger(__name__)

def process_document_job(db: Session, document: models.Document) -> None:
    """Process a document: extract text, generate embeddings and update status."""
    try:
        rag.process_document(doc_id=document.id, file_path=document.file_path)
        document.status = "ready"
        db.add(document)
        db.commit()
    except Exception as exc:
        logger.error("Processing failed for document %s: %s", document.id, exc)
        document.status = "error"
        db.add(document)
        db.commit()
