# docuchat_backend/app/rag.py
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from .core.config import settings

logger = logging.getLogger("docuchat.rag")
logger.setLevel(logging.INFO)

# ---------- optional deps (graceful fallbacks) ----------
try:
    import pdfplumber
except Exception:
    pdfplumber = None  # type: ignore

try:
    import docx  # python-docx
except Exception:
    docx = None  # type: ignore

try:
    import PyPDF2
except Exception:
    PyPDF2 = None  # type: ignore

try:
    import pytesseract
except Exception:
    pytesseract = None  # type: ignore

try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None  # type: ignore

try:
    from PIL import Image  # noqa: F401
except Exception:
    Image = None  # type: ignore

try:
    import chromadb
except Exception as e:
    chromadb = None  # type: ignore
    logger.error("Chroma import failed: %s", e)

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None  # type: ignore
    logger.error("sentence-transformers import failed: %s", e)

# OpenAI (new SDK, no proxies, no legacy kwargs)
_OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False


# ---------- globals ----------
_COLLECTION = None
_S_BERT = None


# ---------- embeddings ----------
def _load_sbert() -> SentenceTransformer | None:
    global _S_BERT
    if _S_BERT is not None:
        return _S_BERT
    if SentenceTransformer is None:
        logger.warning("sentence-transformers not available; RAG will degrade.")
        return None
    logger.info("Loading embeddings model: %s", settings.EMBEDDING_MODEL_NAME)
    _S_BERT = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    return _S_BERT


def _embed_texts(texts: List[str]) -> List[List[float]]:
    model = _load_sbert()
    if model is None:
        logger.warning("Embeddings model missing; using zero vectors.")
        return [[0.0] * 384 for _ in texts]
    vecs = model.encode(texts, convert_to_tensor=False, normalize_embeddings=True)
    return vecs.tolist()  # type: ignore


# ---------- chroma ----------
def _ensure_collection():
    if chromadb is None:
        raise RuntimeError("ChromaDB is not installed.")
    global _COLLECTION
    if _COLLECTION is not None:
        return _COLLECTION
    os.makedirs(settings.CHROMA_DB_DIR, exist_ok=True)
    logger.info("Opening Chroma at %r, collection=%r", settings.CHROMA_DB_DIR, settings.CHROMA_COLLECTION)
    client = chromadb.PersistentClient(path=settings.CHROMA_DB_DIR)
    try:
        collection = client.get_collection(settings.CHROMA_COLLECTION)
    except ValueError:
        collection = client.create_collection(settings.CHROMA_COLLECTION)
    _COLLECTION = collection
    return _COLLECTION


# ---------- text splitting ----------
def _split_text(text: str, chunk_size: int = 900, chunk_overlap: int = 150) -> List[str]:
    text = text.strip().replace("\r\n", "\n")
    if not text:
        return []
    out: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        out.append(text[start:end])
        if end == n:
            break
        start = max(end - chunk_overlap, start + 1)
    return out


# ---------- extraction helpers ----------
def _extract_pdf_text_pdfplumber(path: str) -> str:
    if pdfplumber is None:
        return ""
    try:
        parts: List[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                try:
                    t = page.extract_text() or ""
                except Exception:
                    t = ""
                if t.strip():
                    parts.append(t)
        return "\n\n".join(parts).strip()
    except Exception as e:
        logger.warning("pdfplumber failed on %s: %s", path, e)
        return ""


def _extract_pdf_text_pypdf2(path: str) -> str:
    if PyPDF2 is None:
        return ""
    try:
        parts: List[str] = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                try:
                    t = page.extract_text() or ""
                except Exception:
                    t = ""
                if t.strip():
                    parts.append(t)
        return "\n\n".join(parts).strip()
    except Exception as e:
        logger.warning("PyPDF2 failed on %s: %s", path, e)
        return ""


def _ocr_pdf_pages(path: str) -> str:
    if convert_from_path is None or pytesseract is None:
        return ""

    # configure binaries if explicitly provided
    if settings.TESSERACT_EXE and hasattr(pytesseract, "pytesseract"):
        pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_EXE

    poppler_kw = {}
    if settings.POPLER_BIN_DIR:
        poppler_kw["poppler_path"] = settings.POPLER_BIN_DIR

    try:
        images = convert_from_path(path, **poppler_kw)
    except Exception as e:
        logger.warning("pdf2image failed to rasterize %s: %s", path, e)
        return ""

    ocr_parts: List[str] = []
    for i, img in enumerate(images):
        try:
            text = pytesseract.image_to_string(img, lang="eng") or ""
        except Exception as e:
            logger.warning("pytesseract failed on page %s of %s: %s", i, path, e)
            text = ""
        if text.strip():
            ocr_parts.append(text.strip())

    return "\n\n".join(ocr_parts).strip()


def _read_docx(path: str) -> str:
    if docx is None:
        raise RuntimeError("python-docx not installed; run: pip install python-docx")
    try:
        d = docx.Document(path)
        return ("\n".join(p.text for p in d.paragraphs)).strip() or "[NO_EXTRACTED_TEXT_FROM_DOCX]"
    except Exception as e:
        logger.warning("DOCX read failed for %s: %s", path, e)
        return "[DOCX_READ_ERROR]"


def _read_file_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".txt", ".md", ".log"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()

    if ext == ".pdf":
        text = _extract_pdf_text_pdfplumber(file_path)
        if not text.strip():
            text = _extract_pdf_text_pypdf2(file_path)
        if not text.strip():
            logger.info("Attempting OCR for PDF with little/no text: %s", file_path)
            ocr_text = _ocr_pdf_pages(file_path)
            if ocr_text.strip():
                text = ocr_text
        return text.strip() or "[NO_EXTRACTED_TEXT_FROM_PDF]"

    if ext in [".docx", ".doc"]:
        return _read_docx(file_path)

    with open(file_path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8", "latin-1"):
        try:
            decoded = raw.decode(enc)
            if decoded.strip():
                return decoded
        except Exception:
            pass
    return "[BINARY_OR_EMPTY]"


# ---------- LLM helpers ----------
def _openai_client() -> Optional[OpenAI]:
    if not (_OPENAI_AVAILABLE and settings.OPENAI_API_KEY):
        return None
    # IMPORTANT: no proxies kw; the new SDK doesn’t accept it.
    return OpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL or None,
    )


def _chat(messages: List[Dict[str, str]], max_tokens: int) -> str:
    client = _openai_client()
    if client is None:
        raise RuntimeError("OpenAI not configured")
    resp = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def _build_prompt(question: str, contexts: List[str]) -> List[Dict[str, str]]:
    ctx = "\n\n---\n\n".join(contexts) if contexts else "[NO MATCHING CONTEXT]"
    return [
        {"role": "system", "content": "Answer only from the provided context. If unknown, say so."},
        {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"},
    ]


def _answer_extractive(hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return "I couldn't find relevant context in the selected documents."
    h = hits[0]
    meta = h.get("metadata", {})
    idx = meta.get("chunk_index", "?")
    did = meta.get("doc_id", "?")
    txt = (h.get("document", "") or "").strip()
    if len(txt) > 600:
        txt = txt[:600].rstrip() + "…"
    return f"{txt}\n[cite: doc={did} chunk={idx}]"


# ---------- public: indexing ----------
def store_document(doc_id: int, file_path: str) -> None:
    collection = _ensure_collection()

    full_text = _read_file_text(file_path)
    chunks = _split_text(full_text) or [full_text]
    embeddings = _embed_texts(chunks)

    ids = [f"doc:{doc_id}:chunk:{i}" for i in range(len(chunks))]
    metadatas = [
        {"doc_id": doc_id, "chunk_index": i, "filename": os.path.basename(file_path)}
        for i in range(len(chunks))
    ]

    logger.info("Upserting %d chunks for doc_id=%s", len(chunks), doc_id)
    collection.upsert(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings)


def generate_doc_insights(file_path: str) -> Dict[str, Any]:
    """
    Return {"summary": str, "topics": [..]}.
    If OpenAI fails/unset, fall back to a crude preview.
    """
    full_text = _read_file_text(file_path)
    snippet = full_text[:2000].strip()

    if not snippet:
        return {"summary": "No readable text extracted from this document.", "topics": []}

    client = _openai_client()
    if client is None:
        rough = snippet[:300].replace("\n", " ").strip()
        return {"summary": rough or "No summary generated.", "topics": []}

    prompt = (
        "You will receive an excerpt of a document.\n"
        "Return STRICT JSON only with keys: summary (2–3 sentences, plain, no fluff) and topics (3–6 short tags).\n"
        "{\"summary\":\"...\",\"topics\":[\"tag1\",\"tag2\"]}\n\n"
        f"Document excerpt:\n'''{snippet}'''"
    )
    try:
        resp = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=settings.SUMMARY_MAX_TOKENS,
        )
        raw = resp.choices[0].message.content.strip()
        data = json.loads(raw)
        summary = (data.get("summary") or "").strip()
        topics = data.get("topics") or []
        if not isinstance(topics, list):
            topics = []
        topics = [str(t).strip() for t in topics if str(t).strip()]
        return {"summary": summary or "No summary generated.", "topics": topics}
    except Exception as e:
        logger.warning("OpenAI summary extraction failed: %s", e)
        rough = snippet[:300].replace("\n", " ").strip()
        return {"summary": rough or "No summary generated.", "topics": []}


# ---------- public: querying ----------
def query_documents(question: str, document_ids: Optional[List[int]] = None, top_k: int = 4) -> Dict[str, Any]:
    collection = _ensure_collection()

    where: Dict[str, Any] = {}
    if document_ids:
        where = {"doc_id": {"$in": document_ids}}

    q_emb = _embed_texts([question])[0]
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=max(1, top_k),
        where=where or None,
        include=["documents", "metadatas", "distances"],
    )

    documents: List[str] = res.get("documents", [[]])[0] if res.get("documents") else []
    metadatas: List[Dict[str, Any]] = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
    distances: List[float] = res.get("distances", [[]])[0] if res.get("distances") else []

    hits = [{"document": d, "metadata": m, "distance": float(dist)} for d, m, dist in zip(documents, metadatas, distances)]
    contexts = [h["document"] for h in hits]

    client = _openai_client()
    if client and contexts:
        try:
            messages = _build_prompt(question, contexts)
            answer_text = _chat(messages, settings.ANSWER_MAX_TOKENS)
            # tack on one best citation
            if hits:
                meta = hits[0]["metadata"]
                answer_text = f"{answer_text}\n[cite: doc={meta.get('doc_id')} chunk={meta.get('chunk_index')}]"
        except Exception as e:
            logger.warning("OpenAI call failed (%s); falling back to extractive answer.", e)
            answer_text = _answer_extractive(hits)
    else:
        answer_text = _answer_extractive(hits)

    citations = [
        {
            "doc_id": h["metadata"].get("doc_id"),
            "chunk_index": h["metadata"].get("chunk_index"),
            "filename": h["metadata"].get("filename"),
            "distance": h["distance"],
        }
        for h in hits
    ]
    return {"answer": answer_text, "citations": citations}


# ---------- public: delete ----------
def delete_document_embeddings(doc_id: int) -> None:
    collection = _ensure_collection()
    res = collection.get(where={"doc_id": {"$eq": doc_id}}, include=["ids"])
    ids_to_delete = res.get("ids", [])
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)


# ---------- chat memory helpers ----------
from sqlalchemy.orm import Session  # late import to keep top clean
from . import models  # noqa: E402


def get_recent_chat_history(db: Session, user_id: int, limit: int = 10):
    rows = (
        db.query(models.ChatMessage)
        .filter(models.ChatMessage.user_id == user_id)
        .order_by(models.ChatMessage.created_at.asc())
        .limit(limit)
        .all()
    )
    history = []
    for r in rows:
        history.append({"question": r.question, "answer": r.answer})
    return history


def answer_question_with_memory(
    db: Session,
    user_id: int,
    question_text: str,
    selected_doc_ids: Optional[List[int]] = None,
    top_k: int = 4,
) -> Dict[str, Any]:
    # 1) memory
    history_turns = get_recent_chat_history(db, user_id, limit=10)

    # 2) retrieval
    collection = _ensure_collection()
    where: Dict[str, Any] = {"doc_id": {"$in": selected_doc_ids}} if selected_doc_ids else {}
    q_emb = _embed_texts([question_text])[0]
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=max(1, top_k),
        where=where or None,
        include=["documents", "metadatas", "distances"],
    )
    docs = res.get("documents", [[]])[0] if res.get("documents") else []
    metas = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
    dists = res.get("distances", [[]])[0] if res.get("distances") else []
    hits = [{"document": d, "metadata": m, "distance": float(di)} for d, m, di in zip(docs, metas, dists)]
    contexts = [h["document"] for h in hits]

    citations = [
        {
            "doc_id": h["metadata"].get("doc_id"),
            "chunk_index": h["metadata"].get("chunk_index"),
            "filename": h["metadata"].get("filename"),
            "distance": h["distance"],
        }
        for h in hits
    ]

    # 3) messages
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are DocuChat. Use the user's prior Q&A (memory) for meta questions, "
                "and the provided document context for factual answers. "
                "If the answer isn't in context, say you don't know."
            ),
        }
    ]
    for t in history_turns:
        messages.append({"role": "user", "content": t["question"]})
        messages.append({"role": "assistant", "content": t["answer"]})
    messages.append({"role": "system", "content": "Document context:\n" + ("\n\n---\n\n".join(contexts) if contexts else "[NO MATCHING CONTEXT]")})
    messages.append({"role": "user", "content": question_text})

    client = _openai_client()
    if client:
        try:
            answer_text = _chat(messages, settings.ANSWER_MAX_TOKENS)
            if hits:
                meta = hits[0]["metadata"]
                answer_text = f"{answer_text}\n[cite: doc={meta.get('doc_id')} chunk={meta.get('chunk_index')}]"
        except Exception as e:
            logger.warning("OpenAI call failed (%s); falling back to extractive answer.", e)
            answer_text = _answer_extractive(hits)
    else:
        answer_text = _answer_extractive(hits)

    # 4) persist turn
    new_msg = models.ChatMessage(
        user_id=user_id,
        question=question_text,
        answer=answer_text,
        doc_ids=",".join(str(x) for x in (selected_doc_ids or [])),
    )
    db.add(new_msg)
    db.commit()

    return {"answer": answer_text, "citations": citations}
