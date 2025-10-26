# docuchat_backend/app/routers/health.py
from __future__ import annotations

import os, inspect
from fastapi import APIRouter
from sqlalchemy import text

from ..database import engine
from .. import rag  # only for Chroma ensure; NOT used for OpenAI
from openai import OpenAI
import openai as openai_pkg
import httpx

router = APIRouter(prefix="/health", tags=["health"])

def _make_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("missing_api_key")

    # optional proxy via httpx (new SDK does NOT support 'proxies=' kwarg)
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
    if os.getenv("DOCUCHAT_DISABLE_PROXY") == "1" or not proxy:
        return OpenAI(api_key=api_key)

    http_client = httpx.Client(proxies=proxy, timeout=30.0)
    return OpenAI(api_key=api_key, http_client=http_client)

@router.get("/")
def health():
    status = {
        "database": "unknown",
        "chroma": "unknown",
        "openai": "unknown",
        "openai_module": getattr(openai_pkg, "__file__", "n/a"),
        "openai_version": getattr(openai_pkg, "__version__", "n/a"),
        "collection": os.getenv("CHROMA_COLLECTION", "docuchat"),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "impl": "routers.health.v2",  # so we know THIS file is live
    }

    # 1) DB check
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        status["database"] = "ok"
    except Exception as e:
        status["database"] = f"error: {e!r}"

    # 2) Chroma check
    try:
        coll = rag._ensure_collection()
        _ = coll.count()
        status["chroma"] = "ok"
    except Exception as e:
        status["chroma"] = f"error: {e!r}"

    # 3) OpenAI check (new SDK only, NO proxies kwarg anywhere)
    try:
        client = _make_openai_client()
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=1,
            temperature=0,
        )
        _ = resp.choices[0].message.content
        status["openai"] = "ok"
    except Exception as e:
        status["openai"] = f"error: {e!r}"

    return status
