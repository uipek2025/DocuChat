# docuchat_backend/app/main.py
from __future__ import annotations

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import auth as auth_router
from .routers import documents as documents_router
from .routers import query as query_router
from .routers import health as health_router

logger = logging.getLogger("docuchat.main")
logger.setLevel(logging.INFO)

app = FastAPI(title="DocuChat API", version="0.1.0")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can lock this down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- basic alive probe that does NOT touch Chroma / heavy stuff ---
@app.get("/health/bootcheck")
def bootcheck():
    return {"status": "starting-ok"}

# include routers
app.include_router(auth_router.router)
app.include_router(documents_router.router)
app.include_router(query_router.router)
app.include_router(health_router.router)

# lifecycle hooks for debug
@app.on_event("startup")
async def on_startup():
    logger.info(">>>> FASTAPI STARTUP BEGIN")
    # we won't eagerly load embeddings / chroma here;
    # rag will lazy-load the first time it's called.
    logger.info(">>>> FASTAPI STARTUP COMPLETE")

@app.on_event("shutdown")
async def on_shutdown():
    logger.info(">>>> FASTAPI SHUTDOWN")
