"""DocuChat FastAPI application entry point.

This module creates the FastAPI app, includes routers and sets up
middleware.  To run locally, execute ``uvicorn app.main:app --reload``.
"""
from fastapi import FastAPI

from .routers import auth as auth_router
from .routers import documents as documents_router
from .routers import query as query_router
from .routers import health as health_router
from .database import Base, engine
Base.metadata.create_all(bind=engine)

app = FastAPI(title="DocuChat API", version="0.1.0")

# Include API routers
app.include_router(auth_router.router)
app.include_router(documents_router.router)
app.include_router(query_router.router)
app.include_router(health_router.router)

@app.get("/", tags=["health"])
def read_root():
    """Health check endpoint."""
    return {"status": "ok"}
