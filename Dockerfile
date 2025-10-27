# 1. Use Python 3.11 slim Linux as the base
FROM python:3.11-slim

# 2. Install system packages needed for OCR and PDF processing
#    - tesseract-ocr: actual OCR engine
#    - poppler-utils: lets pdf2image turn PDFs into images
#    - build-essential: lets pip compile anything that needs C
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. Set working directory inside the container
WORKDIR /app

# 4. Copy only requirements first (better layer caching)
COPY requirements.txt /app/requirements.txt

# 5. Install Python deps into the container
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your codebase into the container
COPY . /app

# 7. Expose a port (for local clarity; host platform will still inject $PORT)
EXPOSE 8000

# 8. Set default environment variables (can be overridden in deployment)
ENV CHROMA_DB_DIR="docuchat_backend/chroma_db" \
    CHROMA_COLLECTION="docuchat" \
    DATABASE_URL="sqlite:///./docuchat_backend/docuchat.db" \
    OPENAI_MODEL="gpt-4o-mini" \
    JWT_ALG="HS256"

# 9. Start FastAPI with uvicorn
#    Note: we respect $PORT if the platform gives us one, otherwise default to 8000
CMD ["sh", "-c", "uvicorn docuchat_backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
