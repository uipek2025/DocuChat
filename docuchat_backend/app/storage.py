"""File storage utilities.

This module abstracts saving uploaded files to the local filesystem.  In
production you might replace this with an S3 or Google Cloud Storage
client.  Filenames are randomised to prevent collisions and to avoid
disclosing the original filenames on disk.
"""
import os
import uuid
from fastapi import UploadFile

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploaded_files")

def ensure_upload_dir() -> None:
    """Ensure that the upload directory exists."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_upload_file(upload_file: UploadFile) -> str:
    """Save an uploaded file and return the file path."""
    ensure_upload_dir()
    _, ext = os.path.splitext(upload_file.filename)
    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)
    with open(file_path, "wb") as buffer:
        while True:
            contents = upload_file.file.read(1024 * 1024)
            if not contents:
                break
            buffer.write(contents)
    return file_path

def delete_file(path: str) -> None:
    """Delete a file from the filesystem."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
