import os
import time
import uuid
import requests

API_BASE = os.environ.get("DOCUCHAT_API_BASE", "http://127.0.0.1:8000")


def register_or_login(email, password):
    """
    1. Try register.
    2. If email already exists, try login.
    Return (token, user_id).
    """
    # try register
    r = requests.post(
        f"{API_BASE}/auth/register",
        json={"email": email, "password": password},
        timeout=30,
    )

    if r.status_code in (200, 201):
        data = r.json()
        return data["access_token"], data["user"]["id"]

    # fallback: login
    r = requests.post(
        f"{API_BASE}/auth/login",
        json={"email": email, "password": password},
        timeout=30,
    )
    assert r.status_code == 200, f"login failed after register fail: {r.status_code} {r.text}"
    data = r.json()
    return data["access_token"], data["user"]["id"]


def upload_document(token, filepath):
    with open(filepath, "rb") as f:
        files = {
            "file": (
                os.path.basename(filepath),
                f,
                "text/plain",  # content-type for our sample.txt
            )
        }
        r = requests.post(
            f"{API_BASE}/documents/upload",
            headers={"Authorization": f"Bearer {token}"},
            files=files,
            timeout=60,
        )
    assert r.status_code in (200, 201), f"upload failed: {r.status_code} {r.text}"
    data = r.json()
    return data["id"]  # doc_id


def wait_until_ready(token, doc_id, timeout_s=10):
    """
    Poll /documents until the given doc_id status == 'ready'
    or timeout.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = requests.get(
            f"{API_BASE}/documents/",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        assert r.status_code == 200, f"list documents failed: {r.status_code} {r.text}"
        docs = r.json()

        for d in docs:
            if d["id"] == doc_id:
                if d["status"] == "ready":
                    return d
        time.sleep(0.5)

    raise AssertionError(f"doc {doc_id} not ready after {timeout_s}s")


def ask_question(token, question, doc_ids):
    body = {
        "question": question,
        "document_ids": doc_ids,
        "top_k": 3,
    }
    r = requests.post(
        f"{API_BASE}/query/",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=30,
    )
    assert r.status_code == 200, f"ask question failed: {r.status_code} {r.text}"
    data = r.json()
    # sanity checks
    assert "answer" in data, "no answer in response"
    assert isinstance(data["answer"], str), "answer not string"
    assert "citations" in data, "no citations in response"
    return data


def fetch_health(token):
    r = requests.get(
        f"{API_BASE}/health/",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    assert r.status_code == 200, f"health failed: {r.status_code} {r.text}"
    data = r.json()
    # sanity checks for keys we expect
    assert "database" in data, "health missing database key"
    assert "openai" in data, "health missing openai key"
    return data


def delete_document(token, doc_id):
    r = requests.delete(
        f"{API_BASE}/documents/{doc_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    assert r.status_code == 204, f"delete failed: {r.status_code} {r.text}"


def test_full_e2e_flow(tmp_path):
    """
    This is the full automated smoke test:

    1. Create (or login) test user.
    2. Upload sample file.
    3. Wait until embeddings/summarization done.
    4. Ask a question about that file.
    5. Hit /health to confirm backend status endpoint works.
    6. Delete the file.
    7. Assert no crashes along the way.
    """

    # --- 1. user setup
    unique_email = f"autotest_{uuid.uuid4().hex[:8]}@example.com"
    password = "testpassword123"
    token, user_id = register_or_login(unique_email, password)
    assert token and isinstance(token, str)
    assert isinstance(user_id, int)

    # --- 2. upload
    # we assume tests/sample.txt exists for summary & embedding
    sample_path = os.path.join(os.path.dirname(__file__), "sample.txt")
    assert os.path.exists(sample_path), "tests/sample.txt missing"

    doc_id = upload_document(token, sample_path)
    assert isinstance(doc_id, int)

    # --- 3. wait 'ready'
    doc_info = wait_until_ready(token, doc_id, timeout_s=10)
    assert doc_info["status"] == "ready"

    # basic summarization sanity
    # summary OR summary_html should exist
    assert ("summary" in doc_info) or ("summary_html" in doc_info)

    # --- 4. ask a question
    qdata = ask_question(token,
                         "Give me a short summary in one sentence.",
                         [doc_id])
    assert len(qdata["answer"]) > 0

    # --- 5. /health check
    health = fetch_health(token)
    assert health["database"] == "ok"
    # we don't hard assert openai == "ok" because you might not set OPENAI_API_KEY,
    # but we do assert the key is present already above.

    # --- 6. delete doc
    delete_document(token, doc_id)

    # --- 7. confirm it's gone
    r = requests.get(
        f"{API_BASE}/documents/",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    assert r.status_code == 200
    remaining = r.json()
    assert all(d["id"] != doc_id for d in remaining)
