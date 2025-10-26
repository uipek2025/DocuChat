import uuid
import time
import os
import requests

API_BASE = os.environ.get("DOCUCHAT_API_BASE", "http://127.0.0.1:8000")

def register_or_login(email, password):
    r = requests.post(
        f"{API_BASE}/auth/register",
        json={"email": email, "password": password},
        timeout=30,
    )
    if r.status_code in (200, 201):
        data = r.json()
        return data["access_token"], data["user"]["id"]

    r = requests.post(
        f"{API_BASE}/auth/login",
        json={"email": email, "password": password},
        timeout=30,
    )
    assert r.status_code == 200, f"login failed: {r.status_code} {r.text}"
    data = r.json()
    return data["access_token"], data["user"]["id"]

def upload_doc(token, text_contents: str):
    # we generate the file in-memory just for this test
    filename = f"memtest_{uuid.uuid4().hex[:6]}.txt"
    files = {
        "file": (
            filename,
            text_contents.encode("utf-8"),
            "text/plain",
        )
    }
    r = requests.post(
        f"{API_BASE}/documents/upload",
        headers={"Authorization": f"Bearer {token}"},
        files=files,
        timeout=60,
    )
    assert r.status_code in (200, 201), f"upload failed: {r.status_code} {r.text}"
    return r.json()["id"]

def wait_ready(token, doc_id, timeout_s=10):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = requests.get(
            f"{API_BASE}/documents/",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        assert r.status_code == 200
        docs = r.json()
        for d in docs:
            if d["id"] == doc_id and d["status"] == "ready":
                return
        time.sleep(0.5)
    raise AssertionError("doc never reached ready")

def ask(token, question, doc_ids):
    r = requests.post(
        f"{API_BASE}/query/",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={
            "question": question,
            "document_ids": doc_ids,
            "top_k": 3,
        },
        timeout=30,
    )
    assert r.status_code == 200, f"ask failed: {r.status_code} {r.text}"
    return r.json()["answer"]

def test_memory_flow():
    # 1. BRAND NEW USER
    email = f"memuser_{uuid.uuid4().hex[:8]}@example.com"
    pwd = "memory123"
    token, _ = register_or_login(email, pwd)

    # 2. UPLOAD 1 DOC
    doc_id = upload_doc(
        token,
        "This is a memory flow test document. It mentions the planet Neptune.",
    )
    wait_ready(token, doc_id)

    # 3. ASK FIRST QUESTION
    q1 = "What planet is mentioned in my document?"
    a1 = ask(token, q1, [doc_id])
    assert "neptune" in a1.lower()

    # 4. ASK SECOND QUESTION
    q2 = "Cool. Summarize that again in 5 words."
    a2 = ask(token, q2, [doc_id])
    assert len(a2.strip()) > 0

    # 5. ASK META-QUESTION USING MEMORY
    q3 = "What was my first question today?"
    a3 = ask(token, q3, [doc_id])

    # EXPECTATION:
    # We don't force an exact sentence match,
    # but we DO expect it to reference something like q1.
    assert "planet" in a3.lower() or "neptune" in a3.lower(), (
        f"Memory didn't seem to kick in.\n"
        f"first={q1}\n"
        f"third-answer={a3}"
    )
