import streamlit as st
import requests
import json
from typing import Optional, List

import os
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="DocuChat",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- GLOBAL STYLE OVERRIDES ---
st.markdown(
    """
    <style>
    /* App background */
    .stApp {
        background-color: #f7f9fc;
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif;
    }

    /* Top headers */
    h1, h2, h3, h4, h5, h6 {
        color: #1f2937;
        font-weight: 600;
        letter-spacing: -0.03em;
    }

    /* Sidebar styling */
    div[data-testid="stSidebar"] {
        background-color: #1f2937 !important;
        color: #f9fafb !important;
        border-right: 1px solid #11182722;
    }
    div[data-testid="stSidebar"] h1,
    div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3,
    div[data-testid="stSidebar"] h4,
    div[data-testid="stSidebar"] h5,
    div[data-testid="stSidebar"] h6,
    div[data-testid="stSidebar"] p,
    div[data-testid="stSidebar"] span,
    div[data-testid="stSidebar"] label {
        color: #f9fafb !important;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        background-color: #2563eb;
        color: #fff;
        border: 0;
        padding: 0.6em 0.9em;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #1e40af;
        border: 0;
        color: #fff;
    }

    /* Chat bubbles */
    .chat-user {
        background-color: #2563eb;
        color: #fff;
        align-self: flex-end;
    }
    .chat-assistant {
        background-color: #ffffff;
        color: #111827;
        border: 1px solid rgba(0,0,0,0.08);
    }
    .chat-bubble {
        max-width: 80%;
        padding: 0.8rem 1rem;
        border-radius: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.07);
        margin-bottom: 0.75rem;
        font-size: 0.9rem;
        line-height: 1.4;
    }

    /* Doc cards */
    .doc-card {
        border: 1px solid rgba(0,0,0,0.08);
        background: #ffffff;
        border-radius: 0.75rem;
        padding: 1rem 1.25rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.04);
        margin-bottom: 1.5rem;
    }
    .doc-card-header {
        display: flex;
        flex-wrap: wrap;
        align-items: baseline;
        gap: 0.5rem;
        justify-content: space-between;
    }
    .doc-filename {
        font-size: 1rem;
        font-weight: 600;
        color: #111827;
        line-height: 1.4;
        word-break: break-word;
        display:flex;
        align-items:center;
        gap:0.4rem;
    }
    .doc-meta {
        font-size: .8rem;
        color: #6b7280;
    }

    .status-pill {
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.25rem 0.5rem;
        border-radius: 999px;
        background-color: #e5e7eb;
        color: #374151;
        border: 1px solid rgba(0,0,0,0.05);
        display: inline-block;
        white-space: nowrap;
        line-height:1.2;
    }
    .status-pill.ready {
        background-color: #dcfce7;
        color: #065f46;
        border-color: #065f46;
    }
    .status-pill.processing {
        background-color: #fff7ed;
        color: #78350f;
        border-color: #78350f;
    }
    .status-pill.error {
        background-color: #fee2e2;
        color: #991b1b;
        border-color: #991b1b;
    }

    .doc-section-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #4b5563;
        text-transform: uppercase;
        margin-top: 0.75rem;
        margin-bottom: 0.25rem;
        letter-spacing: .03em;
    }
    .doc-summary-body {
        font-size: 0.9rem;
        color: #1f2937;
        line-height: 1.5;
        white-space: pre-wrap;
        word-break: break-word;
    }

    .chip-row {
        margin-top: 0.5rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
    }
    .chip {
        background-color: #eef2ff;
        color: #3730a3;
        border: 1px solid #3730a322;
        font-size: 0.7rem;
        font-weight: 500;
        padding: 0.3rem 0.5rem;
        border-radius: 999px;
        line-height: 1.2;
        white-space: nowrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- SESSION STATE ---
def init_session():
    if "token" not in st.session_state:
        st.session_state.token = None
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "selected_doc_ids" not in st.session_state:
        st.session_state.selected_doc_ids = []
    if "health" not in st.session_state:
        st.session_state.health = None
    if "last_answer_text" not in st.session_state:
        st.session_state.last_answer_text = ""
    if "last_answer_citations" not in st.session_state:
        st.session_state.last_answer_citations = []
    # chat transcript rendered in Ask tab
    if "chat_ui_history" not in st.session_state:
        st.session_state.chat_ui_history = []

def do_login(email: str, password: str) -> Optional[str]:
    try:
        resp = requests.post(
            f"{API_BASE}/auth/login",
            json={"email": email, "password": password},
            timeout=30,
        )
    except Exception as e:
        st.error(f"Login error: {e}")
        return None

    if resp.status_code == 200:
        data = resp.json()
        token = data.get("access_token")
        if not token:
            st.error("Login succeeded but no token returned.")
            return None
        return token

    st.error(f"Login failed ({resp.status_code}): {resp.text}")
    return None

def register_user(email: str, password: str):
    try:
        resp = requests.post(
            f"{API_BASE}/auth/register",
            json={"email": email, "password": password},
            timeout=30,
        )
    except Exception as e:
        return (False, f"Registration request failed: {e}")

    if resp.status_code in (200, 201):
        data = resp.json()
        token = data.get("access_token")
        if not token:
            return (False, "Registration succeeded but no token returned.")
        return (True, token)

    try:
        detail = resp.json().get("detail", resp.text)
    except Exception:
        detail = resp.text
    return (False, f"Registration failed ({resp.status_code}): {detail}")

def reset_password(email: str, new_password: str):
    try:
        resp = requests.post(
            f"{API_BASE}/auth/reset_password",
            json={"email": email, "new_password": new_password},
            timeout=30,
        )
    except Exception as e:
        return (False, f"Reset request failed: {e}")

    if resp.status_code == 200:
        data = resp.json()
        token = data.get("access_token")
        if not token:
            return (False, "Reset succeeded but no token returned.")
        return (True, token)

    try:
        detail = resp.json().get("detail", resp.text)
    except Exception:
        detail = resp.text
    return (False, f"Reset failed ({resp.status_code}): {detail}")

def fetch_documents(token: str) -> List[dict]:
    try:
        resp = requests.get(
            f"{API_BASE}/documents/",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"Failed to load documents ({resp.status_code}): {resp.text}")
            return []
    except Exception as e:
        st.error(f"Error fetching documents: {e}")
        return []

def fetch_health(token: str) -> dict:
    try:
        resp = requests.get(
            f"{API_BASE}/health/",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            return {"error": f"{resp.status_code} {resp.text}"}
    except Exception as e:
        return {"error": str(e)}

def upload_file(token, uploaded_file):
    headers = {"Authorization": f"Bearer {token}"}
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }

    try:
        resp = requests.post(
            f"{API_BASE}/documents/upload",
            headers=headers,
            files=files,
            timeout=300,
        )
    except Exception:
        return None

    if resp.status_code in (200, 201):
        try:
            data = resp.json()
        except json.JSONDecodeError:
            return False, f"Upload succeeded but response not JSON: {resp.text}"

        doc_id = data.get("id")
        fname = data.get("filename")
        status = data.get("status")
        return True, f"Uploaded: {fname} (id={doc_id}, status={status})"

    try:
        data = resp.json()
        detail = data.get("detail", resp.text)
    except Exception:
        detail = resp.text
    return False, f"Upload failed ({resp.status_code}): {detail}"

def delete_document(token: str, doc_id: int) -> bool:
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.delete(
            f"{API_BASE}/documents/{doc_id}",
            headers=headers,
            timeout=30,
        )
    except Exception as e:
        st.error(f"Delete error: {e}")
        return False

    if resp.status_code == 204:
        return True

    st.error(f"Delete failed ({resp.status_code}): {resp.text}")
    return False

def ask_question_api(token: str, question: str, doc_ids: List[int], top_k: int = 3) -> Optional[dict]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    body = {
        "question": question,
        "document_ids": doc_ids,
        "top_k": top_k,
    }

    try:
        resp = requests.post(
            f"{API_BASE}/query/",
            headers=headers,
            data=json.dumps(body),
            timeout=60,
        )
    except Exception as e:
        st.error(f"Query error: {e}")
        return None

    if resp.status_code == 200:
        return resp.json()

    st.error(f"Query failed ({resp.status_code}): {resp.text}")
    return None

def do_logout():
    st.session_state.token = None
    st.session_state.documents = []
    st.session_state.selected_doc_ids = []
    st.session_state.health = None
    st.session_state.last_answer_text = ""
    st.session_state.last_answer_citations = []
    st.session_state.chat_ui_history = []

# --- RENDER HELPERS ---

def status_pill(status_val: str) -> str:
    cls = "status-pill"
    if status_val == "ready":
        cls += " ready"
    elif status_val == "processing":
        cls += " processing"
    elif status_val == "error":
        cls += " error"
    return f'<span class="{cls}">{status_val}</span>'

def render_document_card(d: dict, delete_cb=None):
    doc_id = d.get("id", "?")
    fname = d.get("filename", "?")
    fsize = d.get("file_size", "?")
    stat = d.get("status", "?")

    # NEW: pull final clean fields from backend
    sum_plain    = d.get("summary")        # plain text
    sum_html     = d.get("summary_html")   # fallback rich HTML if we ever add it
    topics_plain = d.get("topics")         # "a, b, c"
    topics_html  = d.get("topics_html")    # fallback

    # split topics string -> chips
    topics_list = []
    if topics_plain:
        topics_list = [t.strip() for t in topics_plain.split(",") if t.strip()]

    # open card
    st.markdown(
        f"""
        <div class="doc-card">
            <div class="doc-card-header">
                <div style="flex-grow:1;min-width:0;">
                    <div class="doc-filename">📄 {fname}</div>
                    <div class="doc-meta">
                        ID {doc_id} · {fsize} bytes &nbsp;•&nbsp; {status_pill(stat)}
                    </div>
                </div>
            </div>

            <div class="doc-section-label">Summary</div>
        """,
        unsafe_allow_html=True,
    )

    # summary body
    if sum_plain:
        st.markdown(
            f'<div class="doc-summary-body">{sum_plain}</div>',
            unsafe_allow_html=True,
        )
    elif sum_html:
        st.markdown(sum_html, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="doc-summary-body" style="color:#6b7280;"><em>(not available)</em></div>',
            unsafe_allow_html=True,
        )

    # topics
    st.markdown(
        '<div class="doc-section-label">Topics</div>',
        unsafe_allow_html=True,
    )

    if topics_list:
        chip_html = "".join(f'<span class="chip">{t}</span>' for t in topics_list)
        st.markdown(f'<div class="chip-row">{chip_html}</div>', unsafe_allow_html=True)
    elif topics_html:
        st.markdown(topics_html, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="chip-row"><span class="chip">(none)</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # inline delete button
    if delete_cb:
        if st.button(f"🗑 Delete (ID {doc_id})", key=f"del_{doc_id}"):
            delete_cb(doc_id)

def render_chat_history():
    """
    Render chat history bubbles from st.session_state.chat_ui_history
    """
    for turn in st.session_state.chat_ui_history:
        role = turn["role"]
        text = turn["text"]

        if role == "user":
            bubble_class = "chat-bubble chat-user"
            avatar = "👤"
            align = "flex-end"
        else:
            bubble_class = "chat-bubble chat-assistant"
            avatar = "🤖"
            align = "flex-start"

        st.markdown(
            f"""
            <div style="display:flex; flex-direction:column; align-items:{align}; margin-bottom:1rem;">
                <div style="font-size:0.7rem; color:#6b7280; margin-bottom:0.25rem;">
                    {avatar} {role}
                </div>
                <div class="{bubble_class}">{text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def append_chat(role: str, text: str):
    st.session_state.chat_ui_history.append({"role": role, "text": text})


# -------------------- APP BODY --------------------
init_session()

# SIDEBAR
with st.sidebar:
    st.markdown("## 🧠 DocuChat AI")
    if st.session_state.token:
        st.markdown("**Status:** Logged in ✅")
        if st.button("🚪 Log out"):
            do_logout()
            st.rerun()
    else:
        st.markdown("**Status:** Not signed in")
    st.markdown("---")
    st.markdown("**Quick tips**")
    st.caption("• Upload PDFs, DOCX, TXT.\n• Select docs, then ask questions.\n• Ask: 'what was my first question?' to test memory.")
    st.markdown("---")
    if st.session_state.health:
        st.caption("Backend health snapshot:")
        st.json(st.session_state.health, expanded=False)

# HEADER / BRAND
st.markdown(
    """
    <div style="text-align:center; margin-bottom:1rem;">
        <h1 style="margin-bottom:0.2rem;">
            🧠 DocuChat <span style="color:#2563eb;">AI</span>
        </h1>
        <div style="color:#6b7280; font-size:0.9rem;">
            Private, memory-aware RAG assistant for your documents
        </div>
    </div>
    <hr style="margin-top:1rem; margin-bottom:2rem; border:none; border-top:1px solid #e5e7eb;" />
    """,
    unsafe_allow_html=True,
)

# AUTH FLOW
if not st.session_state.token:
    tab_login, tab_register, tab_reset = st.tabs(
        ["🔐 Login", "🆕 Create account", "🔄 Reset password"]
    )

    with tab_login:
        st.subheader("Sign in")
        login_email = st.text_input("Email", value="", key="login_email", autocomplete="username")
        login_password = st.text_input(
            "Password",
            type="password",
            value="",
            key="login_password",
            autocomplete="current-password",
        )

        if st.button("Sign in"):
            if not login_email.strip() or not login_password:
                st.error("Email or password missing. If Chrome auto-filled, please retype your password.")
            else:
                token = do_login(login_email, login_password)
                if token:
                    st.session_state.token = token
                    st.session_state.documents = fetch_documents(token)
                    st.session_state.health = fetch_health(token)
                    st.success("✅ Logged in")
                    st.rerun()
                else:
                    st.error("Login failed")

    with tab_register:
        st.subheader("Create a new account")
        reg_email = st.text_input("New email", key="reg_email")
        reg_password = st.text_input("New password", type="password", key="reg_password")

        if st.button("Create account"):
            ok, result = register_user(reg_email, reg_password)
            if ok:
                st.session_state.token = result
                st.session_state.documents = fetch_documents(result)
                st.session_state.health = fetch_health(result)
                st.success("🎉 Account created and logged in")
                st.rerun()
            else:
                st.error(result)

    with tab_reset:
        st.subheader("Reset your password")
        reset_email = st.text_input("Account email", key="reset_email")
        reset_new_password = st.text_input("New password", type="password", key="reset_new_password")

        if st.button("Reset password"):
            ok, result = reset_password(reset_email, reset_new_password)
            if ok:
                st.session_state.token = result
                st.session_state.documents = fetch_documents(result)
                st.session_state.health = fetch_health(result)
                st.success("🔄 Password updated and logged in")
                st.rerun()
            else:
                st.error(result)

    st.stop()

# MAIN APP (LOGGED IN)
tab_files, tab_ask, tab_status = st.tabs(["📄 Documents", "💬 Ask", "🩺 Status"])

# --- TAB: DOCUMENTS ---
with tab_files:
    st.subheader("Upload & manage files")

    uploaded_file = st.file_uploader(
        "Choose a file to upload (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
    )

    if st.button("Upload file"):
        if uploaded_file is None:
            st.warning("Please choose a file first.")
        else:
            with st.spinner("Uploading & indexing... this can take a minute for big PDFs ⏳"):
                result = upload_file(
                    st.session_state.token,
                    uploaded_file
                )

            if result is None:
                st.error("Upload failed: backend error")
            else:
                ok, msg = result
                if ok:
                    st.success(msg)
                    st.session_state.documents = fetch_documents(st.session_state.token)
                else:
                    st.error(msg)

    st.markdown("")

    st.subheader("Your documents")

    docs = st.session_state.documents
    if not docs:
        st.info("No documents yet.")
    else:
        # label map for multiselect
        doc_label_map = {
            d["id"]: f'{d["id"]} - {d.get("filename","?")} ({d.get("status","?")})'
            for d in docs
        }

        # delete callback
        def _do_delete(doc_id_to_del: int):
            ok = delete_document(st.session_state.token, doc_id_to_del)
            if ok:
                st.success(f"Deleted document {doc_id_to_del}")
                st.session_state.documents = fetch_documents(st.session_state.token)
                st.session_state.selected_doc_ids = [
                    x for x in st.session_state.selected_doc_ids
                    if x != doc_id_to_del
                ]
            else:
                st.error("Delete failed.")

        for d in docs:
            render_document_card(d, delete_cb=_do_delete)

        st.markdown("---")

        st.subheader("Select docs for Q&A")
        chosen_docs = st.multiselect(
            "The assistant will only look in these docs:",
            options=list(doc_label_map.keys()),
            default=st.session_state.selected_doc_ids,
            format_func=lambda _id: doc_label_map.get(_id, str(_id)),
        )
        st.session_state.selected_doc_ids = chosen_docs


# --- TAB: ASK ---
with tab_ask:
    st.subheader("Ask your documents")

    if not st.session_state.selected_doc_ids:
        st.info("First select one or more documents in the Documents tab.")
    else:
        # 1. Show full chat so far (once)
        render_chat_history()

        # 2. Horizontal rule before the composer
        st.markdown(
            "<hr style='border:none;border-top:1px solid #e5e7eb;margin:1.5rem 0;'/>",
            unsafe_allow_html=True,
        )

        # 3. Input composer row
        user_q = st.text_input(
            "Ask a question (or say 'what was my first question?'):",
            key="ask_input",
        )
        top_k_val = st.slider(
            "How many relevant chunks to retrieve (top_k)",
            1, 8, 3,
            key="ask_topk_slider",
        )

        send_clicked = st.button("Send", key="ask_send_btn")

        if send_clicked:
            q = user_q.strip()
            if not q:
                st.warning("Please type a question first.")
            else:
                # Add user message
                append_chat("user", q)

                with st.spinner("Thinking..."):
                    answer = ask_question_api(
                        st.session_state.token,
                        q,
                        st.session_state.selected_doc_ids,
                        top_k=top_k_val,
                    )

                if answer:
                    answer_text = answer.get("answer", "")
                    citations = answer.get("citations", [])

                    # Store last answer for download panel
                    st.session_state.last_answer_text = answer_text
                    st.session_state.last_answer_citations = citations

                    # Add assistant message
                    append_chat("assistant", answer_text)

                else:
                    append_chat("assistant", "⚠️ Query failed.")

                # After updating chat history in session, rerun the page.
                # On rerun: render_chat_history() will include these new turns
                # and the composer will still show at the bottom.
                st.rerun()

        # 4. If we have an answer saved, show citations + download below composer
        if st.session_state.last_answer_text:
            citations = st.session_state.last_answer_citations or []

            # citations accordion
            with st.expander("See citations"):
                st.json(citations)

            # build export text
            export_lines = []
            export_lines.append("QUESTION:")
            export_lines.append(user_q or "")
            export_lines.append("")
            export_lines.append("ANSWER:")
            export_lines.append(st.session_state.last_answer_text.strip())
            export_lines.append("")
            export_lines.append("CITATIONS:")
            for c in citations:
                doc_id = c.get("doc_id", "?")
                chunk_index = c.get("chunk_index", "?")
                filename = c.get("filename", "unknown")
                dist = c.get("distance", None)

                line = f"- doc_id={doc_id}, chunk={chunk_index}, file={filename}"
                if dist is not None:
                    line += f", distance={dist:.4f}"
                export_lines.append(line)

            export_lines.append("")
            export_text = "\n".join(export_lines)

            st.download_button(
                label="💾 Download answer as .txt",
                data=export_text,
                file_name="docuchat_answer.txt",
                mime="text/plain",
            )


# --- TAB: STATUS ---
with tab_status:
    st.subheader("Backend status")
    st.write("This checks `/health/` on the FastAPI server.")

    if st.button("Refresh status"):
        st.session_state.health = fetch_health(st.session_state.token)

    if st.session_state.health:
        st.json(st.session_state.health)
    else:
        st.info("No status yet. Click Refresh status.")
