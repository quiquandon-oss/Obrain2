"""
Obrain2 – Multi-Provider Multimodal Chat Assistant
Supports: Google Gemma/Gemini, OpenRouter, Zhipu AI (GLM), DeepSeek, Alibaba Qwen, SiliconFlow, Moonshot (Kimi),
unified prompt box container, persistent API key management, and chat history via Supabase.
"""

import streamlit as st
from google import genai
from google.genai import types
from openai import OpenAI
from supabase import create_client, Client
from pypdf import PdfReader
from PIL import Image
import io
import base64
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import key_manager

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Obrain2 AI Assistant",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Helper to safely retrieve st.secrets
def get_safe_secrets():
    try:
        if hasattr(st, "secrets"):
            _ = len(st.secrets)
            return st.secrets
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------------
# Constants & model catalogues
# ---------------------------------------------------------------------------
PROVIDERS = {
    "Google (Gemma / Gemini)": {
        "type": "google",
        "key_name": "GEMINI_API_KEY",
        "base_url": None,
        "models": ["gemma-4-26b-a4b-it", "gemma-4-31b-it", "gemini-2.5-flash", "gemini-2.5-pro"],
    },
    "OpenRouter": {
        "type": "openai_compatible",
        "key_name": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "models": [
            "google/gemma-4-26b-a4b-it",
            "google/gemma-4-31b-it",
            "qwen/qwen3-32b",
            "qwen/qwen3-72b",
            "deepseek/deepseek-chat",
            "deepseek/deepseek-reasoner",
            "z-ai/glm-4.5-flash",
        ],
    },
    "Zhipu AI (GLM)": {
        "type": "openai_compatible",
        "key_name": "ZHIPU_API_KEY",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "models": ["glm-4-flash", "glm-4", "glm-4-air", "glm-4-plus"],
    },
    "DeepSeek": {
        "type": "openai_compatible",
        "key_name": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
    "Alibaba Qwen": {
        "type": "openai_compatible",
        "key_name": "QWEN_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": ["qwen-turbo", "qwen-plus", "qwen-max", "qwen-long"],
    },
    "SiliconFlow": {
        "type": "openai_compatible",
        "key_name": "SILICONFLOW_API_KEY",
        "base_url": "https://api.siliconflow.cn/v1",
        "models": [
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
            "deepseek-ai/DeepSeek-V2.5",
            "THUDM/glm-4-9b-chat",
        ],
    },
    "Moonshot (Kimi)": {
        "type": "openai_compatible",
        "key_name": "MOONSHOT_API_KEY",
        "base_url": "https://api.moonshot.cn/v1",
        "models": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
    },
}

DEFAULT_SYSTEM = (
    "You are a helpful, intelligent, and concise AI assistant. "
    "Answer clearly and accurately. When images or documents are provided, "
    "analyse them carefully and base your answer on their content."
)

MAX_FILE_MB = 8          # soft limit per file
WARN_STORAGE_MB = 400    # warn when total uploaded files approach free tier

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "-" + str(uuid.uuid4())[:8]
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "temp_attachments" not in st.session_state:
    st.session_state.temp_attachments = []
if "validation_status" not in st.session_state:
    st.session_state.validation_status = {}  # key_name -> (bool, str)

def clear_attachments():
    st.session_state.temp_attachments = []

# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------
def get_supabase() -> Optional[Client]:
    try:
        sec = get_safe_secrets()
        if sec and "supabase" in sec:
            url = sec["supabase"]["SUPABASE_URL"]
            key = sec["supabase"]["SUPABASE_KEY"]
            return create_client(url, key)
    except Exception:
        pass
    return None

def ensure_conversation(sb: Client, session_id: str) -> str:
    """Return conversation uuid for this session, creating one if needed."""
    res = sb.table("conversations").select("id").eq("session_id", session_id).execute()
    if res.data:
        return res.data[0]["id"]
    new_id = str(uuid.uuid4())
    sb.table("conversations").insert({
        "id": new_id,
        "session_id": session_id,
        "title": f"Chat {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}",
    }).execute()
    return new_id

def load_messages(sb: Client, conversation_id: str) -> List[Dict]:
    res = sb.table("messages").select("role, content").eq("conversation_id", conversation_id).order("created_at").execute()
    return [{"role": r["role"], "content": r["content"]} for r in (res.data or [])]

def save_message(sb: Client, conversation_id: str, role: str, content: str):
    sb.table("messages").insert({
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
    }).execute()

def get_total_storage_mb(sb: Client) -> float:
    try:
        res = sb.table("uploaded_files").select("size_bytes").execute()
        total = sum(r.get("size_bytes", 0) or 0 for r in (res.data or []))
        return total / (1024 * 1024)
    except Exception:
        return 0.0

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip() if text.strip() else "[PDF contained no extractable text]"
    except Exception as e:
        return f"[Could not read PDF: {e}]"

# ---------------------------------------------------------------------------
# Sidebar & Provider Settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🧠 Obrain2 Settings")

    provider_name = st.selectbox("Provider", list(PROVIDERS.keys()))
    provider = PROVIDERS[provider_name]
    model_choice = st.selectbox("Model", provider["models"])

    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
    system_prompt = st.text_area("System Instruction", value=DEFAULT_SYSTEM, height=100)

    st.divider()
    st.subheader("🔑 API Key Management")

    safe_sec = get_safe_secrets()
    active_key_name = provider["key_name"]
    current_key = key_manager.get_api_key(active_key_name, safe_sec)

    if current_key:
        st.success(f"Status: Configured ({key_manager.mask_key(current_key)})")
    else:
        st.warning("Status: Not Configured")

    with st.expander("Manage Provider API Keys", expanded=not bool(current_key)):
        for name, info in PROVIDERS.items():
            k_name = info["key_name"]
            existing_key = key_manager.get_api_key(k_name, safe_sec)

            st.markdown(f"**{name}**")
            status_icon = "🟢" if existing_key else "⚪"
            st.caption(f"{status_icon} {'Configured' if existing_key else 'Not set'}")

            new_k = st.text_input(
                f"Key for {name}",
                value="",
                placeholder=key_manager.mask_key(existing_key) if existing_key else "Enter API key...",
                type="password",
                key=f"input_{k_name}",
                label_visibility="collapsed",
            )

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Save & Test", key=f"save_{k_name}"):
                    val_to_save = new_k.strip() or existing_key or ""
                    if val_to_save:
                        key_manager.save_provider_key(k_name, val_to_save)
                        is_valid, msg = key_manager.validate_key(
                            info["type"], info["base_url"], val_to_save
                        )
                        st.session_state.validation_status[k_name] = (is_valid, msg)
                        st.rerun()
            with c2:
                if st.button("Revoke", key=f"revoke_{k_name}"):
                    key_manager.revoke_provider_key(k_name)
                    if k_name in st.session_state.validation_status:
                        del st.session_state.validation_status[k_name]
                    st.rerun()

            if k_name in st.session_state.validation_status:
                ok, msg = st.session_state.validation_status[k_name]
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
            st.markdown("---")

    st.divider()
    use_persistence = st.checkbox("Persist chats to Supabase", value=True)
    if use_persistence:
        sb = get_supabase()
        if sb:
            used_mb = get_total_storage_mb(sb)
            st.caption(f"Uploaded files storage: ~{used_mb:.1f} MB")
            if used_mb > WARN_STORAGE_MB:
                st.warning("Approaching Supabase free storage limit. Consider deleting old files.")
        else:
            st.caption("Supabase credentials not active or not configured in secrets.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear current chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_id = None
            clear_attachments()
            st.rerun()
    with col2:
        if st.session_state.get("messages"):
            md_lines = ["# Obrain2 Chat Export", f"*Exported: {datetime.now(timezone.utc).isoformat()}*", ""]
            for msg in st.session_state.messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                md_lines.append(f"**{role}:**\n{msg['content']}\n")
            md_content = "\n".join(md_lines)
            st.download_button(
                label="Export MD",
                data=md_content,
                file_name=f"obrain2_chat_{st.session_state.get('session_id', 'export')}.md",
                mime="text/markdown",
                use_container_width=True,
            )

# Load history from Supabase once
if use_persistence and st.session_state.conversation_id is None:
    sb = get_supabase()
    if sb:
        conv_id = ensure_conversation(sb, st.session_state.session_id)
        st.session_state.conversation_id = conv_id
        if not st.session_state.messages:
            st.session_state.messages = load_messages(sb, conv_id)

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("🧠 Obrain2 AI Assistant")
st.caption("Multi-provider · Multimodal · Persistent memory")

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Consolidated Unified Prompt Box Layout
# ---------------------------------------------------------------------------
st.markdown("""
    <style>
    /* Styling Streamlit form as a single dark unified prompt box */
    div[data-testid="stForm"] {
        background-color: #1e1e24 !important;
        border: 1px solid #374151 !important;
        border-radius: 1rem !important;
        padding: 0.5rem 0.75rem !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3) !important;
    }
    div[data-testid="stForm"] section[data-testid="stFileUploader"] {
        padding: 0 !important;
    }
    div[data-testid="stForm"] section[data-testid="stFileUploader"] label {
        display: none !important;
    }
    div[data-testid="stForm"] section[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] {
        display: none !important;
    }
    div[data-testid="stForm"] button[kind="secondaryFormSubmit"] {
        background-color: #374151 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 0.75rem !important;
        padding: 0.5rem 1rem !important;
    }
    div[data-testid="stForm"] button[kind="secondaryFormSubmit"]:hover {
        background-color: #4b5563 !important;
    }
    </style>
""", unsafe_allow_html=True)

with st.form(key="unified_prompt_form", clear_on_submit=True):
    # Dynamic preview chips inside the box container if attachments are selected
    if st.session_state.get("temp_attachments"):
        st.caption("📎 **Attached Files:**")
        cols = st.columns(min(len(st.session_state.temp_attachments), 4))
        for idx, file in enumerate(st.session_state.temp_attachments):
            with cols[idx % 4]:
                st.caption(f"📄 {file.name[:12]}...")

    # Main input row inside container: [Attach Upload Trigger] [Text Input] [Send Button]
    col_attach, col_input, col_submit = st.columns([1.5, 6.5, 1], vertical_alignment="center")

    with col_attach:
        uploaded_files = st.file_uploader(
            "Add Attachments",
            type=["png", "jpg", "jpeg", "pdf", "webp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="file_uploader_widget"
        )

    with col_input:
        prompt_text = st.text_input(
            "Ask anything...",
            placeholder="Ask anything...",
            label_visibility="collapsed",
            key="prompt_text_input"
        )

    with col_submit:
        submitted = st.form_submit_button("↑", use_container_width=True)

    if uploaded_files:
        st.session_state.temp_attachments = uploaded_files

if submitted:
    attachments = st.session_state.temp_attachments or uploaded_files or []
    if prompt_text.strip() or attachments:
        safe_sec = get_safe_secrets()
        provider_key_name = provider["key_name"]
        active_api_key = key_manager.get_api_key(provider_key_name, safe_sec)

        if not active_api_key:
            st.error(f"Please configure an API key for {provider_name} in the sidebar settings first.")
            st.stop()

        # Process attachments
        extra_text_parts = []
        image_parts = []
        pdf_count = 0

        for file in attachments:
            file_bytes = file.getvalue()
            mime_type = file.type or "application/octet-stream"

            if mime_type.startswith("image/"):
                image_parts.append((mime_type, file_bytes))
                extra_text_parts.append(f"[User attached image: {file.name}]")
            elif mime_type == "application/pdf" or file.name.lower().endswith(".pdf"):
                pdf_count += 1
                pdf_text = extract_text_from_pdf(file_bytes)
                extra_text_parts.append(f"[Content of PDF {file.name}]:\n{pdf_text[:12000]}")

        full_user_content = prompt_text
        if extra_text_parts:
            full_user_content = (prompt_text + "\n\n" + "\n\n".join(extra_text_parts)).strip()

        # Display & store user message
        st.session_state.messages.append({"role": "user", "content": full_user_content})
        with st.chat_message("user"):
            st.markdown(prompt_text if prompt_text.strip() else "*(Sent attachment)*")
            if image_parts:
                st.caption(f"📎 {len(image_parts)} image(s) attached")
            if pdf_count > 0:
                st.caption(f"📄 {pdf_count} PDF(s) attached")

        if use_persistence and st.session_state.conversation_id:
            sb = get_supabase()
            if sb:
                save_message(sb, st.session_state.conversation_id, "user", full_user_content)

        # Reset attachments after submission
        clear_attachments()

        # Generate reply
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("Thinking…")

            try:
                assistant_text = ""

                if provider["type"] == "google":
                    client = genai.Client(api_key=active_api_key)

                    history = []
                    for m in st.session_state.messages[:-1]:
                        role = "user" if m["role"] == "user" else "model"
                        history.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))

                    chat = client.chats.create(
                        model=model_choice,
                        history=history,
                        config=types.GenerateContentConfig(
                            temperature=temperature,
                            system_instruction=system_prompt,
                        ),
                    )
                    response = chat.send_message(full_user_content)
                    assistant_text = response.text or "(empty response)"

                else:  # OpenAI-compatible (OpenRouter, Zhipu, DeepSeek, Qwen, SiliconFlow, Moonshot)
                    client = OpenAI(
                        base_url=provider["base_url"],
                        api_key=active_api_key,
                    )
                    msgs = [{"role": "system", "content": system_prompt}]
                    for m in st.session_state.messages[:-1]:
                        msgs.append({"role": m["role"], "content": m["content"]})

                    content_list = [{"type": "text", "text": full_user_content}]
                    for mime, data in image_parts:
                        b64 = base64.b64encode(data).decode()
                        content_list.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        })
                    msgs.append({"role": "user", "content": content_list if image_parts else full_user_content})

                    completion = client.chat.completions.create(
                        model=model_choice,
                        messages=msgs,
                        temperature=temperature,
                    )
                    assistant_text = completion.choices[0].message.content or "(empty response)"

                placeholder.markdown(assistant_text)
                st.session_state.messages.append({"role": "assistant", "content": assistant_text})

                if use_persistence and st.session_state.conversation_id:
                    sb = get_supabase()
                    if sb:
                        save_message(sb, st.session_state.conversation_id, "assistant", assistant_text)

            except Exception as e:
                err = str(e)
                if "503" in err or "high demand" in err.lower() or "unavailable" in err.lower():
                    friendly = "Model temporarily overloaded (503). Please try again in a minute or switch model."
                elif "401" in err or "auth" in err.lower() or "api key" in err.lower() or "unauthorized" in err.lower():
                    friendly = "Authentication failed. Check your API key in settings."
                elif "429" in err or "rate" in err.lower() or "quota" in err.lower():
                    friendly = "Rate limit / quota exceeded. Wait or switch provider."
                else:
                    friendly = f"Error: {err}"
                placeholder.error(friendly)

        st.rerun()
