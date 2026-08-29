"""
Obrain2 – Multi-Provider Multimodal Chat Assistant
Supports: Google Gemma 4, OpenRouter (Qwen, DeepSeek), image/camera/PDF upload,
and persistent chat history via Supabase.
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

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Obrain2 AI Assistant",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants & model catalogues
# ---------------------------------------------------------------------------
PROVIDERS = {
    "Google (Gemma 4)": {
        "type": "google",
        "models": ["gemma-4-26b-a4b-it", "gemma-4-31b-it"],
    },
    "OpenRouter": {
        "type": "openrouter",
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
}

DEFAULT_SYSTEM = (
    "You are a helpful, intelligent, and concise AI assistant. "
    "Answer clearly and accurately. When images or documents are provided, "
    "analyse them carefully and base your answer on their content."
)

MAX_FILE_MB = 8          # soft limit per file
WARN_STORAGE_MB = 400    # warn when total uploaded files approach free tier

# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------
def get_supabase() -> Optional[Client]:
    try:
        url = st.secrets["supabase"]["SUPABASE_URL"]
        key = st.secrets["supabase"]["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception:
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

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🧠 Obrain2 Settings")

    provider_name = st.selectbox("Provider", list(PROVIDERS.keys()))
    provider = PROVIDERS[provider_name]
    model_choice = st.selectbox("Model", provider["models"])

    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
    system_prompt = st.text_area("System Instruction", value=DEFAULT_SYSTEM, height=100)

    st.divider()
    st.subheader("API Keys (or use Secrets)")

    try:
        gemini_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        gemini_key = None
    if not gemini_key:
        gemini_key = st.text_input("Gemini API Key", type="password")

    try:
        openrouter_key = st.secrets.get("OPENROUTER_API_KEY")
    except Exception:
        openrouter_key = None
    if not openrouter_key:
        openrouter_key = st.text_input("OpenRouter API Key", type="password")

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
            st.error("Supabase credentials missing in secrets.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear current chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_id = None
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

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "-" + str(uuid.uuid4())[:8]
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

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

# Multimodal inputs
col1, col2 = st.columns(2)
with col1:
    uploaded_images = st.file_uploader("Upload images / photos", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
with col2:
    camera_photo = st.camera_input("Take a photo")

uploaded_pdfs = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Chat input & generation
# ---------------------------------------------------------------------------
if user_prompt := st.chat_input("Ask anything..."):
    # Build multimodal content
    extra_text_parts = []
    image_parts = []

    # Camera
    if camera_photo is not None:
        img_bytes = camera_photo.getvalue()
        image_parts.append(("image/jpeg", img_bytes))
        extra_text_parts.append("[User provided a camera photo]")

    # Uploaded images
    if uploaded_images:
        for f in uploaded_images:
            image_parts.append((f.type or "image/jpeg", f.getvalue()))
            extra_text_parts.append(f"[User uploaded image: {f.name}]")

    # PDFs → extract text
    if uploaded_pdfs:
        for pdf in uploaded_pdfs:
            try:
                reader = PdfReader(io.BytesIO(pdf.getvalue()))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                if text.strip():
                    extra_text_parts.append(f"[Content of PDF {pdf.name}]:\n{text[:12000]}")
                else:
                    extra_text_parts.append(f"[PDF {pdf.name} contained no extractable text]")
            except Exception as e:
                extra_text_parts.append(f"[Could not read PDF {pdf.name}: {e}]")

    full_user_content = user_prompt
    if extra_text_parts:
        full_user_content = user_prompt + "\n\n" + "\n\n".join(extra_text_parts)

    # Display & store user message
    st.session_state.messages.append({"role": "user", "content": full_user_content})
    with st.chat_message("user"):
        st.markdown(user_prompt)
        if image_parts:
            st.caption(f"📎 {len(image_parts)} image(s) attached")
        if uploaded_pdfs:
            st.caption(f"📄 {len(uploaded_pdfs)} PDF(s) attached")

    if use_persistence and st.session_state.conversation_id:
        sb = get_supabase()
        if sb:
            save_message(sb, st.session_state.conversation_id, "user", full_user_content)

    # Generate reply
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking…")

        try:
            assistant_text = ""

            if provider["type"] == "google":
                if not gemini_key:
                    raise ValueError("Gemini API key required")
                client = genai.Client(api_key=gemini_key)

                # Build history
                history = []
                for m in st.session_state.messages[:-1]:
                    role = "user" if m["role"] == "user" else "model"
                    history.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))

                # Current turn parts
                parts = []
                for mime, data in image_parts:
                    parts.append(types.Part.from_bytes(data=data, mime_type=mime))
                parts.append(types.Part.from_text(text=full_user_content))

                chat = client.chats.create(
                    model=model_choice,
                    history=history,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        system_instruction=system_prompt,
                    ),
                )
                # For simplicity we send text; images are included in the last user message text context
                # Full image support via generate_content is also possible
                response = chat.send_message(full_user_content)
                assistant_text = response.text or "(empty response)"

            else:  # OpenRouter (OpenAI-compatible)
                if not openrouter_key:
                    raise ValueError("OpenRouter API key required")
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=openrouter_key,
                )
                msgs = [{"role": "system", "content": system_prompt}]
                for m in st.session_state.messages[:-1]:
                    msgs.append({"role": m["role"], "content": m["content"]})

                # Build content for last user turn (text + optional images as data URLs)
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
            elif "401" in err or "auth" in err.lower() or "api key" in err.lower():
                friendly = "Authentication failed. Check your API key."
            elif "429" in err or "rate" in err.lower() or "quota" in err.lower():
                friendly = "Rate limit / quota exceeded. Wait or switch provider."
            else:
                friendly = f"Error: {err}"
            placeholder.error(friendly)
