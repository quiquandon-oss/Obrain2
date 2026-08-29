"""
Obrain2 – Multi-Provider Multimodal Chat Assistant
Supports: Google Gemma/Gemini, OpenRouter, Zhipu AI (GLM), DeepSeek, Alibaba Qwen, SiliconFlow, Moonshot (Kimi),
sticky bottom unified prompt container, persistent API key management, named chat sessions, and history via Supabase.
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
        "models": [
            "gemini-3.6-flash",
            "gemma-2-2b-it",
        ],
    },
    "OpenRouter": {
        "type": "openai_compatible",
        "key_name": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "models": [
            "google/gemma-4-26b-a4b-it",
            "google/gemma-4-31b-it",
            "qwen/qwen3.7-plus",
            "qwen/qwen-2.5-72b-instruct",
            "deepseek/deepseek-chat",
            "deepseek/deepseek-reasoner",
            "z-ai/glm-4.5-flash",
        ],
    },
    "Zhipu AI (GLM)": {
        "type": "openai_compatible",
        "key_name": "ZHIPU_API_KEY",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "models": [
            "glm-4-flash",
            "glm-4-air",
            "glm-5.3-flash",
        ],
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

MODEL_REGISTRY = {
    provider_name: info["models"]
    for provider_name, info in PROVIDERS.items()
}

def get_validated_model(provider: str, requested_model: str) -> str:
    supported_models = MODEL_REGISTRY.get(provider, [])
    if requested_model not in supported_models:
        return supported_models[0] if supported_models else "gemini-3.6-flash"
    return requested_model

DEFAULT_SYSTEM = (
    "You are a helpful, intelligent, and concise AI assistant. "
    "Answer clearly and accurately. When images or documents are provided, "
    "analyse them carefully and base your answer on their content."
)

WARN_STORAGE_MB = 400    # warn when total uploaded files approach free tier

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

def load_conversations_from_supabase(sb: Client, session_id: str) -> List[Dict]:
    try:
        res = sb.table("conversations").select("id, title, created_at").eq("session_id", session_id).order("created_at").execute()
        return res.data or []
    except Exception:
        return []

def create_supabase_conversation(sb: Client, session_id: str, title: str) -> str:
    new_id = str(uuid.uuid4())
    sb.table("conversations").insert({
        "id": new_id,
        "session_id": session_id,
        "title": title,
    }).execute()
    return new_id

def update_supabase_conversation_title(sb: Client, conversation_id: str, new_title: str):
    try:
        sb.table("conversations").update({"title": new_title, "updated_at": datetime.now(timezone.utc).isoformat()}).eq("id", conversation_id).execute()
    except Exception:
        pass

def load_messages_from_supabase(sb: Client, conversation_id: str) -> List[Dict]:
    try:
        res = sb.table("messages").select("role, content").eq("conversation_id", conversation_id).order("created_at").execute()
        return [{"role": r["role"], "content": r["content"]} for r in (res.data or [])]
    except Exception:
        return []

def save_message_to_supabase(sb: Client, conversation_id: str, role: str, content: str):
    try:
        sb.table("messages").insert({
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
        }).execute()
    except Exception:
        pass

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
# Session state initialization & chat manager
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "-" + str(uuid.uuid4())[:8]

if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat" not in st.session_state:
    st.session_state.current_chat = "New Chat"

if "temp_attachments" not in st.session_state:
    st.session_state.temp_attachments = []

if "validation_status" not in st.session_state:
    st.session_state.validation_status = {}

def clear_attachments():
    st.session_state.temp_attachments = []

# Check persistence toggle state (default True)
use_persistence = st.session_state.get("use_persistence", True)
sb = get_supabase() if use_persistence else None

# Synchronize session_state.chats with Supabase on initial load if empty
if sb and not st.session_state.chats:
    db_convs = load_conversations_from_supabase(sb, st.session_state.session_id)
    if db_convs:
        for c in db_convs:
            c_title = c.get("title") or "Untitled Chat"
            msgs = load_messages_from_supabase(sb, c["id"])
            st.session_state.chats[c_title] = {"id": c["id"], "messages": msgs}
        st.session_state.current_chat = list(st.session_state.chats.keys())[0]

if not st.session_state.chats:
    default_title = "New Chat"
    conv_id = create_supabase_conversation(sb, st.session_state.session_id, default_title) if sb else None
    st.session_state.chats[default_title] = {"id": conv_id, "messages": []}
    st.session_state.current_chat = default_title

if st.session_state.current_chat not in st.session_state.chats:
    st.session_state.current_chat = list(st.session_state.chats.keys())[0]

# ---------------------------------------------------------------------------
# Sidebar: Navigation & Settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("💬 Chats")

    if st.button("➕ New Chat", use_container_width=True):
        chat_count = len(st.session_state.chats) + 1
        new_name = f"Chat {chat_count}"
        new_conv_id = create_supabase_conversation(sb, st.session_state.session_id, new_name) if sb else None
        st.session_state.chats[new_name] = {"id": new_conv_id, "messages": []}
        st.session_state.current_chat = new_name
        clear_attachments()
        st.rerun()

    st.markdown("---")

    chat_names = list(st.session_state.chats.keys())
    selected_chat = st.radio(
        "Saved Conversations",
        chat_names,
        index=chat_names.index(st.session_state.current_chat) if st.session_state.current_chat in chat_names else 0,
        label_visibility="collapsed",
    )

    if selected_chat != st.session_state.current_chat:
        st.session_state.current_chat = selected_chat
        clear_attachments()
        st.rerun()

    st.divider()
    st.header("🧠 Obrain2 Settings")

    provider_name = st.selectbox("Provider", list(PROVIDERS.keys()))
    provider = PROVIDERS[provider_name]
    raw_model_choice = st.selectbox("Model", provider["models"])
    model_choice = get_validated_model(provider_name, raw_model_choice)

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
    persist_toggle = st.checkbox("Persist chats to Supabase", value=use_persistence, key="use_persistence")
    if persist_toggle and sb:
        used_mb = get_total_storage_mb(sb)
        st.caption(f"Uploaded files storage: ~{used_mb:.1f} MB")
        if used_mb > WARN_STORAGE_MB:
            st.warning("Approaching Supabase free storage limit. Consider deleting old files.")
    elif persist_toggle and not sb:
        st.caption("Supabase credentials not active or not configured in secrets.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear current chat", use_container_width=True):
            current_chat_data = st.session_state.chats[st.session_state.current_chat]
            current_chat_data["messages"] = []
            clear_attachments()
            st.rerun()
    with col2:
        active_messages = st.session_state.chats[st.session_state.current_chat]["messages"]
        if active_messages:
            md_lines = ["# Obrain2 Chat Export", f"*Chat: {st.session_state.current_chat}*", f"*Exported: {datetime.now(timezone.utc).isoformat()}*", ""]
            for msg in active_messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                md_lines.append(f"**{role}:**\n{msg['content']}\n")
            md_content = "\n".join(md_lines)
            st.download_button(
                label="Export MD",
                data=md_content,
                file_name=f"obrain2_chat_{st.session_state.current_chat.replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True,
            )

# ---------------------------------------------------------------------------
# STICKY BOTTOM CSS & STYLING
# ---------------------------------------------------------------------------
st.markdown("""
    <style>
    /* Pin the input wrapper container strictly to viewport bottom */
    .st-fixed-bottom {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #131316;
        padding: 0.75rem 1rem 1rem 1rem;
        z-index: 99999;
        border-top: 1px solid #2d2d35;
        box-shadow: 0 -4px 15px rgba(0,0,0,0.4);
    }
    /* Padding to prevent chat history from hiding behind fixed bottom prompt bar */
    .block-container {
        padding-bottom: 10rem !important;
    }
    div[data-testid="stForm"] {
        background-color: #1e1e24 !important;
        border: 1px solid #374151 !important;
        border-radius: 1rem !important;
        padding: 0.5rem 0.75rem !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3) !important;
        max-width: 800px;
        margin: 0 auto;
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

# ---------------------------------------------------------------------------
# Main UI - Active Chat History Display
# ---------------------------------------------------------------------------
st.title("🧠 Obrain2 AI Assistant")
st.caption(f"Session: **{st.session_state.current_chat}** · Multi-provider · Multimodal · Persistent memory")

current_chat_data = st.session_state.chats[st.session_state.current_chat]
current_messages = current_chat_data["messages"]

for msg in current_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Sticky Prompt Form Container
# ---------------------------------------------------------------------------
st.markdown('<div class="st-fixed-bottom">', unsafe_allow_html=True)

with st.form(key="sticky_prompt_form", clear_on_submit=True):
    if st.session_state.get("temp_attachments"):
        st.caption("📎 **Attached Files:**")
        cols = st.columns(min(len(st.session_state.temp_attachments), 4))
        for idx, file in enumerate(st.session_state.temp_attachments):
            with cols[idx % 4]:
                st.caption(f"📄 {file.name[:12]}...")

    col_attach, col_input, col_submit = st.columns([1.5, 6.5, 1], vertical_alignment="center")

    with col_attach:
        uploaded_files = st.file_uploader(
            "Add Attachments",
            type=["png", "jpg", "jpeg", "pdf", "webp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="bottom_file_uploader",
        )

    with col_input:
        prompt_input = st.text_input(
            "Ask anything...",
            placeholder="Ask anything...",
            label_visibility="collapsed",
            key="bottom_prompt_input",
        )

    with col_submit:
        submitted = st.form_submit_button("↑", use_container_width=True)

    if uploaded_files:
        st.session_state.temp_attachments = uploaded_files

st.markdown('</div>', unsafe_allow_html=True)

if submitted:
    attachments = st.session_state.temp_attachments or uploaded_files or []
    prompt_text = prompt_input.strip()

    if prompt_text or attachments:
        safe_sec = get_safe_secrets()
        provider_key_name = provider["key_name"]
        active_api_key = key_manager.get_api_key(provider_key_name, safe_sec)

        if not active_api_key:
            st.error(f"Please configure an API key for {provider_name} in the sidebar settings first.")
            st.stop()

        # Validate active model ID
        model_choice = get_validated_model(provider_name, model_choice)

        # Model Routing Validation: Gemma vs. Gemini / Multimodal models
        is_gemma = "gemma" in model_choice.lower()

        # Process attachments
        extra_text_parts = []
        image_parts = []
        pdf_count = 0

        for file in attachments:
            file_bytes = file.getvalue()
            mime_type = file.type or "application/octet-stream"

            if mime_type.startswith("image/"):
                if is_gemma:
                    st.warning(f"⚠️ Model '{model_choice}' is text-only. Processing query without image payload '{file.name}'.")
                else:
                    image_parts.append((mime_type, file_bytes))
                extra_text_parts.append(f"[User attached image: {file.name}]")
            elif mime_type == "application/pdf" or file.name.lower().endswith(".pdf"):
                pdf_count += 1
                pdf_text = extract_text_from_pdf(file_bytes)
                extra_text_parts.append(f"[Content of PDF {file.name}]:\n{pdf_text[:12000]}")

        full_user_content = prompt_text
        if extra_text_parts:
            full_user_content = (prompt_text + "\n\n" + "\n\n".join(extra_text_parts)).strip()

        # Dynamic Auto-renaming of generic chat title on initial prompt
        if (st.session_state.current_chat.startswith("Chat ") or st.session_state.current_chat == "New Chat") and len(current_messages) == 0:
            if prompt_text:
                new_title = prompt_text[:22] + "..." if len(prompt_text) > 22 else prompt_text
            else:
                new_title = "Attachment Chat"

            # Avoid collision with existing titles
            base_new_title = new_title
            counter = 1
            while new_title in st.session_state.chats and new_title != st.session_state.current_chat:
                new_title = f"{base_new_title} ({counter})"
                counter += 1

            conv_obj = st.session_state.chats.pop(st.session_state.current_chat)
            st.session_state.chats[new_title] = conv_obj
            st.session_state.current_chat = new_title

            if persist_toggle and sb and conv_obj.get("id"):
                update_supabase_conversation_title(sb, conv_obj["id"], new_title)

        active_chat = st.session_state.chats[st.session_state.current_chat]
        active_msgs = active_chat["messages"]

        # Store user message
        active_msgs.append({"role": "user", "content": full_user_content})

        if persist_toggle and sb and active_chat.get("id"):
            save_message_to_supabase(sb, active_chat["id"], "user", full_user_content)

        clear_attachments()

        # Generate response
        try:
            assistant_text = ""

            if provider["type"] == "google":
                client = genai.Client(api_key=active_api_key)

                history = []
                for m in active_msgs[:-1]:
                    role = "user" if m["role"] == "user" else "model"
                    history.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))

                # Gemini multimodal parts handling vs Gemma text handling
                user_parts = [types.Part.from_text(text=full_user_content)]
                if not is_gemma:
                    for mime, data in image_parts:
                        user_parts.append(types.Part.from_bytes(data=data, mime_type=mime))

                chat = client.chats.create(
                    model=model_choice,
                    history=history,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        system_instruction=system_prompt,
                    ),
                )
                response = chat.send_message(user_parts)
                assistant_text = response.text or "(empty response)"

            else:  # OpenAI-compatible providers
                client = OpenAI(
                    base_url=provider["base_url"],
                    api_key=active_api_key,
                )
                msgs = [{"role": "system", "content": system_prompt}]
                for m in active_msgs[:-1]:
                    msgs.append({"role": m["role"], "content": m["content"]})

                content_list = [{"type": "text", "text": full_user_content}]
                if not is_gemma:
                    for mime, data in image_parts:
                        b64 = base64.b64encode(data).decode()
                        content_list.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        })

                msgs.append({"role": "user", "content": content_list if (image_parts and not is_gemma) else full_user_content})

                completion = client.chat.completions.create(
                    model=model_choice,
                    messages=msgs,
                    temperature=temperature,
                )
                assistant_text = completion.choices[0].message.content or "(empty response)"

            active_msgs.append({"role": "assistant", "content": assistant_text})

            if persist_toggle and sb and active_chat.get("id"):
                save_message_to_supabase(sb, active_chat["id"], "assistant", assistant_text)

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
            active_msgs.append({"role": "assistant", "content": f"⚠️ {friendly}"})

        st.rerun()
