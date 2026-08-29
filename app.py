"""
Gemma 4 AI Assistant
A fluid, mobile-friendly conversational assistant powered by Google Gemma 4.
Supports multi-turn memory, custom system prompts, optional persistence, and export.
"""

import streamlit as st
from google import genai
from google.genai import types
import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Gemma 4 AI Assistant",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AVAILABLE_MODELS = [
    "gemma-4-26b-a4b-it",   # MoE – balanced speed / quality
    "gemma-4-31b-it",       # Dense – stronger reasoning
]
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, intelligent, and concise AI chat assistant "
    "powered by Gemma 4. Answer clearly and accurately."
)
DB_PATH = Path("data/chat_history.db")

# ---------------------------------------------------------------------------
# Optional SQLite persistence helpers
# ---------------------------------------------------------------------------
def init_db():
    """Create the messages table if it does not exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()

def load_messages_from_db(session_id: str) -> list:
    """Load conversation history for a given session."""
    if not DB_PATH.exists():
        return []
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [{"role": r, "content": c} for r, c in rows]

def save_message_to_db(session_id: str, role: str, content: str):
    """Append a single message to the database."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, datetime.utcnow().isoformat()),
        )
        conn.commit()

def clear_db_session(session_id: str):
    """Delete all messages belonging to a session."""
    if not DB_PATH.exists():
        return
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.commit()

# ---------------------------------------------------------------------------
# Sidebar – configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    model_choice = st.selectbox(
        "Model Variant",
        AVAILABLE_MODELS,
        index=0,
        help="26B MoE is faster; 31B Dense offers stronger reasoning.",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Higher values increase creativity; lower values increase determinism.",
    )

    system_prompt = st.text_area(
        "System Instruction",
        value=DEFAULT_SYSTEM_PROMPT,
        height=120,
        help="Controls the assistant persona and behaviour for the entire conversation.",
    )

    use_persistence = st.checkbox(
        "Persist chat to SQLite",
        value=False,
        help="Store conversation history on disk so it survives page reloads.",
    )

    # API key: prefer Streamlit secrets, fall back to manual entry
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        api_key = None

    if not api_key:
        api_key = st.text_input(
            "API Key",
            type="password",
            help="Obtain a free key from https://aistudio.google.com/apikey",
        )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        clear_clicked = st.button("Clear Chat", use_container_width=True)
    with col2:
        export_clicked = st.button("Export MD", use_container_width=True)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    # Simple unique identifier for optional DB persistence
    st.session_state.session_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")

if use_persistence:
    init_db()
    # Load existing history only once when persistence is first enabled
    if not st.session_state.messages:
        st.session_state.messages = load_messages_from_db(st.session_state.session_id)

# Handle clear
if clear_clicked:
    st.session_state.messages = []
    if use_persistence:
        clear_db_session(st.session_state.session_id)
    st.rerun()

# Handle export
if export_clicked and st.session_state.messages:
    md_lines = ["# Gemma 4 Chat Export", f"*Exported: {datetime.utcnow().isoformat()}Z*", ""]
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        md_lines.append(f"**{role}:**\n{msg['content']}\n")
    md_content = "\n".join(md_lines)
    st.sidebar.download_button(
        label="Download Markdown",
        data=md_content,
        file_name=f"gemma_chat_{st.session_state.session_id}.md",
        mime="text/markdown",
    )

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("💬 Gemma 4 AI Assistant")
st.caption(
    "A fluid, mobile-friendly conversational assistant with active chat memory "
    "powered by Google Gemma 4."
)

# Render conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------------------------------------------------
# Chat input & generation
# ---------------------------------------------------------------------------
if user_prompt := st.chat_input("Ask Gemma 4 anything..."):
    if not api_key:
        st.error("Please provide a valid API key in the sidebar or in Streamlit secrets.")
        st.stop()

    # 1. Display and store user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)
    if use_persistence:
        save_message_to_db(st.session_state.session_id, "user", user_prompt)

    # 2. Generate assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking…")

        try:
            client = genai.Client(api_key=api_key)

            # Build history excluding the latest user turn (sent via send_message)
            history = []
            for msg in st.session_state.messages[:-1]:
                sdk_role = "user" if msg["role"] == "user" else "model"
                history.append(
                    types.Content(
                        role=sdk_role,
                        parts=[types.Part.from_text(text=msg["content"])],
                    )
                )

            # Create chat session with accumulated history and system instruction
            chat_session = client.chats.create(
                model=model_choice,
                history=history,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    system_instruction=system_prompt,
                ),
            )

            # Send the new user message; the model sees the full prior context
            response = chat_session.send_message(user_prompt)
            assistant_response = response.text or "(No response generated)"

            message_placeholder.markdown(assistant_response)

            # 3. Store assistant response
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_response}
            )
            if use_persistence:
                save_message_to_db(
                    st.session_state.session_id, "assistant", assistant_response
                )

        except Exception as e:
            error_msg = str(e)
            # Provide more actionable feedback for common failures
            if "API_KEY" in error_msg.upper() or "401" in error_msg or "403" in error_msg:
                friendly = "Authentication failed. Please verify your API key."
            elif "429" in error_msg or "quota" in error_msg.lower():
                friendly = "Rate limit or quota exceeded. Please wait and try again."
            elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                friendly = f"Model '{model_choice}' is unavailable. Try the other variant."
            else:
                friendly = f"Error generating response: {error_msg}"
            message_placeholder.error(friendly)
