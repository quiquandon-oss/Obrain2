# Obrain2 AI Assistant

Obrain2 is a multi-provider, multimodal AI chat assistant powered by **Streamlit**, supporting Google Gemma 4 and OpenRouter models (including Qwen & DeepSeek), image/camera/PDF inputs, persistent chat history via **Supabase**, and Markdown export.

---

## Features

- **Multi-Provider LLM Support:**
  - **Google (Gemma 4):** `gemma-4-26b-a4b-it` (MoE) and `gemma-4-31b-it` (Dense).
  - **OpenRouter:** Models including Gemma 4, Qwen 3 (`qwen/qwen3-32b`, `qwen/qwen3-72b`), DeepSeek (`deepseek/deepseek-chat`, `deepseek/deepseek-reasoner`), and GLM (`z-ai/glm-4.5-flash`).
- **Multimodal Inputs:**
  - **Images & Photos:** Multiple image upload (`PNG`, `JPG`, `JPEG`, `WEBP`).
  - **Camera Capture:** Live photo capture via browser camera.
  - **PDF Documents:** Upload PDF files with automatic text extraction.
- **Persistent Memory with Supabase:**
  - Full-time persistent conversation history.
  - Storage capacity tracking and high-usage alerts (~400 MB warning threshold).
- **Markdown Export:**
  - Export chat history directly into formatted Markdown (`.md`) files.

---

## Setup & Configuration

### 1. Installation

```bash
git clone <repository-url>
cd obrain2
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Streamlit Secrets (`.streamlit/secrets.toml`)

Create `.streamlit/secrets.toml` locally (or configure in Streamlit Cloud → Settings → Secrets):

```toml
GEMINI_API_KEY = "your_gemini_key"
OPENROUTER_API_KEY = "your_openrouter_key"

[supabase]
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_KEY = "your_supabase_anon_or_service_key"
```

### 3. Supabase Database Schema Setup

In your Supabase project's SQL Editor, execute the following schema:

```sql
-- conversations table
create table if not exists conversations (
  id uuid primary key default gen_random_uuid(),
  session_id text not null,
  title text,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- messages table
create table if not exists messages (
  id uuid primary key default gen_random_uuid(),
  conversation_id uuid references conversations(id) on delete cascade,
  role text not null check (role in ('user','assistant')),
  content text not null,
  created_at timestamptz default now()
);

-- optional: uploaded files metadata (with size tracking)
create table if not exists uploaded_files (
  id uuid primary key default gen_random_uuid(),
  conversation_id uuid references conversations(id) on delete cascade,
  filename text,
  mime_type text,
  size_bytes bigint,
  storage_path text,
  created_at timestamptz default now()
);

-- index for fast lookup
create index if not exists idx_messages_conv on messages(conversation_id);
create index if not exists idx_conv_session on conversations(session_id);
```

---

## Running the Application

Launch the app locally:

```bash
streamlit run app.py
```
