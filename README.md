# Gemma 4 AI Assistant

A fluid, mobile-friendly conversational assistant powered by Google Gemma 4 models.

Built with **Streamlit** (Python) and the official **`google-genai` SDK**, this application supports multi-turn conversations with persistent session memory, customizable system prompts, optional SQLite chat history persistence, and chat exports to Markdown.

---

## Features

- **Gemma 4 Models:** Select between `gemma-4-26b-a4b-it` (MoE - fast) and `gemma-4-31b-it` (Dense - strong reasoning).
- **Multi-Turn Memory:** Preserves complete conversation history across turns using Google GenAI chat sessions.
- **Custom System Instructions:** Editable prompt persona to control model behavior dynamically.
- **Optional SQLite Persistence:** Save chat history to disk (`data/chat_history.db`) across page reloads.
- **Markdown Export:** Export your conversations into formatted Markdown files.
- **Streamlit Cloud Ready:** Designed for free zero-cost deployment on Streamlit Community Cloud.

---

## Local Setup & Development

### 1. Prerequisites
- Python 3.10 or higher
- Git

### 2. Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd gemma-chat-assistant
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Secrets:**
   Create `.streamlit/secrets.toml` locally with your Google Gemini API key (obtainable from [Google AI Studio](https://aistudio.google.com/apikey)):
   ```toml
   GEMINI_API_KEY = "your_actual_api_key_here"
   ```

5. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

---

## Deployment to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with your GitHub account.
3. Click **Create app** and select your repository, branch, and `app.py` path.
4. Under **App settings -> Secrets**, enter your key:
   ```toml
   GEMINI_API_KEY = "your_actual_api_key_here"
   ```
5. Click **Deploy**.
