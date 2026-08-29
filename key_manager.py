"""
API Key Storage and Validation Module for Obrain2
"""

import os
import json
from typing import Dict, Tuple, Optional
from google import genai
from openai import OpenAI

CONFIG_FILE_PATH = os.path.expanduser("~/.obrain2_keys.json")

def load_stored_keys() -> Dict[str, str]:
    """Load keys from local JSON config file if it exists."""
    if os.path.exists(CONFIG_FILE_PATH):
        try:
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_stored_keys(keys: Dict[str, str]) -> bool:
    """Save keys dict to local JSON config file."""
    try:
        with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(keys, f, indent=2)
        return True
    except Exception:
        return False

def get_api_key(provider_key_name: str, st_secrets=None) -> Optional[str]:
    """
    Get API key for a provider key name (e.g., 'GEMINI_API_KEY', 'DEEPSEEK_API_KEY').
    Checks:
    1. Saved user configuration file (`~/.obrain2_keys.json`)
    2. Streamlit secrets (`st_secrets`)
    3. Environment variables
    """
    stored_keys = load_stored_keys()
    if stored_keys.get(provider_key_name):
        return stored_keys[provider_key_name]

    if st_secrets:
        try:
            val = st_secrets.get(provider_key_name)
            if val:
                return str(val)
        except Exception:
            pass

    return os.environ.get(provider_key_name)

def save_provider_key(provider_key_name: str, api_key: str) -> bool:
    """Save an individual provider key."""
    keys = load_stored_keys()
    keys[provider_key_name] = api_key.strip()
    return save_stored_keys(keys)

def revoke_provider_key(provider_key_name: str) -> bool:
    """Revoke (delete) a stored key for a provider."""
    keys = load_stored_keys()
    if provider_key_name in keys:
        del keys[provider_key_name]
        return save_stored_keys(keys)
    return True

def mask_key(key: str) -> str:
    """Mask key for display (e.g. 'sk-1234...5678')."""
    if not key:
        return ""
    if len(key) <= 8:
        return "••••••••"
    return key[:4] + "••••••••" + key[-4:]

def validate_key(provider_type: str, base_url: Optional[str], api_key: str) -> Tuple[bool, str]:
    """
    Validate an API key by making a minimal request to provider endpoint.
    Returns (is_valid, message).
    """
    if not api_key or not api_key.strip():
        return False, "API key is empty."

    api_key = api_key.strip()

    try:
        if provider_type == "google":
            client = genai.Client(api_key=api_key)
            # List models to test credentials
            models = list(client.models.list())
            if models:
                return True, "Key validated successfully!"
            return False, "No models returned from Google API."

        else: # OpenAI-compatible
            client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=10.0,
            )
            # Try listing models
            client.models.list()
            return True, "Key validated successfully!"

    except Exception as e:
        err_msg = str(e)
        if "401" in err_msg or "auth" in err_msg.lower() or "invalid" in err_msg.lower() or "api key" in err_msg.lower() or "unauthorized" in err_msg.lower():
            return False, "Authentication failed: Invalid API Key."
        return False, f"Validation failed: {err_msg[:120]}"
