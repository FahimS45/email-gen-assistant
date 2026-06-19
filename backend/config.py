"""
config.py

Configuration values for the email generation pipeline.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Model Configuration 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = "https://api.openai.com/v1"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_API_KEY = "ollama"  # Ollama doesn't need a real key but the client requires one

# Embedding model for fact recall metric
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Judge Model (used in LLM-as-Judge metric) 
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
JUDGE_BASE_URL = OPENAI_BASE_URL

# App Settings
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))

# Tone Options (fixed set exposed to frontend)
TONE_OPTIONS = [
    "formal",
    "casual",
    "urgent",
    "empathetic",
    "persuasive",
    "apologetic",
    "assertive",
    "friendly",
]
