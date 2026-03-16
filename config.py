import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
COLLECTIONS_DIR = os.path.join(DATA_DIR, "collections")

os.makedirs(COLLECTIONS_DIR, exist_ok=True)

DEFAULT_LLM_MODEL = "whatever-you-named-it-in-lmstudio"  # optional – can be empty
LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_URL", "http://localhost:1234/v1")
LMSTUDIO_API_KEY = "lm-studio"  # dummy value – LM Studio ignores it

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUPPORTED_EMBED_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "nomic-embed-text",  # if you also run it in LM Studio / Ollama
]

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
RETRIEVE_K = 4

CHAT_DB_PATH = os.path.join(BASE_DIR, "chat_history.db")