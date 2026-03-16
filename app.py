import streamlit as st
import os
import tempfile
import hashlib
import json
from datetime import datetime
from typing import List, Tuple

from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredPowerPointLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS STYLING
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
""", unsafe_allow_html=True)

st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0f1a 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Chat messages */
    .stChatMessage {
        background: rgba(255,255,255,0.05);
        border-radius: 18px;
        padding: 16px 20px;
        margin: 8px 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    [data-testid="chatAvatarIcon-user"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="chatAvatarIcon-assistant"] {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
    }
    
    /* Custom Chat Input Container */
    .custom-chat-input {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: linear-gradient(180deg, rgba(26, 26, 46, 0.95) 0%, #1a1a2e 100%);
        backdrop-filter: blur(10px);
        border-top: 1px solid rgba(255,255,255,0.1);
        padding: 16px 24px;
        z-index: 1000;
    }
    
    .chat-input-container {
        max-width: 1200px;
        margin: 0 auto;
        display: flex;
        align-items: center;
        gap: 12px;
        background: rgba(255,255,255,0.05);
        border-radius: 30px;
        padding: 8px 16px;
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
    }
    
    .chat-input-container:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
        background: rgba(255,255,255,0.08);
    }
    
    .chat-input-container input {
        flex: 1;
        background: transparent;
        border: none;
        color: white;
        font-size: 16px;
        padding: 12px;
        outline: none;
    }
    
    .chat-input-container input::placeholder {
        color: rgba(255,255,255,0.5);
        font-style: italic;
    }
    
    .chat-input-btn {
        background: none;
        border: none;
        color: rgba(255,255,255,0.7);
        cursor: pointer;
        padding: 10px;
        border-radius: 50%;
        transition: all 0.2s;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
    }
    
    .chat-input-btn:hover {
        background: rgba(255,255,255,0.1);
        color: white;
        transform: scale(1.1);
    }
    
    .chat-input-btn.send-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 45px;
        height: 45px;
    }
    
    .chat-input-btn.send-btn:hover {
        transform: scale(1.1) rotate(5deg);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .chat-input-btn.stop-btn {
        background: #ff6b6b;
        color: white;
        width: 45px;
        height: 45px;
    }
    
    .chat-input-btn.stop-btn:hover {
        background: #ff5252;
    }
    
    /* Hide default Streamlit elements */
    div[data-testid="stChatInput"] {
        display: none !important;
    }
    
    footer {
        display: none !important;
    }
    
    /* Main content padding for fixed input */
    .main-content {
        padding-bottom: 100px !important;
    }
    
    /* Buttons */
    div.stButton > button {
        border-radius: 12px;
        border: none;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 12px 24px;
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Primary button */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00d4aa 0%, #00a884 100%);
    }
    
    div.stButton > button[kind="primary"]:hover {
        box-shadow: 0 8px 25px rgba(0, 212, 170, 0.4);
    }
    
    /* Cards */
    .feature-card {
        background: rgba(255,255,255,0.05);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(255,255,255,0.1);
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        border-color: #667eea;
    }
    
    /* Status indicators */
    .status-connected {
        color: #00d4aa;
        font-weight: bold;
    }
    
    .status-disconnected {
        color: #ff6b6b;
        font-weight: bold;
    }
    
    /* Metrics */
    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* File uploader */
    div[data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.03);
        border-radius: 16px;
        padding: 20px;
        border: 2px dashed rgba(255,255,255,0.2);
        transition: all 0.3s ease;
    }
    
    div[data-testid="stFileUploader"]:hover {
        border-color: #667eea;
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Titles */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Subheader */
    .subheader {
        color: #a0aec0;
        font-size: 14px;
        font-weight: 500;
    }
    
    /* Custom divider */
    hr {
        border-color: rgba(255,255,255,0.1);
        margin: 24px 0;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.05);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.2);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255,255,255,0.3);
    }
    
    /* Loading animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .loading {
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    /* Success/Error/Warning messages */
    .stAlert {
        border-radius: 12px;
        border: none;
    }
    
    /* Select boxes */
    div[data-testid="stSelectbox"] > div > div {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        color: white;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Code blocks */
    code {
        background: rgba(0,0,0,0.3);
        border-radius: 6px;
        padding: 2px 6px;
        color: #00d4aa;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
DOCS_PATH = os.path.join(BASE_DIR, "data", "documents")
METADATA_JSON = os.path.join(BASE_DIR, "processed_files.json")
CHAT_HISTORY_JSON = os.path.join(BASE_DIR, "chat_history.json")

# Create necessary folders
os.makedirs(DOCS_PATH, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
RETRIEVE_K = 4
MAX_HISTORY = 4
MAX_NEW_TOKENS = 600

LMSTUDIO_URL = os.environ.get("LMSTUDIO_URL", "http://localhost:1234/v1")
LMSTUDIO_KEY = "lm-studio"

AVAILABLE_MODELS = [
    "mistralai/ministral-3-3b",
    "google/gemma-3-4b",
    "qwen2.5-7b-instruct-1m"
]

MODEL_DISPLAY_NAMES = {
    "mistralai/ministral-3-3b": "🦙 Ministral 3B",
    "google/gemma-3-4b": "💎 Gemma 3 4B",
    "qwen2.5-7b-instruct-1m": "🔮 Qwen 2.5 7B"
}

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def load_chat_history() -> dict:
    """Load all chat histories from JSON file"""
    if os.path.exists(CHAT_HISTORY_JSON):
        with open(CHAT_HISTORY_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_chat_history(history: dict):
    """Save all chat histories to JSON file"""
    with open(CHAT_HISTORY_JSON, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def get_chat_sessions() -> List[str]:
    """Get list of all chat session names"""
    history = load_chat_history()
    return list(history.keys())


def create_new_chat() -> str:
    """Create a new chat session with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"Chat {timestamp}"


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def get_loader(file_path: str):
    """Get the appropriate loader for a file based on its extension"""
    ext = file_path.lower()
    try:
        if ext.endswith(".pdf"):
            return PyPDFLoader(file_path)
        elif ext.endswith((".txt", ".md", ".py", ".js", ".html", ".css", ".json")):
            return TextLoader(file_path, encoding="utf-8")
        elif ext.endswith(".docx"):
            try:
                return Docx2txtLoader(file_path)
            except ImportError:
                st.error("docx2txt is not installed. Please run: pip install docx2txt")
                return None
        elif ext.endswith((".ppt", ".pptx")):
            try:
                return UnstructuredPowerPointLoader(file_path)
            except ImportError:
                st.error("unstructured is not installed. Please run: pip install unstructured")
                return None
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        st.error(f"Error loading file {os.path.basename(file_path)}: {str(e)}")
        return None


def load_processed_metadata() -> dict:
    if os.path.exists(METADATA_JSON):
        with open(METADATA_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_processed_metadata(metadata: dict):
    with open(METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def process_files(uploaded_files) -> Tuple[int, List[str]]:
    if not uploaded_files:
        return 0, []

    embeddings = get_embeddings()
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    new_chunks = 0
    skipped = []
    newly_processed = []
    failed = []

    for file in uploaded_files:
        try:
            content = file.getvalue()
            file_hash = compute_file_hash(content)
            filename = file.name

            # Check if already processed
            prev = st.session_state.processed_files.get(filename)
            if prev and prev.get("hash") == file_hash:
                skipped.append(filename)
                continue

            # Save file permanently to data/documents/
            file_path = os.path.join(DOCS_PATH, filename)
            with open(file_path, "wb") as f:
                f.write(content)

            # Load and chunk the document
            loader = get_loader(file_path)
            if loader is None:
                failed.append(filename)
                continue
                
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            splits = text_splitter.split_documents(docs)

            # Add to vectorstore with source metadata
            if splits:
                vectorstore.add_documents(splits)
                new_chunks += len(splits)

            # Save metadata
            st.session_state.processed_files[filename] = {
                "hash": file_hash,
                "last_processed": datetime.now().isoformat(),
                "path": file_path,
                "chunks": len(splits)
            }

            newly_processed.append(filename)
            
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            failed.append(file.name)

    # Save metadata
    if newly_processed:
        save_processed_metadata(st.session_state.processed_files)
        # Refresh the document list
        st.session_state.available_docs = list(st.session_state.processed_files.keys())

    return new_chunks, skipped, failed


def get_available_documents() -> List[str]:
    """Get list of available documents from processed files"""
    processed = load_processed_metadata()
    return list(processed.keys())


def build_chain(retriever, selected_doc: str = None):
    llm = get_llm()
    if llm is None:
        return None

    if selected_doc:
        prompt = ChatPromptTemplate.from_template(
            f"""You are a helpful, accurate assistant. Answer using ONLY the provided context from {selected_doc}.
If the context lacks relevant information, say "I don't have enough information."

Context:
{{context}}

Question: {{question}}

Provide a clear, concise answer:"""
        )
    else:
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful, accurate assistant. Answer using ONLY the provided context.
If the context lacks relevant information, say "I don't have enough information."

Context:
{context}

Question: {question}

Provide a clear, concise answer:"""
        )

    def format_docs(docs):
        if not docs:
            return ""
        return "\n\n".join(
            f"[{i}] {d.page_content.strip()} (from {os.path.basename(d.metadata.get('source', 'unknown'))})"
            for i, d in enumerate(docs, 1)
        )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


@st.cache_resource
def get_llm():
    model_name = st.session_state.get("selected_model", AVAILABLE_MODELS[0])

    try:
        llm = ChatOpenAI(
            base_url=LMSTUDIO_URL,
            api_key=LMSTUDIO_KEY,
            model=model_name,
            temperature=0.05,
            streaming=True,
            max_tokens=MAX_NEW_TOKENS
        )
        llm.invoke("hi")
        st.session_state.current_model = model_name
        return llm
    except Exception as e:
        st.session_state.current_model = "Failed"
        return None


def speak(text: str, idx: int):
    if not text.strip():
        return
    escaped = text.replace('"', '\\"').replace("\n", " ")
    st.components.v1.html(f"""
    <script>
    const ut = new SpeechSynthesisUtterance("{escaped}");
    ut.lang = 'en-US';
    ut.rate = 1.1;
    window.speechSynthesis.speak(ut);
    </script>
    """, height=0)


def get_vectorstore_count() -> int:
    if os.path.exists(CHROMA_PATH):
        try:
            embeddings = get_embeddings()
            vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
            return vectorstore._collection.count()
        except:
            return 0
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

if "processed_files" not in st.session_state:
    st.session_state.processed_files = load_processed_metadata()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = AVAILABLE_MODELS[0]

if "current_model" not in st.session_state:
    st.session_state.current_model = "None"

if "selected_document" not in st.session_state:
    st.session_state.selected_document = None

if "available_docs" not in st.session_state:
    st.session_state.available_docs = list(st.session_state.processed_files.keys())

# Chat history management
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = load_chat_history()

if "current_chat" not in st.session_state:
    # Create a new chat session if none exists
    if st.session_state.chat_sessions:
        # Load the most recent chat
        st.session_state.current_chat = list(st.session_state.chat_sessions.keys())[0]
        st.session_state.messages = st.session_state.chat_sessions[st.session_state.current_chat]
    else:
        # Create a new chat
        new_chat_name = create_new_chat()
        st.session_state.current_chat = new_chat_name
        st.session_state.chat_sessions = {new_chat_name: []}
        save_chat_history(st.session_state.chat_sessions)

# Initialize stop flag
if "stop_generation" not in st.session_state:
    st.session_state.stop_generation = False

# Initialize input value
if "input_value" not in st.session_state:
    st.session_state.input_value = ""

# ═══════════════════════════════════════════════════════════════════════════════
# UI - PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="GRL Bot | Local RAG Assistant",
    page_icon="images/robo.png.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
# UI - SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # Logo and Title
    logo_path = "images/robo.png.jpg"
    if os.path.exists(logo_path):
        st.image(logo_path, width=120)
    else:
        st.markdown('<h1 style="font-size: 48px; margin: 0; animation: pulse 2s infinite;">🤖</h1>', unsafe_allow_html=True)
    st.markdown("""
    <h2 style="color: #00d4aa !important; margin: 10px 0; font-size: 28px;">GRL Bot</h2>
    <p style="color: #a0aec0; margin: 0;">Local RAG Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Connection Status
    st.markdown("### 📡 Connection Status")
    
    # Check LM Studio connection
    llm_status = get_llm()
    if llm_status is not None:
        st.markdown("""
        <div class="feature-card" style="text-align: center;">
            <span style="font-size: 32px;">✅</span><br>
            <strong class="status-connected">Connected to LM Studio</strong><br>
            <code style="color: #a0aec0; font-size: 12px;">http://127.0.0.1:1234</code>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="feature-card" style="text-align: center;">
            <span style="font-size: 32px;">❌</span><br>
            <strong class="status-disconnected">LM Studio Not Running</strong><br>
            <code style="color: #a0aec0; font-size: 12px;">Start LM Studio at port 1234</code>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Model Selection
    st.markdown("### 🧠 Model Selection")
    
    # Map display names to actual model names
    model_options = [MODEL_DISPLAY_NAMES[m] for m in AVAILABLE_MODELS]
    current_display = MODEL_DISPLAY_NAMES.get(st.session_state.get("selected_model", AVAILABLE_MODELS[0]))
    current_idx = 0
    for i, name in enumerate(MODEL_DISPLAY_NAMES.values()):
        if name == current_display:
            current_idx = i
            break
    
    selected_display = st.selectbox(
        "Choose Model",
        options=model_options,
        index=current_idx,
        label_visibility="collapsed"
    )
    
    # Find actual model name from display name
    actual_model = next((k for k, v in MODEL_DISPLAY_NAMES.items() if v == selected_display), AVAILABLE_MODELS[0])
    
    if actual_model != st.session_state.get("selected_model"):
        st.session_state.selected_model = actual_model
        get_llm.clear()
        st.rerun()
    
    # Model info
    st.caption(f"Active: **{selected_display}**")
    
    st.divider()
    
    # Document Selection
    st.markdown("### 📄 Select Document")
    
    # Refresh document list
    st.session_state.available_docs = list(load_processed_metadata().keys())
    
    if st.session_state.available_docs:
        doc_options = ["All Documents"] + st.session_state.available_docs
        
        # Find current index
        current_doc = st.session_state.get("selected_document")
        if current_doc and current_doc in doc_options:
            doc_idx = doc_options.index(current_doc)
        else:
            doc_idx = 0
            
        selected_doc = st.selectbox(
            "Choose Document",
            options=doc_options,
            index=doc_idx,
            label_visibility="collapsed"
        )
        
        if selected_doc == "All Documents":
            st.session_state.selected_document = None
        else:
            st.session_state.selected_document = selected_doc
            
        st.caption(f"Selected: **{selected_doc}**")
    else:
        st.warning("No documents uploaded yet")
        st.session_state.selected_document = None
    
    st.divider()
    
    # Chat History
    st.markdown("### 💬 Chat History")
    
    # Refresh chat sessions
    st.session_state.chat_sessions = load_chat_history()
    chat_sessions = list(st.session_state.chat_sessions.keys())
    
    if chat_sessions:
        # Add "New Chat" option
        session_options = ["➕ New Chat"] + chat_sessions
        
        # Find current index
        current_chat = st.session_state.get("current_chat")
        if current_chat and current_chat in session_options:
            chat_idx = session_options.index(current_chat)
        else:
            chat_idx = 0
            
        selected_chat = st.selectbox(
            "Choose Chat",
            options=session_options,
            index=chat_idx,
            label_visibility="collapsed"
        )
        
        # Handle chat selection
        if selected_chat == "➕ New Chat":
            # Create new chat
            new_chat_name = create_new_chat()
            st.session_state.current_chat = new_chat_name
            st.session_state.messages = []
            st.session_state.chat_sessions[new_chat_name] = []
            save_chat_history(st.session_state.chat_sessions)
            st.rerun()
        elif selected_chat != current_chat:
            # Switch to selected chat
            st.session_state.current_chat = selected_chat
            st.session_state.messages = st.session_state.chat_sessions[selected_chat]
            st.rerun()
        
        # Show current chat info
        st.caption(f"Current: **{selected_chat}**")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_chat_btn"):
            st.session_state.messages = []
            st.session_state.chat_sessions[selected_chat] = []
            save_chat_history(st.session_state.chat_sessions)
            st.rerun()
    else:
        if st.button("➕ Start New Chat", use_container_width=True):
            new_chat_name = create_new_chat()
            st.session_state.current_chat = new_chat_name
            st.session_state.messages = []
            st.session_state.chat_sessions = {new_chat_name: []}
            save_chat_history(st.session_state.chat_sessions)
            st.rerun()
    
    st.divider()
    
    # Stats
    st.markdown("### 📊 Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents", len(st.session_state.available_docs))
    with col2:
        st.metric("Chunks", get_vectorstore_count())
    
    st.divider()
    
    # Help
    with st.expander("ℹ️ Help & Instructions"):
        st.markdown("""
        **How to use:**
        1. Start LM Studio and load a model
        2. Upload documents using the file uploader
        3. Click "Process Files" to index them
        4. Select a document from the dropdown
        5. Start chatting with your documents!
        
        **Supported formats:**
        - PDF, TXT, MD, DOCX, PPTX
        
        **Keyboard shortcuts:**
        - Enter: Send message
        - Shift + Enter: New line
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# UI - MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════════

# Main content with padding for fixed input
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Header
st.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h1 style="font-size: 42px; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        💬 Chat with Your Documents
    </h1>
    <p style="color: #a0aec0; font-size: 18px; margin: 10px 0;">
        Upload files, process them, and ask questions - all locally!
    </p>
</div>
""", unsafe_allow_html=True)

# Document Upload Section
with st.container():
    st.markdown("### 📁 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Drag & drop files here",
        type=["pdf", "txt", "md", "docx", "pptx", "py", "js", "html", "css", "json"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        st.markdown("**Selected files:**")
        for f in uploaded_files:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"📄 {f.name}")
            with col2:
                st.write(f"{(f.size/1024):.1f} KB")
            with col3:
                st.write(f"Type: {f.type.split('/')[-1] if '/' in f.type else 'unknown'}")
        
        if st.button("🚀 Process Files", type="primary", use_container_width=True):
            with st.spinner("🔄 Indexing documents..."):
                added, skipped, failed = process_files(uploaded_files)
                
                if added > 0:
                    st.success(f"✅ Successfully indexed {added:,} chunks!")
                if skipped:
                    st.info(f"ℹ️ Skipped {len(skipped)} unchanged files")
                if failed:
                    st.error(f"❌ Failed to process {len(failed)} files")
                if not added and not skipped and not failed:
                    st.warning("⚠️ No files to process")
                
                st.rerun()

# Chat Section
st.divider()
st.markdown("### 💬 Chat")

# Display chat messages
chat_container = st.container()
with chat_container:
    for idx, msg in enumerate(st.session_state.get("messages", [])):
        role = msg["role"]
        content = msg["content"]
        
        with st.chat_message(role, avatar="👤" if role == "user" else "🤖"):
            st.markdown(content)
            
            # Add copy button for assistant messages
            if role == "assistant":
                col1, col2 = st.columns([6, 1])
                with col2:
                    if st.button("📋", key=f"copy_{idx}", help="Copy to clipboard"):
                        st.write(f'<script>navigator.clipboard.writeText(`{content}`)</script>', 
                                unsafe_allow_html=True)
                        st.toast("Copied to clipboard!")

# Close main content div
st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CHAT INPUT
# ═══════════════════════════════════════════════════════════════════════════════

# Custom chat input at the bottom
st.markdown("""
<div class="custom-chat-input">
    <div class="chat-input-container">
        <button class="chat-input-btn" onclick="document.getElementById('chat-input').focus()" title="Focus input">
            <i class="fas fa-keyboard"></i>
        </button>
        <input type="text" id="chat-input" placeholder="💭 Ask a question about your documents..." 
               value="" autocomplete="off">
        <button class="chat-input-btn" onclick="toggleMic()" title="Voice input">
            <i class="fas fa-microphone"></i>
        </button>
        <button class="chat-input-btn send-btn" onclick="sendMessage()" title="Send message (Enter)">
            <i class="fas fa-paper-plane"></i>
        </button>
        <button class="chat-input-btn stop-btn" id="stop-btn" style="display: none;" onclick="stopGeneration()" title="Stop generating">
            <i class="fas fa-stop"></i>
        </button>
    </div>
</div>

<script>
// Get input element
const chatInput = document.getElementById('chat-input');
const stopBtn = document.getElementById('stop-btn');
let isGenerating = false;

// Load saved input value
const savedValue = sessionStorage.getItem('chatInputValue');
if (savedValue) {
    chatInput.value = savedValue;
}

// Save input value on change
chatInput.addEventListener('input', function() {
    sessionStorage.setItem('chatInputValue', this.value);
});

// Handle Enter key
chatInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Focus input on load
window.addEventListener('load', function() {
    chatInput.focus();
});

function sendMessage() {
    const message = chatInput.value.trim();
    if (message && !isGenerating) {
        // Clear input
        chatInput.value = '';
        sessionStorage.removeItem('chatInputValue');
        
        // Send message to Streamlit using query parameters
        const url = new URL(window.location.href);
        url.searchParams.set('message', message);
        window.location.href = url.toString();
        
        // Show stop button
        isGenerating = true;
        stopBtn.style.display = 'flex';
    }
}

function stopGeneration() {
    const url = new URL(window.location.href);
    url.searchParams.set('stop', 'true');
    window.location.href = url.toString();
    stopBtn.style.display = 'none';
    isGenerating = false;
}

function toggleMic() {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        
        recognition.lang = 'en-US';
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;
        
        recognition.start();
        
        recognition.onresult = function(event) {
            const transcript = event.results[0][0].transcript;
            chatInput.value = transcript;
            sessionStorage.setItem('chatInputValue', transcript);
            chatInput.focus();
        };
        
        recognition.onerror = function(event) {
            console.error('Speech recognition error:', event.error);
            alert('Speech recognition error. Please try again.');
        };
    } else {
        alert('Speech recognition is not supported in your browser. Try Chrome or Edge.');
    }
}

// Listen for generation complete
window.addEventListener('load', function() {
    // Check if we're returning from a message
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.has('message')) {
        // Clear the message parameter
        urlParams.delete('message');
        const newUrl = window.location.pathname + (urlParams.toString() ? '?' + urlParams.toString() : '');
        window.history.replaceState({}, '', newUrl);
    }
    
    // Check for stop parameter
    if (urlParams.has('stop')) {
        urlParams.delete('stop');
        const newUrl = window.location.pathname + (urlParams.toString() ? '?' + urlParams.toString() : '');
        window.history.replaceState({}, '', newUrl);
        stopBtn.style.display = 'none';
        isGenerating = false;
    }
});
</script>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HANDLE MESSAGES
# ═══════════════════════════════════════════════════════════════════════════════

# Check for message from URL parameter
params = st.query_params
message = params.get("message", [None])
if message and message[0]:
    prompt = message[0]
    
    # Clear the message parameter
    st.query_params.clear()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    # Save to chat history
    st.session_state.chat_sessions[st.session_state.current_chat] = st.session_state.messages
    save_chat_history(st.session_state.chat_sessions)
    
    # Get vectorstore and build chain
    if os.path.exists(CHROMA_PATH) and get_vectorstore_count() > 0:
        embeddings = get_embeddings()
        vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        
        # Get selected document
        selected_doc = st.session_state.get("selected_document")
        
        # Build retriever with optional filtering
        if selected_doc:
            # Filter by source file
            retriever = vectorstore.as_retriever(
                search_kwargs={
                    "k": RETRIEVE_K,
                    "filter": {"source": {"$contains": selected_doc}}
                }
            )
        else:
            retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVE_K})
        
        chain = build_chain(retriever, selected_doc)
        
        if chain:
            with st.chat_message("assistant", avatar="🤖"):
                placeholder = st.empty()
                full_response = ""
                
                try:
                    for chunk in chain.stream(prompt):
                        # Handle stop generation
                        if "stop" in st.query_params:
                            st.query_params.clear()
                            break
                        
                        # Handle both string chunks and AIMessageChunk objects
                        if isinstance(chunk, str):
                            full_response += chunk
                        elif hasattr(chunk, "content"):
                            chunk_content = chunk.content
                            if chunk_content:
                                full_response += chunk_content
                        elif hasattr(chunk, "text"):
                            chunk_text = chunk.text
                            if chunk_text:
                                full_response += chunk_text
                        
                        # Update placeholder with current response
                        if full_response:
                            placeholder.markdown(full_response + "▌")
                    
                    # Final response without cursor
                    if full_response.strip():
                        placeholder.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        
                        # Save to chat history
                        st.session_state.chat_sessions[st.session_state.current_chat] = st.session_state.messages
                        save_chat_history(st.session_state.chat_sessions)
                        
                        # Auto-speak
                        speak(full_response, len(st.session_state.messages) - 1)
                    else:
                        st.warning("⚠️ Received empty response from model")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Make sure LM Studio is running with a model loaded!")
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.error("❌ Failed to initialize chat chain. Check LM Studio connection.")
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.warning("⚠️ No documents indexed yet. Please upload and process files first!")
    
    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="text-align: center; color: #4a5568; padding: 20px; margin-top: 100px;">
    <p><img src="images/robo.png.jpg" width="24" height="24" style="vertical-align: middle;"> GRL Bot | Local RAG Assistant | Powered by LM Studio & LangChain</p>

    <p style="font-size: 12px;">All processing happens locally on your machine</p>
</div>
""", unsafe_allow_html=True)