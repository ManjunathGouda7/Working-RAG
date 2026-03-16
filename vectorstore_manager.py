from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import Optional, List
import os

# Constants from config
from config import COLLECTIONS_DIR, DEFAULT_EMBED_MODEL

# Function from document_processor
from document_processor import get_embeddings

def get_collection_path(collection_name: str) -> str:
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in collection_name)
    return os.path.join(COLLECTIONS_DIR, safe_name)

def create_or_load_vectorstore(
    collection_name: str,
    documents: Optional[List[Document]] = None,
    embedding_model: str = DEFAULT_EMBED_MODEL,
    use_ollama_emb: bool = False,
    ) -> FAISS:
    path = get_collection_path(collection_name)
    embeddings = get_embeddings(embedding_model, use_ollama_emb)

    if os.path.exists(os.path.join(path, "index.faiss")):
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    
    if documents is None or not documents:
        raise ValueError(f"Collection '{collection_name}' does not exist and no documents provided.")
    
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(path)
    return vectorstore

def add_documents_to_collection(
    collection_name: str,
    documents: List[Document],
    embedding_model: str = DEFAULT_EMBED_MODEL,
    use_ollama_emb: bool = False,
):
    vs = create_or_load_vectorstore(collection_name, embedding_model=embedding_model, use_ollama_emb=use_ollama_emb)
    vs.add_documents(documents)
    vs.save_local(get_collection_path(collection_name))

def list_collections() -> List[str]:
    if not os.path.exists(COLLECTIONS_DIR):
        return []
    return [d for d in os.listdir(COLLECTIONS_DIR) if os.path.isdir(os.path.join(COLLECTIONS_DIR, d))]

def delete_collection(collection_name: str):
    import shutil
    path = get_collection_path(collection_name)
    if os.path.exists(path):
        shutil.rmtree(path)