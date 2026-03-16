from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredPowerPointLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from typing import List
from langchain_core.documents import Document

# Pull the default model name from your config
from config import DEFAULT_EMBED_MODEL

def get_loader(file_path: str):
    ext = file_path.lower()
    if ext.endswith(".pdf"):
        return PyPDFLoader(file_path)
    elif ext.endswith((".txt", ".md")):
        return TextLoader(file_path, encoding="utf-8")
    elif ext.endswith(".docx"):
        return Docx2txtLoader(file_path)
    elif ext.endswith((".ppt", ".pptx")):
        return UnstructuredPowerPointLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def load_and_split_files(file_paths: List[str], chunk_size=800, chunk_overlap=120):
    docs = []
    for path in file_paths:
        loader = get_loader(path)
        docs.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_documents(docs)

def get_embeddings(model_name: str = DEFAULT_EMBED_MODEL, use_ollama=False):
    if use_ollama:
        return OllamaEmbeddings(model=model_name)
    return HuggingFaceEmbeddings(model_name=model_name)