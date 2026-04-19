"""
rag_setup.py — FAISS Vector Store Indexer for HeliosCast Grid Agent
Indexes grid management protocol text files and provides a retrieval tool.
"""

import os
import pickle
from typing import List

# --- LangChain / FAISS Imports ---
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(BASE_DIR, "knowledge_base")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")

# -----------------------------------------------------------------------
# EMBEDDING MODEL (local, no API key needed)
# Uses sentence-transformers/all-MiniLM-L6-v2 (~80 MB download once)
# -----------------------------------------------------------------------
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

# -----------------------------------------------------------------------
# INDEX BUILDER — Run once to create/update the vector store
# -----------------------------------------------------------------------
def build_faiss_index(force_rebuild: bool = False) -> FAISS:
    """
    Loads .txt files from knowledge_base/, splits them into chunks,
    embeds them, and saves a FAISS index to disk.

    Args:
        force_rebuild: If True, rebuilds even if an existing index is found.

    Returns:
        A LangChain FAISS vector store instance.
    """
    if not force_rebuild and os.path.exists(FAISS_INDEX_PATH):
        print("[RAG] Loading existing FAISS index from disk...")
        return FAISS.load_local(
            FAISS_INDEX_PATH,
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )

    print("[RAG] Building new FAISS index from knowledge base...")

    # 1. Load all .txt files from the knowledge_base directory
    loader = DirectoryLoader(
        KB_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    raw_docs: List[Document] = loader.load()

    if not raw_docs:
        raise FileNotFoundError(
            f"No .txt files found in {KB_DIR}. "
            "Add grid protocol files before indexing."
        )

    # 2. Split into overlapping chunks for better retrieval recall
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n---", "\n\n", "\n", " "],
    )
    chunks: List[Document] = splitter.split_documents(raw_docs)
    print(f"[RAG] Indexed {len(raw_docs)} document(s) → {len(chunks)} chunks")

    # 3. Embed & store
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"[RAG] FAISS index saved to: {FAISS_INDEX_PATH}")

    return vectorstore

# -----------------------------------------------------------------------
# RETRIEVAL TOOL — Used by the LangGraph RAG_Retriever node
# -----------------------------------------------------------------------
def get_retriever(k: int = 4):
    """
    Returns a LangChain retriever backed by the FAISS index.
    Builds the index if it doesn't exist yet.

    Args:
        k: Number of top relevant chunks to retrieve.

    Returns:
        A LangChain BaseRetriever instance.
    """
    vectorstore = build_faiss_index()
    return vectorstore.as_retriever(search_kwargs={"k": k})


def retrieve_docs(query: str, k: int = 4) -> str:
    """
    Convenience function: retrieves relevant protocol snippets as a
    single formatted string, ready to be injected into the agent state.

    Args:
        query: The natural-language query string.
        k:     Number of chunks to retrieve.

    Returns:
        A single string containing all retrieved protocol passages.
    """
    retriever = get_retriever(k=k)
    docs = retriever.invoke(query)
    results = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "grid_protocols")
        snippet = doc.page_content.strip()
        results.append(f"[Source {i} — {os.path.basename(source)}]\n{snippet}")
    return "\n\n".join(results)


# -----------------------------------------------------------------------
# STANDALONE TEST — python rag_setup.py
# -----------------------------------------------------------------------
if __name__ == "__main__":
    build_faiss_index(force_rebuild=True)
    print("\n--- Test Query ---")
    result = retrieve_docs("battery charging strategy during cloud cover")
    print(result)
