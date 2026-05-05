"""
Pipeline d'ingestion RAG.
Responsabilité unique : charger les documents, les découper en chunks,
les vectoriser et les stocker dans Chroma.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DOCS_DIR,
    EMBEDDING_MODEL,
    GOOGLE_API_KEY,
    SUPPORTED_EXTENSIONS,
)

logger = logging.getLogger(__name__)



def _load_document(path: Path) -> List[Document]:
    """Charge un fichier unique selon son extension."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix in {".txt", ".md"}:
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(f"Extension non supportée : {suffix}")
    docs = loader.load()
    # Ajoute le nom de fichier comme métadonnée source lisible
    for doc in docs:
        doc.metadata["source"] = path.name
    return docs


def load_documents(directory: Path = DOCS_DIR) -> List[Document]:
    """
    Charge tous les documents supportés depuis le dossier `docs/`.
    Retourne une liste de LangChain Documents bruts (non découpés).
    """
    all_docs: List[Document] = []
    files = [f for f in directory.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        logger.warning("Aucun document trouvé dans %s", directory)
        return []
    for file in files:
        try:
            docs = _load_document(file)
            all_docs.extend(docs)
            logger.info(" Chargé : %s (%d page(s))", file.name, len(docs))
        except Exception as exc:
            logger.error(" Erreur sur %s : %s", file.name, exc)
    logger.info("Total documents chargés : %d", len(all_docs))
    return all_docs




def split_documents(docs: List[Document]) -> List[Document]:
    """
    Découpe les documents en chunks avec RecursiveCharacterTextSplitter.
    Conserve les métadonnées (source, page) sur chaque chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info("Documents découpés : %d chunks", len(chunks))
    return chunks



def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )

def build_vectorstore(chunks: List[Document]) -> Chroma:
    """
    Crée ou recharge un vectorstore Chroma depuis les chunks fournis.
    Si la collection existe déjà, elle est réinitialisée (overwrite).
    """
    embeddings = _get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    logger.info("Vectorstore Chroma créé/mis à jour (%d vecteurs)", len(chunks))
    return vectorstore


def load_vectorstore() -> Chroma:
    """
    Charge un vectorstore Chroma existant depuis le disque.
    Lève une exception si la base n'existe pas encore.
    """
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        raise FileNotFoundError(
            "Vectorstore introuvable. Lance d'abord `ingest()` pour indexer les documents."
        )
    embeddings = _get_embeddings()
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    logger.info("Vectorstore Chroma chargé depuis %s", CHROMA_DIR)
    return vectorstore



def ingest(directory: Path = DOCS_DIR) -> Chroma:
    """
    Pipeline complet d'ingestion :
    1. Charge les documents
    2. Les découpe en chunks
    3. Les vectorise et stocke dans Chroma
    Retourne le vectorstore prêt à l'emploi.
    """
    logger.info("=== Démarrage de l'ingestion ===")
    docs   = load_documents(directory)
    chunks = split_documents(docs)
    vs     = build_vectorstore(chunks)
    logger.info("=== Ingestion terminée ===")
    return vs
