from __future__ import annotations

import logging
from typing import List, Tuple

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config import TOP_K_RETRIEVAL

logger = logging.getLogger(__name__)


def get_retriever(vectorstore: Chroma):
    """
    Retourne un retriever LangChain configuré pour récupérer
    les TOP_K_RETRIEVAL chunks les plus pertinents.
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RETRIEVAL},
    )


def retrieve(vectorstore: Chroma, query: str) -> List[Document]:
    """
    Recherche sémantique directe — retourne les documents bruts.
    Utile pour les tools de l'agent.
    """
    docs = vectorstore.similarity_search(query, k=TOP_K_RETRIEVAL)
    logger.debug("Retrieval pour '%s' → %d chunks", query, len(docs))
    return docs


def format_context(docs: List[Document]) -> Tuple[str, List[str]]:
    """
    Formate les documents récupérés en :
    - context : texte concaténé pour le prompt
    - sources  : liste déduplicée des noms de fichiers sources

    Retourne (context_str, sources_list).
    """
    parts: List[str] = []
    sources: List[str] = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "inconnu")
        page   = doc.metadata.get("page", "")
        ref    = f"{source}" + (f" (p.{page + 1})" if page != "" else "")
        parts.append(f"[{i}] {ref}\n{doc.page_content.strip()}")
        if source not in sources:
            sources.append(source)

    context = "\n\n---\n\n".join(parts)
    return context, sources
