from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from langchain.tools import tool
from langchain_community.vectorstores import Chroma

from config import DOCS_DIR, SUPPORTED_EXTENSIONS
from rag.retrieval import format_context, retrieve

logger = logging.getLogger(__name__)

# Référence globale au vectorstore — injectée au démarrage de l'agent
_vectorstore: Optional[Chroma] = None


def set_vectorstore(vs: Chroma) -> None:
    """Injecte le vectorstore dans le module tools (appelé au boot)."""
    global _vectorstore
    _vectorstore = vs
    logger.info("Vectorstore injecté dans les tools.")



@tool
def list_files(dummy: str = "") -> str:
    """
    Liste tous les fichiers disponibles dans le dossier de documents.
    Utilise ce tool pour découvrir quels documents sont dans la base.
    L'argument est ignoré, passe une chaîne vide si besoin.
    """
    files = [f for f in DOCS_DIR.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        return "Aucun document trouvé dans le dossier docs/."
    lines = [f"• {f.name} ({f.stat().st_size // 1024} Ko)" for f in sorted(files)]
    return "Documents disponibles :\n" + "\n".join(lines)



@tool
def read_file(filename: str) -> str:
    """
    Lit et retourne le contenu brut d'un fichier texte (.txt ou .md).
    Argument : le nom exact du fichier (ex: 'rapport.txt').
    Pour les PDFs, utilise plutôt search_documents.
    """
    path = DOCS_DIR / filename
    if not path.exists():
        return f"Fichier introuvable : '{filename}'. Vérifie avec list_files."
    if path.suffix.lower() == ".pdf":
        return "Lecture directe PDF non supportée. Utilise search_documents avec une question précise."
    try:
        content = path.read_text(encoding="utf-8")
        # Tronque si trop long pour le contexte LLM
        if len(content) > 6000:
            content = content[:6000] + "\n\n[... contenu tronqué ...]"
        return f"=== Contenu de '{filename}' ===\n\n{content}"
    except Exception as exc:
        return f"Erreur lors de la lecture de '{filename}' : {exc}"



@tool
def search_documents(query: str) -> str:
    """
    Recherche sémantique dans la base documentaire vectorisée.
    Retourne les passages les plus pertinents avec leurs sources.
    Argument : la question ou les mots-clés à rechercher.
    C'est le tool principal pour répondre à des questions sur les documents.
    """
    if _vectorstore is None:
        return "Erreur : aucun vectorstore disponible. Lance l'ingestion d'abord."
    try:
        docs = retrieve(_vectorstore, query)
        if not docs:
            return "Aucun passage pertinent trouvé pour cette requête."
        context, sources = format_context(docs)
        sources_str = ", ".join(sources)
        return (
            f"Résultats de recherche pour : '{query}'\n"
            f"Sources : {sources_str}\n\n"
            f"{context}"
        )
    except Exception as exc:
        logger.error("Erreur search_documents : %s", exc)
        return f"Erreur lors de la recherche : {exc}"



@tool
def ingest_document(filename: str) -> str:
    """
    Indexe un nouveau document dans le vectorstore à la volée.
    Le fichier doit être présent dans le dossier docs/.
    Argument : le nom exact du fichier à indexer.
    """
    global _vectorstore
    path = DOCS_DIR / filename
    if not path.exists():
        return f"Fichier introuvable : '{filename}'."
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return f"Extension non supportée. Extensions acceptées : {SUPPORTED_EXTENSIONS}"
    try:
        from rag.ingestion import _load_document, split_documents, _get_embeddings
        docs   = _load_document(path)
        chunks = split_documents(docs)
        if _vectorstore is None:
            from rag.ingestion import build_vectorstore
            _vectorstore = build_vectorstore(chunks)
        else:
            _vectorstore.add_documents(chunks)
        return f"✔ '{filename}' indexé avec succès ({len(chunks)} chunks ajoutés)."
    except Exception as exc:
        return f"Erreur lors de l'indexation de '{filename}' : {exc}"



AGENT_TOOLS = [list_files, read_file, search_documents, ingest_document]
