from __future__ import annotations
import logging
from typing import Optional
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from config import DOCS_DIR, SUPPORTED_EXTENSIONS
from rag.retrieval import format_context, retrieve

logger = logging.getLogger(__name__)
_vectorstore: Optional[Chroma] = None

def set_vectorstore(vs: Chroma) -> None:
    global _vectorstore
    _vectorstore = vs

@tool
def list_files(dummy: str = "") -> str:
    """Liste tous les fichiers disponibles dans le dossier de documents."""
    files = [f for f in DOCS_DIR.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        return "Aucun document trouvé."
    return "Documents disponibles :\n" + "\n".join(f"• {f.name}" for f in sorted(files))

@tool
def read_file(filename: str) -> str:
    """Lit le contenu brut d'un fichier texte. Argument : nom exact du fichier."""
    path = DOCS_DIR / filename
    if not path.exists():
        return f"Fichier introuvable : '{filename}'."
    if path.suffix.lower() == ".pdf":
        return "Utilise search_documents pour les PDFs."
    try:
        content = path.read_text(encoding="utf-8")
        if len(content) > 6000:
            content = content[:6000] + "\n\n[... tronqué ...]"
        return f"=== {filename} ===\n\n{content}"
    except Exception as exc:
        return f"Erreur : {exc}"

@tool
def search_documents(query: str) -> str:
    """Recherche sémantique dans la base documentaire. Argument : question ou mots-clés."""
    if _vectorstore is None:
        return "Erreur : pas de vectorstore."
    try:
        docs = retrieve(_vectorstore, query)
        if not docs:
            return "Aucun résultat."
        context, sources = format_context(docs)
        return f"Résultats pour '{query}'\nSources : {', '.join(sources)}\n\n{context}"
    except Exception as exc:
        return f"Erreur : {exc}"

@tool
def ingest_document(filename: str) -> str:
    """Indexe un nouveau document dans le vectorstore. Argument : nom du fichier."""
    global _vectorstore
    path = DOCS_DIR / filename
    if not path.exists():
        return f"Fichier introuvable : '{filename}'."
    try:
        from rag.ingestion import _load_document, split_documents
        docs = _load_document(path)
        chunks = split_documents(docs)
        _vectorstore.add_documents(chunks)
        return f"✔ '{filename}' indexé ({len(chunks)} chunks)."
    except Exception as exc:
        return f"Erreur : {exc}"

AGENT_TOOLS = [list_files, read_file, search_documents, ingest_document]
