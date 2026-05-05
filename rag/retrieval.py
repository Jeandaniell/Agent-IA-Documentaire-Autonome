from __future__ import annotations
import logging
from typing import List, Tuple
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from config import TOP_K_RETRIEVAL

logger = logging.getLogger(__name__)

def get_retriever(vectorstore: Chroma):
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K_RETRIEVAL})

def retrieve(vectorstore: Chroma, query: str) -> List[Document]:
    return vectorstore.similarity_search(query, k=TOP_K_RETRIEVAL)

def format_context(docs: List[Document]) -> Tuple[str, List[str]]:
    parts, sources = [], []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "inconnu")
        page = doc.metadata.get("page", "")
        ref = f"{source}" + (f" (p.{page+1})" if page != "" else "")
        parts.append(f"[{i}] {ref}\n{doc.page_content.strip()}")
        if source not in sources:
            sources.append(source)
    return "\n\n---\n\n".join(parts), sources
