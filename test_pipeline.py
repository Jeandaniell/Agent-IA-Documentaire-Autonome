import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

# ── Tests ingestion ────────────────────────────────────────────────────────────

class TestSplitDocuments:
    def test_basic_split(self):
        from rag.ingestion import split_documents
        docs = [Document(page_content="A " * 600, metadata={"source": "test.txt"})]
        chunks = split_documents(docs)
        assert len(chunks) >= 1
        # Chaque chunk doit conserver la métadonnée source
        for chunk in chunks:
            assert "source" in chunk.metadata

    def test_overlap(self):
        """Les chunks successifs doivent partager du contenu (overlap)."""
        from rag.ingestion import split_documents
        content = "mot " * 1000
        docs = [Document(page_content=content, metadata={"source": "test.txt"})]
        chunks = split_documents(docs)
        if len(chunks) > 1:
            # Les chunks ne doivent pas être identiques
            assert chunks[0].page_content != chunks[1].page_content

    def test_empty_docs(self):
        from rag.ingestion import split_documents
        chunks = split_documents([])
        assert chunks == []


# ── Tests retrieval ────────────────────────────────────────────────────────────

class TestFormatContext:
    def test_basic_format(self):
        from rag.retrieval import format_context
        docs = [
            Document(page_content="Contenu du doc 1", metadata={"source": "doc1.txt"}),
            Document(page_content="Contenu du doc 2", metadata={"source": "doc2.pdf", "page": 0}),
        ]
        context, sources = format_context(docs)
        assert "doc1.txt" in context
        assert "doc2.pdf" in context
        assert "doc1.txt" in sources
        assert "doc2.pdf" in sources
        assert len(sources) == 2

    def test_deduplication_sources(self):
        from rag.retrieval import format_context
        docs = [
            Document(page_content="A", metadata={"source": "same.pdf"}),
            Document(page_content="B", metadata={"source": "same.pdf"}),
        ]
        _, sources = format_context(docs)
        assert sources.count("same.pdf") == 1

    def test_empty_docs(self):
        from rag.retrieval import format_context
        context, sources = format_context([])
        assert context == ""
        assert sources == []


# ── Tests tools ────────────────────────────────────────────────────────────────

class TestListFiles:
    def test_returns_string(self, tmp_path):
        import config
        original = config.DOCS_DIR
        config.DOCS_DIR = tmp_path
        # Crée un faux fichier
        (tmp_path / "test.txt").write_text("contenu")
        from agent.tools import list_files
        result = list_files.invoke("")
        assert "test.txt" in result
        config.DOCS_DIR = original


class TestReadFile:
    def test_reads_text_file(self, tmp_path):
        import config
        original = config.DOCS_DIR
        config.DOCS_DIR = tmp_path
        (tmp_path / "hello.txt").write_text("Bonjour monde")
        from agent.tools import read_file
        result = read_file.invoke("hello.txt")
        assert "Bonjour monde" in result
        config.DOCS_DIR = original

    def test_missing_file(self, tmp_path):
        import config
        original = config.DOCS_DIR
        config.DOCS_DIR = tmp_path
        from agent.tools import read_file
        result = read_file.invoke("inexistant.txt")
        assert "introuvable" in result.lower()
        config.DOCS_DIR = original


# ── Tests agent response parser ────────────────────────────────────────────────

class TestExtractSources:
    def test_extracts_sources_from_steps(self):
        from agent.agent import extract_sources
        from langchain_core.agents import AgentAction
        steps = [
            (
                AgentAction(tool="search_documents", tool_input="test", log=""),
                "Résultats...\nSources : rapport.pdf, notes.txt\n\nContenu..."
            )
        ]
        sources = extract_sources(steps)
        assert "rapport.pdf" in sources
        assert "notes.txt" in sources

    def test_no_sources(self):
        from agent.agent import extract_sources
        from langchain_core.agents import AgentAction
        steps = [
            (
                AgentAction(tool="list_files", tool_input="", log=""),
                "Aucun document trouvé."
            )
        ]
        sources = extract_sources(steps)
        assert sources == []
