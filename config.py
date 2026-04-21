import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


BASE_DIR   = Path(__file__).parent
DOCS_DIR   = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"

DOCS_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)


GOOGLE_API_KEY      = os.getenv("AIzaSyBWYNBe1k6bch2eyqF8ClbVHQpbE21KUKk", "")
LLM_MODEL           = "gemini-1.5-flash"
EMBEDDING_MODEL     = "models/embedding-001"
LLM_TEMPERATURE     = 0.0
LLM_MAX_TOKENS      = 2048


COLLECTION_NAME     = "rag_documents"
CHUNK_SIZE          = 1000
CHUNK_OVERLAP       = 200
TOP_K_RETRIEVAL     = 5          
MAX_ITERATIONS      = 10 
MEMORY_WINDOW       = 10 

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".md"}
