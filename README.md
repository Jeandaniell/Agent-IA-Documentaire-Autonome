# 🤖 RAG Agent Documentaire Autonome
**Stack : Python · LangChain · Chroma · Gemini · ReAct**

---

## Architecture

```
docs/                        ← Dépose tes PDF/TXT ici
 │
 ▼
rag/ingestion.py             ← Pipeline d'ingestion
 ├─ load_documents()         │  Lecture PDF + TXT
 ├─ split_documents()        │  Chunking (1000 tokens, overlap 200)
 └─ build_vectorstore()      │  Embeddings → Chroma DB

chroma_db/                   ← Vecteurs persistés sur disque
 │
 ▼
rag/retrieval.py             ← Retrieval sémantique
 └─ retrieve()               │  Top-K similarity search

agent/tools.py               ← 4 tools personnalisés
 ├─ list_files               │  Liste les documents disponibles
 ├─ read_file                │  Lit un fichier brut
 ├─ search_documents         │  Recherche RAG (tool principal)
 └─ ingest_document          │  Indexation à la volée

agent/agent.py               ← Agent ReAct (LangChain)
 ├─ Gemini 1.5 Flash         │  LLM backbone
 ├─ ConversationWindowMemory │  Mémoire glissante (10 tours)
 └─ AgentExecutor            │  Boucle Thought → Action → Observation

main.py                      ← CLI interactif
```

## Flux ReAct

```
Question utilisateur
       │
       ▼
  [Thought] Que dois-je faire pour répondre ?
       │
       ▼
  [Action] search_documents("...")
       │
       ▼
  [Observation] Chunks pertinents + sources
       │
       ▼
  [Thought] J'ai assez d'informations
       │
       ▼
  [Final Answer] Réponse + citations [source.pdf]
```

---

## Installation

```bash
# 1. Cloner / créer le dossier
cd rag_agent

# 2. Environnement virtuel
python -m venv .venv
source .venv/bin/activate   # Windows : .venv\Scripts\activate

# 3. Dépendances
pip install -r requirements.txt

# 4. Clé API
cp .env.example .env
# → Édite .env et mets ta GOOGLE_API_KEY
# Obtenir une clé gratuite : https://aistudio.google.com/app/apikey
```

## Utilisation

```bash
# Déposer des documents dans docs/
cp mes_rapports/*.pdf docs/
cp mes_notes.txt docs/

# Lancer (ingestion + chat)
python main.py

# Charger une base existante (sans réindexer)
python main.py --no-ingest

# Juste indexer
python main.py --ingest-only
```

## Commandes en cours de chat

| Commande | Action |
|----------|--------|
| `/docs`  | Liste les documents indexés |
| `/reset` | Efface la mémoire de conversation |
| `/quit`  | Quitter |

## Exemples de questions

```
"Quels documents sont disponibles ?"
"Résume le contenu du rapport trimestriel."
"Quelles sont les conclusions sur les risques de marché ?"
"Compare les sections 2 et 4 du document stratégie.pdf"
```

---

## Structure des fichiers

```
rag_agent/
├── main.py              ← Point d'entrée
├── config.py            ← Configuration centralisée
├── requirements.txt
├── .env.example
├── agent/
│   ├── agent.py         ← Agent ReAct + mémoire
│   └── tools.py         ← Tools personnalisés
├── rag/
│   ├── ingestion.py     ← Chargement + chunking + vectorisation
│   └── retrieval.py     ← Recherche sémantique + formatage sources
├── docs/                ← Dépose tes documents ici
├── chroma_db/           ← Base vectorielle (auto-créée)
└── tests/
    └── test_pipeline.py ← Tests unitaires
```

## Paramètres configurables (config.py)

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `CHUNK_SIZE` | 1000 | Taille des chunks en caractères |
| `CHUNK_OVERLAP` | 200 | Chevauchement entre chunks |
| `TOP_K_RETRIEVAL` | 5 | Nombre de chunks retournés |
| `MEMORY_WINDOW` | 10 | Tours de conversation conservés |
| `MAX_ITERATIONS` | 10 | Limite de boucles ReAct |
| `LLM_TEMPERATURE` | 0.0 | Température LLM (0 = déterministe) |
