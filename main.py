from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import logging
import sys


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)



BANNER = """

        RAG AGENT DOCUMENTAIRE      
        LangChain · Chroma · ReAct · Mémoire conversationnelle


Commandes spéciales :
  /reset    → Efface la mémoire de conversation
  /docs     → Liste les documents indexés
  /quit     → Quitter
"""

COMMANDS = {"/reset", "/docs", "/quit", "/exit"}



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG Agent Documentaire")
    parser.add_argument(
        "--no-ingest", action="store_true",
        help="Charge le vectorstore existant sans réindexer"
    )
    parser.add_argument(
        "--ingest-only", action="store_true",
        help="Indexe les documents puis quitte"
    )
    return parser.parse_args()



def setup_vectorstore(no_ingest: bool):
    """Retourne le vectorstore, en ingérant ou en chargeant selon le flag."""
    if no_ingest:
        from rag.ingestion import load_vectorstore
        logger.info("Chargement du vectorstore existant...")
        return load_vectorstore()
    else:
        from rag.ingestion import ingest
        logger.info("Ingestion des documents...")
        return ingest()



def interactive_loop(agent) -> None:
    """Boucle REPL : lit les questions, affiche réponses + sources."""
    print(BANNER)
    print("Vectorstore prêt. Pose tes questions sur les documents !\n")

    while True:
        try:
            user_input = input(" Vous : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAu revoir !")
            break

        if not user_input:
            continue

        # Commandes spéciales
        if user_input.lower() in {"/quit", "/exit"}:
            print("Au revoir !")
            break

        if user_input.lower() == "/reset":
            agent.reset_memory()
            print(" Mémoire effacée.\n")
            continue

        if user_input.lower() == "/docs":
            from agent.tools import list_files
            print(list_files.invoke(""))
            print()
            continue

        # Requête normale
        print("\n Réflexion en cours...\n")
        try:
            result = agent.chat(user_input)
        except Exception as exc:
            logger.error("Erreur agent : %s", exc)
            print(f" Erreur : {exc}\n")
            continue

        # Affichage de la réponse
        print(f" Agent : {result['answer']}")

        # Sources
        if result["sources"]:
            sources_str = " · ".join(f"[{s}]" for s in result["sources"])
            print(f"\n Sources : {sources_str}")

        print(f"\n{'─' * 60}\n")


# ── Main

def main() -> None:
    args = parse_args()

    # Vérification de la clé API
    from config import GOOGLE_API_KEY
    if not GOOGLE_API_KEY:
        print(" GOOGLE_API_KEY manquante. Crée un fichier .env avec :")
        print("   GOOGLE_API_KEY=ta_clé_ici")
        sys.exit(1)

    # Vectorstore
    try:
        vectorstore = setup_vectorstore(no_ingest=args.no_ingest)
    except FileNotFoundError as e:
        print(f" {e}")
        sys.exit(1)

    if args.ingest_only:
        print(" Ingestion terminée.")
        return

    # Agent
    from agent.agent import DocumentAgent
    agent = DocumentAgent(vectorstore)

    # Loop
    interactive_loop(agent)


if __name__ == "__main__":
    main()
