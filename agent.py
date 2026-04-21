from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.tools import AGENT_TOOLS, set_vectorstore
from config import (
    GOOGLE_API_KEY,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    MAX_ITERATIONS,
    MEMORY_WINDOW,
)

logger = logging.getLogger(__name__)


REACT_PROMPT_TEMPLATE = """Tu es un assistant documentaire intelligent et précis.
Tu as accès à une base de documents que tu peux explorer, lire et interroger via tes outils.

Règles importantes :
- Utilise TOUJOURS search_documents pour répondre aux questions sur le contenu des documents
- Cite TOUJOURS les sources utilisées dans ta réponse finale
- Si tu n'as pas assez d'information, dis-le honnêtement
- Réponds en français sauf si l'utilisateur écrit dans une autre langue

Outils disponibles :
{tools}

Noms des outils : {tool_names}

Historique de conversation :
{chat_history}

Format de raisonnement OBLIGATOIRE (respecte-le strictement) :

Question: la question à laquelle répondre
Thought: réfléchis à ce que tu dois faire
Action: le nom de l'outil à utiliser (exactement parmi : {tool_names})
Action Input: l'argument à passer à l'outil
Observation: le résultat de l'outil
... (répète Thought/Action/Action Input/Observation autant que nécessaire)
Thought: je connais maintenant la réponse finale
Final Answer: la réponse complète avec les sources citées entre [crochets]

Question: {input}
Thought: {agent_scratchpad}"""



def build_agent(vectorstore) -> AgentExecutor:
    """
    Construit et retourne l'AgentExecutor ReAct configuré avec :
    - Gemini 1.5 Flash comme LLM
    - Les 4 tools personnalisés
    - Une mémoire conversationnelle glissante
    """
    set_vectorstore(vectorstore)


    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=LLM_TEMPERATURE,
        max_output_tokens=LLM_MAX_TOKENS,
        convert_system_message_to_human=True,
    )


    prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)


    memory = ConversationBufferWindowMemory(
        k=MEMORY_WINDOW,
        memory_key="chat_history",
        return_messages=False,
    )

    agent = create_react_agent(
        llm=llm,
        tools=AGENT_TOOLS,
        prompt=prompt,
    )

    executor = AgentExecutor(
        agent=agent,
        tools=AGENT_TOOLS,
        memory=memory,
        max_iterations=MAX_ITERATIONS,
        verbose=True,    
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    logger.info("Agent ReAct initialisé avec %d tools", len(AGENT_TOOLS))
    return executor



def extract_sources(intermediate_steps: List) -> List[str]:
    """
    Extrait les noms de fichiers sources depuis les observations des tools.
    Parcourt les étapes intermédiaires (tool calls) de l'agent.
    """
    sources = []
    for action, observation in intermediate_steps:
        if not isinstance(observation, str):
            continue
        
        match = re.search(r"Sources\s*:\s*(.+)", observation)
        if match:
            raw = match.group(1)
            for src in re.split(r",\s*", raw):
                src = src.strip()
                if src and src not in sources:
                    sources.append(src)
    return sources



class DocumentAgent:
    """
    Interface haut niveau pour interagir avec l'agent documentaire.
    Gère le cycle question → raisonnement → réponse + sources.
    """

    def __init__(self, vectorstore):
        self.executor = build_agent(vectorstore)
        self.turn_count = 0
        logger.info("DocumentAgent prêt.")

    def chat(self, user_input: str) -> Dict:
        """
        Envoie une question à l'agent et retourne un dict :
        {
            "answer"  : str,   # Réponse finale
            "sources" : list,  # Fichiers sources cités
            "steps"   : int,   # Nombre d'étapes de raisonnement
        }
        """
        self.turn_count += 1
        logger.info("[Tour %d] Question : %s", self.turn_count, user_input)

        result = self.executor.invoke({"input": user_input})

        answer = result.get("output", "Désolé, je n'ai pas pu générer de réponse.")
        steps  = result.get("intermediate_steps", [])
        sources = extract_sources(steps)

        return {
            "answer" : answer,
            "sources": sources,
            "steps"  : len(steps),
        }

    def reset_memory(self) -> None:
        """Efface l'historique de conversation."""
        self.executor.memory.clear()
        self.turn_count = 0
        logger.info("Mémoire conversationnelle réinitialisée.")
