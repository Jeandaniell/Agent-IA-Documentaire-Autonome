from __future__ import annotations
import logging, re
from typing import Dict, List
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.tools import AGENT_TOOLS, set_vectorstore
from config import GOOGLE_API_KEY, LLM_MAX_TOKENS, LLM_MODEL, LLM_TEMPERATURE, MAX_ITERATIONS, MEMORY_WINDOW

logger = logging.getLogger(__name__)

REACT_PROMPT_TEMPLATE = """Tu es un assistant documentaire intelligent.
Utilise tes outils pour explorer et interroger les documents.
Cite toujours les sources dans ta réponse finale.

Outils disponibles :
{tools}

Noms des outils : {tool_names}

Historique :
{chat_history}

Format OBLIGATOIRE :
Question: la question
Thought: réflexion
Action: nom_de_l_outil
Action Input: argument
Observation: résultat
... (répète si nécessaire)
Thought: je connais la réponse
Final Answer: réponse avec sources [fichier.pdf]

Question: {input}
Thought: {agent_scratchpad}"""

def build_agent(vectorstore) -> AgentExecutor:
    set_vectorstore(vectorstore)
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL, google_api_key=GOOGLE_API_KEY,
        temperature=LLM_TEMPERATURE, max_output_tokens=LLM_MAX_TOKENS,
        convert_system_message_to_human=True,
    )
    prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)
    memory = ConversationBufferWindowMemory(k=MEMORY_WINDOW, memory_key="chat_history", return_messages=False)
    agent = create_react_agent(llm=llm, tools=AGENT_TOOLS, prompt=prompt)
    return AgentExecutor(agent=agent, tools=AGENT_TOOLS, memory=memory,
                         max_iterations=MAX_ITERATIONS, verbose=True,
                         handle_parsing_errors=True, return_intermediate_steps=True)

def extract_sources(steps: List) -> List[str]:
    sources = []
    for _, observation in steps:
        if not isinstance(observation, str):
            continue
        match = re.search(r"Sources\s*:\s*(.+)", observation)
        if match:
            for src in re.split(r",\s*", match.group(1)):
                src = src.strip()
                if src and src not in sources:
                    sources.append(src)
    return sources

class DocumentAgent:
    def __init__(self, vectorstore):
        self.executor = build_agent(vectorstore)
        self.turn_count = 0

    def chat(self, user_input: str) -> Dict:
        self.turn_count += 1
        result = self.executor.invoke({"input": user_input})
        steps = result.get("intermediate_steps", [])
        return {"answer": result.get("output", "Pas de réponse."),
                "sources": extract_sources(steps), "steps": len(steps)}

    def reset_memory(self):
        self.executor.memory.clear()
        self.turn_count = 0
