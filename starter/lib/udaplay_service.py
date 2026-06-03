"""UdaPlay service layer: memory-aware tools, agent, and a thread-safe facade.

This wraps the notebook's agent + tools so they can be reused outside Jupyter (e.g.
from a Flask app) and adds persistent long-term memory:

- ``game_web_search`` stores what it finds into ``KnowledgeMemory`` (vector DB).
- ``retrieve_game`` searches the curated game collection *and* that learned memory,
  so a previously web-searched question is later answered without hitting the web.
- every interaction is logged to ``InteractionStore`` (SQLite) for later display.

Note: no ``from __future__ import annotations`` here on purpose — the ``@tool``
decorator calls ``get_type_hints``/``inspect.signature(eval_str=True)``, so tool
parameter annotations must stay eagerly resolvable (builtins only).
"""

import os
import ast
import json
import uuid
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from pydantic import BaseModel
from tavily import TavilyClient

from lib.agents import Agent
from lib.llm import LLM
from lib.messages import SystemMessage, UserMessage
from lib.tooling import tool
from lib.udaplay_memory import InteractionStore, KnowledgeMemory


# --- Paths (resolved relative to this file, so cwd doesn't matter) ----------------
STARTER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(STARTER_DIR, "chromadb")
INTERACTIONS_DB = os.path.join(STARTER_DIR, "udaplay_interactions.db")

# Pin the embedding model so every collection in the store stays consistent.
EMBEDDING_MODEL = "text-embedding-ada-002"
GAMES_COLLECTION = "udaplay"
LEARNED_SIMILARITY_THRESHOLD = 0.8  # only surface strongly-matching learned memories

AGENT_INSTRUCTIONS = (
    "You are UdaPlay, an AI research assistant for the video game industry. "
    "You answer questions about video games such as release dates, platforms, "
    "genres, publishers, and descriptions.\n\n"
    "Follow this process for every question:\n"
    "1. Call `retrieve_game` to search the internal knowledge base first. Results may "
    "come from the curated game database (source 'game_db') or from facts the agent "
    "previously learned from the web (source 'learned_web') — both are internal knowledge.\n"
    "2. Call `evaluate_retrieval`, passing the original question and the documents "
    "returned by `retrieve_game`, to decide whether they are sufficient.\n"
    "3. If (and only if) the retrieved documents are NOT useful, call `game_web_search` "
    "to look the answer up on the web.\n"
    "4. Give a clear, concise final answer. Include the year and platform when relevant, "
    "and state whether the answer came from the internal knowledge base or a web search. "
    "If you cannot find a reliable answer, say so instead of guessing."
)


class EvaluationReport(BaseModel):
    """Structured result of judging whether retrieved docs answer the question."""
    useful: bool
    description: str


def parse_tool_output(content: str) -> Any:
    """Reliably decode a ToolMessage.content.

    ``Agent._tool_step`` stores ``json.dumps(str(tool_result))`` — i.e. the Python
    ``repr`` of the result wrapped in JSON. So: undo the JSON layer, then use
    ``ast.literal_eval`` (NOT json) to recover the dict/list. Never raises.
    """
    inner = content
    try:
        inner = json.loads(content)
    except Exception:
        pass
    if isinstance(inner, str):
        try:
            return ast.literal_eval(inner)
        except Exception:
            return inner
    return inner


def build_tools(games_collection, knowledge_memory: KnowledgeMemory,
                tavily_client: TavilyClient) -> List:
    """Build the three agent tools as closures over the given clients.

    Annotations are builtins only so the @tool introspection works inside a factory.
    """

    @tool
    def retrieve_game(query: str) -> list[dict]:
        """Semantic search over UdaPlay's internal knowledge.

        Searches both the curated game database and facts previously learned from
        the web. args:
        - query: a question about the game industry.

        Returns a list of results. Game records carry source 'game_db' with fields
        Platform, Name, YearOfRelease, Genre, Publisher, Description. Learned facts
        carry source 'learned_web' with the prior question and its answer.
        """
        results = games_collection.query(query_texts=[query], n_results=5)
        games: list[dict] = []
        for doc_id, meta, dist in zip(
            results["ids"][0], results["metadatas"][0], results["distances"][0]
        ):
            games.append({
                "source": "game_db",
                "id": doc_id,
                "Platform": meta.get("Platform"),
                "Name": meta.get("Name"),
                "YearOfRelease": meta.get("YearOfRelease"),
                "Genre": meta.get("Genre"),
                "Publisher": meta.get("Publisher"),
                "Description": meta.get("Description"),
                "similarity": round(1 - dist, 4),
            })

        # Also recall facts learned from prior web searches (strong matches only).
        learned = [
            m for m in knowledge_memory.recall(query, n_results=3)
            if (m.get("similarity") or 0) >= LEARNED_SIMILARITY_THRESHOLD
        ]
        return games + learned

    @tool
    def evaluate_retrieval(question: str, retrieved_docs: list[dict]) -> dict:
        """Judge whether the retrieved documents are enough to answer the question.

        args:
        - question: original question from the user
        - retrieved_docs: documents returned by retrieve_game

        Returns: useful (bool) and description (str).
        """
        judge = LLM(model="gpt-4o-mini", temperature=0)
        messages = [
            SystemMessage(content=(
                "Your task is to evaluate if the documents are enough to respond to the query. "
                "Give a detailed explanation, so it's possible to take an action to accept it or not."
            )),
            UserMessage(content=(
                f"Question: {question}\n\n"
                f"Retrieved documents:\n{json.dumps(retrieved_docs, indent=2)}"
            )),
        ]
        response = judge.invoke(messages, response_format=EvaluationReport)
        report = EvaluationReport.model_validate_json(response.content)
        return report.model_dump()

    @tool
    def game_web_search(question: str) -> dict:
        """Search the web for game-industry information when internal knowledge is
        insufficient. The finding is stored in long-term memory so the agent can
        answer the same question internally next time.

        args:
        - question: a question about the game industry.
        """
        response = tavily_client.search(question, max_results=5, include_answer=True)
        answer = response.get("answer")
        results = [
            {"title": r.get("title"), "url": r.get("url"), "content": r.get("content")}
            for r in response.get("results", [])
        ]
        # Learn from this search.
        knowledge_memory.remember(
            question=question,
            answer=answer or "",
            urls=[r["url"] for r in results if r.get("url")],
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )
        return {"answer": answer, "results": results}

    return [retrieve_game, evaluate_retrieval, game_web_search]


def extract_retrieval_process(run) -> List[Dict[str, Any]]:
    """Walk a Run's messages into an ordered, displayable trace of reasoning,
    tool calls (with inputs), tool results (with parsed outputs), and the answer."""
    messages = run.get_final_state()["messages"]
    process: List[Dict[str, Any]] = []
    for msg in messages:
        role = getattr(msg, "role", None)
        if role == "assistant":
            if msg.content:
                process.append({"type": "reasoning", "content": msg.content})
            for call in (msg.tool_calls or []):
                try:
                    args = json.loads(call.function.arguments)
                except Exception:
                    args = call.function.arguments
                process.append({"type": "tool_call", "tool": call.function.name, "input": args})
        elif role == "tool":
            process.append({
                "type": "tool_result",
                "tool": msg.name,
                "output": parse_tool_output(msg.content),
            })
    process.append({"type": "final_answer", "content": messages[-1].content})
    return process


def _last_tool_output(process: List[Dict[str, Any]], tool_name: str) -> Optional[Any]:
    for entry in reversed(process):
        if entry.get("type") == "tool_result" and entry.get("tool") == tool_name:
            return entry.get("output")
    return None


def build_structured_answer(run, process: List[Dict[str, Any]], question: str) -> Dict[str, Any]:
    """Assemble a deterministic, integration-friendly JSON answer from the trace."""
    final_state = run.get_final_state()
    answer = final_state["messages"][-1].content

    retrieved = _last_tool_output(process, "retrieve_game")
    retrieved = retrieved if isinstance(retrieved, list) else []
    games = [r for r in retrieved if isinstance(r, dict) and r.get("source") == "game_db"]
    learned = [r for r in retrieved if isinstance(r, dict) and r.get("source") == "learned_web"]

    evaluation = _last_tool_output(process, "evaluate_retrieval")
    evaluation = evaluation if isinstance(evaluation, dict) else None
    eval_useful = evaluation.get("useful") if evaluation else None

    web = _last_tool_output(process, "game_web_search")
    web_called = web is not None
    web_sources = []
    if isinstance(web, dict):
        web_sources = [
            {"title": r.get("title"), "url": r.get("url")}
            for r in (web.get("results") or []) if isinstance(r, dict)
        ]

    if web_called:
        source = "mixed" if eval_useful else "web"
    elif learned:
        source = "learned_memory"
    elif eval_useful:
        source = "internal"
    else:
        source = "unknown"

    return {
        "question": question,
        "answer": answer,
        "source": source,
        "games": games,
        "learned_memory": learned,
        "web_sources": web_sources,
        "evaluation": evaluation,
        "wrote_to_memory": web_called,
        "run_id": run.run_id,
        "total_tokens": final_state.get("total_tokens"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


class UdaPlayService:
    """Thread-safe facade over the UdaPlay agent + persistent memory."""

    def __init__(self):
        load_dotenv(os.path.join(STARTER_DIR, ".env"))
        api_key = os.getenv("OPENAI_API_KEY")

        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key, model_name=EMBEDDING_MODEL
        )
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.games_collection = self.chroma_client.get_or_create_collection(
            name=GAMES_COLLECTION, embedding_function=self.embedding_fn
        )
        self.knowledge_memory = KnowledgeMemory(self.chroma_client, self.embedding_fn)
        self.store = InteractionStore(INTERACTIONS_DB)
        self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

        self.tools = build_tools(self.games_collection, self.knowledge_memory, self.tavily_client)
        self.agent = Agent(
            model_name="gpt-4o-mini",
            instructions=AGENT_INSTRUCTIONS,
            tools=self.tools,
            temperature=0.3,
        )
        self._lock = threading.Lock()

    def ask(self, query: str) -> Dict[str, Any]:
        """Run the agent on a query; persist and return the full result."""
        with self._lock:
            session = "req-" + uuid.uuid4().hex
            run = self.agent.invoke(query, session_id=session)
            process = extract_retrieval_process(run)
            structured = build_structured_answer(run, process, query)
            answer_nl = run.get_final_state()["messages"][-1].content

            interaction_id = uuid.uuid4().hex
            record = {
                "id": interaction_id,
                "timestamp": structured["timestamp"],
                "query": query,
                "answer_nl": answer_nl,
                "answer_json": structured,
                "retrieval_process": process,
                "source": structured["source"],
            }
            self.store.save(record)

            # Don't retain per-request sessions in ShortTermMemory (the SQLite log is
            # the real history). Avoids an unbounded memory leak.
            try:
                self.agent.memory.delete_session(session)
            except Exception:
                pass

            return {
                "id": interaction_id,
                "timestamp": record["timestamp"],
                "query": query,
                "answer_nl": answer_nl,
                "answer_json": structured,
                "retrieval_process": process,
                "source": structured["source"],
            }

    def history(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self.store.list_recent(limit)

    def get(self, interaction_id: str) -> Optional[Dict[str, Any]]:
        return self.store.get(interaction_id)

    def knowledge_base(self) -> List[Dict[str, Any]]:
        """List the curated game records (for the Knowledge Base tab)."""
        data = self.games_collection.get(include=["metadatas"])
        games = []
        for doc_id, meta in zip(data["ids"], data["metadatas"]):
            games.append({
                "id": doc_id,
                "Name": meta.get("Name"),
                "Platform": meta.get("Platform"),
                "Genre": meta.get("Genre"),
                "Publisher": meta.get("Publisher"),
                "YearOfRelease": meta.get("YearOfRelease"),
                "Description": meta.get("Description"),
            })
        return sorted(games, key=lambda g: g["id"])
