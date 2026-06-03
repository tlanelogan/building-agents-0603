"""Persistent long-term memory for the UdaPlay agent.

Two independent stores:

- ``InteractionStore`` — a SQLite log of every interaction (prompt, natural-language
  answer, structured JSON answer, retrieval process, inferred source). This powers the
  "History" tab and lets past prompts/answers be displayed at a later date.

- ``KnowledgeMemory`` — a persistent ChromaDB collection (``udaplay_memory``) that stores
  facts the agent learned from web searches. ``retrieve_game`` queries it alongside the
  curated game collection, so re-asking a previously web-searched question is answered
  from internal memory instead of hitting the web again. This is how the agent "learns".
"""

import json
import sqlite3
import hashlib
from typing import List, Optional, Dict, Any

from chromadb.api.types import EmbeddingFunction


class InteractionStore:
    """SQLite-backed persistent log of agent interactions."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        # A connection per call keeps this usable across Flask worker threads.
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    id                TEXT PRIMARY KEY,
                    timestamp         TEXT NOT NULL,
                    query             TEXT NOT NULL,
                    answer_nl         TEXT,
                    answer_json       TEXT,
                    retrieval_process TEXT,
                    source            TEXT
                )
                """
            )

    def save(self, record: Dict[str, Any]):
        """Persist one interaction. ``answer_json`` and ``retrieval_process`` are
        serialized to JSON text if passed as Python objects."""
        answer_json = record.get("answer_json")
        process = record.get("retrieval_process")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO interactions
                    (id, timestamp, query, answer_nl, answer_json, retrieval_process, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["id"],
                    record["timestamp"],
                    record["query"],
                    record.get("answer_nl"),
                    answer_json if isinstance(answer_json, str) else json.dumps(answer_json),
                    process if isinstance(process, str) else json.dumps(process),
                    record.get("source"),
                ),
            )

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "query": row["query"],
            "answer_nl": row["answer_nl"],
            "answer_json": json.loads(row["answer_json"]) if row["answer_json"] else None,
            "retrieval_process": json.loads(row["retrieval_process"]) if row["retrieval_process"] else [],
            "source": row["source"],
        }

    def list_recent(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM interactions ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get(self, interaction_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM interactions WHERE id = ?", (interaction_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None


def _normalize_question(question: str) -> str:
    return " ".join(question.lower().split())


class KnowledgeMemory:
    """A persistent vector collection of facts learned from web searches."""

    COLLECTION_NAME = "udaplay_memory"

    def __init__(self, chroma_client, embedding_fn: EmbeddingFunction):
        self._collection = chroma_client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=embedding_fn,
        )

    @staticmethod
    def _memory_id(question: str) -> str:
        # Deterministic id so re-learning the same question upserts (no duplicates).
        return "mem-" + hashlib.sha1(_normalize_question(question).encode("utf-8")).hexdigest()[:16]

    def remember(self, question: str, answer: str, urls: Optional[List[str]] = None,
                 timestamp: Optional[str] = None):
        """Store (or update) a learned web answer keyed by the normalized question."""
        if not answer:
            return
        content = f"Q: {question}\nA: {answer}"
        self._collection.upsert(
            ids=[self._memory_id(question)],
            documents=[content],
            metadatas=[{
                "source": "web",
                "question": question,
                "answer": answer,
                "urls": json.dumps(urls or []),
                "timestamp": timestamp or "",
            }],
        )

    def recall(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Return learned web memories most similar to the query."""
        if self._collection.count() == 0:
            return []
        results = self._collection.query(
            query_texts=[query],
            n_results=min(n_results, self._collection.count()),
            include=["documents", "distances", "metadatas"],
        )
        memories = []
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            memories.append({
                "source": "learned_web",
                "question": meta.get("question"),
                "answer": meta.get("answer"),
                "urls": json.loads(meta.get("urls", "[]")),
                "timestamp": meta.get("timestamp"),
                "similarity": round(1 - dist, 4),
            })
        return memories

    def count(self) -> int:
        return self._collection.count()

    def reset(self):
        """Delete all learned memories (handy for reproducible demos/tests)."""
        ids = self._collection.get()["ids"]
        if ids:
            self._collection.delete(ids=ids)
