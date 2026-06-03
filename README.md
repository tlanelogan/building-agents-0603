# UdaPlay — AI Game Research Agent

UdaPlay is an AI research agent for the video game industry. It answers questions
about video games (release dates, platforms, genres, publishers, descriptions) by
first consulting a local vector database and falling back to web search when the
internal knowledge isn't sufficient.

The project is built on a small, from-scratch agent framework (`starter/lib/`) — no
LangChain/LlamaIndex — and is split into two parts:

- **Part 1 — Offline RAG:** build a persistent [ChromaDB](https://www.trychroma.com/)
  vector store from the provided game JSON files and query it with semantic search.
- **Part 2 — Agent:** a stateful, tool-using agent (implemented as a state machine)
  that retrieves from the vector DB, evaluates the retrieval, and web-searches as a
  fallback, while maintaining conversation memory across turns.

## Getting Started

### Dependencies

- Python 3.11+
- `chromadb`, `openai`, `pydantic`, `python-dotenv`, `tavily-python`, `pdfplumber`
- `jupyter` (plus `nbconvert` to run notebooks headlessly)

### Installation

```bash
# from the repo root
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install chromadb openai pydantic python-dotenv tavily-python pdfplumber \
            jupyter nbconvert ipykernel
```

Create a `.env` file inside `starter/` with your API keys (the notebooks call
`load_dotenv()` relative to their own directory):

```
OPENAI_API_KEY="YOUR_KEY"
TAVILY_API_KEY="YOUR_KEY"
# Optional: Chroma's OpenAI embedding function reads this if OPENAI_API_KEY is unset
CHROMA_OPENAI_API_KEY="YOUR_KEY"
# Optional: only if routing OpenAI calls through a proxy (e.g. the Udacity/Vocareum proxy)
OPENAI_BASE_URL="https://openai.vocareum.com/v1"
```

> **Note:** start Jupyter from inside `starter/` (or select that folder) so the
> notebooks can import `lib/` and resolve `games/`, `chromadb/`, and `.env`.

## Running the project

```bash
cd starter
jupyter notebook        # or: jupyter lab
```

1. **Part 1 — `Udaplay_01_solution_project.ipynb`**
   Loads and formats the game JSON files, embeds them with OpenAI embeddings, and
   stores them in a persistent ChromaDB collection (`udaplay`) under
   `starter/chromadb/`. Ends with a semantic-search demonstration.

2. **Part 2 — `Udaplay_02_solution_project.ipynb`**
   Builds the UdaPlay agent and runs it on example queries, printing each query's
   reasoning, tool calls/results, and final cited answer — plus a multi-turn
   memory demonstration.

> Run Part 1 first: it creates the persistent `chromadb/` store that Part 2 reads.

## Project structure

```
project/
├── README.md
└── starter/
    ├── games/                              # game JSON files (one document each)
    ├── lib/                                # the agent framework
    │   ├── llm.py            messages.py   tooling.py    parsers.py
    │   ├── agents.py         state_machine.py            memory.py
    │   ├── vector_db.py      loaders.py    documents.py  rag.py
    │   └── evaluation.py
    ├── Udaplay_01_starter_project.ipynb    # Part 1 (with TODOs)
    ├── Udaplay_01_solution_project.ipynb   # Part 1 deliverable (executed)
    ├── Udaplay_02_starter_project.ipynb    # Part 2 (with TODOs)
    └── Udaplay_02_solution_project.ipynb   # Part 2 deliverable (executed)
```

## The agent's tools

| Tool | Purpose |
|---|---|
| `retrieve_game` | Semantic search over the ChromaDB game collection |
| `evaluate_retrieval` | LLM-as-judge that decides whether the retrieved docs answer the question (returns a structured `EvaluationReport`) |
| `game_web_search` | Tavily web search, used only when retrieval is judged insufficient |

The agent's workflow: **retrieve → evaluate → (web search if needed) → cited answer.**

## Built With

* [ChromaDB](https://www.trychroma.com/) — persistent vector database
* [OpenAI API](https://platform.openai.com/) — embeddings and chat completions (`gpt-4o-mini`)
* [Tavily](https://tavily.com/) — web search API
* [Pydantic](https://docs.pydantic.dev/) — structured outputs and message models

## License

Course materials for Udacity's "Building AI Agents." See [LICENSE](../LICENSE.md) if present.
