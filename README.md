# MultiAgentic RAG

**A self-correcting, multi-agent Retrieval-Augmented Generation system built with LangGraph**

## Overview

MultiAgentic RAG is a **Multi-Agent Research RAG (Retrieval-Augmented Generation) Tool** that answers questions over a document (by default, Google's 2024 Environmental Report) using a graph of cooperating LLM agents rather than a single retrieve-then-generate call.

Instead of naively embedding a question and stuffing the top-k chunks into a prompt, the system:

1. **Routes** the query — deciding if it needs more information, is on-topic, or is a general/off-topic question.
2. **Plans** a short multi-step research strategy for on-topic questions.
3. **Researches** each step in parallel — expanding it into several search queries, retrieving with an ensemble of retrievers, and re-ranking with Cohere.
4. **Responds** using only the retrieved evidence, with inline citations.
5. **Grades itself** for hallucinations and — if the answer isn't well-grounded — pauses for **human-in-the-loop approval** before deciding whether to retry.

This turns RAG from a single hop into an auditable, self-correcting pipeline.

### Key Features

- **LLM-based Query Router** — classifies each query as `environmental`, `more-info`, or `general` before doing any retrieval work.
- **Multi-Step Research Planning** — an LLM breaks the question into a short (1–2 step) research plan.
- **Query Fan-Out** — each research step is expanded into multiple sub-queries and retrieved **in parallel** via LangGraph's `Send` API.
- **Hybrid Ensemble Retrieval** — combines dense similarity search, MMR search, and BM25 (sparse/keyword) search.
- **Cohere Contextual Re-Ranking** — re-ranks the ensemble's candidates for higher precision before they reach the LLM.
- **Hallucination Grading + Human-in-the-Loop** — a grader LLM checks whether the final answer is supported by the retrieved documents; ungrounded answers trigger a LangGraph `interrupt()` so a human can approve a retry.
- **Streaming CLI** — token-by-token streaming responses in the terminal via `app.py`.
- **Config-Driven** — all retrieval/LLM parameters live in `config.yaml`, not hardcoded in source.
- **Document Ingestion Pipeline** — PDF → Markdown (via Docling) → header-aware chunking → Chroma vector store, run once as an offline indexing step.

---

## Quick Start

### Prerequisites

- Python 3.10+
- A [Groq](https://groq.com/) API key (LLM inference)
- A [Cohere](https://cohere.com/) API key (re-ranking)

### Installation

```bash
git clone <link>
cd MultiAgenticRAG

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file in the project root with your API keys:

```bash
GROQ_API_KEY=your_groq_api_key
COHERE_API_KEY=your_cohere_api_key
```

### 1. Index the document

Open `config.yaml` and confirm `load_documents: True` (this tells the indexer to (re)process the source PDF into the vector store):

```yaml
retriever:
  file: "retriever/google-2024-environmental-report.pdf"
  load_documents: True
  collection_name: rag-chroma-google
  directory: vector_db
```

Then build the index:

```bash
python3 -m retriever.retriever
```

This converts the PDF to Markdown with Docling, splits it by header, embeds each chunk with `BAAI/bge-small-en-v1.5`, and persists everything into a local Chroma DB at `vector_db/`.

> Once the index exists, you can set `load_documents: False` to skip re-processing on subsequent runs.

### 2. Run the app

```bash
python3 app.py
```

```
Enter your query (type '-q' to quit):
> What is the data center PUE efficiency value in Dublin in 2021?
```

Ask questions grounded in the indexed document, e.g. one based on: https://sustainability.google/reports/google-2024-environmental-report/

---

## Architecture

The system is composed of two LangGraph graphs: a **main conversational graph** that owns routing, planning and response generation, and a **researcher subgraph** that it calls into for each research step.

```
                              ┌────────────────────────────┐
                              │     analyze_and_route_query  │
                              │  (Router: more-info /        │
                              │   environmental / general)    │
                              └───────────────┬──────────────┘
                     ┌────────────────────────┼───────────────────────┐
                     ▼                        ▼                       ▼
          ┌────────────────────┐  ┌─────────────────────┐  ┌────────────────────────┐
          │ ask_for_more_info   │  │ create_research_plan │  │ respond_to_general_query│
          │ (asks 1 follow-up)  │  │ (LLM plans ≤2 steps) │  │ (politely declines)     │
          └─────────┬───────────┘  └───────────┬──────────┘  └────────────┬────────────┘
                    END                         ▼                        END
                                    ┌────────────────────────┐
                                    │    conduct_research     │◄────────────┐
                                    │ (calls researcher_graph │             │
                                    │  for one step at a time)│             │
                                    └────────────┬─────────────┘            │
                                                 ▼                         │
                                      steps remaining? ─── yes ────────────┘
                                                 │ no
                                                 ▼
                                    ┌────────────────────────┐
                                    │         respond          │
                                    │ (answers using retrieved │
                                    │  docs, cites sources)    │
                                    └────────────┬─────────────┘
                                                 ▼
                                    ┌────────────────────────┐
                                    │   check_hallucinations   │
                                    │ (LLM grades groundedness)│
                                    └────────────┬─────────────┘
                                                 ▼
                                    binary_score == 1? ── yes ── END
                                                 │ no
                                                 ▼
                                    interrupt(): human approves retry?
                                       y → respond   |   n → END
```

### Researcher Subgraph (per research step)

```
┌──────────────────┐        ┌──────────────────────────────┐
│  generate_queries │ ─────▶ │  retrieve_and_rerank_documents │  (fanned out in parallel
│  (LLM expands the │  Send  │  • Ensemble retriever:         │   via LangGraph Send,
│   step into N      │  x N   │    - similarity search         │   one call per query)
│   search queries)   │        │    - MMR search                │
└──────────────────┘        │    - BM25 keyword search       │
                              │  • Cohere re-rank (top_k)      │
                              └──────────────────────────────┘
```

### Document Ingestion Pipeline (offline, `retriever/retriever.py`)

```
PDF (Docling, CPU accelerator)
   │  convert → export_to_markdown()
   ▼
Markdown
   │  MarkdownHeaderTextSplitter (splits on #, ##)
   ▼
Chunks
   │  HuggingFaceEmbeddings ("BAAI/bge-small-en-v1.5")
   ▼
Chroma vector store (persisted to vector_db/)
```

---

## How the Router & Decision Logic Work

The router (`ROUTER_SYSTEM_PROMPT`) classifies every incoming message into one of three types before any retrieval happens:

| Classification | Meaning | Next Node |
|---|---|---|
| `more-info` | The question is ambiguous or missing a key detail (e.g. no region/year specified) | `ask_for_more_info` — asks exactly one clarifying follow-up |
| `environmental` | The question can be answered from the Environmental Report | `create_research_plan` → `conduct_research` |
| `general` | Off-topic / unrelated to the report | `respond_to_general_query` — politely declines |

For on-topic questions, `create_research_plan` produces a short (1–2 step) plan, and each step is executed sequentially through `conduct_research`, which internally calls the researcher subgraph to fan a single step out into multiple parallel retrieval queries.

### Retrieval Strategy

Each generated query is retrieved through an **ensemble** of three retrievers, weighted and combined, then compressed/re-ranked:

| Retriever | Role | Weight (default) |
|---|---|---|
| Similarity (dense) | Semantic nearest-neighbour search | 0.3 |
| MMR (dense) | Diversity-aware semantic search | 0.3 |
| BM25 (sparse) | Keyword/lexical matching | 0.4 |

The combined candidate pool is then passed through **Cohere Rerank** (`rerank-english-v3.0`) to select the most relevant `top_k_compression` documents, all configurable in `config.yaml`.

### Groundedness Check

After `respond` generates an answer, `check_hallucinations` asks a grader LLM to output a binary score (`1` = grounded, `0` = not grounded) by comparing the answer against the retrieved documents. A `0` score triggers a LangGraph `interrupt()`, pausing execution and asking the human user whether to retry generation — a lightweight human-in-the-loop safety net rather than silently returning an unsupported answer.

---

## Configuration

All tunable parameters live in `config.yaml` rather than being hardcoded:

```yaml
retriever:
  file: "retriever/google-2024-environmental-report.pdf"
  headers_to_split_on:
    - ["#", "Header 1"]
    - ["##", "Header 2"]
  load_documents: True
  collection_name: rag-chroma-google
  directory: vector_db
  top_k: 3
  top_k_compression: 3
  ensemble_weights: [0.3, 0.3, 0.4]
  cohere_rerank_model: rerank-english-v3.0
llm:
  gpt_4o_mini: gpt-4o-mini-2024-07-18
  gpt_4o: gpt-4o-2024-08-06
  groq_model: llama-3.3-70b-versatile
  temperature: 0
```

- `.env` → secrets (API keys, tokens)
- `config.yaml` → everything else (model names, chunk sizes, file paths, retrieval weights, temperature)

By default the system runs on **Groq** (`llama-3.3-70b-versatile`) for all LLM calls (routing, planning, query generation, response, grading); OpenAI (`gpt-4o` / `gpt-4o-mini`) client code is present but currently commented out and can be swapped back in per-node.

---

## Project Structure

```
MultiAgenticRAG/
├── app.py                        # CLI entrypoint — streams responses, handles retry interrupt
├── config.yaml                   # All retrieval/LLM configuration
├── requirements.txt
├── main_graph/
│   ├── graph_builder.py          # Router, planner, respond, hallucination check, human approval
│   └── graph_states.py           # AgentState, Router, GradeHallucinations, InputState
├── subgraph/
│   ├── graph_builder.py          # generate_queries + retrieve_and_rerank_documents (parallel)
│   └── graph_states.py           # ResearcherState, QueryState
├── retriever/
│   ├── retriever.py              # Offline PDF → Markdown → Chroma indexing pipeline
│   └── google-2024-environmental-report.pdf
├── utils/
│   ├── prompt.py                 # All system prompts (router, planner, response, hallucination)
│   └── utils.py                  # config loader, UUID helpers, reduce_docs state reducer
└── vector_db/                    # Persisted Chroma store (generated, git-ignored)
```

---

## Example Interaction

```
Enter your query (type '-q' to quit):
> What is the data center PUE efficiency value in Dublin in 2021?

[router] → environmental
[plan]   → 1. Find Dublin data center PUE for 2021 in the report
[research] → generates sub-queries → retrieves via ensemble → Cohere re-rank
[respond]  → "Google's Dublin data center reported a PUE of ... [1]"
[check_hallucinations] → binary_score = 1 → END
```

If the grader instead returns `binary_score = 0`:

```
The response may contain uncertain information. Retry the generation? If yes, press 'y':
```

---

## Limitations & Future Work

### Current Limitations
- Single fixed document source (the Google Environmental Report PDF); no multi-document / multi-collection support yet.
- Research plan is capped at ~2 steps and executed sequentially, not in parallel across steps.
- No conversation-level memory persistence beyond the in-process `MemorySaver` checkpointer (state isn't durable across process restarts).
- Hallucination grading is a single binary LLM judgment with no explanation surfaced to the user beyond the retry prompt.
- CLI-only interface — no API server or web UI yet.

### Future Enhancements
- **Multi-document ingestion** — support arbitrary PDFs/URLs, not just the hardcoded environmental report.
- **Parallel step execution** — run independent research-plan steps concurrently instead of sequentially.
- **Persistent checkpointing** — swap `MemorySaver` for a durable backend (SQLite/Postgres) for cross-session memory.
- **Richer hallucination feedback** — surface *why* an answer was flagged as ungrounded, not just a binary score.
- **Web/API interface** — expose the graph via FastAPI or a simple chat UI instead of a terminal loop.
- **Evaluation harness** — automated RAG quality metrics (faithfulness, answer relevance, context precision/recall).

---

## Contributing

Want to extend MultiAgentic RAG?

1. Fork or clone the repository
2. Add new nodes/routes in `main_graph/graph_builder.py` or `subgraph/graph_builder.py`
3. Add or tune prompts in `utils/prompt.py`
4. Update `config.yaml` for any new tunables
5. Submit a PR with example queries demonstrating the change

---

## License
This project is private. All rights reserved by [Gufran](https://github.com/Gufran-wordlybee)

## Contact
**Gufran Alam**
- **Email:** <a href="mailto:justgufran07@gmail.com">justgufran07@gmail.com</a>
- **LinkedIn:** <a href="https://www.linkedin.com/in/gufran-alam-a25717321/" target="_blank">linkedin.com/in/gufran-alam-a25717321</a>

For bug reports or feature requests, please open an issue in this repository.



