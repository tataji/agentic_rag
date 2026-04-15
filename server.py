"""
FastAPI Backend for Agentic RAG
Endpoints: ingest, ask, stream, stats
"""

import os
import json
import asyncio
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent dir to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.orchestrator import AgenticRAG, AgentStep


# ── Global agent instance ────────────────────────────────────────────────

agent: Optional[AgenticRAG] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    agent = AgenticRAG(
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        verbose=True,
    )
    # Seed some demo knowledge
    agent.ingest(DEMO_KNOWLEDGE, source="demo_knowledge_base")
    print("[API] AgenticRAG ready.")
    yield
    print("[API] Shutting down.")


app = FastAPI(title="Agentic RAG API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────

class IngestRequest(BaseModel):
    text: str
    source: str = "user_upload"


class AskRequest(BaseModel):
    question: str


class IngestResponse(BaseModel):
    chunks_added: int
    total_chunks: int
    source: str


class AskResponse(BaseModel):
    question: str
    answer: str
    iterations: int
    sources_used: List[str]
    steps: List[dict]


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    if not agent:
        raise HTTPException(503, "Agent not initialized")
    count = agent.ingest(req.text, source=req.source)
    return IngestResponse(
        chunks_added=count,
        total_chunks=agent.rag_engine.doc_count,
        source=req.source,
    )


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if not agent:
        raise HTTPException(503, "Agent not initialized")
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    result = agent.ask(req.question)
    return AskResponse(**result.to_dict())


@app.get("/ask/stream")
async def ask_stream(question: str):
    """
    Server-Sent Events endpoint that streams agent steps in real-time.
    Frontend can connect with EventSource('/ask/stream?question=...')
    """
    if not agent:
        raise HTTPException(503, "Agent not initialized")

    queue: asyncio.Queue = asyncio.Queue()

    def on_step(step: AgentStep):
        asyncio.get_event_loop().call_soon_threadsafe(
            queue.put_nowait,
            json.dumps({"type": step.step_type, "tool": step.tool_name, "content": str(step.content)[:500]}),
        )

    async def generate():
        loop = asyncio.get_event_loop()

        async def run_agent():
            result = await loop.run_in_executor(None, lambda: agent.ask(question))
            await queue.put(json.dumps({"type": "done", "answer": result.answer, "sources": result.sources_used}))
            await queue.put(None)  # sentinel

        asyncio.create_task(run_agent())

        while True:
            item = await queue.get()
            if item is None:
                break
            yield f"data: {item}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/stats")
async def stats():
    if not agent:
        raise HTTPException(503, "Agent not initialized")
    return agent.stats()


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Demo seed data ─────────────────────────────────────────────────────────

DEMO_KNOWLEDGE = """
# Agentic RAG Overview

Agentic Retrieval-Augmented Generation (RAG) combines large language models with dynamic 
tool-use loops to go beyond single-shot retrieval. Traditional RAG pipelines retrieve 
once and generate — agentic RAG can retrieve multiple times, evaluate quality, decompose 
complex questions, and self-correct.

## Key Components

### 1. Vector Store
Documents are chunked, embedded, and stored in a vector database. At query time, 
the agent performs cosine-similarity search to find relevant passages. Popular 
vector stores include Chroma, Qdrant, Pinecone, Weaviate, and FAISS.

### 2. Embedding Model
Text is converted to high-dimensional vectors that capture semantic meaning. 
OpenAI's text-embedding-3-small, Cohere's embed-v3, and sentence-transformers 
are common choices. The same model must be used for both indexing and querying.

### 3. Agent Orchestrator
The orchestrator is an LLM (e.g. Claude) with access to retrieval tools. It decides:
- Whether to retrieve once or multiple times
- When to decompose a complex question into sub-queries
- When retrieved context is sufficient to generate an answer
- How to synthesize across multiple retrieved chunks

### 4. Tool Use Loop
The agentic loop: Plan → Tool Call → Observe → Plan → ... → Answer
This continues until the agent has enough information or hits a max-iteration limit.

## Agentic RAG Strategies

### Multi-hop Retrieval
For questions requiring information from multiple documents, the agent chains 
retrieval calls. Example: "What did the CEO say about Q3 revenue, and how does 
that compare to analyst forecasts?" requires two separate retrievals.

### Query Rewriting
When initial retrieval yields low-relevance results, the agent rewrites the query 
using different terminology, synonyms, or broader/narrower scope.

### Iterative Refinement
The agent checks whether retrieved context fully answers the question. If not, 
it refines the query and retrieves again, up to a configured limit.

### Hybrid Search
Combining dense vector search (semantic) with sparse BM25 search (keyword) 
improves recall. This is especially useful for technical queries with specific 
terminology.

## Production Considerations

- **Chunking strategy**: Chunk size and overlap significantly affect retrieval quality.
  Smaller chunks (128–256 tokens) improve precision; larger chunks preserve context.
- **Reranking**: After initial retrieval, a cross-encoder reranker (e.g. Cohere Rerank)
  improves top-k precision.
- **Caching**: Cache embeddings and frequent query results to reduce latency and cost.
- **Guardrails**: Validate tool calls, limit iterations, and handle empty-context gracefully.
- **Observability**: Log every tool call, retrieved chunk, and agent decision for debugging.

## Indian Market Applications

Agentic RAG is particularly valuable for:
- GST circular and compliance Q&A across thousands of CBIC notifications
- SEBI regulation research and cross-referencing
- MSME credit policy analysis across RBI master circulars
- Legal contract analysis combining statute lookup with specific clause retrieval
- Agricultural commodity price research combining news with historical data
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
