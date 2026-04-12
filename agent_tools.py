"""
Agent Tools - Actions available to the agentic RAG orchestrator
"""

import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: Any
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "tool": self.tool_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
        }


class ToolRegistry:
    """Registry of all tools available to the agent."""

    def __init__(self, rag_engine):
        self.rag_engine = rag_engine
        self._tools = {
            "retrieve_documents": self._retrieve_documents,
            "check_relevance": self._check_relevance,
            "refine_query": self._refine_query,
            "generate_sub_queries": self._generate_sub_queries,
            "ingest_text": self._ingest_text,
            "get_store_stats": self._get_store_stats,
        }

    def get_tool_schemas(self) -> List[Dict]:
        """Returns Anthropic-compatible tool schemas."""
        return [
            {
                "name": "retrieve_documents",
                "description": (
                    "Search the vector store and retrieve the most relevant document chunks "
                    "for a given query. Use this as the primary retrieval step."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to retrieve documents for.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of chunks to retrieve (default: 5).",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "check_relevance",
                "description": (
                    "Assess whether the retrieved context is sufficient to answer the question, "
                    "or if additional retrieval steps are needed. Returns a relevance score and reasoning."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The original user question."},
                        "context": {"type": "string", "description": "The retrieved context to evaluate."},
                    },
                    "required": ["question", "context"],
                },
            },
            {
                "name": "refine_query",
                "description": (
                    "Reformulate or expand the search query to improve retrieval quality. "
                    "Use when initial retrieval returns low-relevance results."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "original_query": {"type": "string", "description": "The original search query."},
                        "feedback": {
                            "type": "string",
                            "description": "Reason why the original query was insufficient.",
                        },
                    },
                    "required": ["original_query"],
                },
            },
            {
                "name": "generate_sub_queries",
                "description": (
                    "Decompose a complex question into multiple focused sub-queries "
                    "for multi-hop retrieval. Use for complex questions requiring multiple lookups."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The complex question to decompose.",
                        }
                    },
                    "required": ["question"],
                },
            },
            {
                "name": "ingest_text",
                "description": "Add new text to the knowledge base at runtime.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text content to ingest."},
                        "source": {"type": "string", "description": "Source label for the text."},
                    },
                    "required": ["text"],
                },
            },
            {
                "name": "get_store_stats",
                "description": "Get statistics about the current knowledge base (doc count, etc.).",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ]

    def execute(self, tool_name: str, tool_input: Dict) -> ToolResult:
        if tool_name not in self._tools:
            return ToolResult(tool_name=tool_name, success=False, output=None, error=f"Unknown tool: {tool_name}")
        try:
            result = self._tools[tool_name](**tool_input)
            return ToolResult(tool_name=tool_name, success=True, output=result)
        except Exception as e:
            return ToolResult(tool_name=tool_name, success=False, output=None, error=str(e))

    # ── Tool implementations ─────────────────────────────────────────────

    def _retrieve_documents(self, query: str, top_k: int = 5) -> Dict:
        chunks = self.rag_engine.retrieve(query, top_k=top_k)
        return {
            "query": query,
            "chunks_retrieved": len(chunks),
            "context": self.rag_engine.retrieve_as_context(query, top_k=top_k),
            "scores": [{"rank": c.rank, "score": round(c.score, 4), "source": c.document.metadata.get("source", "?")} for c in chunks],
        }

    def _check_relevance(self, question: str, context: str) -> Dict:
        # Heuristic: measure keyword overlap as a proxy relevance signal
        q_words = set(question.lower().split())
        c_words = set(context.lower().split())
        overlap = len(q_words & c_words) / max(len(q_words), 1)
        is_sufficient = overlap > 0.15 and len(context) > 100
        return {
            "is_sufficient": is_sufficient,
            "keyword_overlap": round(overlap, 3),
            "context_length": len(context),
            "recommendation": "proceed_to_answer" if is_sufficient else "retrieve_more",
        }

    def _refine_query(self, original_query: str, feedback: str = "") -> Dict:
        # Simple expansion heuristics — replaced by LLM call in production
        expansions = {
            "what": "explain describe define",
            "how": "process steps method approach",
            "why": "reason cause explanation rationale",
            "when": "time date period timeline",
            "who": "person entity organization",
        }
        first_word = original_query.split()[0].lower() if original_query else ""
        extra = expansions.get(first_word, "details context information")
        refined = f"{original_query} {extra}".strip()
        return {
            "original": original_query,
            "refined": refined,
            "strategy": "keyword_expansion",
            "feedback_applied": bool(feedback),
        }

    def _generate_sub_queries(self, question: str) -> Dict:
        # Rule-based decomposition — LLM should handle this in production
        connectors = [" and ", " as well as ", " along with ", " while also "]
        sub_queries = [question]
        for conn in connectors:
            if conn in question.lower():
                parts = question.lower().split(conn, 1)
                sub_queries = [parts[0].strip(), parts[1].strip()]
                break
        # Always add a general "background" query
        topic = question.split()[:4]
        sub_queries.append(" ".join(topic) + " overview background")
        return {
            "original_question": question,
            "sub_queries": list(dict.fromkeys(sub_queries)),  # deduplicate
            "strategy": "conjunction_split",
        }

    def _ingest_text(self, text: str, source: str = "runtime_input") -> Dict:
        count = self.rag_engine.ingest(text, metadata={"source": source})
        return {"chunks_added": count, "source": source, "total_docs": self.rag_engine.doc_count}

    def _get_store_stats(self) -> Dict:
        return {
            "total_chunks": self.rag_engine.doc_count,
            "embedding_dim": self.rag_engine.embedder.dim,
            "chunk_size": self.rag_engine.splitter.chunk_size,
        }
