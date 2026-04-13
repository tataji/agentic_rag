"""
Agentic RAG Orchestrator
Uses Anthropic tool-use (function calling) in an agentic loop:
  Plan → Retrieve → Evaluate → Re-retrieve (if needed) → Synthesize
"""

import os
import json
import anthropic
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field

from core.rag_engine import RAGEngine
from tools.agent_tools import ToolRegistry, ToolResult


@dataclass
class AgentStep:
    step_type: str          # "thinking" | "tool_call" | "tool_result" | "answer"
    content: Any
    tool_name: Optional[str] = None


@dataclass
class AgentResponse:
    question: str
    answer: str
    steps: List[AgentStep] = field(default_factory=list)
    iterations: int = 0
    sources_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "iterations": self.iterations,
            "sources_used": self.sources_used,
            "steps": [
                {"type": s.step_type, "tool": s.tool_name, "content": str(s.content)[:300]}
                for s in self.steps
            ],
        }


SYSTEM_PROMPT = """You are an expert Agentic RAG assistant. Your job is to answer questions 
accurately using the provided tools to retrieve and reason over a knowledge base.

## Your Decision Process:
1. **Analyze** the question — is it simple (single retrieval) or complex (multi-hop)?
2. **For complex questions**: use `generate_sub_queries` to decompose, then retrieve each.
3. **Retrieve** relevant chunks using `retrieve_documents`.
4. **Evaluate** retrieved context with `check_relevance`.
5. **If insufficient**: refine the query with `refine_query` and retrieve again.
6. **Synthesize** a precise, grounded answer from accumulated context.

## Rules:
- Always base your final answer on retrieved context, not prior knowledge.
- Cite the source labels when answering.
- If context is truly empty, say so honestly.
- Maximum 4 retrieval iterations per question to avoid loops.
- Think step-by-step before calling tools.
"""


class AgenticRAG:
    """
    Agentic RAG system with full tool-use loop.
    
    Usage:
        rag = AgenticRAG(api_key="sk-ant-...")
        rag.ingest("Your document text here", source="my_doc")
        response = rag.ask("What does the document say about X?")
        print(response.answer)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-opus-4-5",
        max_iterations: int = 6,
        verbose: bool = True,
        on_step: Optional[Callable[[AgentStep], None]] = None,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.on_step = on_step  # Optional callback for streaming step updates

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.rag_engine = RAGEngine()
        self.tool_registry = ToolRegistry(self.rag_engine)

    # ── Ingestion ────────────────────────────────────────────────────────

    def ingest(self, text: str, source: str = "default") -> int:
        """Add text to the knowledge base. Returns number of chunks created."""
        count = self.rag_engine.ingest(text, metadata={"source": source})
        if self.verbose:
            print(f"[Ingest] Added {count} chunks from '{source}' (total: {self.rag_engine.doc_count})")
        return count

    def ingest_file(self, filepath: str) -> int:
        count = self.rag_engine.ingest_file(filepath)
        if self.verbose:
            print(f"[Ingest] File '{filepath}' → {count} chunks")
        return count

    # ── Agentic Query Loop ────────────────────────────────────────────────

    def ask(self, question: str) -> AgentResponse:
        """
        Run the full agentic RAG loop for a question.
        Returns an AgentResponse with answer + full trace.
        """
        response_obj = AgentResponse(question=question)
        messages: List[Dict] = [{"role": "user", "content": question}]
        sources_seen = set()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[Agent] Question: {question}")
            print(f"{'='*60}")

        for iteration in range(self.max_iterations):
            response_obj.iterations = iteration + 1

            # ── Call Anthropic with tools ─────────────────────────────
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                tools=self.tool_registry.get_tool_schemas(),
                messages=messages,
            )

            # ── Process response blocks ───────────────────────────────
            tool_use_blocks = []
            text_blocks = []

            for block in response.content:
                if block.type == "text":
                    text_blocks.append(block.text)
                    step = AgentStep(step_type="thinking", content=block.text)
                    response_obj.steps.append(step)
                    if self.on_step:
                        self.on_step(step)
                    if self.verbose:
                        print(f"\n[Think] {block.text[:200]}...")

                elif block.type == "tool_use":
                    tool_use_blocks.append(block)
                    step = AgentStep(
                        step_type="tool_call",
                        content=block.input,
                        tool_name=block.name,
                    )
                    response_obj.steps.append(step)
                    if self.on_step:
                        self.on_step(step)
                    if self.verbose:
                        print(f"\n[Tool Call] {block.name}({json.dumps(block.input)[:150]})")

            # ── If no tool calls → agent is done, extract final answer ─
            if response.stop_reason == "end_turn" or not tool_use_blocks:
                final_answer = " ".join(text_blocks).strip()
                response_obj.answer = final_answer
                response_obj.sources_used = list(sources_seen)
                step = AgentStep(step_type="answer", content=final_answer)
                response_obj.steps.append(step)
                if self.on_step:
                    self.on_step(step)
                if self.verbose:
                    print(f"\n[Answer] {final_answer[:300]}")
                return response_obj

            # ── Execute tools and collect results ─────────────────────
            messages.append({"role": "assistant", "content": response.content})
            tool_results_content = []

            for tool_block in tool_use_blocks:
                tool_result: ToolResult = self.tool_registry.execute(
                    tool_name=tool_block.name,
                    tool_input=tool_block.input,
                )

                # Track sources from retrieve calls
                if tool_result.success and isinstance(tool_result.output, dict):
                    for score_info in tool_result.output.get("scores", []):
                        sources_seen.add(score_info.get("source", "?"))

                result_str = json.dumps(tool_result.output) if tool_result.success else f"ERROR: {tool_result.error}"

                step = AgentStep(
                    step_type="tool_result",
                    content=result_str[:500],
                    tool_name=tool_block.name,
                )
                response_obj.steps.append(step)
                if self.on_step:
                    self.on_step(step)
                if self.verbose:
                    print(f"[Tool Result] {tool_block.name} → {result_str[:200]}")

                tool_results_content.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": result_str,
                })

            messages.append({"role": "user", "content": tool_results_content})

        # ── Max iterations reached ─────────────────────────────────────
        response_obj.answer = "Maximum retrieval iterations reached. Partial context may be incomplete."
        response_obj.sources_used = list(sources_seen)
        return response_obj

    # ── Batch QA ──────────────────────────────────────────────────────────

    def ask_batch(self, questions: List[str]) -> List[AgentResponse]:
        return [self.ask(q) for q in questions]

    def stats(self) -> Dict:
        return {
            "model": self.model,
            "chunks_in_store": self.rag_engine.doc_count,
            "embedding_dim": self.rag_engine.embedder.dim,
            "max_iterations": self.max_iterations,
        }
