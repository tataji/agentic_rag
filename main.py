"""
Agentic RAG - Main Entry Point
Usage:
    python main.py                    # Interactive CLI
    python main.py --demo             # Run demo with sample questions
    python main.py --question "..."   # Ask a single question
    python main.py --serve            # Start FastAPI server
"""

import os
import sys
import argparse
import json

# ── Path setup ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from agents.orchestrator import AgenticRAG

DEMO_TEXT = """
ArthAI is a multi-agent AI trading system designed for Indian equity markets (NSE/BSE).
The system consists of six specialized agents:

1. Orchestrator Agent: Coordinates all agents and makes final trade decisions.
2. Technical Analysis Agent: Analyzes price patterns, RSI, MACD, Bollinger Bands.
3. Fundamental Screener Agent: Evaluates P/E ratios, earnings growth, debt-equity.
4. News Sentiment Agent: Processes financial news and social signals.
5. Risk Manager Agent: Enforces position sizing, stop-losses, and drawdown limits.
6. F&O Strategy Agent: Handles options strategies including iron condor and straddles.

The system uses paper trading mode by default for safe experimentation.
It connects to NSE data feeds during market hours (9:15 AM to 3:30 PM IST).
The backend is built with FastAPI and the frontend with React/Vite.
Docker is used for deployment with CI/CD via GitHub Actions.
Backtesting uses 5 years of historical OHLCV data with Sharpe ratio optimization.
"""

DEMO_QUESTIONS = [
    "What are the six agents in ArthAI and what does each one do?",
    "What risk management features does ArthAI have?",
    "How does ArthAI handle options trading?",
]


def run_demo(rag: AgenticRAG):
    print("\n" + "=" * 60)
    print("AGENTIC RAG DEMO")
    print("=" * 60)

    print("\n[Demo] Ingesting knowledge base...")
    rag.ingest(DEMO_TEXT, source="arthai_docs")
    print(f"[Demo] Knowledge base ready. Stats: {rag.stats()}")

    for q in DEMO_QUESTIONS:
        response = rag.ask(q)
        print(f"\n{'─'*60}")
        print(f"Q: {q}")
        print(f"A: {response.answer}")
        print(f"Sources: {response.sources_used} | Iterations: {response.iterations}")


def run_interactive(rag: AgenticRAG):
    print("\n" + "=" * 60)
    print("AGENTIC RAG - Interactive Mode")
    print("Commands: /ingest <text>, /stats, /quit")
    print("=" * 60 + "\n")

    print("First, let's add some knowledge. Type /ingest followed by text,")
    print("or just start asking questions (the agent will retrieve from an empty store).\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
                print("Goodbye!")
                break

            elif user_input.startswith("/ingest "):
                text = user_input[8:]
                count = rag.ingest(text, source="interactive_input")
                print(f"[System] Added {count} chunks. Total: {rag.rag_engine.doc_count}")

            elif user_input == "/stats":
                print(json.dumps(rag.stats(), indent=2))

            else:
                response = rag.ask(user_input)
                print(f"\nAgent: {response.answer}")
                print(f"[Sources: {response.sources_used} | {response.iterations} iterations]\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"[Error] {e}")


def main():
    parser = argparse.ArgumentParser(description="Agentic RAG System")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--question", "-q", type=str, help="Ask a single question")
    parser.add_argument("--ingest", type=str, help="Text file to ingest")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI server")
    parser.add_argument("--model", default="claude-opus-4-5", help="Claude model to use")
    args = parser.parse_args()

    if args.serve:
        import uvicorn
        uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("[WARNING] ANTHROPIC_API_KEY not set. Tool calls will fail.")

    rag = AgenticRAG(api_key=api_key, model=args.model, verbose=True)

    if args.ingest:
        rag.ingest_file(args.ingest)

    if args.demo:
        run_demo(rag)
    elif args.question:
        response = rag.ask(args.question)
        print(f"\nAnswer: {response.answer}")
        print(f"Sources: {response.sources_used}")
    else:
        run_interactive(rag)


if __name__ == "__main__":
    main()
