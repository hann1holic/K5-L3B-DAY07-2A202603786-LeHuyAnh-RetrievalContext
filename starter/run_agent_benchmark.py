"""Run the real tool-calling agent on one frozen split; API use may incur cost."""
import argparse
import json
import time
from pathlib import Path
from day6_adapter import EMBEDDING_STATS, ROOT as STARTER, load_env, day6_root
from model_controller import OpenAIController
from retrievers import Day6DenseRetriever, HybridRetriever, LexicalRetriever
from agentic import run_agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=STARTER.parent / "data" / "day7_questions.json")
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--strategy", choices=("dense", "lexical", "hybrid"), default="dense")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--max-searches", type=int, default=3)
    ap.add_argument("--budget", type=int, default=2400, help="Context budget in characters")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if min(args.top_k, args.max_searches, args.budget) < 1:
        ap.error("top-k, max-searches, and budget must be positive")
    load_env(day6_root() / "starter" / ".env")
    load_env(STARTER / ".env")
    dense = Day6DenseRetriever()
    if args.strategy == "dense": retriever = dense
    elif args.strategy == "lexical": retriever = LexicalRetriever(dense.chunks)
    else: retriever = HybridRetriever(dense, LexicalRetriever(dense.chunks))
    controller = OpenAIController()
    embedding_before = dict(EMBEDDING_STATS)
    questions = [q for q in json.loads(args.questions.read_text(encoding="utf-8")) if q["split"] == args.split]
    if args.split == "test" and any(q.get("audit_required") for q in questions):
        raise ValueError("Test benchmark has unreviewed absence labels; audit them against the full source before running test")
    traces, cases = [], []
    for q in questions:
        cases.append({"id": q["id"], "split": q["split"], "answerable": q["answerable"]})
        started = time.perf_counter()
        try:
            # Deliberately pass only question and user-visible filters; gold never enters the agent.
            result = run_agent(q["question"], controller, retriever, filters=q.get("filter") or None,
                               max_searches=args.max_searches, top_k=args.top_k, budget=args.budget)
            trace = {"query_id": q["id"], "split": q["split"], "question": q["question"],
                     **result, "elapsed_seconds": time.perf_counter() - started}
        except Exception as exc:
            trace = {"query_id": q["id"], "split": q["split"], "question": q["question"],
                     "status": "error", "answer": None, "citations": [], "events": [],
                     "error_type": type(exc).__name__, "elapsed_seconds": time.perf_counter() - started}
        traces.append(trace)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata = {"split": args.split, "strategy": args.strategy, "top_k": args.top_k,
                "max_searches": args.max_searches, "context_budget_chars": args.budget,
                "chat_model": controller.model, "chat_base_url": controller.base_url,
                "embedding_model": __import__("os").getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
                "model_calls": controller.calls, "prompt_tokens_reported": controller.prompt_tokens,
                "completion_tokens_reported": controller.completion_tokens,
                "provider_reported_cost_usd": (round(controller.reported_cost_usd, 8)
                    if controller.cost_reporting_calls == controller.calls and controller.calls else None),
                "provider_cost_reporting_calls": controller.cost_reporting_calls,
                "query_embedding_requests": EMBEDDING_STATS["requests"] - embedding_before["requests"],
                "query_embedding_tokens": EMBEDDING_STATS["tokens"] - embedding_before["tokens"],
                "query_embedding_seconds": round(EMBEDDING_STATS["elapsed_seconds"] - embedding_before["elapsed_seconds"], 3),
                "query_embedding_cost_usd": None,
                "errors": sum(t["status"] == "error" for t in traces)}
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
