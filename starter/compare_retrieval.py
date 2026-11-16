"""Compare Day 6 dense, TF-IDF lexical, and hybrid RRF on one benchmark split."""
import argparse
import json
import statistics
import time
from pathlib import Path
from baseline import hit, pack_context
from day6_adapter import EMBEDDING_STATS, ROOT as STARTER, load_day6_chunks
from retrievers import Day6DenseRetriever, HybridRetriever, LexicalRetriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=STARTER.parent / "data" / "day7_questions.json")
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--top-k", type=int, default=10, help="Must be >=3; metrics report top 1 and top 3")
    ap.add_argument("--budget", type=int, default=2400, help="Context budget in characters")
    ap.add_argument("--methods", default="dense,lexical,hybrid",
                    help="Comma-separated subset of dense,lexical,hybrid")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.top_k < 3 or args.budget < 1:
        ap.error("--top-k must be >=3 and --budget must be positive")
    questions = [q for q in json.loads(args.questions.read_text(encoding="utf-8"))
                 if q["split"] == args.split and q["answerable"]]
    selected_methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not selected_methods or set(selected_methods) - {"dense", "lexical", "hybrid"}:
        ap.error("--methods must contain dense, lexical, and/or hybrid")
    _, chunks = load_day6_chunks()
    lexical = LexicalRetriever(chunks)
    methods = {}
    if "dense" in selected_methods or "hybrid" in selected_methods:
        collection, chunks = load_day6_chunks()
        dense = Day6DenseRetriever(collection, chunks)
    if "dense" in selected_methods: methods["dense_day6_b"] = dense
    if "lexical" in selected_methods: methods["tfidf_lexical"] = lexical
    if "hybrid" in selected_methods: methods["hybrid_rrf"] = HybridRetriever(dense, lexical)
    traces, summary = [], {}
    for name, method in methods.items():
        before = dict(EMBEDDING_STATS)
        method_traces, latencies = [], []
        for q in questions:
            start = time.perf_counter()
            results = method.search(q["question"], filters=q.get("filter") or None, top_k=args.top_k)
            context, ids = pack_context(results, args.budget)
            ms = (time.perf_counter() - start) * 1000
            top3 = results[:3]
            row = {"query_id": q["id"], "split": q["split"], "question": q["question"],
                   "strategy": name, "filter": q.get("filter", {}), "top_k": args.top_k,
                   "budget_chars": args.budget, "results": results, "context": context,
                   "context_ids": ids, "elapsed_ms": ms, "gold": q["gold"],
                   "hit1": bool(results and hit(results[0], q["gold"])),
                   "hit3": any(hit(c, q["gold"]) for c in top3),
                   "context_hit": any(c["id"] in ids and hit(c, q["gold"]) for c in results)}
            traces.append(row); method_traces.append(row); latencies.append(ms)
        n = len(method_traces)
        summary[name] = {"split": args.split, "n_answerable": n,
                         "hit1": sum(t["hit1"] for t in method_traces),
                         "hit3": sum(t["hit3"] for t in method_traces),
                         "context_evidence": sum(t["context_hit"] for t in method_traces),
                         "median_retrieval_ms": statistics.median(latencies) if latencies else None,
                         "query_embedding_requests": EMBEDDING_STATS["requests"] - before["requests"],
                         "query_embedding_tokens": EMBEDDING_STATS["tokens"] - before["tokens"],
                         "query_embedding_seconds": round(EMBEDDING_STATS["elapsed_seconds"] - before["elapsed_seconds"], 3),
                         "embedding_cost_usd": None,
                         "context_budget_chars": args.budget, "candidate_top_k": args.top_k}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
