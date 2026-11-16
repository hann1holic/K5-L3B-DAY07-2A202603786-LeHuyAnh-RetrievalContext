"""Dense, lexical, and reciprocal-rank-fusion retrieval over Day 6 chunks."""
from typing import Any
from baseline import TfidfIndex
from day6_adapter import embed_query, load_day6_chunks


class Day6DenseRetriever:
    def __init__(self, collection=None, chunks=None):
        if collection is None or chunks is None:
            collection, chunks = load_day6_chunks()
        self.collection, self.chunks = collection, chunks
        self.by_id = {c["id"]: c for c in chunks}

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3):
        options: dict[str, Any] = {"query_embeddings": [embed_query(query)], "n_results": top_k,
                                   "include": ["documents", "metadatas", "distances"]}
        if filters:
            options["where"] = filters
        result = self.collection.query(**options)
        rows = []
        for cid, text, meta, distance in zip(result["ids"][0], result["documents"][0],
                                               result["metadatas"][0], result["distances"][0]):
            rows.append({"id": cid, "source_id": meta.get("document_id", "magnifica_humanitas"),
                         "source": meta.get("source", "Magnifica_Humanitas.pdf"), "page": meta.get("page"),
                         "text": text, "metadata": meta, "score": 1.0 - float(distance),
                         "strategy": "dense_day6_b"})
        return rows


class LexicalRetriever:
    """Offline TF-IDF lexical ranking over the exact same indexed chunks."""
    def __init__(self, chunks=None):
        if chunks is None:
            _, chunks = load_day6_chunks()
        self.chunks = chunks
        self.index = TfidfIndex(chunks)

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3):
        rows = self.index.search(query, top_k=len(self.chunks))
        if filters:
            rows = [r for r in rows if all(str(r["metadata"].get(k)) == str(v) for k, v in filters.items())]
        return [{**r, "strategy": "tfidf_lexical"} for r in rows[:top_k]]


class HybridRetriever:
    """Fuse dense and lexical rankings using reciprocal rank fusion (RRF)."""
    def __init__(self, dense=None, lexical=None, rrf_k: int = 60):
        self.dense = dense or Day6DenseRetriever()
        self.lexical = lexical or LexicalRetriever(self.dense.chunks)
        self.rrf_k = rrf_k

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3):
        candidates = {}
        for retriever in (self.dense, self.lexical):
            rows = retriever.search(query, filters=filters, top_k=max(top_k, 10))
            for rank, row in enumerate(rows, 1):
                cid = row["id"]
                item = candidates.setdefault(cid, {**row, "rrf_score": 0.0, "rankings": {}})
                item["rrf_score"] += 1.0 / (self.rrf_k + rank)
                item["rankings"][row.get("strategy", retriever.__class__.__name__)] = rank
        ranked = sorted(candidates.values(), key=lambda x: (-x["rrf_score"], x["id"]))[:top_k]
        return [{**row, "score": row["rrf_score"], "strategy": "hybrid_rrf"} for row in ranked]
