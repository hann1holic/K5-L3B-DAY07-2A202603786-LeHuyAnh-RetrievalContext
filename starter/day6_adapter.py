"""Read the persisted Day 6 Chroma index and benchmark without copying vectors."""
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parent
DEFAULT_DAY6 = Path(r"D:\VinUni\K5-L3B-DAY06-LeHuyAnh-2A202603786-DataIngestion")
EMBEDDING_STATS = {"requests": 0, "tokens": 0, "elapsed_seconds": 0.0}


def load_env(path: Path) -> None:
    """Load local KEY=VALUE config without logging secret values."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ[key.strip()] = value


def day6_root() -> Path:
    return Path(os.getenv("DAY6_ROOT", str(DEFAULT_DAY6))).expanduser().resolve()


def chroma_path() -> Path:
    configured = Path(os.getenv("DAY6_CHROMA_PATH", "starter/results/chroma"))
    return configured if configured.is_absolute() else day6_root() / configured


def embedding_config() -> tuple[str, str, str]:
    """Return (api_key, model, base_url), inheriting Day 6's local settings."""
    load_env(day6_root() / "starter" / ".env")
    load_env(ROOT / ".env")
    key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"
    base = os.getenv("EMBEDDING_BASE_URL")
    if not base:
        base = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY") else "https://api.openai.com/v1"
    if not key:
        raise RuntimeError("Thiếu OPENAI_API_KEY/OPENROUTER_API_KEY trong .env Day 6 hoặc Day 7")
    return key, model, base.rstrip("/")


def load_questions(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or (day6_root() / "starter" / "benchmark" / "questions.json")
    return json.loads(path.read_text(encoding="utf-8"))


def as_day7_question(q: dict[str, Any]) -> dict[str, Any]:
    pages = q.get("gold_pages", [])
    sources = q.get("gold_source", [])
    evidence = q.get("gold_evidence", [])
    gold = []
    for i, quote in enumerate(evidence):
        gold.append({"source_id": "magnifica_humanitas", "source": sources[min(i, len(sources)-1)],
                     "page": pages[min(i, len(pages)-1)], "evidence": quote})
    return {"id": q["id"], "split": q["split"], "question": q["question"],
            "answerable": True, "gold": gold, "filter": q.get("filters", {}),
            "expected_answer": "Review the cited evidence in the source document."}


def load_day6_chunks(collection_name: str | None = None) -> tuple[Any, list[dict[str, Any]]]:
    """Return the Chroma collection and normalized chunk records; no re-embedding."""
    import chromadb
    path = chroma_path()
    if not path.exists():
        raise FileNotFoundError(f"Day 6 Chroma index not found: {path}")
    client = chromadb.PersistentClient(path=str(path))
    name = collection_name or os.getenv("DAY6_COLLECTION", "day6_strategy_b")
    collection = client.get_collection(name)
    result = collection.get(include=["documents", "metadatas"])
    chunks = []
    for cid, text, meta in zip(result["ids"], result["documents"], result["metadatas"]):
        meta = meta or {}
        chunks.append({"id": cid, "source_id": meta.get("document_id", "magnifica_humanitas"),
                       "source": meta.get("source", "Magnifica_Humanitas.pdf"),
                       "page": meta.get("page"), "text": text, "metadata": meta})
    return collection, chunks


def embed_query(text: str) -> list[float]:
    """Create a query vector compatible with the Day 6 index."""
    import json as _json
    from urllib.request import Request, urlopen
    key, model, base = embedding_config()
    body = _json.dumps({"model": model, "input": text}).encode()
    req = Request(base + "/embeddings", body,
                  {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    try:
        with urlopen(req, timeout=90) as response:
            payload = _json.loads(response.read())
    except HTTPError as exc:
        if exc.code == 401:
            raise RuntimeError("Embedding API rejected the configured credential (HTTP 401); update the local .env key") from None
        raise RuntimeError(f"Embedding API returned HTTP {exc.code}; check provider/model configuration") from None
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"Embedding API is unreachable ({type(exc).__name__})") from None
    EMBEDDING_STATS["requests"] += 1
    EMBEDDING_STATS["tokens"] += int(payload.get("usage", {}).get("total_tokens", 0) or 0)
    EMBEDDING_STATS["elapsed_seconds"] += time.perf_counter() - started
    vector = payload["data"][0]["embedding"]
    if len(vector) != 1536:
        raise ValueError(f"Embedding dimension {len(vector)} does not match Day 6 index dimension 1536")
    return vector
