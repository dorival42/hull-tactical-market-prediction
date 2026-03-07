"""
DAG 2 — RAG Update (hourly)
==============================
Schedule : every hour during market hours (Mon–Fri, 14:00–22:00 UTC)
Tasks    :
  1. fetch_financial_news  — pull latest articles from NewsAPI
  2. chunk_and_embed       — split text, embed with HuggingFace sentence-transformers
  3. update_chromadb       — upsert embeddings into ChromaDB collection

Airflow Variables required (set via UI or airflow-init):
  - NEWS_API_KEY  : NewsAPI.org API key

ChromaDB connection:
  - CHROMADB_HOST (env var, default 'chromadb')
  - CHROMADB_PORT (env var, default '8000')
"""
from __future__ import annotations

import json
import logging
import os
import sys
import hashlib
from datetime import datetime, timedelta
from typing import Any

_project_root = os.getenv("PYTHONPATH", "/opt/airflow/project")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from dags.utils.alert_utils import on_failure_callback

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_ARGS = {
    "owner": "hull-tactical",
    "depends_on_past": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=1),
    "retry_exponential_backoff": True,          # HIGH-15: 1m → 2m → 4m
    "max_retry_delay": timedelta(minutes=30),
    "email_on_failure": False,
    "email_on_retry": False,
    "on_failure_callback": on_failure_callback,  # HIGH-14: Slack + log alerting
}

CHROMADB_HOST = os.getenv("CHROMADB_HOST", "chromadb")
CHROMADB_PORT = int(os.getenv("CHROMADB_PORT", "8000"))
COLLECTION_NAME = "financial_news"
EMBED_MODEL = "all-MiniLM-L6-v2"   # ~80 MB, fast CPU inference

# Queries used to search for S&P 500 / macro news
NEWS_QUERIES = [
    "S&P 500 stock market",
    "Federal Reserve interest rates",
    "US economy GDP",
    "Wall Street earnings",
]

# ---------------------------------------------------------------------------
# Task implementations
# ---------------------------------------------------------------------------

def _fetch_financial_news(**ctx) -> str:
    """
    Task 1 — Fetch recent financial news articles from NewsAPI.

    Reads NEWS_API_KEY from Airflow Variables.
    Returns path (str) of a JSON file containing the raw articles.
    """
    import requests
    from pathlib import Path

    api_key = Variable.get("NEWS_API_KEY", default_var=os.getenv("NEWS_API_KEY", ""))
    if not api_key:
        raise ValueError(
            "NEWS_API_KEY not set. Add it via Airflow UI → Admin → Variables."
        )

    articles: list[dict[str, Any]] = []
    base_url = "https://newsapi.org/v2/everything"

    for query in NEWS_QUERIES:
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 20,         # max 100 on free plan
            "from": (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%S"),
            "apiKey": api_key,
        }
        resp = requests.get(base_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for art in data.get("articles", []):
            if art.get("title") and art.get("description"):
                articles.append({
                    "id": hashlib.md5(art["url"].encode()).hexdigest(),
                    "title": art["title"],
                    "description": art.get("description", ""),
                    "content": art.get("content", ""),
                    "source": art.get("source", {}).get("name", "unknown"),
                    "published_at": art.get("publishedAt", ""),
                    "url": art.get("url", ""),
                    "query": query,
                })

    logger.info("[fetch_news] Collected %d articles across %d queries", len(articles), len(NEWS_QUERIES))

    out_dir = Path(os.getenv("PYTHONPATH", "/opt/airflow/project")) / "artifacts" / "data" / "news"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"news_{ctx['ds_nodash']}_{ctx['ts_nodash']}.json"
    out_path.write_text(json.dumps(articles, ensure_ascii=False, indent=2))

    return str(out_path)


def _chunk_and_embed(**ctx) -> str:
    """
    Task 2 — Chunk articles into passages and embed with sentence-transformers.

    Strategy:
      - Each article produces 1–3 chunks: title, description, content (if long enough).
      - Embeddings computed locally with 'all-MiniLM-L6-v2' (384 dims).

    Returns:
        Path (str) of a JSON file with chunks + embeddings (as lists).
    """
    import json
    from pathlib import Path
    from sentence_transformers import SentenceTransformer

    raw_path: str = ctx["ti"].xcom_pull(task_ids="fetch_financial_news")
    articles: list[dict] = json.loads(Path(raw_path).read_text())

    if not articles:
        logger.warning("[embed] No articles to embed — skipping.")
        return ""

    model = SentenceTransformer(EMBED_MODEL)
    chunks: list[dict] = []

    for art in articles:
        # Build text passages to embed
        passages = {
            "title": art["title"].strip(),
        }
        if art["description"] and len(art["description"]) > 30:
            passages["description"] = art["description"].strip()
        if art["content"] and len(art["content"]) > 80:
            # Truncate very long content to 512 chars to stay within model limits
            passages["content"] = art["content"][:512].strip()

        for chunk_type, text in passages.items():
            chunk_id = f"{art['id']}_{chunk_type}"
            chunks.append({
                "id": chunk_id,
                "text": text,
                "metadata": {
                    "article_id": art["id"],
                    "chunk_type": chunk_type,
                    "source": art["source"],
                    "published_at": art["published_at"],
                    "url": art["url"],
                    "query": art["query"],
                },
            })

    # Batch encode for performance
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)

    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()

    out_dir = Path(raw_path).parent
    out_path = out_dir / f"embedded_{ctx['ds_nodash']}_{ctx['ts_nodash']}.json"
    out_path.write_text(json.dumps(chunks, ensure_ascii=False))
    logger.info("[embed] Embedded %d chunks (%d articles)", len(chunks), len(articles))

    return str(out_path)


def _update_chromadb(**ctx) -> dict:
    """
    Task 3 — Upsert embedded chunks into ChromaDB.

    Uses 'upsert' so re-runs are idempotent (same chunk IDs → overwrite).

    Returns:
        dict with collection name and count of upserted documents.
    """
    import json
    from pathlib import Path
    import chromadb

    embedded_path: str = ctx["ti"].xcom_pull(task_ids="chunk_and_embed")
    if not embedded_path:
        logger.warning("[chromadb] No embedded path received — nothing to upsert.")
        return {"upserted": 0}

    chunks: list[dict] = json.loads(Path(embedded_path).read_text())
    if not chunks:
        return {"upserted": 0}

    # Connect to ChromaDB (HTTP client → container)
    client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Upsert in batches of 100 (ChromaDB recommendation)
    batch_size = 100
    total_upserted = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.upsert(
            ids=[c["id"] for c in batch],
            embeddings=[c["embedding"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )
        total_upserted += len(batch)

    logger.info(
        "[chromadb] Upserted %d documents into collection '%s'. Total in collection: %d",
        total_upserted, COLLECTION_NAME, collection.count(),
    )
    return {"collection": COLLECTION_NAME, "upserted": total_upserted, "total": collection.count()}


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="dag_rag_update",
    description="Hourly: fetch financial news, embed with HuggingFace, update ChromaDB",
    schedule="30 14-22 * * 1-5",           # HIGH-3: schedule_interval deprecated in 2.4+
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["rag", "nlp", "chromadb", "news"],
    default_args=DEFAULT_ARGS,
    doc_md=__doc__,
) as dag:

    t1_fetch = PythonOperator(
        task_id="fetch_financial_news",
        python_callable=_fetch_financial_news,
        doc_md="Fetch S&P 500 related news articles from NewsAPI.",
    )

    t2_embed = PythonOperator(
        task_id="chunk_and_embed",
        python_callable=_chunk_and_embed,
        doc_md="Chunk articles and embed with sentence-transformers (all-MiniLM-L6-v2).",
    )

    t3_chroma = PythonOperator(
        task_id="update_chromadb",
        python_callable=_update_chromadb,
        doc_md="Upsert embeddings into ChromaDB 'financial_news' collection.",
    )

    t1_fetch >> t2_embed >> t3_chroma
