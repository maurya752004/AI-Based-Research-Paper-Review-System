from __future__ import annotations

import os
import re
import shutil
import time
from collections import Counter

import chromadb
from chromadb.api.types import EmbeddingFunction


def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        chunks.append(clean[start:end])
        if end == len(clean):
            break
        start = max(0, end - overlap)
    return chunks


class HashEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: list[str]) -> list[list[float]]:
        dims = 256
        out: list[list[float]] = []
        for text in input:
            vec = [0.0] * dims
            tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
            counts = Counter(tokens)
            norm = max(1.0, float(sum(counts.values())))
            for tok, cnt in counts.items():
                idx = hash(tok) % dims
                vec[idx] += cnt / norm
            out.append(vec)
        return out


RAG_DIR = os.getenv("RAG_DB_DIR", "./rag_store")


def _init_collection():
    try:
        client = chromadb.PersistentClient(path=RAG_DIR)
        collection = client.get_or_create_collection(
            name="paper_chunks",
            embedding_function=HashEmbeddingFunction(),
        )
        return client, collection
    except BaseException:
        # Recover from local Chroma storage corruption by rotating the directory.
        if os.path.exists(RAG_DIR):
            backup_dir = f"{RAG_DIR}_corrupt_{int(time.time())}"
            shutil.move(RAG_DIR, backup_dir)
        try:
            client = chromadb.PersistentClient(path=RAG_DIR)
            collection = client.get_or_create_collection(
                name="paper_chunks",
                embedding_function=HashEmbeddingFunction(),
            )
            return client, collection
        except BaseException:
            # Last resort: in-memory client keeps app alive for demo usage.
            client = chromadb.Client()
            collection = client.get_or_create_collection(
                name="paper_chunks",
                embedding_function=HashEmbeddingFunction(),
            )
            return client, collection


_client, _collection = _init_collection()


def index_submission(submission_id: int, title: str, full_text: str) -> int:
    chunks = _chunk_text(full_text)
    if not chunks:
        return 0

    ids = [f"{submission_id}-{i}" for i in range(len(chunks))]
    metadatas = [{"submission_id": submission_id, "title": title, "chunk_index": i} for i in range(len(chunks))]
    _collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
    return len(chunks)


def retrieve_context(query: str, submission_id: int | None = None, top_k: int = 5) -> list[dict]:
    where = {"submission_id": submission_id} if submission_id is not None else None
    result = _collection.query(query_texts=[query], n_results=top_k, where=where)

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]
    items: list[dict] = []
    for doc, meta, dist in zip(docs, metas, dists):
        items.append(
            {
                "text": doc,
                "metadata": meta,
                "distance": float(dist) if dist is not None else 0.0,
            }
        )
    return items
