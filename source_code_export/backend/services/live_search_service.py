from __future__ import annotations

import requests

from backend.services.feed_service import fetch_arxiv_papers


def search_semantic_scholar(topic: str, limit: int = 10) -> list[dict]:
    """Fetch live paper metadata from Semantic Scholar Graph API."""
    query = (topic or "").strip()
    if not query:
        return []

    bounded_limit = max(1, min(limit, 20))
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": bounded_limit,
        "fields": "title,abstract,year,url,authors",
    }

    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        # Service-friendly fallback for demo reliability.
        fallback_query = f"all:{query}"
        arxiv_items = fetch_arxiv_papers(query=fallback_query, max_results=bounded_limit)
        return [
            {
                "title": item.get("title", "Untitled"),
                "summary": item.get("summary", "No abstract available."),
                "url": item.get("url", ""),
                "published_at": item.get("published_at"),
                "source": "arxiv_fallback",
                "authors": [],
            }
            for item in arxiv_items
        ]

    items: list[dict] = []
    for row in payload.get("data", []):
        authors = [a.get("name", "") for a in row.get("authors", []) if a.get("name")]
        items.append(
            {
                "title": row.get("title", "Untitled"),
                "summary": row.get("abstract") or "No abstract available.",
                "url": row.get("url") or "",
                "published_at": str(row.get("year") or ""),
                "source": "semantic_scholar",
                "authors": authors,
            }
        )
    return items
