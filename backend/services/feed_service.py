from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import requests
from apscheduler.schedulers.background import BackgroundScheduler

from backend.services.db_service import add_feed_items

_scheduler: BackgroundScheduler | None = None


def fetch_arxiv_papers(query: str, max_results: int = 10) -> list[dict]:
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    items: list[dict] = []
    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()
        links = entry.findall("atom:link", ns)
        link = ""
        for l in links:
            href = l.attrib.get("href", "")
            rel = l.attrib.get("rel", "")
            if rel == "alternate" and href:
                link = href
                break
        if not link and links:
            link = links[0].attrib.get("href", "")
        items.append(
            {
                "title": title,
                "summary": summary,
                "url": link,
                "published_at": published,
            }
        )
    return items


def fetch_and_store_papers() -> int:
    query = os.getenv("PAPER_FEED_QUERY", "cat:cs.LG OR cat:cs.CL")
    max_results = int(os.getenv("PAPER_FEED_MAX_RESULTS", "10"))
    items = fetch_arxiv_papers(query=query, max_results=max_results)
    return add_feed_items(items, source="arxiv")


def start_feed_scheduler() -> None:
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        return

    _scheduler = BackgroundScheduler()
    # Run daily by default, configurable via env.
    hour = int(os.getenv("PAPER_FEED_HOUR_UTC", "6"))
    _scheduler.add_job(fetch_and_store_papers, "cron", hour=hour, minute=0)
    _scheduler.start()
