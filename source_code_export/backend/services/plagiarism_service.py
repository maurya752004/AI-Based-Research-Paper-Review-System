from __future__ import annotations

import re


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z]{4,}", (text or "").lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def check_similarity(target: dict, others: list[dict], top_k: int = 5) -> dict:
    target_tokens = _token_set(target.get("source_text", ""))
    rows: list[dict] = []

    for item in others:
        sim = _jaccard(target_tokens, _token_set(item.get("source_text", "")))
        if sim <= 0:
            continue
        rows.append(
            {
                "submission_id": item.get("id"),
                "title": item.get("title"),
                "similarity": round(sim, 4),
            }
        )

    rows.sort(key=lambda x: x["similarity"], reverse=True)
    top = rows[:top_k]
    max_sim = top[0]["similarity"] if top else 0.0
    if max_sim >= 0.55:
        risk = "high"
    elif max_sim >= 0.3:
        risk = "medium"
    else:
        risk = "low"

    return {
        "target_submission_id": target.get("id"),
        "risk_level": risk,
        "max_similarity": max_sim,
        "matches": top,
    }
