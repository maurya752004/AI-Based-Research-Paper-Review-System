from __future__ import annotations


def rank_submissions(items: list[dict]) -> list[dict]:
    ranked: list[dict] = []
    for item in items:
        ai_score = float(item.get("score", 0.0))
        citation_quality = float(item.get("citation_metrics", {}).get("reference_quality_score", 0.0))
        novelty_score = float(item.get("novelty_score", 0.0))
        confidence = float(item.get("confidence_score", 0.0))
        final_score = round(0.45 * ai_score + 0.2 * citation_quality + 0.2 * novelty_score + 0.15 * confidence, 3)
        ranked.append(
            {
                "submission_id": item.get("id"),
                "title": item.get("title"),
                "owner": item.get("submitted_by"),
                "ai_score": ai_score,
                "citation_quality": citation_quality,
                "novelty_score": novelty_score,
                "confidence_score": confidence,
                "ranking_score": final_score,
            }
        )

    ranked.sort(key=lambda x: x["ranking_score"], reverse=True)
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i
    return ranked
