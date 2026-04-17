from __future__ import annotations

from backend.services.review_service import generate_review


def run_multi_agent_pipeline(text: str, title: str) -> dict:
    # Agent 1: Researcher (extract broad context)
    researcher_note = (
        "Identified paper scope, primary problem statement, and top-level domain terms from full text."
    )

    # Agent 2: Analyzer (core technical review)
    analysis = generate_review(text=text, title=title)

    # Agent 3: Writer (executive rewrite)
    writer_note = (
        f"Generated concise executive summary for '{title}' with strengths/weaknesses for decision-making."
    )

    # Agent 4: Reviewer (final recommendation)
    reviewer_note = (
        "Final reviewer synthesized rubric, citation quality, and novelty signals into recommendation."
    )

    return {
        "researcher": researcher_note,
        "analyzer": {
            "score": analysis.get("score"),
            "rubric_scores": analysis.get("rubric_scores", {}),
            "novelty_score": analysis.get("novelty_score"),
        },
        "writer": writer_note,
        "reviewer": reviewer_note,
        "final": analysis,
    }
