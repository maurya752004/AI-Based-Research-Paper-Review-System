from __future__ import annotations


def run_multi_agent_pipeline(review: dict) -> dict:
    researcher_output = {
        "problem_statement": "Identify contribution, methods, and limitations from paper content.",
        "key_terms": review.get("keywords", [])[:8],
    }

    analyzer_output = {
        "rubric": review.get("rubric_scores", {}),
        "section_scores": review.get("section_scores", {}),
        "confidence": {
            "level": review.get("confidence_level", "medium"),
            "score": review.get("confidence_score", 0.0),
        },
    }

    writer_output = {
        "draft": (
            f"This paper titled '{review.get('title', 'Untitled')}' demonstrates "
            f"an overall score of {review.get('score', 0)}/10 with recommendation: "
            f"{review.get('recommendation', 'N/A')}"
        )
    }

    reviewer_output = {
        "final_observation": "Human validation required for final acceptance decision.",
        "strengths": review.get("strengths", []),
        "weaknesses": review.get("weaknesses", []),
    }

    return {
        "researcher_agent": researcher_output,
        "analyzer_agent": analyzer_output,
        "writer_agent": writer_output,
        "reviewer_agent": reviewer_output,
    }
