"""Utilities for automated research paper analysis."""

from __future__ import annotations

import re
from typing import Dict, List, TypedDict

SCORE_WEIGHT = 1.2


class ScoreBreakdown(TypedDict):
    novelty: float
    methodology: float
    clarity: float
    impact: float
    overall: float


class PaperAnalysis(TypedDict):
    title: str
    summary: str
    strengths: List[str]
    weaknesses: List[str]
    scores: ScoreBreakdown


class RankedPaperAnalysis(PaperAnalysis):
    rank: int


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase.lower())}(?!\w)", text.lower()) is not None


def _contains_any_phrase(text: str, phrases: List[str]) -> bool:
    return any(_contains_phrase(text, phrase) for phrase in phrases)


def generate_summary(text: str, max_sentences: int = 2) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return "No content available to summarize."
    return " ".join(sentences[:max_sentences])


def identify_strengths_weaknesses(text: str) -> Dict[str, List[str]]:
    strengths: List[str] = []
    weaknesses: List[str] = []

    if _contains_any_phrase(text, ["novel", "innovative", "original"]):
        strengths.append("Demonstrates novelty or innovation.")
    if _contains_any_phrase(text, ["dataset", "experiment", "evaluation", "benchmark"]):
        strengths.append("Provides empirical validation.")
    if _contains_any_phrase(text, ["clear", "well-structured", "organized", "readable"]):
        strengths.append("Presents ideas clearly.")

    if _contains_any_phrase(text, ["small sample", "limited dataset", "insufficient data"]):
        weaknesses.append("Uses limited data for validation.")
    if _contains_any_phrase(text, ["unclear", "ambiguous", "poorly explained"]):
        weaknesses.append("Contains unclear or ambiguous explanations.")
    if _contains_any_phrase(text, ["future work", "not implemented", "missing baseline"]):
        weaknesses.append("Leaves important implementation or comparison gaps.")

    if not strengths:
        strengths.append("Addresses a relevant research problem.")
    if not weaknesses:
        weaknesses.append("Could further strengthen experimental depth.")

    return {"strengths": strengths, "weaknesses": weaknesses}


def _score_category(text: str, positive_words: List[str], negative_words: List[str]) -> float:
    score = 5.0
    score += sum(1 for word in positive_words if _contains_phrase(text, word)) * SCORE_WEIGHT
    score -= sum(1 for word in negative_words if _contains_phrase(text, word)) * SCORE_WEIGHT
    return max(1.0, min(10.0, round(score, 2)))


def score_paper(text: str) -> ScoreBreakdown:
    novelty = _score_category(
        text,
        positive_words=["novel", "innovative", "original", "state-of-the-art"],
        negative_words=["incremental", "limited novelty", "repetitive"],
    )
    methodology = _score_category(
        text,
        positive_words=["rigorous", "robust", "benchmark", "ablation", "evaluation"],
        negative_words=["weak methodology", "missing baseline", "insufficient data"],
    )
    clarity = _score_category(
        text,
        positive_words=["clear", "organized", "well-structured", "readable"],
        negative_words=["unclear", "ambiguous", "poorly explained"],
    )
    impact = _score_category(
        text,
        positive_words=["impact", "practical", "real-world", "scalable"],
        negative_words=["narrow scope", "limited impact"],
    )

    overall = round((novelty + methodology + clarity + impact) / 4, 2)
    return {
        "novelty": novelty,
        "methodology": methodology,
        "clarity": clarity,
        "impact": impact,
        "overall": overall,
    }


def analyze_paper(title: str, text: str) -> PaperAnalysis:
    insights = identify_strengths_weaknesses(text)
    scores = score_paper(text)
    return {
        "title": title,
        "summary": generate_summary(text),
        "strengths": insights["strengths"],
        "weaknesses": insights["weaknesses"],
        "scores": scores,
    }


def compare_papers(papers: List[Dict[str, str]]) -> List[RankedPaperAnalysis]:
    analyses = [analyze_paper(paper["title"], paper["text"]) for paper in papers]
    ranked = sorted(analyses, key=lambda item: item["scores"]["overall"], reverse=True)

    for index, paper in enumerate(ranked, start=1):
        paper["rank"] = index

    return ranked
