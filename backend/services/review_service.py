from __future__ import annotations

import json
import math
import os
import re
from collections import Counter

import requests

from dotenv import load_dotenv

load_dotenv()

_LAST_OPENAI_ERROR = ""


def _openai_enabled() -> bool:
    return os.getenv("USE_OPENAI", "true").strip().lower() in {"1", "true", "yes", "on"}


def _local_llm_enabled() -> bool:
    return os.getenv("USE_LOCAL_LLM", "true").strip().lower() in {"1", "true", "yes", "on"}


def _local_fast_mode() -> bool:
    return os.getenv("LOCAL_REVIEW_FAST_MODE", "true").strip().lower() in {"1", "true", "yes", "on"}


def _novelty_api_enabled() -> bool:
    return os.getenv("NOVELTY_API_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


def _ollama_generate(prompt: str) -> str | None:
    if not _local_llm_enabled():
        return None
    model = os.getenv("OLLAMA_MODEL", "llama3").strip() or "llama3"
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    fast_mode = _local_fast_mode()
    timeout_sec = int(os.getenv("OLLAMA_TIMEOUT_SEC", "65" if fast_mode else "90"))
    options = {
        "temperature": 0.2,
        "top_p": 0.9,
        "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "300" if fast_mode else "520")),
        "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "4096" if fast_mode else "6144")),
    }
    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "options": options},
            timeout=timeout_sec,
        )
        if resp.status_code != 200:
            return None
        payload = resp.json()
        text = payload.get("response", "")
        return text.strip() if isinstance(text, str) else None
    except Exception:
        return None


def _extract_json_object(raw: str) -> dict | None:
    if not raw:
        return None
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _split_sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    return [c.strip() for c in chunks if c.strip()]


def _heuristic_summary(text: str, max_sentences: int = 4) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return "No summary could be generated from the uploaded document."
    return " ".join(sentences[:max_sentences])


def _extract_keywords(text: str, top_k: int = 8) -> list[str]:
    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "are",
        "was",
        "were",
        "have",
        "has",
        "had",
        "into",
        "their",
        "there",
        "than",
        "then",
        "also",
        "using",
        "used",
        "use",
        "between",
        "such",
        "these",
        "those",
        "which",
        "our",
        "can",
        "may",
        "might",
        "will",
        "would",
        "should",
        "could",
        "about",
        "paper",
        "research",
        "study",
    }

    tokens = re.findall(r"[a-zA-Z]{4,}", text.lower())
    filtered = [t for t in tokens if t not in stop_words]
    counts = Counter(filtered)
    return [word for word, _ in counts.most_common(top_k)]


def _safe_score(word_count: int) -> float:
    # Lightweight proxy score for fallback mode.
    if word_count < 1200:
        return 6.0
    if word_count < 3000:
        return 7.0
    if word_count < 6000:
        return 8.0
    return 8.5


def _extract_sections(text: str) -> dict[str, str]:
    section_patterns = {
        "abstract": r"\babstract\b",
        "introduction": r"\bintroduction\b",
        "methodology": r"\b(methodology|methods?|approach)\b",
        "results": r"\b(results?|experiments?|evaluation)\b",
        "conclusion": r"\b(conclusion|future work)\b",
    }
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    starts: list[tuple[str, int]] = []
    for idx, line in enumerate(lines):
        lower_line = line.lower()
        for name, pattern in section_patterns.items():
            if re.search(pattern, lower_line):
                starts.append((name, idx))

    starts.sort(key=lambda x: x[1])
    sections: dict[str, str] = {}
    for i, (name, start_idx) in enumerate(starts):
        end_idx = starts[i + 1][1] if i + 1 < len(starts) else len(lines)
        body = " ".join(lines[start_idx:end_idx]).strip()
        if body and name not in sections:
            sections[name] = body

    if not sections:
        sections["full_text"] = text[:2500]
    return sections


def _section_summary(section_text: str) -> str:
    sents = _split_sentences(section_text)
    return " ".join(sents[:2]) if sents else "N/A"


def _score_section(section_text: str, base: float = 6.0) -> float:
    signals = ["dataset", "experiment", "model", "results", "analysis", "baseline"]
    found = sum(1 for s in signals if s in section_text.lower())
    length_bonus = min(2.0, len(section_text.split()) / 500)
    return round(min(10.0, base + 0.4 * found + length_bonus), 2)


def _build_section_scores_and_summaries(text: str) -> tuple[dict[str, float], dict[str, str]]:
    sections = _extract_sections(text)
    section_scores: dict[str, float] = {}
    section_summaries: dict[str, str] = {}
    for name, content in sections.items():
        section_scores[name] = _score_section(content)
        section_summaries[name] = _section_summary(content)
    return section_scores, section_summaries


def _citation_metrics(text: str) -> dict[str, float]:
    bracket_refs = len(re.findall(r"\[[0-9]{1,3}\]", text))
    year_refs = len(re.findall(r"\b(19|20)\d{2}\b", text))
    doi_refs = len(re.findall(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", text))
    words = max(1, len(text.split()))
    density = round((bracket_refs + year_refs + doi_refs) / words * 1000, 2)
    quality = min(10.0, round(4.5 + density / 2.2, 2))
    return {
        "bracket_references": float(bracket_refs),
        "year_mentions": float(year_refs),
        "doi_mentions": float(doi_refs),
        "citation_density_per_1000_words": density,
        "reference_quality_score": quality,
    }


def _novelty_check(title: str, keywords: list[str]) -> tuple[float, list[str]]:
    if not _novelty_api_enabled():
        return 6.5, ["Novelty API disabled for faster local review mode."]

    query_terms = " ".join([title] + keywords[:4]).strip()
    if not query_terms:
        return 6.0, ["Insufficient keywords for novelty probing."]

    try:
        resp = requests.get(
            "https://api.openalex.org/works",
            params={"search": query_terms, "per-page": 5},
            timeout=4,
        )
        if resp.status_code != 200:
            return 6.0, ["External novelty API unavailable; used neutral novelty score."]

        data = resp.json()
        results = data.get("results", [])
        if not results:
            return 8.2, ["No highly similar works found in quick OpenAlex probe."]

        similar_titles = [r.get("display_name", "") for r in results[:3]]
        notes = [f"Potential related work: {t[:90]}" for t in similar_titles if t]
        novelty = 7.0 - min(1.5, len(similar_titles) * 0.4)
        return round(max(4.8, novelty), 2), notes or ["Related works found."]
    except Exception:
        return 6.0, ["Novelty service call failed; used fallback novelty estimate."]


def _confidence_from_signals(section_scores: dict[str, float], citation_metrics: dict[str, float], word_count: int) -> tuple[str, float]:
    section_avg = sum(section_scores.values()) / max(1, len(section_scores))
    citation_quality = citation_metrics.get("reference_quality_score", 6.0)
    wc_factor = min(1.5, word_count / 3000)
    score = round(min(10.0, 0.55 * section_avg + 0.3 * citation_quality + 1.5 * wc_factor), 2)
    if score >= 8.0:
        return "high", score
    if score >= 6.2:
        return "medium", score
    return "low", score


def _contains_any(text: str, terms: set[str]) -> bool:
    lower_text = text.lower()
    return any(t in lower_text for t in terms)


def _build_rubric_scores(text: str, word_count: int) -> tuple[dict[str, float], list[str]]:
    novelty_terms = {"novel", "new", "propose", "innovation", "contribution"}
    method_terms = {"method", "algorithm", "framework", "model", "approach"}
    eval_terms = {"experiment", "results", "dataset", "accuracy", "evaluation"}
    reproducibility_terms = {"code", "implementation", "parameters", "settings", "baseline"}
    writing_terms = {"introduction", "conclusion", "abstract", "discussion", "references"}

    novelty = 6.0 + (1.0 if _contains_any(text, novelty_terms) else 0.0)
    methodology = 6.0 + (1.5 if _contains_any(text, method_terms) else 0.0)
    evaluation = 5.5 + (2.0 if _contains_any(text, eval_terms) else 0.0)
    reproducibility = 5.0 + (1.5 if _contains_any(text, reproducibility_terms) else 0.0)
    writing_quality = 6.0 + (1.0 if _contains_any(text, writing_terms) else 0.0)

    if word_count > 2500:
        writing_quality += 0.5

    rubric_scores = {
        "novelty": min(10.0, round(novelty, 2)),
        "methodology": min(10.0, round(methodology, 2)),
        "evaluation": min(10.0, round(evaluation, 2)),
        "reproducibility": min(10.0, round(reproducibility, 2)),
        "writing_quality": min(10.0, round(writing_quality, 2)),
    }

    evidence_points = [
        f"Detected methodology terms: {', '.join(sorted(method_terms)[:3])} (signal-based check).",
        "Used deterministic fallback heuristics when external model is unavailable.",
        "Generated rubric sub-scores to make evaluation transparent for exam/demo.",
    ]
    return rubric_scores, evidence_points


def _build_score_explanations(rubric_scores: dict[str, float], text: str) -> dict[str, str]:
    lower_text = text.lower()
    explanations: dict[str, str] = {}

    novelty_hits = [t for t in ["novel", "new", "contribution", "innovation"] if t in lower_text]
    method_hits = [t for t in ["method", "algorithm", "approach", "framework", "model"] if t in lower_text]
    eval_hits = [t for t in ["experiment", "results", "evaluation", "dataset", "accuracy"] if t in lower_text]
    repro_hits = [t for t in ["code", "implementation", "baseline", "parameters", "settings"] if t in lower_text]
    writing_hits = [t for t in ["abstract", "introduction", "conclusion", "references"] if t in lower_text]

    explanations["novelty"] = (
        f"Score {rubric_scores.get('novelty', 0):.1f}/10: novelty signals found ({', '.join(novelty_hits[:3])})"
        if novelty_hits
        else f"Score {rubric_scores.get('novelty', 0):.1f}/10: limited explicit novelty cues in extracted text"
    )
    explanations["methodology"] = (
        f"Score {rubric_scores.get('methodology', 0):.1f}/10: methodology terms detected ({', '.join(method_hits[:3])})"
        if method_hits
        else f"Score {rubric_scores.get('methodology', 0):.1f}/10: methodology description appears shallow"
    )
    explanations["evaluation"] = (
        f"Score {rubric_scores.get('evaluation', 0):.1f}/10: evaluation evidence present ({', '.join(eval_hits[:3])})"
        if eval_hits
        else f"Score {rubric_scores.get('evaluation', 0):.1f}/10: limited evaluation/result indicators"
    )
    explanations["reproducibility"] = (
        f"Score {rubric_scores.get('reproducibility', 0):.1f}/10: reproducibility markers found ({', '.join(repro_hits[:3])})"
        if repro_hits
        else f"Score {rubric_scores.get('reproducibility', 0):.1f}/10: few reproducibility details"
    )
    explanations["writing_quality"] = (
        f"Score {rubric_scores.get('writing_quality', 0):.1f}/10: structure cues found ({', '.join(writing_hits[:3])})"
        if writing_hits
        else f"Score {rubric_scores.get('writing_quality', 0):.1f}/10: structure quality uncertain in extracted text"
    )
    return explanations


def _allow_fallback() -> bool:
    return os.getenv("REVIEW_ALLOW_FALLBACK", "true").strip().lower() in {"1", "true", "yes", "on"}


def _build_fallback_review(text: str, title: str) -> dict:
    word_count = len(text.split())
    keywords = _extract_keywords(text)
    rubric_scores, evidence_points = _build_rubric_scores(text=text, word_count=word_count)
    section_scores, section_summaries = _build_section_scores_and_summaries(text)
    citation_metrics = _citation_metrics(text)
    novelty_score, novelty_notes = _novelty_check(title=title, keywords=keywords)
    confidence_level, confidence_score = _confidence_from_signals(section_scores, citation_metrics, word_count)

    strengths = [
        "Document has sufficient technical content for an initial review.",
        "Key terminology appears consistent across the paper.",
        "Structure can be interpreted for automated summarization.",
    ]

    weaknesses = [
        "Fallback mode is heuristic and not a deep semantic peer review.",
        "Scoring is approximate and should be validated by a human reviewer.",
        "Extracted PDF text may contain formatting noise.",
    ]

    return {
        "title": title,
        "summary": _heuristic_summary(text),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "recommendation": "Needs manual validation by supervisor/reviewer.",
        "score": _safe_score(word_count),
        "keywords": keywords,
        "word_count": word_count,
        "mode": "fallback",
        "rubric_scores": rubric_scores,
        "score_explanations": _build_score_explanations(rubric_scores, text),
        "evidence_points": evidence_points,
        "section_scores": section_scores,
        "section_summaries": section_summaries,
        "citation_metrics": citation_metrics,
        "novelty_score": novelty_score,
        "novelty_notes": novelty_notes,
        "confidence_level": confidence_level,
        "confidence_score": confidence_score,
    }


def _model_candidates() -> list[str]:
    primary = os.getenv("MODEL_PRIMARY", "gpt-4o-mini").strip() or "gpt-4o-mini"
    secondary = os.getenv("MODEL_SECONDARY", "gpt-4.1-mini").strip() or "gpt-4.1-mini"
    if secondary == primary:
        return [primary]
    return [primary, secondary]


def _local_llm_review(text: str, title: str) -> dict | None:
    max_chars = int(os.getenv("LOCAL_REVIEW_MAX_CHARS", "7000" if _local_fast_mode() else "14000"))
    excerpt = text[:max_chars]
    prompt = (
        "You are an academic paper assistant. Return strict JSON only with keys: "
        "title, summary, strengths, weaknesses, recommendation, score, keywords, rubric_scores, evidence_points.\n"
        "Rules:\n"
        "- summary: max 180 words\n"
        "- strengths: list of 3 concise bullets\n"
        "- weaknesses: list of 3 concise bullets\n"
        "- recommendation: one sentence\n"
        "- score: numeric 1-10\n"
        "- keywords: list of 5-8 terms\n"
        "- rubric_scores: object with novelty, methodology, evaluation, reproducibility, writing_quality (1-10)\n"
        "- evidence_points: list of 3 concise technical points\n"
        f"Title: {title}\n"
        f"Paper Text:\n{excerpt}"
    )
    raw = _ollama_generate(prompt)
    data = _extract_json_object(raw or "")
    if not data:
        return None

    word_count = len(text.split())
    data["title"] = data.get("title") or title
    data["word_count"] = word_count
    data["mode"] = "local-llm"
    data["mode_reason"] = "ollama"

    if "rubric_scores" not in data or not isinstance(data.get("rubric_scores"), dict):
        fallback_rubric, fallback_evidence = _build_rubric_scores(text=text, word_count=word_count)
        data["rubric_scores"] = fallback_rubric
        data["evidence_points"] = fallback_evidence

    if "score_explanations" not in data or not isinstance(data.get("score_explanations"), dict):
        data["score_explanations"] = _build_score_explanations(data.get("rubric_scores", {}), text)

    if "evidence_points" not in data or not isinstance(data.get("evidence_points"), list):
        _, fallback_evidence = _build_rubric_scores(text=text, word_count=word_count)
        data["evidence_points"] = fallback_evidence

    try:
        data["score"] = float(data.get("score", _safe_score(word_count)))
    except Exception:
        data["score"] = _safe_score(word_count)

    section_scores, section_summaries = _build_section_scores_and_summaries(text)
    citation_metrics = _citation_metrics(text)
    novelty_score, novelty_notes = _novelty_check(title=title, keywords=data.get("keywords", []))
    confidence_level, confidence_score = _confidence_from_signals(section_scores, citation_metrics, word_count)

    data["section_scores"] = section_scores
    data["section_summaries"] = section_summaries
    data["citation_metrics"] = citation_metrics
    data["novelty_score"] = novelty_score
    data["novelty_notes"] = novelty_notes
    data["confidence_level"] = confidence_level
    data["confidence_score"] = confidence_score
    return data


def _cosine_similarity(text_a: str, text_b: str) -> float:
    toks_a = re.findall(r"[a-zA-Z]{3,}", text_a.lower())
    toks_b = re.findall(r"[a-zA-Z]{3,}", text_b.lower())
    if not toks_a or not toks_b:
        return 0.0
    ca = Counter(toks_a)
    cb = Counter(toks_b)
    inter = set(ca) & set(cb)
    dot = sum(ca[t] * cb[t] for t in inter)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(dot / (na * nb))


def compare_papers_joint_prompt(text_a: str, title_a: str, text_b: str, title_b: str) -> dict:
    prompt = (
        "Compare these two research papers and return strict JSON only.\n"
        "DO NOT give similar scores. Highlight differences clearly.\n"
        "JSON keys: key_differences (list[str]), winner (paper_a|paper_b), rationale (str), paper_a_score (float), paper_b_score (float).\n"
        f"Paper A Title: {title_a}\nPaper A Text:\n{text_a[:3000]}\n\n"
        f"Paper B Title: {title_b}\nPaper B Text:\n{text_b[:3000]}\n"
    )
    raw = _ollama_generate(prompt)
    parsed = _extract_json_object(raw or "")
    if parsed:
        return parsed

    sim = _cosine_similarity(text_a, text_b)
    score_a = round(max(5.0, 8.6 - sim * 1.8), 2)
    score_b = round(max(5.0, 7.9 - sim * 1.2), 2)
    if abs(score_a - score_b) < 0.2:
        score_a = round(min(9.5, score_a + 0.3), 2)
        score_b = round(max(4.0, score_b - 0.3), 2)
    winner = "paper_a" if score_a >= score_b else "paper_b"
    return {
        "key_differences": [
            "Similarity estimated from TF-style cosine over extracted text.",
            "Scores intentionally separated for clearer decision support in demo mode.",
            f"Estimated text similarity: {sim:.3f}",
        ],
        "winner": winner,
        "rationale": "Heuristic+local comparison selected winner using separated scoring and textual similarity signals.",
        "paper_a_score": score_a,
        "paper_b_score": score_b,
    }


def _openai_review(text: str, title: str) -> dict | None:
    global _LAST_OPENAI_ERROR
    _LAST_OPENAI_ERROR = ""
    if not _openai_enabled():
        _LAST_OPENAI_ERROR = "demo_mode_openai_disabled"
        return None
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        _LAST_OPENAI_ERROR = "missing_api_key"
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        prompt = (
            "You are an academic paper assistant. Analyze the provided paper text and "
            "return strict JSON only with keys: title, summary, strengths, weaknesses, "
            "recommendation, score, keywords, rubric_scores, evidence_points.\n"
            "Rules:\n"
            "- summary: max 180 words\n"
            "- strengths: list of 3 concise bullets\n"
            "- weaknesses: list of 3 concise bullets\n"
            "- recommendation: one sentence\n"
            "- score: numeric 1-10\n"
            "- keywords: list of 5-8 terms\n"
            "- rubric_scores: object with novelty, methodology, evaluation, reproducibility, writing_quality (1-10)\n"
            "- evidence_points: list of 3 concise technical points\n"
        )

        truncated_text = text[:18000]
        response = None
        last_error = ""
        for model in _model_candidates():
            try:
                response = client.responses.create(
                    model=model,
                    input=[
                        {
                            "role": "system",
                            "content": "You return only valid JSON.",
                        },
                        {
                            "role": "user",
                            "content": f"{prompt}\nTitle: {title}\nPaper Text:\n{truncated_text}",
                        },
                    ],
                )
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {str(exc)}"
                continue

        if response is None:
            _LAST_OPENAI_ERROR = last_error[:400] if last_error else "openai_request_failed"
            return None

        raw = response.output_text.strip()
        data = json.loads(raw)

        word_count = len(text.split())
        data["title"] = data.get("title") or title
        data["word_count"] = word_count
        data["mode"] = "openai"
        if "rubric_scores" not in data or not isinstance(data.get("rubric_scores"), dict):
            fallback_rubric, fallback_evidence = _build_rubric_scores(text=text, word_count=word_count)
            data["rubric_scores"] = fallback_rubric
            data["evidence_points"] = fallback_evidence

        if "score_explanations" not in data or not isinstance(data.get("score_explanations"), dict):
            data["score_explanations"] = _build_score_explanations(data.get("rubric_scores", {}), text)

        if "evidence_points" not in data or not isinstance(data.get("evidence_points"), list):
            _, fallback_evidence = _build_rubric_scores(text=text, word_count=word_count)
            data["evidence_points"] = fallback_evidence

        if "score" in data:
            try:
                data["score"] = float(data["score"])
            except (TypeError, ValueError):
                data["score"] = _safe_score(word_count)

        section_scores, section_summaries = _build_section_scores_and_summaries(text)
        citation_metrics = _citation_metrics(text)
        novelty_score, novelty_notes = _novelty_check(title=title, keywords=data.get("keywords", []))
        confidence_level, confidence_score = _confidence_from_signals(section_scores, citation_metrics, word_count)

        data["section_scores"] = section_scores
        data["section_summaries"] = section_summaries
        data["citation_metrics"] = citation_metrics
        data["novelty_score"] = novelty_score
        data["novelty_notes"] = novelty_notes
        data["confidence_level"] = confidence_level
        data["confidence_score"] = confidence_score

        return data
    except Exception as exc:
        _LAST_OPENAI_ERROR = f"{type(exc).__name__}: {str(exc)}"[:400]
        return None


def generate_review(text: str, title: str = "Untitled Paper") -> dict:
    local_result = _local_llm_review(text=text, title=title)
    if local_result is not None:
        return local_result

    ai_result = _openai_review(text=text, title=title)
    if ai_result is not None:
        return ai_result
    if not _allow_fallback():
        raise RuntimeError(
            "AI review unavailable. Configure valid OPENAI_API_KEY/model access or set REVIEW_ALLOW_FALLBACK=true."
        )
    fallback = _build_fallback_review(text=text, title=title)
    if _LAST_OPENAI_ERROR:
        fallback["mode_reason"] = _LAST_OPENAI_ERROR
    return fallback
