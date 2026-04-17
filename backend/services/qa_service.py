from __future__ import annotations

import os
import re
import requests

from openai import OpenAI

from backend.services.rag_service import retrieve_context


def _openai_enabled() -> bool:
    return os.getenv("USE_OPENAI", "true").strip().lower() in {"1", "true", "yes", "on"}


def _local_llm_enabled() -> bool:
    return os.getenv("USE_LOCAL_LLM", "true").strip().lower() in {"1", "true", "yes", "on"}


def _ollama_answer(question: str, context: str) -> str | None:
    if not _local_llm_enabled():
        return None

    model = os.getenv("OLLAMA_MODEL", "llama3.2:1b").strip() or "llama3.2:1b"
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    prompt = (
        "Answer the question only using the context below. "
        "If context is insufficient, say so briefly.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context[:12000]}"
    )

    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        if response.status_code != 200:
            return None
        payload = response.json()
        text = payload.get("response", "")
        if isinstance(text, str) and text.strip():
            return text.strip()
    except Exception:
        return None
    return None


def _looks_insufficient(answer: str) -> bool:
    text = (answer or "").strip().lower()
    if not text:
        return True
    weak_patterns = [
        "insufficient context",
        "insufficient information",
        "does not provide sufficient information",
        "i can't answer",
        "i cannot answer",
        "not enough context",
        "not enough information",
        "cannot determine",
        "don't have enough information",
        "i'm sorry, but i cannot",
        "sorry, but i cannot",
        "i'm sorry, but i can't",
        "cannot provide a response",
        "unable to provide",
        "cannot determine from context",
        "context does not mention",
        "context does not provide",
        "is not mentioned in",
        "is not specified in",
    ]
    if any(p in text for p in weak_patterns):
        return True
    # Also flag verbose deflections that are likely insufficient
    if len(text) > 300 and ("sorry" in text or "however" in text or "but" in text) and len(text.split()) > 50:
        return True
    return False


def _extractive_answer(question: str, hits: list[dict]) -> str:
    """Generate factual answer from top retrieved chunks with keyword scoring."""
    if not hits:
        return "No relevant context found for this question."

    # Extract key keywords from question (skip stop words)
    tokens = [t for t in re.findall(r"[a-zA-Z]{4,}", question.lower()) 
              if t not in {"what", "which", "where", "when", "why", "how", "this", "that", "from", "with"}]
    
    # Combine top chunks
    top_text = "\n".join([h.get("text", "") for h in hits[:3]])
    
    # Split into sentences
    sents = re.split(r"(?<=[.!?])\s+", top_text)
    
    # Score and rank sentences by keyword match
    scored: list[tuple[int, str]] = []
    for s in sents:
        ls = s.lower()
        score = sum(1 for t in tokens if t in ls)
        if s.strip():
            scored.append((score, s.strip()))
    
    # Sort by relevance (keyword matches)
    scored.sort(key=lambda x: (-x[0], -len(x[1].split())))
    
    # Build answer from top 2-3 relevant sentences
    best = [s for score, s in scored if s and score > 0][:3]
    if not best:
        best = [s for score, s in scored if s][:2]
    
    joined = " ".join(best).strip()
    if not joined:
        joined = top_text[:400].strip()
    
    return joined if len(joined) > 20 else f"Topic mentioned in context: {top_text[:300]}"


def answer_question(question: str, submission_id: int | None = None) -> dict:
    hits = retrieve_context(query=question, submission_id=submission_id, top_k=5)
    context = "\n\n".join([h["text"] for h in hits])
    if not context.strip():
        return {
            "answer": "No relevant context found. Please review indexed papers first.",
            "citations": [],
            "mode": "rag-fallback",
            "mode_reason": "no_context",
        }

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    local_answer = _ollama_answer(question=question, context=context)
    if local_answer:
        if _looks_insufficient(local_answer):
            local_answer = _extractive_answer(question=question, hits=hits)
        return {
            "answer": local_answer,
            "citations": [h["metadata"] for h in hits],
            "mode": "rag-local-llm",
            "mode_reason": "ollama",
        }

    if not _openai_enabled():
        return {
            "answer": f"Based on retrieved context: {context[:500]}",
            "citations": [h["metadata"] for h in hits],
            "mode": "rag-fallback",
            "mode_reason": "demo_mode_openai_disabled",
        }

    if not api_key:
        return {
            "answer": f"Based on retrieved context: {context[:500]}",
            "citations": [h["metadata"] for h in hits],
            "mode": "rag-fallback",
            "mode_reason": "missing_api_key",
        }

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("MODEL_PRIMARY", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "Answer only from provided context. Be concise."},
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nContext:\n{context[:12000]}",
                },
            ],
        )
        return {
            "answer": response.choices[0].message.content.strip(),
            "citations": [h["metadata"] for h in hits],
            "mode": "rag-openai",
        }
    except Exception:
        reason = "openai_request_failed"
        try:
            import traceback
            reason = traceback.format_exc().splitlines()[-1][:240]
        except Exception:
            pass
        return {
            "answer": f"Based on retrieved context: {context[:500]}",
            "citations": [h["metadata"] for h in hits],
            "mode": "rag-fallback",
            "mode_reason": reason,
        }
