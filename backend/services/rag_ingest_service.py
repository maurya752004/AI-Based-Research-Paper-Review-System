from __future__ import annotations

from backend.services.rag_service import index_submission


def ingest_submission_to_rag(submission_id: int, title: str, source_text: str) -> int:
    if not source_text.strip():
        return 0
    return index_submission(submission_id=submission_id, title=title, full_text=source_text)
