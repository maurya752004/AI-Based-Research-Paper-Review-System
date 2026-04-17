from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "paper_reviews.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                strengths_json TEXT NOT NULL,
                weaknesses_json TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                score REAL NOT NULL,
                keywords_json TEXT NOT NULL,
                word_count INTEGER NOT NULL,
                mode TEXT NOT NULL,
                section_scores_json TEXT NOT NULL DEFAULT '{}',
                section_summaries_json TEXT NOT NULL DEFAULT '{}',
                citation_metrics_json TEXT NOT NULL DEFAULT '{}',
                novelty_score REAL NOT NULL DEFAULT 0.0,
                novelty_notes_json TEXT NOT NULL DEFAULT '[]',
                confidence_level TEXT NOT NULL DEFAULT 'medium',
                confidence_score REAL NOT NULL DEFAULT 0.0,
                submitted_by TEXT NOT NULL,
                submitted_role TEXT NOT NULL,
                human_score REAL,
                reviewer_comment TEXT,
                review_status TEXT NOT NULL DEFAULT 'ai_generated',
                source_text TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        # Apply lightweight migrations for older DB files.
        existing_cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(submissions)").fetchall()
        }
        migrations = {
            "section_scores_json": "ALTER TABLE submissions ADD COLUMN section_scores_json TEXT NOT NULL DEFAULT '{}'",
            "section_summaries_json": "ALTER TABLE submissions ADD COLUMN section_summaries_json TEXT NOT NULL DEFAULT '{}'",
            "citation_metrics_json": "ALTER TABLE submissions ADD COLUMN citation_metrics_json TEXT NOT NULL DEFAULT '{}'",
            "novelty_score": "ALTER TABLE submissions ADD COLUMN novelty_score REAL NOT NULL DEFAULT 0.0",
            "novelty_notes_json": "ALTER TABLE submissions ADD COLUMN novelty_notes_json TEXT NOT NULL DEFAULT '[]'",
            "confidence_level": "ALTER TABLE submissions ADD COLUMN confidence_level TEXT NOT NULL DEFAULT 'medium'",
            "confidence_score": "ALTER TABLE submissions ADD COLUMN confidence_score REAL NOT NULL DEFAULT 0.0",
            "human_score": "ALTER TABLE submissions ADD COLUMN human_score REAL",
            "reviewer_comment": "ALTER TABLE submissions ADD COLUMN reviewer_comment TEXT",
            "review_status": "ALTER TABLE submissions ADD COLUMN review_status TEXT NOT NULL DEFAULT 'ai_generated'",
            "source_text": "ALTER TABLE submissions ADD COLUMN source_text TEXT",
        }
        for col, sql in migrations.items():
            if col not in existing_cols:
                conn.execute(sql)

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS review_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                submitted_by TEXT NOT NULL,
                submitted_role TEXT NOT NULL,
                status TEXT NOT NULL,
                error_message TEXT,
                submission_id INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                actor_username TEXT NOT NULL,
                actor_role TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_details_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_feed (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                url TEXT NOT NULL,
                published_at TEXT,
                source TEXT NOT NULL,
                fetched_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_submission(filename: str, review: dict, submitted_by: str, submitted_role: str) -> int:
    strengths_json = json.dumps(review.get("strengths", []))
    weaknesses_json = json.dumps(review.get("weaknesses", []))
    keywords_json = json.dumps(review.get("keywords", []))
    section_scores_json = json.dumps(review.get("section_scores", {}))
    section_summaries_json = json.dumps(review.get("section_summaries", {}))
    citation_metrics_json = json.dumps(review.get("citation_metrics", {}))
    novelty_notes_json = json.dumps(review.get("novelty_notes", []))
    created_at = datetime.now(timezone.utc).isoformat()

    with _get_conn() as conn:
        cursor = conn.execute(
            """
            INSERT INTO submissions (
                filename, title, summary, strengths_json, weaknesses_json,
                recommendation, score, keywords_json, word_count, mode,
                section_scores_json, section_summaries_json, citation_metrics_json,
                novelty_score, novelty_notes_json, confidence_level, confidence_score,
                submitted_by, submitted_role, review_status, source_text, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                review.get("title", "Untitled Paper"),
                review.get("summary", ""),
                strengths_json,
                weaknesses_json,
                review.get("recommendation", ""),
                float(review.get("score", 0.0)),
                keywords_json,
                int(review.get("word_count", 0)),
                review.get("mode", "fallback"),
                section_scores_json,
                section_summaries_json,
                citation_metrics_json,
                float(review.get("novelty_score", 0.0)),
                novelty_notes_json,
                review.get("confidence_level", "medium"),
                float(review.get("confidence_score", 0.0)),
                submitted_by,
                submitted_role,
                "ai_generated",
                review.get("_raw_text", ""),
                created_at,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def list_submissions(user_role: str, username: str, limit: int = 50) -> list[dict]:
    query = "SELECT * FROM submissions ORDER BY id DESC LIMIT ?"
    params: tuple = (limit,)

    if user_role == "student":
        query = "SELECT * FROM submissions WHERE submitted_by = ? ORDER BY id DESC LIMIT ?"
        params = (username, limit)

    with _get_conn() as conn:
        rows = conn.execute(query, params).fetchall()

    items: list[dict] = []
    for row in rows:
        items.append(
            {
                "id": row["id"],
                "filename": row["filename"],
                "title": row["title"],
                "summary": row["summary"],
                "strengths": json.loads(row["strengths_json"]),
                "weaknesses": json.loads(row["weaknesses_json"]),
                "recommendation": row["recommendation"],
                "score": row["score"],
                "keywords": json.loads(row["keywords_json"]),
                "word_count": row["word_count"],
                "mode": row["mode"],
                "section_scores": json.loads(row["section_scores_json"]),
                "section_summaries": json.loads(row["section_summaries_json"]),
                "citation_metrics": json.loads(row["citation_metrics_json"]),
                "novelty_score": row["novelty_score"],
                "novelty_notes": json.loads(row["novelty_notes_json"]),
                "confidence_level": row["confidence_level"],
                "confidence_score": row["confidence_score"],
                "submitted_by": row["submitted_by"],
                "submitted_role": row["submitted_role"],
                "human_score": row["human_score"],
                "reviewer_comment": row["reviewer_comment"],
                "review_status": row["review_status"],
                "source_text": row["source_text"] or "",
                "created_at": row["created_at"],
            }
        )
    return items


def get_submission(submission_id: int, user_role: str, username: str) -> dict | None:
    with _get_conn() as conn:
        if user_role == "student":
            row = conn.execute(
                "SELECT * FROM submissions WHERE id = ? AND submitted_by = ?",
                (submission_id, username),
            ).fetchone()
        else:
            row = conn.execute("SELECT * FROM submissions WHERE id = ?", (submission_id,)).fetchone()
    if row is None:
        return None
    return {
        "id": int(row["id"]),
        "title": row["title"],
        "summary": row["summary"],
        "score": float(row["score"]),
        "keywords": json.loads(row["keywords_json"]),
        "source_text": row["source_text"] or "",
        "submitted_by": row["submitted_by"],
        "created_at": row["created_at"],
    }


def list_submission_texts(user_role: str, username: str, exclude_submission_id: int | None = None) -> list[dict]:
    query = "SELECT id, title, source_text FROM submissions"
    params: list = []
    clauses: list[str] = []
    if user_role == "student":
        clauses.append("submitted_by = ?")
        params.append(username)
    if exclude_submission_id is not None:
        clauses.append("id != ?")
        params.append(exclude_submission_id)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY id DESC LIMIT 200"

    with _get_conn() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
    return [{"id": int(r["id"]), "title": r["title"], "source_text": r["source_text"] or ""} for r in rows]


def add_feed_items(items: list[dict], source: str = "arxiv") -> int:
    now = datetime.now(timezone.utc).isoformat()
    inserted = 0
    with _get_conn() as conn:
        for item in items:
            title = (item.get("title") or "").strip()
            summary = (item.get("summary") or "").strip()
            url = (item.get("url") or "").strip()
            published_at = item.get("published_at")
            if not title or not url:
                continue
            dup = conn.execute(
                "SELECT 1 FROM paper_feed WHERE title = ? AND url = ? LIMIT 1",
                (title, url),
            ).fetchone()
            if dup:
                continue
            conn.execute(
                """
                INSERT INTO paper_feed (title, summary, url, published_at, source, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (title, summary, url, published_at, source, now),
            )
            inserted += 1
        conn.commit()
    return inserted


def list_feed_items(limit: int = 50) -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM paper_feed ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {
            "id": int(r["id"]),
            "title": r["title"],
            "summary": r["summary"],
            "url": r["url"],
            "published_at": r["published_at"],
            "source": r["source"],
            "fetched_at": r["fetched_at"],
        }
        for r in rows
    ]


def update_human_review(submission_id: int, human_score: float, reviewer_comment: str) -> dict | None:
    with _get_conn() as conn:
        cursor = conn.execute(
            """
            UPDATE submissions
            SET human_score = ?, reviewer_comment = ?, review_status = 'human_reviewed'
            WHERE id = ?
            """,
            (human_score, reviewer_comment, submission_id),
        )
        if cursor.rowcount == 0:
            return None
        row = conn.execute(
            "SELECT id, human_score, reviewer_comment, review_status FROM submissions WHERE id = ?",
            (submission_id,),
        ).fetchone()
        conn.commit()

    return {
        "submission_id": int(row["id"]),
        "human_score": float(row["human_score"]),
        "reviewer_comment": row["reviewer_comment"],
        "review_status": row["review_status"],
    }


def create_review_job(filename: str, submitted_by: str, submitted_role: str) -> int:
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        cursor = conn.execute(
            """
            INSERT INTO review_jobs (
                filename, submitted_by, submitted_role, status,
                error_message, submission_id, created_at, updated_at
            ) VALUES (?, ?, ?, 'pending', NULL, NULL, ?, ?)
            """,
            (filename, submitted_by, submitted_role, now, now),
        )
        conn.commit()
        return int(cursor.lastrowid)


def set_review_job_status(
    job_id: int,
    status: str,
    error_message: str | None = None,
    submission_id: int | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.execute(
            """
            UPDATE review_jobs
            SET status = ?, error_message = ?, submission_id = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, error_message, submission_id, now, job_id),
        )
        conn.commit()


def get_review_job(job_id: int, user_role: str, username: str) -> dict | None:
    with _get_conn() as conn:
        if user_role == "student":
            row = conn.execute(
                """
                SELECT * FROM review_jobs
                WHERE id = ? AND submitted_by = ?
                """,
                (job_id, username),
            ).fetchone()
        else:
            row = conn.execute("SELECT * FROM review_jobs WHERE id = ?", (job_id,)).fetchone()

    if row is None:
        return None

    return {
        "id": int(row["id"]),
        "filename": row["filename"],
        "submitted_by": row["submitted_by"],
        "submitted_role": row["submitted_role"],
        "status": row["status"],
        "error_message": row["error_message"],
        "submission_id": row["submission_id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def write_audit_event(actor_username: str, actor_role: str, event_type: str, details: dict) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO audit_events (
                actor_username, actor_role, event_type, event_details_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (actor_username, actor_role, event_type, json.dumps(details), now),
        )
        conn.commit()


def get_metrics_summary() -> dict:
    with _get_conn() as conn:
        submissions_count = conn.execute("SELECT COUNT(*) AS c FROM submissions").fetchone()["c"]
        jobs_count = conn.execute("SELECT COUNT(*) AS c FROM review_jobs").fetchone()["c"]
        pending_jobs = conn.execute(
            "SELECT COUNT(*) AS c FROM review_jobs WHERE status IN ('pending', 'processing')"
        ).fetchone()["c"]
        avg_score_row = conn.execute("SELECT AVG(score) AS avg_score FROM submissions").fetchone()
        avg_human_row = conn.execute(
            "SELECT AVG(human_score) AS avg_human_score FROM submissions WHERE human_score IS NOT NULL"
        ).fetchone()

    return {
        "submissions_count": int(submissions_count or 0),
        "jobs_count": int(jobs_count or 0),
        "pending_jobs": int(pending_jobs or 0),
        "average_ai_score": round(float(avg_score_row["avg_score"] or 0.0), 3),
        "average_human_score": round(float(avg_human_row["avg_human_score"] or 0.0), 3),
    }
