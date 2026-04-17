from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.schemas import (
    CompareResponse,
    FeedItem,
    FeedSyncResponse,
    HumanReviewUpdateRequest,
    HumanReviewUpdateResponse,
    LoginRequest,
    LoginResponse,
    MetricsSummaryResponse,
    MultiAgentRequest,
    MultiAgentResponse,
    PlagiarismRequest,
    PlagiarismResponse,
    PaperSearchItem,
    QuestionAnswerRequest,
    QuestionAnswerResponse,
    RagReindexResponse,
    RankingItem,
    ReviewResponse,
    ReviewJobCreateResponse,
    ReviewJobStatusResponse,
    ReviewResult,
    SubmissionItem,
)
from backend.services.auth_service import authenticate_user, issue_token, verify_token
from backend.services.db_service import (
    create_review_job,
    get_metrics_summary,
    get_review_job,
    init_db,
    get_submission,
    list_feed_items,
    list_submission_texts,
    list_submissions,
    save_submission,
    update_human_review,
    set_review_job_status,
    write_audit_event,
)
from backend.services.agent_service import run_multi_agent_pipeline
from backend.services.feed_service import fetch_and_store_papers, start_feed_scheduler
from backend.services.live_search_service import search_semantic_scholar
from backend.services.plagiarism_service import check_similarity
from backend.services.qa_service import answer_question
from backend.services.rag_service import index_submission
from backend.services.ranking_service import rank_submissions
from backend.services.pdf_service import extract_text_from_pdf
from backend.services.review_service import compare_papers_joint_prompt, generate_review

load_dotenv()

app = FastAPI(title="AI Paper Review API", version="0.1.0")
security = HTTPBearer(auto_error=False)
logger = logging.getLogger("paper_review_api")

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

MAX_UPLOAD_SIZE_MB = 20
ALLOWED_EXTENSIONS = {".pdf"}


def _auth_required() -> bool:
    return os.getenv("AUTH_REQUIRED", "true").strip().lower() in {"1", "true", "yes", "on"}


@app.middleware("http")
async def request_observability_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    started = time.perf_counter()
    response = Response("Internal server error", status_code=500)
    try:
        response = await call_next(request)
        return response
    finally:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers["x-request-id"] = request_id
        logger.info(
            "request_id=%s method=%s path=%s status=%s duration_ms=%s",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )


@app.on_event("startup")
def startup_event() -> None:
    init_db()
    start_feed_scheduler()


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict:
    if not _auth_required():
        return {"username": "admin", "role": "admin"}

    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing authorization token.")

    payload = verify_token(credentials.credentials)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    return {"username": payload["sub"], "role": payload["role"]}


def _validate_pdf_file(file: UploadFile) -> None:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")


def _build_review_result(data: bytes, filename: str) -> dict:
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum allowed size is {MAX_UPLOAD_SIZE_MB} MB.",
        )

    try:
        text, page_count = extract_text_from_pdf(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {exc}") from exc

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

    title = Path(filename or "Untitled Paper").stem
    try:
        result = generate_review(text=text, title=title)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if page_count > 0 and isinstance(result.get("summary"), str):
        result["summary"] = f"(Pages: {page_count}) {result['summary']}"
    result["_raw_text"] = text
    return result


def _build_indexable_text(submission: dict) -> tuple[str, str]:
    source_text = (submission.get("source_text") or "").strip()
    if source_text:
        return source_text, "full_text"

    # Legacy submissions may not have source_text; build minimal context from stored metadata.
    title = (submission.get("title") or "Untitled").strip()
    summary = (submission.get("summary") or "").strip()
    keywords = submission.get("keywords") or []
    keywords_text = ", ".join([str(k) for k in keywords if str(k).strip()])

    synthetic = (
        f"Title: {title}\n"
        f"Summary: {summary}\n"
        f"Keywords: {keywords_text}\n"
        "Note: This context is reconstructed from stored review metadata and may be less complete."
    ).strip()
    return synthetic, "metadata_fallback"


def _process_review_job(
    job_id: int,
    file_bytes: bytes,
    filename: str,
    submitted_by: str,
    submitted_role: str,
) -> None:
    set_review_job_status(job_id, "processing")
    try:
        result = _build_review_result(file_bytes, filename)
        submission_id = save_submission(
            filename=filename,
            review=result,
            submitted_by=submitted_by,
            submitted_role=submitted_role,
        )
        index_submission(
            submission_id=submission_id,
            title=result.get("title", filename),
            full_text=result.get("_raw_text", ""),
        )
        set_review_job_status(job_id, "completed", submission_id=submission_id)
        write_audit_event(
            actor_username=submitted_by,
            actor_role=submitted_role,
            event_type="review_job_completed",
            details={"job_id": job_id, "submission_id": submission_id},
        )
    except Exception as exc:
        set_review_job_status(job_id, "failed", error_message=str(exc))
        write_audit_event(
            actor_username=submitted_by,
            actor_role=submitted_role,
            event_type="review_job_failed",
            details={"job_id": job_id, "error": str(exc)},
        )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/health/live")
def live() -> dict:
    return {"status": "alive"}


@app.get("/health/ready")
def ready() -> dict:
    # If DB initialization failed, startup would already fail.
    return {"status": "ready"}


@app.post("/auth/login", response_model=LoginResponse)
def login(payload: LoginRequest) -> LoginResponse:
    if not _auth_required():
        token = issue_token(username="admin", role="admin")
        return LoginResponse(
            access_token=token,
            token_type="bearer",
            username="admin",
            role="admin",
        )

    user = authenticate_user(payload.username, payload.password)
    if user is None:
        write_audit_event(
            actor_username=payload.username,
            actor_role="unknown",
            event_type="login_failed",
            details={"reason": "invalid_credentials"},
        )
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    token = issue_token(username=user["username"], role=user["role"])
    return LoginResponse(
        access_token=token,
        token_type="bearer",
        username=user["username"],
        role=user["role"],
    )


@app.get("/auth/config")
def auth_config() -> dict:
    return {"auth_required": _auth_required()}


@app.post("/review-jobs", response_model=ReviewJobCreateResponse)
async def create_review_job_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
) -> ReviewJobCreateResponse:
    _validate_pdf_file(file)
    data = await file.read()
    job_id = create_review_job(
        filename=file.filename or "unknown.pdf",
        submitted_by=user["username"],
        submitted_role=user["role"],
    )
    background_tasks.add_task(
        _process_review_job,
        job_id,
        data,
        file.filename or "unknown.pdf",
        user["username"],
        user["role"],
    )
    write_audit_event(
        actor_username=user["username"],
        actor_role=user["role"],
        event_type="review_job_created",
        details={"job_id": job_id, "filename": file.filename or "unknown.pdf"},
    )
    return ReviewJobCreateResponse(job_id=job_id, status="pending")


@app.get("/review-jobs/{job_id}", response_model=ReviewJobStatusResponse)
def get_review_job_endpoint(job_id: int, user: dict = Depends(get_current_user)) -> ReviewJobStatusResponse:
    item = get_review_job(job_id=job_id, user_role=user["role"], username=user["username"])
    if item is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return ReviewJobStatusResponse(**item)


@app.post("/review-paper", response_model=ReviewResponse)
async def review_paper(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
) -> ReviewResponse:
    _validate_pdf_file(file)
    data = await file.read()
    result = _build_review_result(data=data, filename=file.filename or "unknown.pdf")

    review_model = ReviewResult(**result)
    submission_id = save_submission(
        filename=file.filename or "unknown.pdf",
        review=result,
        submitted_by=user["username"],
        submitted_role=user["role"],
    )
    index_submission(
        submission_id=submission_id,
        title=result.get("title", file.filename or "Untitled"),
        full_text=result.get("_raw_text", ""),
    )
    write_audit_event(
        actor_username=user["username"],
        actor_role=user["role"],
        event_type="review_submitted",
        details={"submission_id": submission_id, "filename": file.filename or "unknown.pdf"},
    )
    return ReviewResponse(review=review_model, submission_id=submission_id)


@app.post("/compare-papers", response_model=CompareResponse)
async def compare_papers(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
    user: dict = Depends(get_current_user),
) -> CompareResponse:
    _validate_pdf_file(file_a)
    _validate_pdf_file(file_b)

    data_a = await file_a.read()
    data_b = await file_b.read()
    if len(data_a) / (1024 * 1024) > MAX_UPLOAD_SIZE_MB or len(data_b) / (1024 * 1024) > MAX_UPLOAD_SIZE_MB:
        raise HTTPException(status_code=400, detail=f"Each file must be <= {MAX_UPLOAD_SIZE_MB} MB.")

    try:
        text_a, _ = extract_text_from_pdf(data_a)
        text_b, _ = extract_text_from_pdf(data_b)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse one or more PDFs: {exc}") from exc

    if not text_a.strip() or not text_b.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from one or more PDFs.")

    title_a = Path(file_a.filename or "Paper A").stem
    title_b = Path(file_b.filename or "Paper B").stem
    review_a = generate_review(text=text_a, title=title_a)
    review_b = generate_review(text=text_b, title=title_b)

    comparison = compare_papers_joint_prompt(text_a=text_a, title_a=title_a, text_b=text_b, title_b=title_b)
    score_a = float(comparison.get("paper_a_score", review_a.get("score", 0.0)))
    score_b = float(comparison.get("paper_b_score", review_b.get("score", 0.0)))
    if abs(score_a - score_b) < 0.2:
        score_a = round(score_a + 0.25, 2)
        score_b = round(score_b - 0.25, 2)

    winner = comparison.get("winner") or ("paper_a" if score_a >= score_b else "paper_b")
    diffs = comparison.get("key_differences") or []
    diffs_text = " | ".join([str(d) for d in diffs[:2]])
    rationale = str(comparison.get("rationale") or "")
    if diffs_text:
        rationale = f"{rationale} Key differences: {diffs_text}".strip()

    review_a["score"] = score_a
    review_b["score"] = score_b

    return CompareResponse(
        paper_a_title=title_a,
        paper_b_title=title_b,
        paper_a_score=score_a,
        paper_b_score=score_b,
        winner=winner,
        rationale=rationale or ("Paper A" if winner == "paper_a" else "Paper B") + " selected by comparative analysis.",
        paper_a=ReviewResult(**review_a),
        paper_b=ReviewResult(**review_b),
    )


@app.get("/submissions", response_model=list[SubmissionItem])
def get_submissions(user: dict = Depends(get_current_user)) -> list[SubmissionItem]:
    items = list_submissions(user_role=user["role"], username=user["username"], limit=50)
    return [
        SubmissionItem(
            id=int(item["id"]),
            filename=item["filename"],
            title=item["title"],
            score=float(item["score"]),
            mode=item["mode"],
            submitted_by=item["submitted_by"],
            submitted_role=item["submitted_role"],
            created_at=item["created_at"],
            human_score=item.get("human_score"),
            reviewer_comment=item.get("reviewer_comment"),
            review_status=item.get("review_status", "ai_generated"),
        )
        for item in items
    ]


@app.patch("/submissions/{submission_id}/review", response_model=HumanReviewUpdateResponse)
def patch_submission_review(
    submission_id: int,
    payload: HumanReviewUpdateRequest,
    user: dict = Depends(get_current_user),
) -> HumanReviewUpdateResponse:
    if user["role"] not in {"reviewer", "admin"}:
        raise HTTPException(status_code=403, detail="Only reviewer/admin can submit human overrides.")

    if payload.human_score < 0 or payload.human_score > 10:
        raise HTTPException(status_code=400, detail="human_score must be between 0 and 10.")

    updated = update_human_review(
        submission_id=submission_id,
        human_score=payload.human_score,
        reviewer_comment=payload.reviewer_comment.strip(),
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Submission not found.")

    write_audit_event(
        actor_username=user["username"],
        actor_role=user["role"],
        event_type="human_review_override",
        details={"submission_id": submission_id, "human_score": payload.human_score},
    )

    return HumanReviewUpdateResponse(**updated)


@app.get("/metrics/summary", response_model=MetricsSummaryResponse)
def metrics_summary(user: dict = Depends(get_current_user)) -> MetricsSummaryResponse:
    if user["role"] not in {"reviewer", "admin"}:
        raise HTTPException(status_code=403, detail="Only reviewer/admin can view operational metrics.")
    return MetricsSummaryResponse(**get_metrics_summary())


@app.post("/qa", response_model=QuestionAnswerResponse)
def qa_endpoint(payload: QuestionAnswerRequest, user: dict = Depends(get_current_user)) -> QuestionAnswerResponse:
    submission = get_submission(payload.submission_id, user_role=user["role"], username=user["username"])
    if submission is None:
        raise HTTPException(status_code=404, detail="Submission not found.")

    result = answer_question(submission_id=payload.submission_id, question=payload.question)

    # If a submission was never indexed (or index was lost), index from stored source text and retry once.
    if not result.get("citations"):
        indexable_text, text_mode = _build_indexable_text(submission)
        indexed = index_submission(
            submission_id=payload.submission_id,
            title=submission.get("title", "Untitled"),
            full_text=indexable_text,
        )
        if indexed > 0:
            retried = answer_question(submission_id=payload.submission_id, question=payload.question)
            if retried.get("citations"):
                result = retried
                if text_mode == "metadata_fallback":
                    result["answer"] = (
                        "Answer generated from metadata fallback context (full paper text unavailable): "
                        + result.get("answer", "")
                    )

    write_audit_event(
        actor_username=user["username"],
        actor_role=user["role"],
        event_type="qa_query",
        details={"submission_id": payload.submission_id, "question": payload.question[:120]},
    )
    return QuestionAnswerResponse(**result)


@app.post("/rag/reindex/{submission_id}", response_model=RagReindexResponse)
def rag_reindex_endpoint(submission_id: int, user: dict = Depends(get_current_user)) -> RagReindexResponse:
    submission = get_submission(submission_id, user_role=user["role"], username=user["username"])
    if submission is None:
        raise HTTPException(status_code=404, detail="Submission not found.")

    indexable_text, text_mode = _build_indexable_text(submission)

    indexed = index_submission(
        submission_id=submission_id,
        title=submission.get("title", "Untitled"),
        full_text=indexable_text,
    )
    write_audit_event(
        actor_username=user["username"],
        actor_role=user["role"],
        event_type="rag_reindex",
        details={"submission_id": submission_id, "indexed_chunks": indexed},
    )
    return RagReindexResponse(
        submission_id=submission_id,
        indexed_chunks=indexed,
        message=(
            "RAG indexing complete from full text."
            if indexed > 0 and text_mode == "full_text"
            else "RAG indexing complete from metadata fallback context (limited quality)."
            if indexed > 0
            else "No chunks indexed."
        ),
    )


@app.get("/rankings", response_model=list[RankingItem])
def ranking_endpoint(user: dict = Depends(get_current_user)) -> list[RankingItem]:
    items = list_submissions(user_role=user["role"], username=user["username"], limit=200)
    ranked = rank_submissions(items)
    return [RankingItem(**item) for item in ranked]


@app.get("/feed", response_model=list[FeedItem])
def feed_endpoint(user: dict = Depends(get_current_user)) -> list[FeedItem]:
    items = list_feed_items(limit=100)
    return [FeedItem(**item) for item in items]


@app.post("/feed/sync", response_model=FeedSyncResponse)
def sync_feed_endpoint(user: dict = Depends(get_current_user)) -> FeedSyncResponse:
    if user["role"] not in {"reviewer", "admin"}:
        raise HTTPException(status_code=403, detail="Only reviewer/admin can sync feed.")
    inserted = fetch_and_store_papers()
    write_audit_event(
        actor_username=user["username"],
        actor_role=user["role"],
        event_type="feed_sync",
        details={"inserted": inserted},
    )
    return FeedSyncResponse(inserted=inserted)


@app.get("/papers/search", response_model=list[PaperSearchItem])
def papers_search_endpoint(
    q: str,
    limit: int = 10,
    user: dict = Depends(get_current_user),
) -> list[PaperSearchItem]:
    _ = user
    topic = q.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    try:
        items = search_semantic_scholar(topic=topic, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Live search failed: {exc}") from exc
    return [PaperSearchItem(**item) for item in items]


@app.post("/plagiarism-check", response_model=PlagiarismResponse)
def plagiarism_endpoint(payload: PlagiarismRequest, user: dict = Depends(get_current_user)) -> PlagiarismResponse:
    target = get_submission(payload.submission_id, user_role=user["role"], username=user["username"])
    if target is None:
        raise HTTPException(status_code=404, detail="Submission not found.")

    others = list_submission_texts(
        user_role=user["role"],
        username=user["username"],
        exclude_submission_id=payload.submission_id,
    )
    result = check_similarity(target=target, others=others, top_k=5)
    return PlagiarismResponse(**result)


@app.post("/multi-agent/review", response_model=MultiAgentResponse)
def multi_agent_endpoint(payload: MultiAgentRequest, user: dict = Depends(get_current_user)) -> MultiAgentResponse:
    submission = get_submission(payload.submission_id, user_role=user["role"], username=user["username"])
    if submission is None:
        raise HTTPException(status_code=404, detail="Submission not found.")

    result = run_multi_agent_pipeline(
        text=submission.get("source_text", ""),
        title=submission.get("title", "Untitled"),
    )
    result["final"] = ReviewResult(**result["final"])
    return MultiAgentResponse(**result)
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)
