from pydantic import BaseModel


class ReviewResult(BaseModel):
    title: str
    summary: str
    strengths: list[str]
    weaknesses: list[str]
    recommendation: str
    score: float
    keywords: list[str]
    word_count: int
    mode: str
    mode_reason: str | None = None
    rubric_scores: dict[str, float]
    score_explanations: dict[str, str]
    evidence_points: list[str]
    section_scores: dict[str, float]
    section_summaries: dict[str, str]
    citation_metrics: dict[str, float]
    novelty_score: float
    novelty_notes: list[str]
    confidence_level: str
    confidence_score: float


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    username: str
    role: str


class SubmissionItem(BaseModel):
    id: int
    filename: str
    title: str
    score: float
    mode: str
    submitted_by: str
    submitted_role: str
    created_at: str
    human_score: float | None = None
    reviewer_comment: str | None = None
    review_status: str


class ReviewResponse(BaseModel):
    review: ReviewResult
    submission_id: int


class CompareResponse(BaseModel):
    paper_a_title: str
    paper_b_title: str
    paper_a_score: float
    paper_b_score: float
    winner: str
    rationale: str
    paper_a: ReviewResult
    paper_b: ReviewResult


class HumanReviewUpdateRequest(BaseModel):
    human_score: float
    reviewer_comment: str


class HumanReviewUpdateResponse(BaseModel):
    submission_id: int
    human_score: float
    reviewer_comment: str
    review_status: str


class ReviewJobCreateResponse(BaseModel):
    job_id: int
    status: str


class ReviewJobStatusResponse(BaseModel):
    id: int
    filename: str
    submitted_by: str
    submitted_role: str
    status: str
    error_message: str | None = None
    submission_id: int | None = None
    created_at: str
    updated_at: str


class MetricsSummaryResponse(BaseModel):
    submissions_count: int
    jobs_count: int
    pending_jobs: int
    average_ai_score: float
    average_human_score: float


class QuestionAnswerRequest(BaseModel):
    submission_id: int
    question: str


class QuestionAnswerResponse(BaseModel):
    answer: str
    citations: list[int | dict]
    mode: str
    mode_reason: str | None = None


class RagReindexResponse(BaseModel):
    submission_id: int
    indexed_chunks: int
    message: str


class RankingItem(BaseModel):
    submission_id: int
    title: str
    owner: str
    ai_score: float
    citation_quality: float
    novelty_score: float
    confidence_score: float
    ranking_score: float
    rank: int


class FeedItem(BaseModel):
    id: int
    title: str
    summary: str
    url: str
    published_at: str | None = None
    source: str
    fetched_at: str


class PaperSearchItem(BaseModel):
    title: str
    summary: str
    url: str
    published_at: str | None = None
    source: str
    authors: list[str] = []


class FeedSyncResponse(BaseModel):
    inserted: int


class PlagiarismRequest(BaseModel):
    submission_id: int


class PlagiarismMatch(BaseModel):
    submission_id: int
    title: str
    similarity: float


class PlagiarismResponse(BaseModel):
    target_submission_id: int
    risk_level: str
    max_similarity: float
    matches: list[PlagiarismMatch]


class MultiAgentRequest(BaseModel):
    submission_id: int


class MultiAgentResponse(BaseModel):
    researcher: str
    analyzer: dict
    writer: str
    reviewer: str
    final: ReviewResult
