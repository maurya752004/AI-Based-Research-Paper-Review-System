# AI Paper Review System (MVP)

This project is a starter implementation of an AI system to automatically review and summarize research papers.

## What this MVP does

- Upload a research paper PDF
- Extract text from the PDF
- Login with role-based accounts (`student`, `reviewer`, `admin`)
- Generate:
  - concise summary
  - strengths
  - weaknesses
  - recommendation
  - numeric score (1-10)
  - rubric scores (novelty, methodology, evaluation, reproducibility, writing quality)
  - section-wise scores and section summaries
  - citation metrics and novelty score
  - confidence level and confidence score
  - evidence points for explainability
  - keywords
- Show results in a simple Streamlit interface
- Store submissions in local SQLite DB (`paper_reviews.db`)
- Download review report as PDF from UI
- View score-trend dashboard charts from submission history
- Compare two papers side-by-side
- Allow reviewer/admin human override score and comments
- Support async review jobs (`pending -> processing -> completed/failed`)
- Keep audit events for security and compliance traces
- Provide operational metrics and readiness/liveness health endpoints
- Add RAG indexing + retrieval for scalable QA on large papers
- Add smart research assistant Q&A endpoint
- Add submission ranking endpoint for decision support
- Add scheduled real-time paper feed sync from arXiv
- Add plagiarism similarity risk checks
- Add multi-agent pipeline endpoint (researcher/analyzer/writer/reviewer)

## Tech Stack

- Backend API: FastAPI
- Frontend UI: Streamlit
- PDF parsing: pypdf
- AI provider: OpenAI (optional, via `OPENAI_API_KEY`)
- Free local LLM option: Ollama (`USE_LOCAL_LLM=true`, e.g. `OLLAMA_MODEL=llama3`)
- Fallback mode: heuristic review if API/model is unavailable (`REVIEW_ALLOW_FALLBACK=true`)
- Strict AI mode: disable fallback (`REVIEW_ALLOW_FALLBACK=false`)

## Project Structure

```text
backend/
  main.py
  schemas.py
  services/
    pdf_service.py
    review_service.py
frontend/
  app.py
requirements.txt
.env.example
```

## Setup

1. Create and activate virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables:

```bash
cp .env.example .env
```

Then set `OPENAI_API_KEY` in `.env` if you want AI-powered review.

For fully free local demo mode:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama run llama3
```

Then set in `.env`:

```bash
USE_OPENAI=false
USE_LOCAL_LLM=true
OLLAMA_MODEL=llama3
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

## Run backend

```bash
uvicorn backend.main:app --reload --port 8000
```

## Run frontend

In another terminal:

```bash
streamlit run frontend/app.py
```

## API Endpoint

- `GET /health`
- `GET /health/live`
- `GET /health/ready`
- `POST /auth/login` (JSON username/password)
- `POST /review-paper` (multipart form with `file`, requires bearer token)
- `POST /review-jobs` (async processing job, requires bearer token)
- `GET /review-jobs/{id}` (job status polling, requires bearer token)
- `POST /compare-papers` (multipart form with `file_a` and `file_b`, requires bearer token)
- `GET /submissions` (requires bearer token)
- `PATCH /submissions/{id}/review` (reviewer/admin only)
- `GET /metrics/summary` (reviewer/admin only)
- `POST /qa` (RAG-based question answering)
- `GET /rankings` (paper scoring and ranking)
- `GET /feed` (latest fetched research feed)
- `POST /feed/sync` (manual feed sync, reviewer/admin)
- `GET /papers/search?q=<topic>&limit=<n>` (live Semantic Scholar search)
- `POST /plagiarism-check` (submission similarity risk)
- `POST /multi-agent/review` (multi-agent analysis workflow)

## Demo Login Accounts

- student: `student1 / student123`
- reviewer: `reviewer1 / reviewer123`
- admin: `admin1 / admin123`

You can override users with env variable `APP_USERS_JSON`.

## Notes

- This is an MVP for internship/final-year demonstration.
- For production use, add authentication, persistent storage, better evaluation logic, and model safety checks.
- For production use, replace demo credentials and token logic with secure OAuth/JWT.

## Industry-Oriented Patterns Included

- Request observability middleware (`x-request-id`, request duration logging)
- Async job orchestration for long-running document analysis
- Audit trail events for login, reviews, overrides, and job lifecycle
- Operational metrics endpoint for dashboards and monitoring
