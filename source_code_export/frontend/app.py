from __future__ import annotations

import io
import os
from datetime import datetime

import altair as alt
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

load_dotenv()

st.set_page_config(page_title="AI Paper Review System", page_icon="R", layout="wide")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def _api_headers() -> dict[str, str]:
    token = st.session_state.get("token", "")
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _fetch_auth_config() -> bool:
    try:
        response = requests.get(f"{BACKEND_URL}/auth/config", timeout=10)
        if response.status_code == 200:
            return bool(response.json().get("auth_required", True))
    except requests.RequestException:
        pass
    return True


def _fetch_history() -> list[dict]:
    try:
        res = requests.get(
            f"{BACKEND_URL}/submissions",
            headers=_api_headers(),
            timeout=20,
        )
        if res.status_code == 200:
            return res.json()
    except requests.RequestException:
        return []
    return []


def _latest_submission_id() -> int | None:
    # Prefer latest known in memory; fallback to API history.
    sid = st.session_state.get("last_submission_id")
    if isinstance(sid, int) and sid > 0:
        return sid

    history = st.session_state.get("history") or _fetch_history()
    if history:
        try:
            return int(history[0].get("id"))
        except Exception:
            return None
    return None


def _post_review(file_name: str, file_bytes: bytes) -> tuple[dict, int | None, str | None]:
    try:
        response = requests.post(
            f"{BACKEND_URL}/review-paper",
            files={"file": (file_name, file_bytes, "application/pdf")},
            headers=_api_headers(),
            timeout=120,
        )
    except requests.RequestException as exc:
        return {}, None, str(exc)

    if response.status_code != 200:
        try:
            return {}, None, response.json().get("detail", "Unknown error")
        except ValueError:
            return {}, None, response.text

    data = response.json()
    return data.get("review", {}), data.get("submission_id"), None


def _create_review_job(file_name: str, file_bytes: bytes) -> tuple[int | None, str | None]:
    try:
        response = requests.post(
            f"{BACKEND_URL}/review-jobs",
            files={"file": (file_name, file_bytes, "application/pdf")},
            headers=_api_headers(),
            timeout=30,
        )
    except requests.RequestException as exc:
        return None, str(exc)

    if response.status_code != 200:
        try:
            return None, response.json().get("detail", "Unknown error")
        except ValueError:
            return None, response.text

    payload = response.json()
    return int(payload.get("job_id")), None


def _fetch_job_status(job_id: int) -> tuple[dict, str | None]:
    try:
        response = requests.get(
            f"{BACKEND_URL}/review-jobs/{job_id}",
            headers=_api_headers(),
            timeout=20,
        )
    except requests.RequestException as exc:
        return {}, str(exc)

    if response.status_code != 200:
        try:
            return {}, response.json().get("detail", "Unknown error")
        except ValueError:
            return {}, response.text

    return response.json(), None


def _fetch_metrics_summary() -> tuple[dict, str | None]:
    try:
        response = requests.get(
            f"{BACKEND_URL}/metrics/summary",
            headers=_api_headers(),
            timeout=20,
        )
    except requests.RequestException as exc:
        return {}, str(exc)

    if response.status_code != 200:
        try:
            return {}, response.json().get("detail", "Unknown error")
        except ValueError:
            return {}, response.text

    return response.json(), None


def _ask_question(submission_id: int, question: str) -> tuple[dict, str | None]:
    try:
        response = requests.post(
            f"{BACKEND_URL}/qa",
            json={"submission_id": submission_id, "question": question},
            headers=_api_headers(),
            timeout=180,
        )
    except requests.RequestException as exc:
        return {}, str(exc)

    if response.status_code != 200:
        try:
            return {}, response.json().get("detail", "Unknown error")
        except ValueError:
            return {}, response.text
    return response.json(), None


def _reindex_submission(submission_id: int) -> tuple[dict, str | None]:
    try:
        response = requests.post(
            f"{BACKEND_URL}/rag/reindex/{submission_id}",
            headers=_api_headers(),
            timeout=45,
        )
    except requests.RequestException as exc:
        return {}, str(exc)

    if response.status_code != 200:
        try:
            return {}, response.json().get("detail", "Unknown error")
        except ValueError:
            return {}, response.text
    return response.json(), None


def _fetch_rankings() -> tuple[list[dict], str | None]:
    try:
        response = requests.get(
            f"{BACKEND_URL}/rankings",
            headers=_api_headers(),
            timeout=30,
        )
    except requests.RequestException as exc:
        return [], str(exc)

    if response.status_code != 200:
        try:
            return [], response.json().get("detail", "Unknown error")
        except ValueError:
            return [], response.text
    return response.json(), None


def _fetch_feed() -> tuple[list[dict], str | None]:
    try:
        response = requests.get(
            f"{BACKEND_URL}/feed",
            headers=_api_headers(),
            timeout=30,
        )
    except requests.RequestException as exc:
        return [], str(exc)

    if response.status_code != 200:
        try:
            return [], response.json().get("detail", "Unknown error")
        except ValueError:
            return [], response.text
    return response.json(), None


def _sync_feed() -> tuple[dict, str | None]:
    try:
        response = requests.post(
            f"{BACKEND_URL}/feed/sync",
            headers=_api_headers(),
            timeout=60,
        )
    except requests.RequestException as exc:
        return {}, str(exc)

    if response.status_code != 200:
        try:
            return {}, response.json().get("detail", "Unknown error")
        except ValueError:
            return {}, response.text
    return response.json(), None


def _search_live_papers(query: str, limit: int = 10) -> tuple[list[dict], str | None]:
    try:
        response = requests.get(
            f"{BACKEND_URL}/papers/search",
            params={"q": query, "limit": limit},
            headers=_api_headers(),
            timeout=40,
        )
    except requests.RequestException as exc:
        return [], str(exc)

    if response.status_code != 200:
        try:
            return [], response.json().get("detail", "Unknown error")
        except ValueError:
            return [], response.text
    return response.json(), None


def _check_plagiarism(submission_id: int) -> tuple[dict, str | None]:
    try:
        response = requests.post(
            f"{BACKEND_URL}/plagiarism-check",
            json={"submission_id": submission_id},
            headers=_api_headers(),
            timeout=45,
        )
    except requests.RequestException as exc:
        return {}, str(exc)

    if response.status_code != 200:
        try:
            return {}, response.json().get("detail", "Unknown error")
        except ValueError:
            return {}, response.text
    return response.json(), None


def _run_multi_agent(submission_id: int) -> tuple[dict, str | None]:
    try:
        response = requests.post(
            f"{BACKEND_URL}/multi-agent/review",
            json={"submission_id": submission_id},
            headers=_api_headers(),
            timeout=120,
        )
    except requests.RequestException as exc:
        return {}, str(exc)

    if response.status_code != 200:
        try:
            return {}, response.json().get("detail", "Unknown error")
        except ValueError:
            return {}, response.text
    return response.json(), None


def _is_fallback_mode_active() -> bool:
    review_mode = (st.session_state.get("last_review") or {}).get("mode", "")
    qa_mode = (st.session_state.get("qa_result") or {}).get("mode", "")
    return review_mode == "fallback" or qa_mode == "rag-fallback"


def _fallback_reason_text() -> str:
    review_reason = (st.session_state.get("last_review") or {}).get("mode_reason", "")
    qa_reason = (st.session_state.get("qa_result") or {}).get("mode_reason", "")
    reason = qa_reason or review_reason
    if not reason:
        return "OpenAI unavailable, running in fallback mode."
    if "ollama" in reason:
        return "Local LLM active (Ollama)."
    if "demo_mode_openai_disabled" in reason:
        return "Free demo mode active: local AI + RAG is running (no paid API required)."
    if "insufficient_quota" in reason or "RateLimitError" in reason or "429" in reason:
        return "OpenAI quota exhausted (429 insufficient_quota), running in fallback mode."
    if "missing_api_key" in reason:
        return "OpenAI API key missing, running in fallback mode."
    return f"OpenAI unavailable ({reason[:120]}), running in fallback mode."


def _add_wrapped_line(pdf: canvas.Canvas, text: str, x: int, y: int, max_chars: int = 95) -> int:
    remaining = text.strip()
    while remaining:
        chunk = remaining[:max_chars]
        if len(remaining) > max_chars:
            split_at = chunk.rfind(" ")
            if split_at > 20:
                chunk = chunk[:split_at]
        pdf.drawString(x, y, chunk)
        y -= 14
        remaining = remaining[len(chunk) :].strip()
        if y < 60:
            pdf.showPage()
            pdf.setFont("Helvetica", 10)
            y = 800
    return y


def _build_pdf_report(review: dict, submission_id: int | None, username: str) -> bytes | None:
    if not REPORTLAB_AVAILABLE:
        return None

    buff = io.BytesIO()
    pdf = canvas.Canvas(buff, pagesize=A4)
    pdf.setTitle("Paper Review Report")
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(40, 820, "AI Paper Review Report")
    pdf.setFont("Helvetica", 10)
    y = 798

    meta_lines = [
        f"Generated by: {username}",
        f"Generated on: {datetime.utcnow().isoformat()} UTC",
        f"Submission ID: {submission_id or 'N/A'}",
        f"Title: {review.get('title', 'Untitled')}",
        f"Score: {review.get('score', 0)}/10",
        f"Mode: {review.get('mode', 'unknown')}",
    ]
    for line in meta_lines:
        y = _add_wrapped_line(pdf, line, 40, y)

    y -= 8
    sections = [
        ("Summary", [review.get("summary", "N/A")]),
        ("Strengths", review.get("strengths", [])),
        ("Weaknesses", review.get("weaknesses", [])),
        ("Recommendation", [review.get("recommendation", "N/A")]),
        ("Evidence Points", review.get("evidence_points", [])),
    ]

    rubric = review.get("rubric_scores", {})
    if rubric:
        sections.append(
            (
                "Rubric Scores",
                [f"{k.replace('_', ' ').title()}: {v}/10" for k, v in rubric.items()],
            )
        )

    score_explanations = review.get("score_explanations", {})
    if score_explanations:
        sections.append(
            (
                "Score Explanations",
                [f"{k.replace('_', ' ').title()}: {v}" for k, v in score_explanations.items()],
            )
        )

    for title, lines in sections:
        pdf.setFont("Helvetica-Bold", 11)
        y = _add_wrapped_line(pdf, title, 40, y)
        pdf.setFont("Helvetica", 10)
        for line in lines:
            y = _add_wrapped_line(pdf, f"- {line}", 50, y)
        y -= 4

    pdf.save()
    buff.seek(0)
    return buff.read()


def _render_dashboard(history: list[dict]) -> None:
    st.markdown("### Score Dashboard")
    if not history:
        st.info("No data yet. Analyze at least one paper and refresh history.")
        return

    frame = pd.DataFrame(history)
    frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce")
    frame = frame.sort_values("created_at")

    st.dataframe(frame, use_container_width=True)

    trend = (
        alt.Chart(frame)
        .mark_line(point=True)
        .encode(
            x=alt.X("created_at:T", title="Submission Time"),
            y=alt.Y("score:Q", title="Score"),
            color=alt.Color("submitted_by:N", title="User"),
            tooltip=["title", "submitted_by", "score", "mode", "created_at"],
        )
        .properties(height=280)
    )
    st.altair_chart(trend, use_container_width=True)

    mode_count = frame.groupby("mode", as_index=False).size().rename(columns={"size": "count"})
    bar = (
        alt.Chart(mode_count)
        .mark_bar()
        .encode(
            x=alt.X("mode:N", title="Review Mode"),
            y=alt.Y("count:Q", title="Count"),
            tooltip=["mode", "count"],
        )
        .properties(height=220)
    )
    st.altair_chart(bar, use_container_width=True)


def _render_failure_analysis(history: list[dict]) -> None:
    st.markdown("### Failure Analysis")
    if not history:
        st.info("No data available for failure analysis.")
        return

    low = [h for h in history if float(h.get("score", 0)) < 7.0]
    st.write(f"Low-score submissions (<7): **{len(low)}** / {len(history)}")
    if not low:
        st.success("No low-score submissions found. Current quality trend is stable.")
        return

    weakness_counter: dict[str, int] = {}
    for row in low:
        for item in row.get("weaknesses", []):
            key = item.strip().lower()[:80]
            if not key:
                continue
            weakness_counter[key] = weakness_counter.get(key, 0) + 1

    top_items = sorted(weakness_counter.items(), key=lambda x: x[1], reverse=True)[:5]
    if not top_items:
        st.info("Weakness details are limited for current low-score set.")
        return
    st.write("Most frequent weak points:")
    for text, count in top_items:
        st.write(f"- {text} ({count})")


def _render_cohort_insights(history: list[dict]) -> None:
    st.markdown("### Cohort Insights")
    if not history:
        st.info("No cohort data available yet.")
        return

    frame = pd.DataFrame(history)
    if "submitted_by" in frame.columns:
        user_stats = (
            frame.groupby("submitted_by", as_index=False)["score"]
            .mean()
            .rename(columns={"score": "avg_score"})
            .sort_values("avg_score", ascending=False)
        )
        st.dataframe(user_stats, use_container_width=True)

    if "review_status" in frame.columns:
        status_counts = frame.groupby("review_status", as_index=False).size().rename(columns={"size": "count"})
        st.bar_chart(status_counts.set_index("review_status"))


def _render_review_details(data: dict, submission_id: int | None) -> None:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(data.get("title", "Paper"))
        st.markdown("### Summary")
        st.write(data.get("summary", "N/A"))

        st.markdown("### Strengths")
        for item in data.get("strengths", []):
            st.write(f"- {item}")

        st.markdown("### Weaknesses")
        for item in data.get("weaknesses", []):
            st.write(f"- {item}")

        st.markdown("### Recommendation")
        st.info(data.get("recommendation", "N/A"))

    with col2:
        st.metric("Score", f"{data.get('score', 0)}/10")
        st.metric("Confidence", f"{data.get('confidence_level', 'medium')} ({data.get('confidence_score', 0)}/10)")
        st.metric("Novelty", f"{data.get('novelty_score', 0)}/10")
        st.metric("Word Count", data.get("word_count", 0))
        st.metric("Submission ID", submission_id or "N/A")
        st.write(f"Mode: {data.get('mode', 'unknown')}")
        st.markdown("### Keywords")
        for word in data.get("keywords", []):
            st.write(f"`{word}`")

    st.markdown("### Rubric Scores")
    rubric_scores = data.get("rubric_scores", {})
    if rubric_scores:
        st.json(rubric_scores)

    st.markdown("### Score Explanations")
    score_explanations = data.get("score_explanations", {})
    if score_explanations:
        for metric, explanation in score_explanations.items():
            st.write(f"- **{metric.replace('_', ' ').title()}**: {explanation}")

    st.markdown("### Section Scores")
    section_scores = data.get("section_scores", {})
    if section_scores:
        st.json(section_scores)

    st.markdown("### Section Summaries")
    section_summaries = data.get("section_summaries", {})
    if section_summaries:
        for sec, txt in section_summaries.items():
            st.write(f"- **{sec.title()}**: {txt}")

    st.markdown("### Citation Metrics")
    citation_metrics = data.get("citation_metrics", {})
    if citation_metrics:
        st.json(citation_metrics)

    st.markdown("### Novelty Notes")
    for note in data.get("novelty_notes", []):
        st.write(f"- {note}")

    st.markdown("### Evidence Points")
    for point in data.get("evidence_points", []):
        st.write(f"- {point}")

    st.markdown("### Download Report")
    pdf_bytes = _build_pdf_report(
        review=data,
        submission_id=submission_id,
        username=st.session_state.username,
    )
    if pdf_bytes is None:
        st.warning("Install `reportlab` to enable PDF download.")
    else:
        safe_title = data.get("title", "paper_review").replace(" ", "_")[:50]
        st.download_button(
            label="Download Review PDF",
            data=pdf_bytes,
            file_name=f"{safe_title}_review_report.pdf",
            mime="application/pdf",
        )

st.title("AI System to Review and Summarize Research Papers")
st.caption("Advanced mode: section-aware scoring, compare mode, and human override")

if "token" not in st.session_state:
    st.session_state.token = ""
if "username" not in st.session_state:
    st.session_state.username = ""
if "role" not in st.session_state:
    st.session_state.role = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "last_review" not in st.session_state:
    st.session_state.last_review = {}
if "last_submission_id" not in st.session_state:
    st.session_state.last_submission_id = None
if "compare_result" not in st.session_state:
    st.session_state.compare_result = {}
if "last_job_id" not in st.session_state:
    st.session_state.last_job_id = None
if "qa_result" not in st.session_state:
    st.session_state.qa_result = {}
if "rankings" not in st.session_state:
    st.session_state.rankings = []
if "feed_items" not in st.session_state:
    st.session_state.feed_items = []
if "live_search_items" not in st.session_state:
    st.session_state.live_search_items = []
if "plag_result" not in st.session_state:
    st.session_state.plag_result = {}
if "agent_result" not in st.session_state:
    st.session_state.agent_result = {}
if "auth_required" not in st.session_state:
    st.session_state.auth_required = _fetch_auth_config()

if not st.session_state.auth_required and not st.session_state.token:
    st.session_state.token = "demo-no-auth"
    st.session_state.username = "admin"
    st.session_state.role = "admin"

with st.sidebar:
    if st.session_state.auth_required:
        st.subheader("Login")
        username = st.text_input("Username", value=st.session_state.username)
        password = st.text_input("Password", type="password")

        if st.button("Sign In"):
            try:
                login_res = requests.post(
                    f"{BACKEND_URL}/auth/login",
                    json={"username": username, "password": password},
                    timeout=20,
                )
                if login_res.status_code == 200:
                    data = login_res.json()
                    st.session_state.token = data["access_token"]
                    st.session_state.username = data["username"]
                    st.session_state.role = data["role"]
                    st.success(f"Logged in as {data['username']} ({data['role']})")
                else:
                    st.error("Login failed. Check credentials.")
            except requests.RequestException as exc:
                st.error(f"Login request failed: {exc}")

        if st.session_state.token and st.button("Sign Out"):
            st.session_state.token = ""
            st.session_state.username = ""
            st.session_state.role = ""
            st.info("Logged out.")
    else:
        st.subheader("Login")
        st.info("Login disabled (demo mode): using admin access.")

if st.session_state.auth_required and not st.session_state.token:
    st.warning("Please sign in from the sidebar. Demo users: student1, reviewer1, admin1")
    st.stop()

st.caption(f"Logged in: {st.session_state.username} ({st.session_state.role})")

if _is_fallback_mode_active():
    banner_text = _fallback_reason_text()
    if "Free demo mode active" in banner_text or "Local LLM active" in banner_text:
        st.info(banner_text)
    else:
        st.warning(banner_text)

tab_single, tab_compare, tab_assistant, tab_ranking, tab_feed, tab_plag, tab_agents, tab_analytics, tab_override = st.tabs(
    [
        "Single Review",
        "Compare Papers",
        "Research Assistant",
        "Ranking",
        "Live Feed",
        "Plagiarism",
        "Multi-Agent",
        "Analytics",
        "Human Override",
    ]
)

with tab_single:
    st.markdown("Upload a paper in PDF format to generate advanced review insights.")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], key="single_upload")

    if uploaded_file is not None and st.button("Analyze Paper", type="primary"):
        with st.spinner("Analyzing... this can take a few seconds"):
            review, submission_id, err = _post_review(uploaded_file.name, uploaded_file.getvalue())
            if err:
                st.error(f"Review failed: {err}")
            else:
                st.session_state.last_review = review
                st.session_state.last_submission_id = submission_id
                st.session_state.history = _fetch_history()

    if uploaded_file is not None and st.button("Submit Async Review Job"):
        job_id, err = _create_review_job(uploaded_file.name, uploaded_file.getvalue())
        if err:
            st.error(f"Job creation failed: {err}")
        else:
            st.session_state.last_job_id = job_id
            st.success(f"Review job created. Job ID: {job_id}")

    st.markdown("#### Async Job Status")
    default_job_id = int(st.session_state.last_job_id or 1)
    job_id_input = st.number_input("Job ID", min_value=1, step=1, value=default_job_id, key="job_id_input")
    if st.button("Check Job Status"):
        job, err = _fetch_job_status(int(job_id_input))
        if err:
            st.error(f"Status check failed: {err}")
        else:
            st.json(job)

    if st.session_state.last_review:
        _render_review_details(st.session_state.last_review, st.session_state.last_submission_id)

with tab_compare:
    st.markdown("Upload two papers and compare scores side by side.")
    file_a = st.file_uploader("Paper A (PDF)", type=["pdf"], key="cmp_a")
    file_b = st.file_uploader("Paper B (PDF)", type=["pdf"], key="cmp_b")

    if file_a and file_b and st.button("Compare Papers"):
        try:
            res = requests.post(
                f"{BACKEND_URL}/compare-papers",
                files={
                    "file_a": (file_a.name, file_a.getvalue(), "application/pdf"),
                    "file_b": (file_b.name, file_b.getvalue(), "application/pdf"),
                },
                headers=_api_headers(),
                timeout=180,
            )
            if res.status_code == 200:
                st.session_state.compare_result = res.json()
            else:
                st.error(f"Compare failed: {res.text}")
        except requests.RequestException as exc:
            st.error(f"Compare request failed: {exc}")

    cmp = st.session_state.compare_result
    if cmp:
        st.success(f"Winner: {cmp.get('winner')} | {cmp.get('rationale')}")
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**{cmp.get('paper_a_title')}**")
            st.metric("Score", cmp.get("paper_a_score", 0))
            st.json(cmp.get("paper_a", {}).get("rubric_scores", {}))
        with c2:
            st.write(f"**{cmp.get('paper_b_title')}**")
            st.metric("Score", cmp.get("paper_b_score", 0))
            st.json(cmp.get("paper_b", {}).get("rubric_scores", {}))

with tab_assistant:
    st.markdown("Ask targeted questions on indexed paper chunks (RAG).")
    st.caption("Tip: Use a recent Submission ID from Single Review. If Q&A says no context, click Reindex Submission.")
    latest_id = _latest_submission_id()
    if latest_id and not st.session_state.get("qa_submission_id"):
        st.session_state.qa_submission_id = int(latest_id)

    qa_submission_id = st.number_input("Submission ID for Q&A", min_value=1, step=1, key="qa_submission_id")
    if latest_id:
        st.caption(f"Latest available submission ID: {latest_id}")
    if st.button("Use Latest Submission ID"):
        if latest_id:
            st.session_state.qa_submission_id = int(latest_id)
            st.success(f"Using submission ID {latest_id}")
        else:
            st.warning("No submission found yet. Upload a paper first in Single Review.")
    qa_question = st.text_input(
        "Your question",
        value="What is the main contribution of this paper?",
        key="qa_question",
    )
    if st.button("Ask Question"):
        if not qa_question.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner("Generating answer from indexed chunks..."):
                res, err = _ask_question(int(qa_submission_id), qa_question)
            if err:
                st.error(f"Q&A failed: {err}")
                st.info("Try: 1) Use latest submission ID, 2) Reindex submission, 3) Ask shorter question.")
                st.session_state.qa_result = {}
            else:
                st.session_state.qa_result = res

    if st.button("Reindex Submission"):
        out, err = _reindex_submission(int(qa_submission_id))
        if err:
            st.error(f"Reindex failed: {err}")
        else:
            st.success(
                f"Indexed {out.get('indexed_chunks', 0)} chunks for submission {out.get('submission_id')}"
            )

    if st.session_state.qa_result:
        st.write(f"Mode: {st.session_state.qa_result.get('mode', 'unknown')}")
        if st.session_state.qa_result.get("mode") == "rag-fallback":
            banner_text = _fallback_reason_text()
            if "Free demo mode active" in banner_text or "Local LLM active" in banner_text:
                st.info(banner_text)
            else:
                st.warning(banner_text)
        st.info(st.session_state.qa_result.get("answer", ""))
        st.write("Citations:")
        st.write(st.session_state.qa_result.get("citations", []))
    else:
        st.info("No answer yet. Select a valid submission ID and click 'Ask Question'.")

with tab_ranking:
    st.markdown("Paper scoring/ranking for decision support.")
    if st.button("Refresh Ranking"):
        data, err = _fetch_rankings()
        if err:
            st.error(f"Ranking fetch failed: {err}")
        else:
            st.session_state.rankings = data

    if st.session_state.rankings:
        st.dataframe(st.session_state.rankings, use_container_width=True)
        rank_df = pd.DataFrame(st.session_state.rankings)
        if not rank_df.empty:
            st.bar_chart(rank_df.set_index("title")["ranking_score"])

with tab_feed:
    st.markdown("Real-time research monitoring from arXiv.")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Sync Feed Now"):
            out, err = _sync_feed()
            if err:
                st.error(f"Feed sync failed: {err}")
            else:
                st.success(f"Inserted {out.get('inserted', 0)} new papers")
    with col_b:
        if st.button("Refresh Feed List"):
            items, err = _fetch_feed()
            if err:
                st.error(f"Feed fetch failed: {err}")
            else:
                st.session_state.feed_items = items

    if st.session_state.feed_items:
        st.dataframe(st.session_state.feed_items, use_container_width=True)

    st.markdown("### Semantic Scholar Live Search")
    topic = st.text_input("Topic", value="large language models for education", key="live_topic")
    limit = st.slider("Results", min_value=3, max_value=20, value=10, step=1, key="live_limit")
    if st.button("Search Live Papers"):
        items, err = _search_live_papers(topic, limit)
        if err:
            st.error(f"Live search failed: {err}")
        else:
            st.session_state.live_search_items = items

    if st.session_state.live_search_items:
        st.dataframe(st.session_state.live_search_items, use_container_width=True)

with tab_plag:
    st.markdown("Similarity-based plagiarism risk check.")
    plag_submission_id = st.number_input("Submission ID", min_value=1, step=1, key="plag_submission")
    if st.button("Run Plagiarism Check"):
        out, err = _check_plagiarism(int(plag_submission_id))
        if err:
            st.error(f"Plagiarism check failed: {err}")
        else:
            st.session_state.plag_result = out

    if st.session_state.plag_result:
        st.write(f"Risk: **{st.session_state.plag_result.get('risk_level', 'unknown')}**")
        st.write(f"Max similarity: {st.session_state.plag_result.get('max_similarity', 0)}")
        st.dataframe(st.session_state.plag_result.get("matches", []), use_container_width=True)

with tab_agents:
    st.markdown("Multi-agent workflow (Researcher -> Analyzer -> Writer -> Reviewer).")
    agent_submission_id = st.number_input("Submission ID", min_value=1, step=1, key="agent_submission")
    if st.button("Run Multi-Agent Pipeline"):
        out, err = _run_multi_agent(int(agent_submission_id))
        if err:
            st.error(f"Agent pipeline failed: {err}")
        else:
            st.session_state.agent_result = out

    if st.session_state.agent_result:
        st.json(st.session_state.agent_result)

with tab_analytics:
    st.markdown("### Submission History")
    if st.button("Refresh History"):
        st.session_state.history = _fetch_history()

    if st.session_state.history:
        st.dataframe(st.session_state.history, use_container_width=True)
    else:
        st.info("No submissions found yet.")

    _render_dashboard(st.session_state.history)
    _render_failure_analysis(st.session_state.history)
    _render_cohort_insights(st.session_state.history)

    st.markdown("### Operational Metrics")
    if st.button("Refresh Operational Metrics"):
        metrics, err = _fetch_metrics_summary()
        if err:
            st.error(f"Metrics fetch failed: {err}")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Submissions", metrics.get("submissions_count", 0))
            c2.metric("Jobs", metrics.get("jobs_count", 0))
            c3.metric("Pending Jobs", metrics.get("pending_jobs", 0))
            c4, c5 = st.columns(2)
            c4.metric("Avg AI Score", metrics.get("average_ai_score", 0.0))
            c5.metric("Avg Human Score", metrics.get("average_human_score", 0.0))

with tab_override:
    st.markdown("Reviewer/Admin can override AI score with final human score.")
    if st.session_state.role not in {"reviewer", "admin"}:
        st.warning("Only reviewer/admin accounts can perform human override.")
    else:
        submission_id_input = st.number_input("Submission ID", min_value=1, step=1)
        human_score_input = st.slider("Human Final Score", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
        reviewer_comment_input = st.text_area("Reviewer Comment", placeholder="Final decision notes")

        if st.button("Submit Human Override"):
            try:
                res = requests.patch(
                    f"{BACKEND_URL}/submissions/{int(submission_id_input)}/review",
                    json={
                        "human_score": float(human_score_input),
                        "reviewer_comment": reviewer_comment_input,
                    },
                    headers=_api_headers(),
                    timeout=20,
                )
                if res.status_code == 200:
                    st.success("Human override saved successfully.")
                    st.json(res.json())
                    st.session_state.history = _fetch_history()
                else:
                    st.error(f"Override failed: {res.text}")
            except requests.RequestException as exc:
                st.error(f"Override request failed: {exc}")

st.markdown("---")
if os.getenv("USE_OPENAI", "true").strip().lower() in {"1", "true", "yes", "on"}:
    st.caption("Tip: set OPENAI_API_KEY in .env for better AI quality.")
else:
    st.caption("Tip: Local demo mode active (Ollama + RAG).")
