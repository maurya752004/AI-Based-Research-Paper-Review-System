# AI Paper Review and Research Assistant System

## Title Page

**Project Title:** AI Paper Review and Research Assistant System  
**Program:** B.Tech Internship/Project Report  
**Student Name:** [Your Name]  
**Roll Number:** [Your Roll Number]  
**Department:** [Your Department]  
**Institute:** [Institute Name]  
**Session:** [Academic Year]  
**Supervisor:** [Faculty Name]  
**Organization/Internship Place:** [Organization Name]

---

## Certificate (Format Draft)

This is to certify that **[Student Name]**, Roll No. **[Roll Number]**, has successfully completed the internship/project titled **"AI Paper Review and Research Assistant System"** under my supervision during the academic session **[Year]**. The work presented in this report is original and carried out as part of the internship/project requirements for the award of B.Tech degree.

**Supervisor Signature:** ____________________  
**Head of Department:** ____________________  
**Date:** ____________________

---

## Declaration

I hereby declare that this report titled **"AI Paper Review and Research Assistant System"** is my original work carried out during the internship/project period under the guidance of **[Supervisor Name]**. This report has not been submitted to any other university or institute for the award of any degree or diploma.

**Student Signature:** ____________________  
**Name:** [Your Name]  
**Date:** ____________________

---

## Acknowledgement

I would like to express my sincere gratitude to my project supervisor **[Supervisor Name]** for continuous guidance and support throughout this work. I also thank the department faculty and my classmates for their valuable suggestions and encouragement. Finally, I thank my family for their support and motivation.

---

## Abstract

This project presents an AI-powered paper review and research assistant platform designed to support academic paper evaluation, summary generation, question answering, ranking, and decision support. The system allows users to upload research papers in PDF format, extract text, generate structured reviews, compare papers, perform plagiarism-style similarity checks, and ask context-aware questions over indexed paper content.

The solution is implemented using a **FastAPI backend**, **Streamlit frontend**, **SQLite database**, and a **RAG (Retrieval-Augmented Generation)** pipeline built on **ChromaDB**. For AI processing, the system supports both **OpenAI models** and a free local model option through **Ollama**, along with deterministic fallback logic for demo reliability. The platform also includes role-based login, async review jobs, audit logging, analytics dashboards, PDF report export, live paper feed integration, and human override of AI scores.

This project demonstrates a practical, low-cost, and explainable AI system for academic decision support. It is especially suitable for internship demonstrations, final-year projects, and classroom evaluation workflows where transparency and offline-friendly operation are important.

---

## Table of Contents

1. Introduction  
2. Objectives  
3. Problem Statement  
4. System Requirements  
5. Tools, Libraries, and APIs Used  
6. System Design and Architecture  
7. Module Description  
8. Implementation Details  
9. Week-Wise Development Report  
10. Results and Observations  
11. Challenges and Fixes  
12. Conclusion  
13. Future Scope  
14. References  
15. Appendix

---

## 1. Introduction

Research paper review is a time-consuming and often subjective process. Reviewers must read long documents, identify contributions, assess methodology, examine evidence, and make quality judgments under time constraints. This becomes more difficult when handling multiple submissions in academic or research environments.

The proposed system addresses this problem by building a complete AI-assisted paper review workflow. A user can upload a PDF, generate a structured review, store the submission, index paper content for retrieval, ask questions over the indexed content, compare papers side by side, view analytics, and allow a human reviewer to override the final decision when necessary.

The system is designed as a practical MVP with real implementation of backend APIs, frontend interaction, database persistence, local AI support, and operational reliability features.

---

## 2. Objectives

- Build an end-to-end AI-based academic paper review platform.
- Generate structured review outputs such as summary, strengths, weaknesses, recommendation, and numeric score.
- Provide rubric-based scoring for novelty, methodology, evaluation, reproducibility, and writing quality.
- Support section-wise summaries and section-wise scoring.
- Enable RAG-based question answering with citation metadata.
- Support paper comparison, ranking, live search, and research feed monitoring.
- Provide plagiarism-style similarity checks across stored submissions.
- Allow human override for final reviewer decision.
- Run in a low-cost local mode using Ollama for demos and academic use.

---

## 3. Problem Statement

Manual paper review is slow, difficult to scale, and may vary significantly between reviewers. In many student or demo settings, there is also a need for low-cost tooling that can run without continuous paid API usage. Existing basic review tools often lack transparency, citations, human control, and integrated workflow support.

Therefore, there is a need for a practical AI system that can:

- automate first-level paper analysis,
- provide structured and explainable review output,
- support question answering over paper content,
- help compare and rank submissions,
- and still keep the human reviewer in control of the final decision.

---

## 4. System Requirements

### 4.1 Hardware Requirements

- Laptop/Desktop with at least 8 GB RAM
- Multi-core CPU
- Stable internet connection for OpenAI, Semantic Scholar, arXiv, or OpenAlex based features
- Optional GPU for faster local model inference through Ollama

### 4.2 Software Requirements

- Operating System: Linux / Windows / macOS
- Python 3.10 or above
- FastAPI and Uvicorn
- Streamlit
- SQLite
- `pypdf` for PDF text extraction
- `chromadb` for vector storage
- `reportlab` for PDF report generation
- `apscheduler` for scheduled feed synchronization
- Ollama for local LLM mode
- OpenAI API key for cloud AI mode

---

## 5. Tools, Libraries, and APIs Used

### 5.1 Development Tools and Frameworks

| Tool/Framework | Purpose |
|---|---|
| FastAPI | Backend API development |
| Uvicorn | ASGI server for running backend |
| Streamlit | Frontend dashboard and user interface |
| SQLite | Local storage for submissions, jobs, audit logs, and paper feed |
| Ollama | Local LLM execution for free/offline-friendly demo mode |
| OpenAI API | Optional cloud model support for review and Q&A |

### 5.2 Python Libraries Used

| Library | Purpose in Project |
|---|---|
| `pypdf` | Extract text and page count from uploaded PDF files |
| `python-multipart` | Handle file upload in FastAPI |
| `python-dotenv` | Load environment variables from `.env` |
| `requests` | Call external APIs and local Ollama API |
| `reportlab` | Generate downloadable PDF review reports |
| `chromadb` | Store and retrieve indexed text chunks for RAG |
| `apscheduler` | Run scheduled background feed synchronization |
| `pydantic` | Data validation and response models |
| `pandas` | Analytics table preparation in frontend |
| `altair` | Score dashboard and chart visualization |

### 5.3 External APIs and Services

| API/Service | Usage |
|---|---|
| Ollama REST API | Local generation for review and Q&A |
| OpenAI Responses / Chat API | Cloud AI review and question answering |
| arXiv API | Fetch recent papers for live feed |
| Semantic Scholar Graph API | Search papers by topic |
| OpenAlex API | Optional novelty probing |

### 5.4 Core Modules Implemented

- `backend/main.py` for API routing, middleware, auth checks, and endpoint orchestration.
- `backend/services/review_service.py` for AI review generation, heuristic fallback, rubric scoring, novelty, confidence, and comparison.
- `backend/services/pdf_service.py` for PDF parsing.
- `backend/services/db_service.py` for SQLite schema creation and data operations.
- `backend/services/rag_service.py` for chunking, indexing, and retrieval.
- `backend/services/qa_service.py` for question answering over retrieved context.
- `backend/services/feed_service.py` and `live_search_service.py` for research feed sync and live paper search.
- `backend/services/plagiarism_service.py` for similarity checks.
- `backend/services/agent_service.py` for multi-agent style analysis flow.
- `frontend/app.py` for the complete user-facing dashboard.

---

## 6. System Design and Architecture

### 6.1 High-Level Workflow

1. User logs into the system using student, reviewer, or admin credentials.
2. User uploads a research paper in PDF format.
3. Backend extracts text and page count from the PDF.
4. Review service generates structured review data using local LLM, OpenAI, or fallback heuristics.
5. Submission data is stored in SQLite.
6. Full text is chunked and indexed into ChromaDB for retrieval.
7. User can ask questions, compare papers, rank submissions, or run plagiarism checks.
8. Reviewer/admin can override AI-generated score and add final comments.

### 6.2 Architecture Components

- **Frontend Layer:** Streamlit-based tabbed interface
- **API Layer:** FastAPI endpoints for review, QA, ranking, feed, plagiarism, and admin actions
- **Service Layer:** Modular Python services for review, RAG, feed sync, ranking, auth, and analytics
- **Persistence Layer:** SQLite for structured records and ChromaDB for text chunk retrieval
- **AI Layer:** Ollama local model, OpenAI cloud model, and deterministic fallback logic

### 6.3 Reliability Features

- Health, liveness, and readiness endpoints
- Request observability middleware with request ID and duration logging
- Async review jobs for long-running paper processing
- Audit trail events for login, review, override, QA, and feed sync
- ChromaDB corruption recovery by rotating damaged storage
- Metadata-based reindex fallback for older submissions

---

## 7. Module Description

### 7.1 Single Review Module

This module accepts one PDF and generates:

- summary
- strengths
- weaknesses
- recommendation
- overall score
- rubric scores
- section summaries
- section scores
- citation metrics
- novelty score
- confidence level
- evidence points

### 7.2 Compare Papers Module

This module compares two uploaded papers and produces:

- paper-wise scores
- winner decision
- rationale
- rubric comparison support

### 7.3 Research Assistant Module

This module performs RAG-based Q&A by retrieving relevant indexed chunks from ChromaDB and generating an answer with citation metadata.

### 7.4 Ranking Module

This module ranks submissions using a weighted combination of:

- AI review score
- citation quality
- novelty score
- confidence score

### 7.5 Live Feed and Search Module

This module supports:

- scheduled arXiv feed synchronization
- manual feed refresh
- live topic-based paper search using Semantic Scholar
- fallback to arXiv when live search fails

### 7.6 Plagiarism Risk Module

This module compares token overlap across stored submissions and reports:

- top matching submissions
- maximum similarity
- low/medium/high risk level

### 7.7 Multi-Agent Module

This module simulates a staged AI workflow:

- Researcher agent
- Analyzer agent
- Writer agent
- Reviewer agent

### 7.8 Analytics Module

This module shows:

- submission history
- score trend chart
- mode-wise count chart
- failure analysis
- cohort insights
- operational metrics

### 7.9 Human Override Module

Reviewer and admin users can override the AI score, add reviewer comments, and mark the record as human reviewed.

---

## 8. Implementation Details

### 8.1 Backend Implementation

The backend is implemented using FastAPI. It includes endpoints for authentication, PDF review, async jobs, submission history, metrics, QA, rankings, feed synchronization, live search, plagiarism checks, and multi-agent review. The backend also validates uploaded PDFs, enforces role-based access, writes audit logs, and stores processed results in SQLite.

### 8.2 Review Engine

The review engine supports three execution paths:

- **Local LLM mode** through Ollama
- **OpenAI mode** through cloud API
- **Fallback mode** using heuristic scoring and rule-based summarization

The review pipeline computes:

- overall score
- rubric scores
- score explanations
- section summaries
- section scores
- citation metrics
- novelty score
- confidence score

### 8.3 RAG Implementation

The RAG pipeline uses custom text chunking and a lightweight hash-based embedding function for local indexing in ChromaDB. For Q&A, the system retrieves top matching chunks and generates answers through Ollama or OpenAI. If the model response is weak or insufficient, the system falls back to an extractive answer built from relevant retrieved sentences.

### 8.4 Frontend Implementation

The frontend is implemented using Streamlit and organized into tabs for:

- Single Review
- Compare Papers
- Research Assistant
- Ranking
- Live Feed
- Plagiarism
- Multi-Agent
- Analytics
- Human Override

It also supports session-based login, result caching in session state, PDF report download, and analytics visualization.

### 8.5 Database Design

SQLite is used to store:

- submissions
- review jobs
- audit events
- paper feed records

The database layer also includes lightweight migration logic so older database files can be upgraded with newly added columns such as section scores, citation metrics, novelty score, confidence score, human score, reviewer comment, and source text.

### 8.6 Reliability and Demo Readiness

The project includes multiple improvements to make the system stable for real demonstrations:

- configurable local fast mode
- fallback when AI provider is unavailable
- weak-answer detection in Q&A
- metadata-based reindex support
- corruption recovery for local vector store
- async review job handling
- operational metrics and health checks

---

## 9. Week-Wise Development Report

The implementation can be presented as a 6-week development cycle based on the modules and functionality available in the project.

| Week | Work Completed | Tools/Libraries Used | Deliverable |
|---|---|---|---|
| Week 1 | Requirement analysis, architecture planning, project structure creation, backend and frontend skeleton setup | FastAPI, Uvicorn, Streamlit, python-dotenv | Basic project skeleton with API and UI startup |
| Week 2 | PDF upload pipeline, text extraction, single paper review flow, rubric scoring, section analysis, citation metrics, confidence and novelty logic | `pypdf`, `requests`, Ollama, OpenAI SDK, Pydantic | End-to-end review generation for one paper |
| Week 3 | SQLite schema design, submission storage, login system, role-based access, submission history, async review jobs, audit logging, metrics endpoints | SQLite, FastAPI security, Pydantic | Persistent and role-aware review system |
| Week 4 | RAG chunking, ChromaDB indexing, retrieval-based question answering, reindex feature, ranking logic | `chromadb`, custom hash embeddings, OpenAI/Ollama, requests | Research assistant and paper ranking features |
| Week 5 | Compare papers workflow, plagiarism similarity check, live paper feed, Semantic Scholar search, multi-agent review pipeline | APScheduler, arXiv API, Semantic Scholar API, requests | Advanced review support features |
| Week 6 | Dashboard analytics, PDF report download, fallback refinement, vector-store recovery, UI polishing, testing, and documentation | `pandas`, `altair`, `reportlab`, ChromaDB, Streamlit | Demo-ready and report-ready final system |

### 9.1 Detailed Weekly Description

#### Week 1: Planning and Setup

- Studied the problem of manual paper review and defined system goals.
- Finalized architecture with FastAPI backend and Streamlit frontend.
- Created project folders, schema structure, and startup configuration.
- Added environment-variable support for configurable deployment.

#### Week 2: Core Review Pipeline

- Implemented PDF upload and extraction flow.
- Built review generation service with support for local LLM, OpenAI, and fallback mode.
- Added summary, strengths, weaknesses, recommendation, score, keywords, rubric scores, and section-level analysis.
- Added citation metrics, novelty score, and confidence estimation.

#### Week 3: Persistence and Access Control

- Designed SQLite tables for submissions, review jobs, audit events, and paper feed.
- Added role-based login for student, reviewer, and admin accounts.
- Implemented submission history and protected endpoints.
- Added async review jobs to avoid blocking for large reviews.
- Added audit logging and operational metrics.

#### Week 4: RAG and Decision Support

- Implemented text chunking and indexing in ChromaDB.
- Built question answering endpoint with citations.
- Added reindexing logic for missing or legacy content.
- Implemented ranking logic using weighted academic quality signals.

#### Week 5: Advanced Features

- Built paper comparison with side-by-side scoring and winner selection.
- Added plagiarism-style similarity check across submissions.
- Integrated arXiv feed synchronization and Semantic Scholar live search.
- Implemented multi-agent review pipeline for staged analysis output.

#### Week 6: Reliability, Visualization, and Finalization

- Added analytics dashboard with charts and trend analysis.
- Added downloadable PDF review report.
- Improved demo reliability with fallback strategies and local fast mode.
- Added vector-store corruption recovery and extractive QA fallback.
- Prepared final documentation and project report material.

---

## 10. Results and Observations

- The system successfully accepts PDF papers and generates structured review results.
- RAG-based Q&A provides contextual answers with citation metadata.
- The compare module produces separate scores and a winner rationale.
- Ranking and analytics help summarize submission quality across multiple papers.
- Local mode enables a useful classroom/demo experience without mandatory paid API usage.
- Human override keeps the reviewer in control of the final decision.

### Sample Observations

- Better PDF extraction leads to better review quality.
- Response time depends on model size and context length.
- Local fast mode improves usability during demonstration.
- Citation-backed retrieval improves trust in the Q&A output.

---

## 11. Challenges and Fixes

### 11.1 Slow Local Model Response

**Challenge:** Local inference may become slow on large paper text.  
**Fix:** Reduced prompt size, made context configurable, and added local fast mode settings.

### 11.2 Weak or Insufficient Q&A Responses

**Challenge:** Model sometimes returned vague answers.  
**Fix:** Added weak-answer detection and extractive fallback from retrieved chunks.

### 11.3 Older Submissions Missing Full Text

**Challenge:** Some submissions may not have indexable full text available later.  
**Fix:** Added metadata-based fallback reindexing logic.

### 11.4 Local Vector Store Failure

**Challenge:** ChromaDB local storage can fail or become corrupted.  
**Fix:** Added automatic recovery by rotating the damaged directory and reinitializing storage.

### 11.5 Need for Human Validation

**Challenge:** AI-generated review should not be treated as a final academic verdict.  
**Fix:** Added human override with reviewer comments and role-based control.

---

## 12. Conclusion

The project successfully delivers a practical AI paper review and research assistant platform that combines document analysis, retrieval-based question answering, comparison, ranking, plagiarism-style similarity checks, analytics, and human review control in one integrated system. The design balances automation with transparency by exposing rubric scores, citation metadata, evidence points, and reviewer override capability.

Because the system supports both cloud AI and local AI execution, it is suitable for internship presentations, final-year demonstrations, and small academic workflows where cost, explainability, and reliability matter.

---

## 13. Future Scope

- Integrate stronger embedding models for better semantic retrieval.
- Add in-document citation highlighting and PDF annotation.
- Improve plagiarism detection using semantic similarity instead of token overlap only.
- Add reviewer collaboration workflow and shared comments.
- Extend ranking with conference-specific rubrics.
- Add Docker deployment and production-grade authentication.

---

## 14. References

1. FastAPI Documentation. https://fastapi.tiangolo.com/  
2. Streamlit Documentation. https://docs.streamlit.io/  
3. Ollama Documentation. https://ollama.com/  
4. ChromaDB Documentation. https://docs.trychroma.com/  
5. PyPDF Documentation. https://pypdf.readthedocs.io/  
6. OpenAI API Documentation. https://platform.openai.com/docs/  
7. APScheduler Documentation. https://apscheduler.readthedocs.io/  
8. Semantic Scholar API Documentation. https://api.semanticscholar.org/  
9. arXiv API Documentation. https://info.arxiv.org/help/api/index.html  
10. ReportLab User Guide. https://www.reportlab.com/documentation/  

---

## 15. Appendix

### A. Sample Environment Configuration

- `USE_OPENAI=false`
- `USE_LOCAL_LLM=true`
- `OLLAMA_MODEL=llama3`
- `OLLAMA_BASE_URL=http://127.0.0.1:11434`
- `AUTH_REQUIRED=true`
- `REVIEW_ALLOW_FALLBACK=true`
- `LOCAL_REVIEW_FAST_MODE=true`

### B. Major Features Visible in the Frontend

1. Single paper review
2. Async review jobs
3. Compare papers
4. Research assistant Q&A
5. Ranking dashboard
6. Live feed and live search
7. Plagiarism check
8. Multi-agent workflow
9. Analytics dashboard
10. Human override

### C. Suggested Screenshots to Insert

1. Login page and sidebar
2. Single review output page
3. Research assistant response with citations
4. Compare papers result
5. Ranking and analytics page
6. Human override form
