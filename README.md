# AI-Based-Research-Paper-Review-System
Fast, accurate, and structured paper evaluation system

## Automated paper analysis

The repository now includes a lightweight analysis engine in
`paper_analysis.py` that can:

- Generate paper summaries automatically
- Identify strengths and weaknesses from content signals
- Provide multi-dimensional scoring (novelty, methodology, clarity, impact)
- Compare multiple papers and rank them by overall score

### Quick usage

```python
from paper_analysis import analyze_paper, compare_papers

analysis = analyze_paper("Paper 1", "Your paper text...")
ranking = compare_papers(
    [
        {"title": "Paper 1", "text": "Paper text 1"},
        {"title": "Paper 2", "text": "Paper text 2"},
    ]
)
```
