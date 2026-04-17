import unittest

from paper_analysis import analyze_paper, compare_papers, generate_summary


class PaperAnalysisTests(unittest.TestCase):
    def test_generate_summary_uses_first_sentences(self):
        text = "Sentence one. Sentence two! Sentence three?"
        self.assertEqual(generate_summary(text), "Sentence one. Sentence two!")

    def test_analyze_paper_finds_strengths_and_weaknesses(self):
        text = (
            "This novel method is clear and organized. "
            "Evaluation on a benchmark is included. "
            "However, the study has a limited dataset and missing baseline."
        )
        analysis = analyze_paper("Paper A", text)

        self.assertIn("Demonstrates novelty or innovation.", analysis["strengths"])
        self.assertIn("Provides empirical validation.", analysis["strengths"])
        self.assertIn("Uses limited data for validation.", analysis["weaknesses"])
        self.assertIn("Leaves important implementation or comparison gaps.", analysis["weaknesses"])

    def test_compare_papers_ranks_by_overall_score(self):
        papers = [
            {
                "title": "High Score",
                "text": "An innovative and robust benchmark study with clear and practical impact.",
            },
            {
                "title": "Low Score",
                "text": "An incremental study with unclear writing, narrow scope and missing baseline.",
            },
        ]

        ranked = compare_papers(papers)
        self.assertEqual(ranked[0]["title"], "High Score")
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertEqual(ranked[1]["title"], "Low Score")
        self.assertEqual(ranked[1]["rank"], 2)
        self.assertNotIn("Presents ideas clearly.", ranked[1]["strengths"])
        self.assertIn("Contains unclear or ambiguous explanations.", ranked[1]["weaknesses"])


if __name__ == "__main__":
    unittest.main()
