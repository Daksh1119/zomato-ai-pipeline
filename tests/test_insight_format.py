import json
import unittest

from ai.insight_format import normalise_review_insight, render_review_insight


class InsightFormatTests(unittest.TestCase):
    def test_removes_markdown_and_generic_ai_heading(self):
        insight = normalise_review_insight(
            '{"answer":"**Executive Takeaway** The sample is too narrow.",'
            '"findings":[{"finding":"* Limited evidence",'
            '"evidence":"- Five reviews were supplied.",'
            '"implication":"--- Do not rank cities."}]}'
        )

        self.assertEqual(insight["answer"], "The sample is too narrow.")
        self.assertEqual(insight["findings"][0]["finding"], "Limited evidence")
        self.assertEqual(insight["findings"][0]["implication"], "Do not rank cities.")

    def test_escapes_model_html_before_rendering(self):
        insight = normalise_review_insight(
            '{"answer":"<script>alert(1)</script>Evidence is limited.", "findings": []}'
        )
        rendered = render_review_insight(insight)

        self.assertNotIn("<script", rendered)
        self.assertIn("Evidence is limited.", rendered)

    def test_limits_rendered_findings_to_three(self):
        findings = [
            {"finding": f"Finding {index}", "evidence": "Signal", "implication": "Meaning"}
            for index in range(4)
        ]
        insight = normalise_review_insight(
            json.dumps({"answer": "Answer", "findings": findings})
        )

        self.assertEqual(len(insight["findings"]), 3)


if __name__ == "__main__":
    unittest.main()
