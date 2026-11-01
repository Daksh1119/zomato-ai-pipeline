import json
import unittest

import pandas as pd

from ai.insight_format import normalise_review_insight, render_review_insight
from ai.review_queries import build_city_catalog, requests_city_catalog


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

    def test_city_catalog_queries_bypass_review_sampling(self):
        self.assertTrue(requests_city_catalog("List all cities"))
        self.assertTrue(requests_city_catalog("How many cities are covered?"))
        self.assertTrue(requests_city_catalog("Display cities in the dataset"))
        self.assertFalse(requests_city_catalog("Which city has the worst ratings?"))
        self.assertFalse(requests_city_catalog("What do Pune customers complain about?"))

    def test_city_catalog_uses_every_city_and_counts_reviews(self):
        reviews = pd.DataFrame(
            {"city": ["Pune", "Mumbai", "Pune", None, " ", "Bangalore"]}
        )

        catalog = build_city_catalog(reviews)

        self.assertEqual(catalog["city"].tolist(), ["Bangalore", "Mumbai", "Pune"])
        self.assertEqual(catalog["review_count"].tolist(), [1, 1, 2])


if __name__ == "__main__":
    unittest.main()
