import unittest

import numpy as np
import pandas as pd

from lpcv2026.evaluation import (
    ImageRecord,
    dataset_integrity,
    evaluate_stage,
    manifest_digest,
    per_query_retrieval,
    quantization_acceptance,
)


class RetrievalMetricTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            ImageRecord("a.jpg", "/unused/a.jpg", (10, 11), "a"),
            ImageRecord("b.jpg", "/unused/b.jpg", (12,), "b"),
        ]
        self.text_positions = {10: 0, 11: 1, 12: 2, 13: 3}

    def test_fractional_and_hit_recall_are_distinct(self):
        similarity = np.array(
            [
                [0.9, 0.1, 0.8, 0.7],
                [0.9, 0.8, 0.1, 0.7],
            ],
            dtype=np.float32,
        )
        fractional, hit, ranks = per_query_retrieval(
            similarity, self.records, self.text_positions, k=2
        )
        np.testing.assert_allclose(fractional, [0.5, 0.0])
        np.testing.assert_allclose(hit, [1.0, 0.0])
        np.testing.assert_array_equal(ranks, [1, 4])

    def test_evaluate_stage_normalizes_embeddings(self):
        image = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
        text = np.array(
            [[4.0, 0.0], [3.0, 0.0], [0.0, 7.0], [-1.0, 0.0]],
            dtype=np.float32,
        )
        result = evaluate_stage(image, text, self.records, self.text_positions, ks=(1,))
        self.assertEqual(result["fractional_recall_at_1"], 0.75)
        self.assertEqual(result["hit_recall_at_1"], 1.0)

    def test_missing_positive_matches_organizer_denominator(self):
        records = [ImageRecord("a.jpg", "/unused/a.jpg", (10, 99), "a")]
        similarity = np.array([[0.9, 0.1]], dtype=np.float32)
        fractional, hit, ranks = per_query_retrieval(
            similarity, records, {10: 0, 11: 1}, k=1
        )
        np.testing.assert_allclose(fractional, [0.5])
        np.testing.assert_allclose(hit, [1.0])
        np.testing.assert_array_equal(ranks, [1])
        integrity = dataset_integrity(records, {10: 0, 11: 1}, k=1)
        self.assertEqual(integrity["dangling_positive_text_ids"], [99])
        self.assertEqual(integrity["fractional_recall_at_1_attainable_ceiling"], 0.5)

    def test_quantization_gate_rejects_low_fidelity(self):
        result = quantization_acceptance(
            {"fractional_recall_at_10": 0.90},
            {"fractional_recall_at_10": 0.89},
            {"cosine_mean": 0.81},
            min_cosine=0.95,
            max_fractional_r10_drop=0.02,
        )
        self.assertEqual(result["status"], "rejected")
        self.assertFalse(result["checks"]["image_cosine"]["passed"])
        self.assertTrue(result["checks"]["fractional_recall_at_10_drop"]["passed"])

    def test_manifest_digest_does_not_depend_on_absolute_path(self):
        texts = pd.DataFrame({"Text_nums": [10, 11], "Unique_Texts": ["alpha", "beta"]})
        records_a = [ImageRecord("a.jpg", "/one/a.jpg", (10, 11), "abc")]
        records_b = [ImageRecord("a.jpg", "/two/a.jpg", (10, 11), "abc")]
        self.assertEqual(
            manifest_digest(records_a, texts), manifest_digest(records_b, texts)
        )


if __name__ == "__main__":
    unittest.main()
