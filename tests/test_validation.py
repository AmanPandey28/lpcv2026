import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

from lpcv2026.evaluation import (
    corresponding_fidelity,
    load_track1_manifest,
    preprocess_track1_image,
    resolve_onnx_artifact,
    similarity_matrix,
)
from lpcv2026.hub import profile_summary


class ValidationTests(unittest.TestCase):
    def test_manifest_preserves_csv_order_and_sorts_text_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("Z.jpg", "a.jpg"):
                Image.new("RGB", (9, 7), (128, 64, 32)).save(root / name)
            pd.DataFrame(
                {"Image_names": ["a.jpg", "Z.jpg"], "Text_nums": ["12", "10;11"]}
            ).to_csv(root / "images.csv", index=False)
            pd.DataFrame(
                {"Text_nums": [12, 10, 11], "Unique_Texts": ["c", "a", "b"]}
            ).to_csv(root / "texts.csv", index=False)
            records, texts, positions, digest = load_track1_manifest(
                root / "images.csv", root / "texts.csv", root
            )
            self.assertEqual([r.name for r in records], ["a.jpg", "Z.jpg"])
            self.assertEqual(texts.Text_nums.tolist(), [10, 11, 12])
            self.assertEqual(positions, {10: 0, 11: 1, 12: 2})
            self.assertEqual(len(digest), 64)
            value = preprocess_track1_image(root / "a.jpg")
            self.assertEqual(value.shape, (1, 3, 224, 224))
            self.assertEqual(value.dtype, np.float32)
            self.assertTrue((value >= 0).all() and (value <= 1).all())

    def test_schema_validation_precedes_sort(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame({"Image_names": ["a.jpg"], "Text_nums": [10]}).to_csv(
                root / "images.csv", index=False
            )
            pd.DataFrame({"wrong": [10]}).to_csv(root / "texts.csv", index=False)
            with self.assertRaisesRegex(ValueError, "text CSV"):
                load_track1_manifest(root / "images.csv", root / "texts.csv", root)

    def test_similarity_and_fidelity_reject_nonfinite_or_wrong_shape(self):
        for candidate in (np.ones((3, 2)), np.full((2, 2), np.nan), np.array([])):
            with self.assertRaises(ValueError):
                corresponding_fidelity(np.ones((2, 2)), candidate)
        with self.assertRaises(ValueError):
            similarity_matrix(np.ones((2, 2)), np.full((3, 2), np.inf))

    def test_archive_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "bad.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../../../escaped.onnx", "invalid")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                resolve_onnx_artifact(archive_path, root / "cache")
            self.assertFalse((root / "escaped.onnx").exists())

    def test_profile_units_and_placement(self):
        result = profile_summary(
            {
                "execution_summary": {
                    "all_inference_times": [1000, 2000, 3000],
                    "estimated_inference_time": 1000,
                },
                "execution_detail": [{"compute_unit": "NPU"}, {"compute_unit": "CPU"}],
            }
        )
        self.assertEqual(result["latency_median_ms"], 2.0)
        self.assertAlmostEqual(result["latency_p90_ms"], 2.8)
        self.assertEqual(result["placement_operation_counts"], {"NPU": 1, "CPU": 1})

    def test_missing_profile_detail_is_not_npu_evidence(self):
        result = profile_summary({})
        self.assertEqual(result["placement_operation_counts"], {})
        self.assertEqual(result["iterations"], 0)
        self.assertNotIn("latency_p90_ms", result)


class HistoricalRecordTests(unittest.TestCase):
    def test_final_record_is_consistent(self):
        path = Path(__file__).resolve().parents[1] / "results/open_submission_fp16.json"
        report = json.loads(path.read_text())
        self.assertEqual(report["acceptance"]["status"], "accepted")
        self.assertEqual(report["dataset"]["num_images"], 56)
        self.assertEqual(report["dataset"]["num_texts"], 211)
        self.assertEqual(report["device"], "Samsung Galaxy S22 (Family)")
        self.assertEqual(
            report["profiles"]["image"]["placement_operation_counts"], {"NPU": 405}
        )
        self.assertEqual(
            report["profiles"]["text"]["placement_operation_counts"], {"NPU": 389}
        )
        actual_sum = sum(
            report["profiles"][branch]["latency_p90_ms"] for branch in ("image", "text")
        )
        self.assertAlmostEqual(actual_sum, 18.4977)
        self.assertAlmostEqual(actual_sum, report["profiles"]["sum_of_branch_p90_ms"])
        self.assertEqual(
            report["reference"]["metrics"]["fractional_recall_at_10"],
            report["device_metrics"]["fractional_recall_at_10"],
        )
        self.assertTrue(
            all(item["passed"] for item in report["acceptance"]["checks"].values())
        )

    def test_failed_integer_route_is_not_presented_as_runtime_success(self):
        path = (
            Path(__file__).resolve().parents[1] / "results/w8a16_runtime_lineage.json"
        )
        report = json.loads(path.read_text())
        for branch in ("image", "text"):
            self.assertEqual(report[branch]["compile_status"], "SUCCESS")
            self.assertEqual(report[branch]["profile_status"], "FAILED")
            self.assertIn("MODEL_GRAPH_ERROR", report[branch]["failure"])


if __name__ == "__main__":
    unittest.main()
