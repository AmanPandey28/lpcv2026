from pathlib import Path
import tempfile
import unittest

import numpy as np

from lpcv2026.deploy import check
from lpcv2026.quantize import calibration_samples


class CalibrationTests(unittest.TestCase):
    def samples(self, values, branch):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.npy"
            np.save(path, values)
            return calibration_samples(path, branch)

    def test_image_boundary_and_batch_one_upload(self):
        values = np.zeros((2, 3, 224, 224), dtype=np.float32)
        values[1] = 1
        samples = self.samples(values, "image")
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].shape, (1, 3, 224, 224))
        self.assertEqual(samples[1].dtype, np.float32)

    def test_image_rejects_normalized_or_wrong_dtype(self):
        for values in (
            np.full((1, 3, 224, 224), -1, dtype=np.float32),
            np.zeros((1, 3, 224, 224), dtype=np.float64),
        ):
            with self.assertRaises(ValueError):
                self.samples(values, "image")

    def test_text_boundary_and_token_range(self):
        values = np.full((2, 77), 49407, dtype=np.int64)
        samples = self.samples(values, "text")
        self.assertEqual(samples[0].shape, (1, 77))
        self.assertEqual(samples[0].dtype, np.int64)
        with self.assertRaises(ValueError):
            self.samples(np.full((1, 77), 49408, dtype=np.int64), "text")

    def test_empty_wrong_shape_and_nonfinite_rejected(self):
        for values in (
            np.zeros((0, 77), dtype=np.int64),
            np.zeros((1, 76), dtype=np.int64),
            np.full((1, 77), np.nan),
        ):
            with self.assertRaises(ValueError):
                self.samples(values, "text")

    def test_latency_gate_is_strict(self):
        self.assertFalse(check(35.0, 35.0, "maximum_exclusive")["passed"])
        self.assertTrue(check(34.9, 35.0, "maximum_exclusive")["passed"])
        self.assertFalse(check(float("nan"), 0.99, "minimum")["passed"])


if __name__ == "__main__":
    unittest.main()
