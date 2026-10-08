import sys
import types
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    import torch.nn.functional as F
    from lpcv2026.models import explicit_sdpa, batch_one_text_pool, export_rewrites


@unittest.skipIf(torch is None, "Install the model extra to test inference rewrites")
class RewriteTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.q = torch.randn(1, 2, 5, 8)
        self.k = torch.randn(1, 2, 7, 8)
        self.v = torch.randn(1, 2, 7, 8)

    def compare(self, **kwargs):
        expected = F.scaled_dot_product_attention(self.q, self.k, self.v, **kwargs)
        actual = explicit_sdpa(self.q, self.k, self.v, **kwargs)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_noncausal_attention(self):
        self.compare()

    def test_additive_mask(self):
        mask = torch.zeros(5, 7)
        mask[:, -1] = float("-inf")
        self.compare(attn_mask=mask)

    def test_boolean_mask_and_fully_masked_row(self):
        mask = torch.ones(5, 7, dtype=torch.bool)
        mask[0] = False
        mask[:, -1] = False
        self.compare(attn_mask=mask)

    def test_causal_nonsquare_attention(self):
        self.compare(is_causal=True)

    def test_explicit_scale(self):
        self.compare(scale=0.25)

    def test_training_dropout_and_gqa_rejected(self):
        for kwargs in (
            {"dropout_p": 0.1},
            {"enable_gqa": True},
            {"is_causal": True, "attn_mask": torch.ones(5, 7)},
        ):
            with self.assertRaises(ValueError):
                explicit_sdpa(self.q, self.k, self.v, **kwargs)

    def test_pool_preserves_eos_position(self):
        x = torch.randn(1, 5, 8)
        tokens = torch.tensor([[49406, 320, 49407, 0, 0]])
        torch.testing.assert_close(batch_one_text_pool(x, tokens, "argmax"), x[:, 2, :])

    def test_pool_rejects_nonstatic_batch(self):
        with self.assertRaises(ValueError):
            batch_one_text_pool(torch.randn(2, 5, 8), torch.zeros(2, 5), "argmax")

    def test_patch_context_restores_on_exception(self):
        # No OpenCLIP install or model weights are required for lifecycle coverage.
        parent = types.ModuleType("open_clip")
        child = types.ModuleType("open_clip.transformer")
        original_pool = lambda *args: None
        child.text_global_pool = original_pool
        parent.transformer = child
        original_sdpa = F.scaled_dot_product_attention
        original_fastpath = torch.backends.mha.get_fastpath_enabled()
        with patch.dict(
            sys.modules, {"open_clip": parent, "open_clip.transformer": child}
        ):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                with export_rewrites():
                    self.assertIs(F.scaled_dot_product_attention, explicit_sdpa)
                    self.assertFalse(torch.backends.mha.get_fastpath_enabled())
                    raise RuntimeError("test failure")
            self.assertIs(child.text_global_pool, original_pool)
        self.assertIs(F.scaled_dot_product_attention, original_sdpa)
        self.assertEqual(torch.backends.mha.get_fastpath_enabled(), original_fastpath)


if __name__ == "__main__":
    unittest.main()
