import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None
    RestormerLiteStudent = None
else:
    from uie_dis import RestormerLiteStudent


@unittest.skipIf(torch is None, "torch is not installed in the current environment")
class RestormerLiteStudentSmokeTest(unittest.TestCase):
    def test_forward_returns_expected_keys_and_shapes(self) -> None:
        model = RestormerLiteStudent(enable_color_head=True).eval()
        x_t = torch.randn(2, 3, 256, 256)
        y = torch.randn(2, 3, 256, 256)
        t = torch.tensor([0.0, 32.0])

        with torch.no_grad():
            outputs = model(x_t, y, t)

        self.assertEqual(set(outputs.keys()), {"x0_pred", "residual_pred", "color_map", "features", "meta"})
        self.assertEqual(outputs["x0_pred"].shape, x_t.shape)
        self.assertEqual(outputs["residual_pred"].shape, x_t.shape)
        self.assertEqual(outputs["color_map"].shape, x_t.shape)
        self.assertEqual(len(outputs["features"]["encoder"]), 4)
        self.assertEqual(len(outputs["features"]["condition"]), 5)
        self.assertEqual(len(outputs["meta"]["stage_specs_256"]), 9)
        self.assertEqual(outputs["meta"]["fusion_mode"], "gated")
        self.assertIn("gated-cross-attn", outputs["meta"]["available_fusion_modes"])

    def test_forward_supports_non_256_resolution_via_padding(self) -> None:
        model = RestormerLiteStudent(enable_color_head=False).eval()
        x_t = torch.randn(1, 3, 320, 448)
        y = torch.randn(1, 3, 320, 448)
        t = torch.tensor([8.0])

        with torch.no_grad():
            outputs = model(x_t, y, t)

        self.assertEqual(outputs["x0_pred"].shape, x_t.shape)
        self.assertEqual(outputs["residual_pred"].shape, x_t.shape)
        self.assertIsNone(outputs["color_map"])
        self.assertEqual(outputs["meta"]["original_shape"], (320, 448))

    def test_forward_supports_cross_attention_fusion_mode(self) -> None:
        model = RestormerLiteStudent(enable_color_head=False, fusion_mode="cross-attn").eval()
        x_t = torch.randn(1, 3, 64, 64)
        y = torch.randn(1, 3, 64, 64)
        t = torch.tensor([4.0])

        with torch.no_grad():
            outputs = model(x_t, y, t)

        self.assertEqual(outputs["x0_pred"].shape, x_t.shape)
        self.assertEqual(outputs["meta"]["fusion_mode"], "cross-attn")
        self.assertIn("cross-attention", outputs["meta"]["fusion"])

    def test_forward_supports_gated_cross_attention_fusion_mode(self) -> None:
        model = RestormerLiteStudent(enable_color_head=False, fusion_mode="gated-cross-attn").eval()
        x_t = torch.randn(1, 3, 64, 64)
        y = torch.randn(1, 3, 64, 64)
        t = torch.tensor([12.0])

        with torch.no_grad():
            outputs = model(x_t, y, t)

        self.assertEqual(outputs["residual_pred"].shape, x_t.shape)
        self.assertEqual(outputs["meta"]["fusion_mode"], "gated-cross-attn")
        self.assertIn("gated lightweight conditional cross-attention", outputs["meta"]["fusion"])

    def test_invalid_fusion_mode_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            RestormerLiteStudent(fusion_mode="invalid-mode")


if __name__ == "__main__":
    unittest.main()
