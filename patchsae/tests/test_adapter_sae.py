"""Checks for the joint LoRA + adapter-SAE building blocks."""
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sae_training.adapter_sae import (AdapterSAE, FrozenReLUSAE, apply_vision_lora, export_lora,
                                         merge_into_sae, scale_grad, set_lora_enabled, shift_subspace,
                                         youden_thresholds, backward_projected_aux, calibration_scale)


def tiny_gsae(d_in=8, d_sae=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    sd = dict(W_enc=torch.randn(d_in, d_sae, generator=g), b_enc=torch.randn(d_sae, generator=g) * 0.1,
              W_dec=F.normalize(torch.randn(d_sae, d_in, generator=g), dim=1), b_dec=torch.randn(d_in, generator=g) * 0.1)
    cfg = dict(d_in=d_in, d_sae=d_sae, l1_coefficient=8e-5, dtype=torch.float32, gated_sae=False,
               block_layer=-2, module_name='resid', class_token=False, expansion_factor=d_sae // d_in,
               model_name='openai/clip-vit-base-patch16', use_ghost_grads=False)
    return {'cfg': cfg, 'state_dict': sd}


def tiny_adapter(ck, k=3, d=6, seed=1):
    g = torch.Generator().manual_seed(seed)
    U, _ = torch.linalg.qr(torch.randn(ck['cfg']['d_in'], k, generator=g))
    return AdapterSAE(U, torch.randn(d, k, generator=g), ck['state_dict']['b_dec'])


class AdapterSAETest(unittest.TestCase):
    def test_calibration_bounds_nearly_silent_latents(self):
        residual = torch.ones(4, 8)
        decoded = residual * 1e-4
        applied, raw = calibration_scale(residual, decoded, max_scale=10)
        self.assertAlmostEqual(raw.item(), 10000, delta=.01)
        self.assertEqual(applied.item(), 10)
        exact, _ = calibration_scale(residual, residual / 2, max_scale=10)
        self.assertEqual(exact.item(), 2)
        silent, _ = calibration_scale(residual, torch.zeros_like(decoded), max_scale=10)
        self.assertEqual(silent.item(), 0)

    def test_youden_ties_report_achievable_separation(self):
        target = torch.tensor([[1., 2.], [1., 3.], [1., 3.]])
        reference = torch.tensor([[1., 1.], [1., 2.], [1., 2.]])
        theta, score = youden_thresholds(target, reference, min_target_fire=1 / 3)
        actual = (target > theta).float().mean(0) - (reference > theta).float().mean(0)
        torch.testing.assert_close(score, actual)
        self.assertEqual(score[0].item(), 0.)  # identical distributions
        self.assertAlmostEqual(score[1].item(), 2 / 3, places=6)

    def test_youden_matches_exhaustive_threshold_search(self):
        gen = torch.Generator().manual_seed(9)
        target = torch.randint(0, 4, (11, 7), generator=gen).float()
        reference = torch.randint(0, 4, (13, 7), generator=gen).float()
        theta, score = youden_thresholds(target, reference, min_target_fire=.3)
        for j in range(7):
            thresholds = torch.arange(-.5, 4, 1.)
            tpr = (target[:, j, None] > thresholds).float().mean(0)
            fpr = (reference[:, j, None] > thresholds).float().mean(0)
            best = (tpr - fpr)[tpr >= .3].max()
            torch.testing.assert_close(score[j], best)
            self.assertGreaterEqual((target[:, j] > theta[j]).float().mean().item(), .3)
        empty = reference[:0]
        no_ref, null = youden_thresholds(target, empty)
        self.assertTrue(torch.isfinite(no_ref).all())
        self.assertEqual(null.sum().item(), 0.)

    def test_projected_gradients_preserve_task_and_adapter_gradients(self):
        p = torch.nn.Parameter(torch.zeros(2))
        ad = torch.nn.Parameter(torch.tensor(2.))
        stats = backward_projected_aux(p[0], -3 * p[0] + 4 * p[1] + ad.square(), [p], [ad], .25)
        torch.testing.assert_close(p.grad, torch.tensor([1., .25]))
        torch.testing.assert_close(ad.grad, torch.tensor(4.))
        self.assertLess(stats['sae_task_cosine'], 0)
        self.assertAlmostEqual(stats['sae_grad_scale'], .0625)

    def test_projected_gradients_zero_task_and_unused_lora(self):
        p = torch.nn.Parameter(torch.zeros(2))
        unused = torch.nn.Parameter(torch.ones(1))
        ad = torch.nn.Parameter(torch.tensor(2.))
        backward_projected_aux(0 * p.sum(), p.sum() + ad.square(), [p, unused], [ad])
        torch.testing.assert_close(p.grad, torch.zeros(2))
        torch.testing.assert_close(unused.grad, torch.zeros(1))
        torch.testing.assert_close(ad.grad, torch.tensor(4.))
        p.grad = ad.grad = None
        backward_projected_aux(p.sum(), ad.square(), [p], [ad])
        torch.testing.assert_close(p.grad, torch.ones(2))
        torch.testing.assert_close(ad.grad, torch.tensor(4.))

    def test_merge_equals_gsae_plus_adapter(self):
        from eval_masked_sae import load_masked_sae
        ck = tiny_gsae(); ad = tiny_adapter(ck)
        g = FrozenReLUSAE(ck['state_dict'], 'cpu')
        x = torch.randn(5, 8)
        expected = g.decode(g.encode(x)) + ad.decode(ad.encode(x))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'merged.pt'
            torch.save(merge_into_sae(ck, ad), path)
            sae, cfg, mask = load_masked_sae(str(path), 'cpu')
            out, acts = sae(x)[:2]
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(acts[:, 16:], ad.encode(x), atol=1e-6, rtol=1e-6)
        self.assertEqual(int(mask.sum()), 16)
        self.assertEqual(cfg.d_sae, 22)

    def test_decoder_in_subspace_and_refresh_keeps_directions(self):
        ck = tiny_gsae(); ad = tiny_adapter(ck)
        U = ad.U
        resid = ad.W_dec - ad.W_dec @ U @ U.T
        self.assertLess(resid.abs().max().item(), 1e-6)
        torch.testing.assert_close(ad.W_dec.norm(dim=1), torch.ones(6))
        before = ad.W_dec.clone()
        Q, _ = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(5)))
        ad.refresh_subspace_(U @ Q)                   # same subspace, rotated basis
        torch.testing.assert_close(ad.W_dec, before, atol=1e-5, rtol=1e-5)

    def test_shift_subspace_energy(self):
        g = torch.Generator().manual_seed(2)
        basis, _ = torch.linalg.qr(torch.randn(16, 2, generator=g))
        delta = torch.randn(500, 2, generator=g) @ basis.T + 1e-3 * torch.randn(500, 16, generator=g)
        U, k, _ = shift_subspace(delta, energy=0.95)
        self.assertEqual(k, 2)
        self.assertGreater((U.T @ basis).svd().S.min().item(), 0.99)

    def test_scale_grad(self):
        for gamma, expected in [(0.0, 0.0), (0.1, 0.1), (1.0, 1.0)]:
            x = torch.ones(3, requires_grad=True)
            y = scale_grad(x, gamma)
            torch.testing.assert_close(y.detach(), x.detach())
            (y.sum() if y.requires_grad else (y.sum() + 0 * x.sum())).backward()
            torch.testing.assert_close(x.grad, torch.full((3,), expected))

    def test_reseed_only_touches_flagged(self):
        ck = tiny_gsae(); ad = tiny_adapter(ck)
        C, W, b = ad.C.clone(), ad.W_enc.clone(), ad.b_enc.clone()
        idx = torch.tensor([1, 4])
        ad.reseed_(idx, torch.randn(2, 3, generator=torch.Generator().manual_seed(7)))
        keep = torch.tensor([0, 2, 3, 5])
        torch.testing.assert_close(ad.C[keep], C[keep])
        torch.testing.assert_close(ad.W_enc[:, keep], W[:, keep])
        self.assertFalse(torch.allclose(ad.C[idx], C[idx]))

    def test_lora_export_reloads_to_same_activations(self):
        from transformers import CLIPConfig, CLIPModel
        from tasks.train_sae_lora_clip import _apply_custom_clip_lora
        cfg = CLIPConfig(vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                                            num_attention_heads=4, image_size=32, patch_size=8),
                         text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                          num_attention_heads=4), projection_dim=16)
        torch.manual_seed(0)
        base = CLIPModel(cfg).eval()
        lora_model = CLIPModel(cfg).eval()
        lora_model.load_state_dict(base.state_dict())
        wrapped = apply_vision_lora(lora_model, r=2, alpha=1, dropout=0.0)
        for _, _, m in wrapped:
            torch.nn.init.normal_(m.lora_B, std=0.5)
        px = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            ref = lora_model.vision_model(pixel_values=px).last_hidden_state
            set_lora_enabled(wrapped, False)
            off = lora_model.vision_model(pixel_values=px).last_hidden_state
            _apply_custom_clip_lora(base, export_lora(wrapped, 2, 1), 'cpu')
            merged = base.vision_model(pixel_values=px).last_hidden_state
        torch.testing.assert_close(merged, ref, atol=1e-5, rtol=1e-5)
        self.assertGreater((ref - off).abs().max().item(), 1e-3)   # LoRA actually changed something


if __name__ == '__main__':
    unittest.main()
