"""Focused checks for the optional causal within-window cache."""

import unittest

import torch

from student_cache import build_model


class CacheContractTests(unittest.TestCase):
    def test_recency_weighted_cache_is_normalized(self):
        ids = torch.tensor([[1, 2, 3, 1, 2, 4, 1, 2]])
        for mode in ('unigram', 'following', 'bigram', 'trigram', 'variable'):
            with self.subTest(mode=mode):
                model = build_model(dict(vocab=2048, width=32, heads=4, depth=2,
                                         context=256, cache_mode=mode,
                                         cache_strength=.2, cache_tau=64.0)).eval()
                with torch.no_grad():
                    logp = model.predict_log_probs(ids)
                torch.testing.assert_close(logp.logsumexp(-1),
                                           torch.zeros_like(logp[..., 0]), atol=1e-5, rtol=1e-5)

    def test_causality_normalization_and_independence(self):
        torch.set_num_threads(2)
        for backbone in ('student', 'baseline'):
            for mode in ('unigram', 'following', 'bigram', 'trigram', 'variable'):
                with self.subTest(backbone=backbone, mode=mode):
                    torch.manual_seed(17)
                    model = build_model(dict(vocab=2048, width=32, heads=4, depth=2,
                                             context=256, backbone=backbone,
                                             cache_mode=mode, cache_strength=.2)).eval()
                    ids = torch.randint(0, 2048, (2, 12))
                    ids[:, 1] = ids[:, 0]
                    changed = ids.clone()
                    changed[:, 7:] = (changed[:, 7:] + 19) % 2048
                    with torch.no_grad():
                        original = model.predict_log_probs(ids)
                        future_changed = model.predict_log_probs(changed)
                        alone = model.predict_log_probs(ids[:1])
                        repeat = model.predict_log_probs(ids)
                    self.assertTrue(torch.isfinite(original).all())
                    torch.testing.assert_close(original.logsumexp(-1),
                                               torch.zeros_like(original[..., 0]), atol=1e-5, rtol=1e-5)
                    torch.testing.assert_close(original[:, :7], future_changed[:, :7], atol=1e-6, rtol=1e-6)
                    torch.testing.assert_close(original[:1], alone, atol=1e-5, rtol=1e-5)
                    torch.testing.assert_close(original, repeat, atol=1e-6, rtol=1e-6)
