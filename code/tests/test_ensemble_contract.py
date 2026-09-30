"""Checks specific to the selected two-model logit ensemble."""

import unittest

import torch

from student_ensemble import build_model


class EnsembleContractTests(unittest.TestCase):
    def test_logit_mix_is_causal_normalized_and_independent(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        model = build_model(dict(vocab=2048, width=32, heads=4, depth=2,
                                 context=256, ffn_hidden=88, gelu_weight=0.5,
                                 ensemble_method='logit')).eval()
        ids = torch.randint(0, 2048, (2, 12))
        changed = ids.clone()
        changed[:, 7:] = (changed[:, 7:] + 19) % 2048
        with torch.no_grad():
            original = model.predict_log_probs(ids)
            future_changed = model.predict_log_probs(changed)
            alone = model.predict_log_probs(ids[:1])
            repeat = model.predict_log_probs(ids)
        self.assertTrue(torch.isfinite(original).all())
        torch.testing.assert_close(original.logsumexp(-1),
                                   torch.zeros_like(original[..., 0]),
                                   atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(original[:, :7], future_changed[:, :7],
                                   atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(original[:1], alone, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(original, repeat, atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
