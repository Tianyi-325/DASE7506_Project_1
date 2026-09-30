"""Small independent-window checks for RoPE and the mixed ensemble."""

import unittest

import torch

from student_mixed_ensemble import build_model as build_mixed
from student_rope import build_model as build_rope


class RoPEContractTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(37)
        config = dict(vocab=2048, width=32, heads=4, depth=2, context=256, dropout_p=0.1)
        self.rope = build_rope(config).eval()
        self.mixed = build_mixed(dict(
            vocab=2048, context=256, first_implementation='student_rope',
            first_config=config, second_implementation='model',
            second_config={key: value for key, value in config.items() if key != 'dropout_p'},
            first_weight=0.55, logit_temperature=1.075,
        )).eval()

    def test_causality_normalization_and_reset(self):
        x = torch.randint(2048, (2, 13))
        changed = x.clone()
        changed[:, 8:] = (changed[:, 8:] + 17) % 2048
        for model in (self.rope, self.mixed):
            with self.subTest(model=type(model).__name__), torch.no_grad():
                original = model.predict_log_probs(x)
                torch.testing.assert_close(original[:, :8], model.predict_log_probs(changed)[:, :8])
                torch.testing.assert_close(original.logsumexp(-1), torch.zeros_like(x, dtype=torch.float))
                torch.testing.assert_close(original[:1], model.predict_log_probs(x[:1]))
                model.predict_log_probs(changed)
                torch.testing.assert_close(original, model.predict_log_probs(x))


if __name__ == '__main__':
    unittest.main()
