"""Contract checks for the exploratory routed-FFN GPT."""

import unittest

import torch
from torch.nn import functional as F

from student_moe import build_model


class MoEContractTests(unittest.TestCase):
    def test_router_trains_and_predictions_are_causal(self):
        torch.set_num_threads(2)
        torch.manual_seed(41)
        model = build_model(dict(vocab=2048, width=32, heads=4, depth=2,
                                 context=256, dropout_p=0.1, experts=4,
                                 expert_hidden=64, shared_hidden=32))
        ids = torch.randint(2048, (2, 13))
        loss = F.cross_entropy(model(ids[:, :-1]).flatten(0, 1), ids[:, 1:].flatten())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.blocks[0].mlp.router.weight.grad.abs().sum().item(), 0)
        model.eval()
        changed = ids.clone()
        changed[:, 8:] = (changed[:, 8:] + 1) % 2048
        with torch.no_grad():
            original = model.predict_log_probs(ids)
            altered = model.predict_log_probs(changed)
        torch.testing.assert_close(original[:, :8], altered[:, :8], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(original.logsumexp(-1), torch.zeros_like(ids, dtype=torch.float))


if __name__ == '__main__':
    unittest.main()
