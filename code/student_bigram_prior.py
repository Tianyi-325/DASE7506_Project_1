"""Self-trained dual GELU with a compact train-text bigram logit correction."""

import torch
from torch import nn
from torch.nn import functional as F

from student_dual_gelu import build_model as build_dual_gelu


class BigramPriorEnsemble(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context = config['context']
        self.vocab = config['vocab']
        self.base = build_dual_gelu(config)
        self.prior_strength = float(config.get('prior_strength', 0.0))
        self.register_buffer('log_ratio', torch.zeros(self.vocab, self.vocab))
        if self.prior_strength < 0:
            raise ValueError('prior_strength must be nonnegative.')

    def forward(self, ids):
        return self.base(ids)

    def predict_log_probs(self, ids):
        logits = self.base(ids).float() / self.base.logit_temperature
        logits = logits + self.prior_strength * self.log_ratio[ids]
        return F.log_softmax(logits, dim=-1)


def build_model(config):
    return BigramPriorEnsemble(config)
