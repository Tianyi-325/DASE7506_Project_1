"""Causal, token-local gate over two separately trained language models."""

import math
from importlib import import_module

import torch
from torch import nn
from torch.nn import functional as F


class GatedEnsemble(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context = config['context']
        self.first = import_module(config['first_implementation']).build_model(config['first_config'])
        self.second = import_module(config['second_implementation']).build_model(config['second_config'])
        if self.context != 256 or self.first.context != 256 or self.second.context != 256:
            raise ValueError('All models must use independent 256-token windows.')
        self.first_weight = float(config['first_weight'])
        self.logit_temperature = float(config['logit_temperature'])
        self.gate_strength = float(config.get('gate_strength', 1.0))
        if not 0 < self.first_weight < 1 or self.logit_temperature <= 0:
            raise ValueError('Invalid mixture configuration.')
        # Bias starts at the existing fixed-mixture optimum; the learned terms
        # use only each model's current-prefix logit margin and window position.
        self.gate_delta = nn.Parameter(torch.zeros(3))

    def forward(self, ids):
        a = self.first(ids)
        b = self.second(ids)
        margin_a = a.topk(2, dim=-1).values.diff(dim=-1).squeeze(-1)
        margin_b = b.topk(2, dim=-1).values.diff(dim=-1).squeeze(-1)
        position = (torch.arange(ids.shape[1], device=ids.device, dtype=a.dtype) + .5) / 256 - .5
        d = self.gate_delta * self.gate_strength
        gate = torch.sigmoid(math.log(self.first_weight / (1 - self.first_weight))
                             + d[0] + d[1] * (margin_a - margin_b)
                             + d[2] * position[None, :])
        return gate.unsqueeze(-1) * a + (1 - gate.unsqueeze(-1)) * b

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float() / self.logit_temperature, dim=-1)


def build_model(config):
    return GatedEnsemble(config)
