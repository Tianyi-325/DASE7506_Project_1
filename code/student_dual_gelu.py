"""Two independently trained GELU GPTs combined in logit space."""

import torch
from torch import nn
from torch.nn import functional as F

from model import GPT


class DualGELU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context = config['context']
        self.first = GPT(config)
        self.second = GPT(config)
        self.first_weight = float(config.get('first_weight', 0.5))
        self.logit_temperature = float(config.get('logit_temperature', 1.0))
        if not 0 < self.first_weight < 1:
            raise ValueError('first_weight must be strictly between 0 and 1.')
        if self.logit_temperature <= 0:
            raise ValueError('logit_temperature must be positive.')

    def forward(self, ids):
        return self.first_weight * self.first(ids) + (1 - self.first_weight) * self.second(ids)

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float() / self.logit_temperature, dim=-1)


def build_model(config):
    return DualGELU(config)
