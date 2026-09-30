"""Probability ensemble of independently trained GELU and SwiGLU GPTs."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from model import GPT
from student import StudentGPT


class Ensemble(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        self.gelu = GPT(config)
        self.swiglu = StudentGPT(config)
        self.gelu_weight = float(config.get('gelu_weight', 0.5))
        self.ensemble_method = config.get('ensemble_method', 'prob')
        self.logit_temperature = float(config.get('logit_temperature', 1.0))
        if not 0 < self.gelu_weight < 1:
            raise ValueError('gelu_weight must be strictly between 0 and 1.')
        if self.ensemble_method not in ('prob', 'logit'):
            raise ValueError('ensemble_method must be prob or logit.')
        if self.logit_temperature <= 0:
            raise ValueError('logit_temperature must be positive.')

    def forward(self, ids):
        return self.gelu_weight * self.gelu(ids) + (1 - self.gelu_weight) * self.swiglu(ids)

    def predict_log_probs(self, ids):
        if self.ensemble_method == 'logit':
            return F.log_softmax(self(ids).float() / self.logit_temperature, dim=-1)
        a = F.log_softmax(self.gelu(ids).float(), dim=-1)
        b = F.log_softmax(self.swiglu(ids).float(), dim=-1)
        return torch.logaddexp(a + math.log(self.gelu_weight),
                               b + math.log1p(-self.gelu_weight))


def build_model(config):
    return Ensemble(config)
