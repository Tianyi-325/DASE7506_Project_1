"""Logit or probability ensemble of two train-only GPT checkpoints."""

import math
from importlib import import_module

import torch
from torch import nn
from torch.nn import functional as F


class MixedEnsemble(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context = config['context']
        self.first = import_module(config['first_implementation']).build_model(config['first_config'])
        self.second = import_module(config['second_implementation']).build_model(config['second_config'])
        self.first_weight = float(config['first_weight'])
        self.logit_temperature = float(config['logit_temperature'])
        self.ensemble_method = config.get('ensemble_method', 'logit')
        if self.context != 256 or self.first.context != 256 or self.second.context != 256:
            raise ValueError('All models must use independent 256-token windows.')
        if not 0 < self.first_weight < 1 or self.logit_temperature <= 0:
            raise ValueError('Invalid ensemble weight or temperature.')
        if self.ensemble_method not in ('logit', 'prob'):
            raise ValueError('ensemble_method must be logit or prob.')

    def forward(self, ids):
        return self.first_weight * self.first(ids) + (1 - self.first_weight) * self.second(ids)

    def predict_log_probs(self, ids):
        if self.ensemble_method == 'prob':
            a = F.log_softmax(self.first(ids).float() / self.logit_temperature, dim=-1)
            b = F.log_softmax(self.second(ids).float() / self.logit_temperature, dim=-1)
            return torch.logaddexp(a + math.log(self.first_weight),
                                   b + math.log1p(-self.first_weight))
        return F.log_softmax(self(ids).float() / self.logit_temperature, dim=-1)


def build_model(config):
    return MixedEnsemble(config)
