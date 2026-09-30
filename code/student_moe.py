"""Small train-from-scratch GPT with a shared FFN and token-routed FFN experts."""

import torch
from torch import nn

from student_dropout import DropoutGPT


def feed_forward(width, hidden):
    return nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))


class RoutedFFN(nn.Module):
    def __init__(self, width, experts, expert_hidden, shared_hidden):
        super().__init__()
        self.router = nn.Linear(width, experts)
        self.shared = feed_forward(width, shared_hidden)
        self.experts = nn.ModuleList(feed_forward(width, expert_hidden) for _ in range(experts))

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        scores = self.router(flat).sigmoid()
        chosen = scores.argmax(dim=-1)
        gate = scores.gather(1, chosen[:, None])
        result = self.shared(flat)
        for index, expert in enumerate(self.experts):
            positions = torch.nonzero(chosen == index).flatten()
            if positions.numel():
                values = expert(flat.index_select(0, positions))
                values = values * gate.index_select(0, positions)
                result = result.index_add(0, positions, values)
        return result.reshape(shape)


class MoEGPT(DropoutGPT):
    def __init__(self, config):
        super().__init__(config)
        width = config['width']
        for block in self.blocks:
            block.mlp = RoutedFFN(width, config['experts'],
                                  config['expert_hidden'], config['shared_hidden'])
            block.mlp.apply(self.initialize)


def build_model(config):
    return MoEGPT(config)
