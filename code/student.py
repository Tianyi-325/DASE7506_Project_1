"""A parameter-matched SwiGLU alternative to the classroom GPT baseline.

Only the feed-forward sublayer changes. Attention, learned positions, model
width/depth, and the training/evaluation interfaces are inherited from GPT.
"""

from torch import nn
from torch.nn import functional as F

from model import GPT


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.gate_and_value = nn.Linear(width, 2 * hidden)
        self.proj = nn.Linear(hidden, width)

    def forward(self, x):
        gate, value = self.gate_and_value(x).chunk(2, dim=-1)
        return self.proj(F.silu(gate) * value)


class StudentGPT(GPT):
    def __init__(self, config):
        super().__init__(config)
        width = config['width']
        hidden = config.get('ffn_hidden', round(8 * width / 3))
        for block in self.blocks:
            block.mlp = SwiGLU(width, hidden)
            block.mlp.apply(self.initialize)


def build_model(config):
    return StudentGPT(config)
