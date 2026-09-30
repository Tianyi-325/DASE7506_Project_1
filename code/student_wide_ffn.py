"""GELU GPT with a configurable feed-forward expansion at fixed attention width."""

from torch import nn

from model import GPT


class WideFFNGPT(GPT):
    def __init__(self, config):
        super().__init__(config)
        width = config['width']
        expansion = float(config.get('ffn_expansion', 4.0))
        if expansion < 1:
            raise ValueError('ffn_expansion must be at least 1.')
        hidden = round(width * expansion)
        for block in self.blocks:
            block.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(),
                                      nn.Linear(hidden, width))
            block.mlp.apply(self.initialize)


def build_model(config):
    return WideFFNGPT(config)
