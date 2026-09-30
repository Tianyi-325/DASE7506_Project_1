"""Train-time dropout variant of the GELU GPT; inference uses the same weights."""

import torch
from torch.nn import functional as F

from model import Block, GPT


class DropoutBlock(Block):
    def __init__(self, width, heads, dropout_p):
        super().__init__(width, heads)
        self.dropout_p = dropout_p

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                  dropout_p=self.dropout_p if self.training else 0.0)
        attention_out = self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        x = x + F.dropout(attention_out, p=self.dropout_p, training=self.training)
        mlp_out = self.mlp(self.norm2(x))
        return x + F.dropout(mlp_out, p=self.dropout_p, training=self.training)


class DropoutGPT(GPT):
    def __init__(self, config):
        super().__init__(config)
        dropout_p = float(config.get('dropout_p', 0.1))
        if not 0 <= dropout_p < 1:
            raise ValueError('dropout_p must be in [0, 1).')
        width, heads = config['width'], config['heads']
        for index, old in enumerate(self.blocks):
            new = DropoutBlock(width, heads, dropout_p)
            new.load_state_dict(old.state_dict())
            self.blocks[index] = new


def build_model(config):
    return DropoutGPT(config)
