"""Causal GPT with rotary query/key positions and optional train-time dropout."""

import torch
from torch import nn
from torch.nn import functional as F

from model import Block, GPT


class RoPEBlock(Block):
    def __init__(self, width, heads, context, dropout_p=0.0):
        super().__init__(width, heads)
        self.dropout_p = float(dropout_p)
        head_dim = width // heads
        if width % heads or head_dim % 2:
            raise ValueError('RoPE requires an even head dimension.')
        inverse = 10000.0 ** (-torch.arange(0, head_dim, 2).float() / head_dim)
        angles = torch.arange(context).float()[:, None] * inverse[None, :]
        angles = torch.cat((angles, angles), dim=-1)
        self.register_buffer('rope_cos', angles.cos()[None, None], persistent=False)
        self.register_buffer('rope_sin', angles.sin()[None, None], persistent=False)

    def rotate(self, tensor):
        half = tensor.shape[-1] // 2
        rotated = torch.cat((-tensor[..., half:], tensor[..., :half]), dim=-1)
        return tensor * self.rope_cos[:, :, :tensor.shape[-2]] + rotated * self.rope_sin[:, :, :tensor.shape[-2]]

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        q, k = self.rotate(q), self.rotate(k)
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        attention_out = self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        x = x + F.dropout(attention_out, p=self.dropout_p, training=self.training)
        return x + F.dropout(self.mlp(self.norm2(x)), p=self.dropout_p, training=self.training)


class RoPEGPT(GPT):
    def __init__(self, config):
        super().__init__(config)
        dropout_p = float(config.get('dropout_p', 0.0))
        if not 0 <= dropout_p < 1:
            raise ValueError('dropout_p must be in [0, 1).')
        self.pos = None
        for index, old in enumerate(self.blocks):
            new = RoPEBlock(config['width'], config['heads'], config['context'], dropout_p)
            new.load_state_dict(old.state_dict())
            self.blocks[index] = new

    def features(self, ids):
        x = self.token(ids)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def build_model(config):
    return RoPEGPT(config)
