"""RoPE GPT with unit-length query/key vectors before attention."""

import math

from torch.nn import functional as F

from student_rope import RoPEBlock, RoPEGPT


class QKNormRoPEBlock(RoPEBlock):
    def forward(self, x):
        batch, length, width = x.shape
        head_dim = width // self.heads
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, head_dim).permute(2, 0, 3, 1, 4)
        scale = math.sqrt(head_dim)
        q = self.rotate(F.normalize(q, dim=-1) * scale)
        k = self.rotate(F.normalize(k, dim=-1) * scale)
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        attention_out = self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        x = x + F.dropout(attention_out, p=self.dropout_p, training=self.training)
        return x + F.dropout(self.mlp(self.norm2(x)), p=self.dropout_p, training=self.training)


class QKNormRoPEGPT(RoPEGPT):
    def __init__(self, config):
        super().__init__(config)
        for index, old in enumerate(self.blocks):
            new = QKNormRoPEBlock(config['width'], config['heads'], config['context'], config.get('dropout_p', 0.0))
            new.load_state_dict(old.state_dict())
            self.blocks[index] = new


def build_model(config):
    return QKNormRoPEGPT(config)
