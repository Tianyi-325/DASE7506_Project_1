"""SwiGLU GPT with a causal, window-local token cache at inference.

The cache is reconstructed from each input window on every call. No state or
information passes between windows, and no reference targets are accessed.
"""

import torch
from torch.nn import functional as F

from model import GPT
from student import StudentGPT


class CacheMixin:
    def __init__(self, config):
        super().__init__(config)
        self.cache_mode = config.get('cache_mode', 'none')
        self.cache_strength = float(config.get('cache_strength', 0.0))
        self.cache_tau = float(config.get('cache_tau', 0.0))
        self.cache_bonus = float(config.get('cache_bonus', 0.0))
        self.cache_count_scale = float(config.get('cache_count_scale', 0.0))
        if self.cache_mode not in ('none', 'unigram', 'following', 'bigram',
                                   'trigram', 'variable'):
            raise ValueError(f'Unknown cache mode: {self.cache_mode}')
        if not 0 <= self.cache_strength < 1:
            raise ValueError('cache_strength must be in [0, 1).')
        if self.cache_bonus < 0 or self.cache_count_scale < 0:
            raise ValueError('Cache bonus and count scale must be nonnegative.')

    def predict_log_probs(self, ids):
        neural_logp = F.log_softmax(self(ids).float(), dim=-1)
        if self.cache_mode == 'none' or self.cache_strength == 0 or ids.shape[1] < 2:
            return neural_logp

        batch, length = ids.shape
        t = torch.arange(length, device=ids.device)[:, None]
        s = torch.arange(length - 1, device=ids.device)[None, :]
        causal = s < t

        if self.cache_mode == 'unigram':
            # Prefix tokens themselves supply the distribution of repeat tokens.
            cache_targets = ids[:, :-1]
            matches = causal.unsqueeze(0).expand(batch, -1, -1)
        else:
            # For each earlier occurrence of the current context, use its
            # observed successor. The most recent eligible successor is ids[t].
            cache_targets = ids[:, 1:]
            matches = (ids[:, :, None] == ids[:, None, :-1]) & causal
            bonus_mask = None
            if self.cache_mode in ('bigram', 'trigram', 'variable'):
                previous = torch.cat((ids[:, :1], ids[:, :-1]), dim=1)
                same_previous = (previous[:, :, None] == previous[:, None, :-1])
                same_previous = same_previous & (t > 0).unsqueeze(0) & (s > 0).unsqueeze(0)
                if self.cache_mode == 'variable':
                    bonus_mask = same_previous
                else:
                    matches = matches & same_previous
            if self.cache_mode == 'trigram':
                previous2 = torch.cat((ids[:, :2], ids[:, :-2]), dim=1)
                matches = matches & (previous2[:, :, None] == previous2[:, None, :-1])
                matches = matches & (t > 1).unsqueeze(0) & (s > 1).unsqueeze(0)

        weights = matches.float()
        if self.cache_mode == 'variable':
            weights = weights * (1 + self.cache_bonus * bonus_mask.float())
        if self.cache_tau > 0:
            weights = weights * torch.exp(-(t - s).clamp_min(0).float() / self.cache_tau)
        mass = weights.sum(dim=-1, keepdim=True)
        if self.cache_count_scale > 0:
            strength = self.cache_strength * mass / (mass + self.cache_count_scale)
        else:
            strength = self.cache_strength * (mass > 0).float()
        mixed = neural_logp.exp() * (1 - strength)
        cache_additions = strength * weights / mass.clamp_min(1e-12)
        mixed.scatter_add_(2, cache_targets[:, None, :].expand(-1, length, -1), cache_additions)
        return mixed.log()


class CachedStudent(CacheMixin, StudentGPT):
    pass


class CachedBaseline(CacheMixin, GPT):
    pass


def build_model(config):
    if config.get('backbone', 'student') == 'baseline':
        return CachedBaseline(config)
    return CachedStudent(config)
