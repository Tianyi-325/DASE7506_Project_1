"""Fit train-only bigram ratios and sweep their logit weight on validation."""

import argparse
import json
from pathlib import Path

import torch

from common import load_data, setup, sha
from evaluate import score
from student_bigram_prior import build_model


def train_bigram_log_ratio(tokens, vocab, alpha):
    tokens = tokens.long().cpu()
    counts = torch.bincount(tokens[:-1] * vocab + tokens[1:], minlength=vocab*vocab)
    counts = counts.reshape(vocab, vocab).double()
    unigram_counts = torch.bincount(tokens[1:], minlength=vocab).double()
    unigram = (unigram_counts + 1) / (unigram_counts.sum() + vocab)
    row_counts = counts.sum(-1, keepdim=True)
    conditional = (counts + alpha * unigram[None, :]) / (row_counts + alpha)
    return (conditional.log() - unigram.log()[None, :]).float()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--alphas', nargs='+', type=float, default=[100.0, 1000.0])
    parser.add_argument('--strengths', nargs='+', type=float,
                        default=[0.0, 0.05, 0.1, 0.2, 0.3])
    args = parser.parse_args()
    device, _ = setup('cuda', 'fp32', 4)
    checkpoint = torch.load(args.base, map_location='cpu', weights_only=True)
    if checkpoint['implementation'] != 'student_dual_gelu':
        raise ValueError('Expected a dual-GELU checkpoint.')
    model = build_model(checkpoint['config']).to(device)
    model.base.load_state_dict(checkpoint['model'])
    data = load_data()
    rows = []
    for alpha in args.alphas:
        if alpha <= 0:
            raise ValueError('Smoothing alpha must be positive.')
        model.log_ratio.copy_(train_bigram_log_ratio(data['train'][0], model.vocab, alpha).to(device))
        for strength in args.strengths:
            model.prior_strength = strength
            result = score(model, *data['validation'], device, 'fp32')
            row = {'alpha': alpha, 'prior_strength': strength,
                   'bpb': result['bpb'], 'seconds': result['seconds']}
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'split': 'validation', 'precision': 'fp32',
                                       'base_sha256': sha(args.base),
                                       'train_scored_transitions': len(data['train'][0])-1,
                                       'candidates': rows}, indent=2)+'\n')


if __name__ == '__main__':
    main()
