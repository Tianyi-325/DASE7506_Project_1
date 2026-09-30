"""Bundle a train-only dense bigram correction with a frozen dual GELU."""

import argparse
from pathlib import Path

import torch

from bigram_prior_sweep import train_bigram_log_ratio
from common import load_data, sha
from student_bigram_prior import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--alpha', required=True, type=float)
    parser.add_argument('--strength', required=True, type=float)
    args = parser.parse_args()
    base = torch.load(args.base, map_location='cpu', weights_only=True)
    if base['implementation'] != 'student_dual_gelu':
        raise ValueError('Expected a dual-GELU checkpoint.')
    config = dict(base['config'])
    config.update(prior_strength=args.strength, prior_alpha=args.alpha)
    model = build_model(config)
    model.base.load_state_dict(base['model'])
    train_tokens = load_data()['train'][0]
    model.log_ratio.copy_(train_bigram_log_ratio(train_tokens, config['vocab'], args.alpha))
    checkpoint = {'protocol': base['protocol'], 'implementation': 'student_bigram_prior',
                  'config': config, 'model': model.state_dict(),
                  'parent_checkpoints': [sha(args.base)],
                  'train_tokens': base['train_tokens'],
                  'prior_train_transitions': len(train_tokens)-1,
                  'processed_targets_including_ancestry': base['train_tokens']+len(train_tokens)-1,
                  'seed': base['seed']}
    args.output.parent.mkdir(parents=True, exist_ok=False)
    torch.save(checkpoint, args.output)
    print(f'{args.output}: {sha(args.output)}')


if __name__ == '__main__':
    main()
