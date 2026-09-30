"""Attach a validation-selected causal cache to a trained classroom checkpoint."""

import argparse
from pathlib import Path

import torch

from common import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', choices=['unigram', 'following', 'bigram'], required=True)
    parser.add_argument('--strength', type=float, required=True)
    parser.add_argument('--tau', type=float, default=0.0)
    args = parser.parse_args()
    checkpoint = torch.load(args.base, map_location='cpu', weights_only=True)
    config = dict(checkpoint['config'])
    config.update(backbone='baseline' if checkpoint['implementation'] == 'model' else 'student',
                  cache_mode=args.mode, cache_strength=args.strength,
                  cache_tau=args.tau)
    checkpoint.update(implementation='student_cache', config=config,
                      parent_checkpoint_sha256=sha(args.base))
    args.output.parent.mkdir(parents=True, exist_ok=False)
    torch.save(checkpoint, args.output)
    print(f'{args.output}: {sha(args.output)}')


if __name__ == '__main__':
    main()
