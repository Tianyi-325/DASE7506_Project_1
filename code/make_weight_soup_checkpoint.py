"""Save one same-architecture weight interpolation selected on validation."""

import argparse
from pathlib import Path

import torch

from common import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', required=True, type=Path)
    parser.add_argument('--second', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--second-weight', required=True, type=float)
    args = parser.parse_args()
    a = torch.load(args.first, map_location='cpu', weights_only=True)
    b = torch.load(args.second, map_location='cpu', weights_only=True)
    if a['implementation'] != 'model' or b['implementation'] != 'model' or a['config'] != b['config']:
        raise ValueError('Both checkpoints must use the same baseline architecture.')
    if not 0 <= args.second_weight <= 1:
        raise ValueError('second-weight must be in [0, 1].')
    w = args.second_weight
    checkpoint = dict(a)
    checkpoint['model'] = {name: (1 - w) * tensor + w * b['model'][name]
                           for name, tensor in a['model'].items()}
    checkpoint['parent_checkpoints'] = [sha(args.first), sha(args.second)]
    checkpoint['parent_weight'] = [1 - w, w]
    checkpoint['train_tokens'] = a['train_tokens'] + b['train_tokens']
    args.output.parent.mkdir(parents=True, exist_ok=False)
    torch.save(checkpoint, args.output)
    print(f'{args.output}: {sha(args.output)}')


if __name__ == '__main__':
    main()
