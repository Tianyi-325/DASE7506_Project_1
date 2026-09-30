"""Repackage a dropout-trained GPT as the identical dropout-free GELU inference model."""

import argparse
from pathlib import Path

import torch

from common import sha
from model import GPT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    base = torch.load(args.base, map_location='cpu', weights_only=True)
    if base['implementation'] != 'student_dropout':
        raise ValueError('Expected a dropout-trained GPT checkpoint.')
    config = {key: value for key, value in base['config'].items() if key != 'dropout_p'}
    model = GPT(config)
    model.load_state_dict(base['model'])
    model.eval()
    checkpoint = dict(base)
    checkpoint.update(implementation='model', config=config,
                      model=model.state_dict(), parent_checkpoints=[sha(args.base)])
    args.output.parent.mkdir(parents=True, exist_ok=False)
    torch.save(checkpoint, args.output)
    print(f'{args.output}: {sha(args.output)}')


if __name__ == '__main__':
    main()
