"""Bundle the selected mixed-architecture ensemble for direct evaluation."""

import argparse
from pathlib import Path

import torch

from common import sha
from mixed_ensemble_sweep import configuration
from student_mixed_ensemble import build_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first', type=Path, required=True)
    p.add_argument('--second', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--first-weight', type=float, required=True)
    p.add_argument('--temperature', type=float, required=True)
    args = p.parse_args()
    a = torch.load(args.first, map_location='cpu', weights_only=True)
    b = torch.load(args.second, map_location='cpu', weights_only=True)
    if a['protocol'] != b['protocol']:
        raise ValueError('Incompatible evaluation protocols.')
    config = configuration(a, b, args.first_weight, args.temperature)
    model = build_model(config)
    model.first.load_state_dict(a['model'])
    model.second.load_state_dict(b['model'])
    checkpoint = dict(protocol=a['protocol'], implementation='student_mixed_ensemble',
                      config=config, model=model.state_dict(),
                      parent_checkpoints=[sha(args.first), sha(args.second)],
                      train_tokens=a['train_tokens'] + b['train_tokens'],
                      seed=[a['seed'], b['seed']])
    args.output.parent.mkdir(parents=True, exist_ok=False)
    torch.save(checkpoint, args.output)
    print(f'{args.output}: {sha(args.output)}')


if __name__ == '__main__':
    main()
