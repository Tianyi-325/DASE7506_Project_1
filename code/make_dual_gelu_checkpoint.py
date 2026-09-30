"""Bundle two independently trained GELU checkpoints for direct evaluation."""

import argparse
from pathlib import Path

import torch

from common import sha
from student_dual_gelu import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', required=True, type=Path)
    parser.add_argument('--second', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--first-weight', required=True, type=float)
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()
    a = torch.load(args.first, map_location='cpu', weights_only=True)
    b = torch.load(args.second, map_location='cpu', weights_only=True)
    if a['implementation'] != 'model' or b['implementation'] != 'model' or a['config'] != b['config']:
        raise ValueError('Both checkpoints must use the same GELU GPT architecture.')
    config = dict(a['config'])
    config.update(first_weight=args.first_weight, logit_temperature=args.temperature)
    model = build_model(config)
    model.first.load_state_dict(a['model'])
    model.second.load_state_dict(b['model'])
    checkpoint = {'protocol': a['protocol'], 'implementation': 'student_dual_gelu',
                  'config': config, 'model': model.state_dict(),
                  'parent_checkpoints': [sha(args.first), sha(args.second)],
                  'train_tokens': a['train_tokens'] + b['train_tokens'],
                  'seed': [a['seed'], b['seed']]}
    args.output.parent.mkdir(parents=True, exist_ok=False)
    torch.save(checkpoint, args.output)
    print(f'{args.output}: {sha(args.output)}')


if __name__ == '__main__':
    main()
