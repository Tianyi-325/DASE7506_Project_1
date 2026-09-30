"""Bundle two trained checkpoints into one directly evaluable ensemble."""

import argparse
from pathlib import Path

import torch

from common import sha
from student_ensemble import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gelu', required=True, type=Path)
    parser.add_argument('--swiglu', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gelu-weight', required=True, type=float)
    parser.add_argument('--method', choices=['prob', 'logit'], default='prob')
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()
    g = torch.load(args.gelu, map_location='cpu', weights_only=True)
    s = torch.load(args.swiglu, map_location='cpu', weights_only=True)
    config = dict(g['config'])
    config.update(ffn_hidden=s['config']['ffn_hidden'], gelu_weight=args.gelu_weight,
                  ensemble_method=args.method, logit_temperature=args.temperature)
    model = build_model(config)
    model.gelu.load_state_dict(g['model'])
    model.swiglu.load_state_dict(s['model'])
    checkpoint = {
        'protocol': g['protocol'], 'implementation': 'student_ensemble',
        'config': config, 'model': model.state_dict(),
        'parent_checkpoints': [sha(args.gelu), sha(args.swiglu)],
        'train_tokens': g['train_tokens'] + s['train_tokens'],
        'seed': [g['seed'], s['seed']],
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    torch.save(checkpoint, args.output)
    print(f'{args.output}: {sha(args.output)}')


if __name__ == '__main__':
    main()
