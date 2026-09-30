"""Validation-only weight interpolation between two same-architecture runs."""

import argparse
import json
from pathlib import Path

import torch

from common import load_data, setup, sha
from evaluate import score
from model import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', required=True, type=Path)
    parser.add_argument('--second', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    device, _ = setup('cuda', 'fp32', 4)
    a = torch.load(args.first, map_location='cpu', weights_only=True)
    b = torch.load(args.second, map_location='cpu', weights_only=True)
    if a['implementation'] != 'model' or b['implementation'] != 'model' or a['config'] != b['config']:
        raise ValueError('Both checkpoints must use the same baseline architecture.')
    model = build_model(a['config']).to(device)
    data = load_data()
    rows = []
    for second_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        averaged = {name: (1 - second_weight) * tensor + second_weight * b['model'][name]
                    for name, tensor in a['model'].items()}
        model.load_state_dict(averaged)
        result = score(model, *data['validation'], device, 'fp32')
        row = {'second_weight': second_weight, 'bpb': result['bpb'],
               'seconds': result['seconds']}
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps({
        'split': 'validation', 'precision': 'fp32',
        'first_checkpoint_sha256': sha(args.first),
        'second_checkpoint_sha256': sha(args.second),
        'candidates': rows,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
