"""Validation-only sweep of two independently trained GELU GPTs."""

import argparse
import json
from pathlib import Path

import torch

from common import load_data, setup, sha
from evaluate import score
from student_dual_gelu import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', required=True, type=Path)
    parser.add_argument('--second', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--weights', nargs='+', type=float, default=[0.25, 0.5, 0.75])
    parser.add_argument('--temperatures', nargs='+', type=float, default=[1.0])
    args = parser.parse_args()
    device, _ = setup('cuda', 'fp32', 4)
    a = torch.load(args.first, map_location='cpu', weights_only=True)
    b = torch.load(args.second, map_location='cpu', weights_only=True)
    if a['implementation'] != 'model' or b['implementation'] != 'model' or a['config'] != b['config']:
        raise ValueError('Both checkpoints must use the same GELU GPT architecture.')
    model = build_model(a['config']).to(device)
    model.first.load_state_dict(a['model'])
    model.second.load_state_dict(b['model'])
    validation = load_data()['validation']
    rows = []
    for weight in args.weights:
        model.first_weight = weight
        for temperature in args.temperatures:
            model.logit_temperature = temperature
            result = score(model, *validation, device, 'fp32')
            row = {'first_weight': weight, 'temperature': temperature,
                   'bpb': result['bpb'], 'seconds': result['seconds']}
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'split': 'validation', 'precision': 'fp32',
                                       'first_sha256': sha(args.first),
                                       'second_sha256': sha(args.second),
                                       'candidates': rows}, indent=2) + '\n')


if __name__ == '__main__':
    main()
