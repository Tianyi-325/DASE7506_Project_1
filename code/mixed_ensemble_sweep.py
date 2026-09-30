"""Validation-only sweep over two frozen, separately trained models."""

import argparse
import json
from pathlib import Path

import torch

from common import load_data, setup, sha
from evaluate import score
from student_mixed_ensemble import build_model


def configuration(a, b, weight=0.5, temperature=1.0):
    return dict(context=256, vocab=2048,
                first_implementation=a['implementation'], first_config=a['config'],
                second_implementation=b['implementation'], second_config=b['config'],
                first_weight=weight, logit_temperature=temperature)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first', type=Path, required=True)
    p.add_argument('--second', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--weights', type=float, nargs='+', default=[0.4, 0.5, 0.6])
    p.add_argument('--temperatures', type=float, nargs='+', default=[1.05, 1.075, 1.1])
    p.add_argument('--ensemble-method', choices=['logit','prob'], default='logit')
    args = p.parse_args()
    a = torch.load(args.first, map_location='cpu', weights_only=True)
    b = torch.load(args.second, map_location='cpu', weights_only=True)
    if a['protocol'] != b['protocol'] or a['config']['vocab'] != 2048 or b['config']['vocab'] != 2048:
        raise ValueError('Incompatible checkpoints.')
    device, _ = setup('cuda', 'fp32', 4)
    model = build_model(configuration(a, b)).to(device).eval()
    model.ensemble_method = args.ensemble_method
    model.first.load_state_dict(a['model'])
    model.second.load_state_dict(b['model'])
    validation = load_data()['validation']
    rows = []
    for weight in args.weights:
        model.first_weight = weight
        for temperature in args.temperatures:
            model.logit_temperature = temperature
            result = score(model, *validation, device, 'fp32')
            row = dict(first_weight=weight, temperature=temperature,
                       bpb=result['bpb'], seconds=result['seconds'])
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(split='validation', precision='fp32',
        ensemble_method=args.ensemble_method,
        first_sha256=sha(args.first), second_sha256=sha(args.second), candidates=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
