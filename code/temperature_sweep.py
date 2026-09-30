"""Validation-only logit temperature sweep for a frozen ensemble."""

import argparse
import json
from pathlib import Path

import torch

from common import load_data, setup, sha
from evaluate import score
from student_ensemble import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--temperatures', nargs='+', type=float,
                        default=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    args = parser.parse_args()
    device, _ = setup('cuda', 'fp32', 4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['implementation'] != 'student_ensemble' or checkpoint['config']['ensemble_method'] != 'logit':
        raise ValueError('Expected a logit-ensemble checkpoint.')
    model = build_model(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model'])
    validation = load_data()['validation']
    rows = []
    for temperature in args.temperatures:
        model.logit_temperature = temperature
        result = score(model, *validation, device, 'fp32')
        row = {'temperature': temperature, 'bpb': result['bpb'],
               'seconds': result['seconds']}
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'split': 'validation', 'precision': 'fp32',
                                       'checkpoint_sha256': sha(args.checkpoint),
                                       'candidates': rows}, indent=2) + '\n')


if __name__ == '__main__':
    main()
