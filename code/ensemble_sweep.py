"""Validation-only probability-mixture sweep for two frozen checkpoints."""

import argparse
import json
from pathlib import Path

import torch

from common import load_data, setup, sha
from evaluate import score
from student_ensemble import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gelu', required=True, type=Path)
    parser.add_argument('--swiglu', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--method', choices=['prob', 'logit'], default='prob')
    parser.add_argument('--weights', nargs='+', type=float, default=[0.25, 0.5, 0.75])
    parser.add_argument('--temperatures', nargs='+', type=float, default=[1.0])
    args = parser.parse_args()
    device, _ = setup('cuda', 'fp32', 4)
    g = torch.load(args.gelu, map_location='cpu', weights_only=True)
    s = torch.load(args.swiglu, map_location='cpu', weights_only=True)
    if g['config']['width'] != s['config']['width'] or g['config']['depth'] != s['config']['depth']:
        raise ValueError('The ensemble members must have matching width and depth.')
    config = dict(g['config'])
    config['ffn_hidden'] = s['config']['ffn_hidden']
    config['ensemble_method'] = args.method
    model = build_model(config).to(device)
    model.gelu.load_state_dict(g['model'])
    model.swiglu.load_state_dict(s['model'])
    data = load_data()
    rows = []
    for weight in args.weights:
        model.gelu_weight = weight
        for temperature in args.temperatures:
            model.logit_temperature = temperature
            result = score(model, *data['validation'], device, 'fp32')
            row = {'gelu_weight': weight, 'temperature': temperature,
                   'bpb': result['bpb'], 'seconds': result['seconds']}
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps({'split': 'validation', 'precision': 'fp32',
                                       'ensemble_method': args.method,
                                       'gelu_sha256': sha(args.gelu),
                                       'swiglu_sha256': sha(args.swiglu),
                                       'candidates': rows}, indent=2) + '\n')


if __name__ == '__main__':
    main()
