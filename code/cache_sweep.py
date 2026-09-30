"""Validation-only sweep for a causal cache on one frozen student checkpoint."""

import argparse
import json
from pathlib import Path

import torch

from common import ROOT, load_data, setup, sha
from evaluate import score
from student_cache import build_model


CANDIDATES = [
    ('none', 0.0, 0.0),
    ('unigram', 0.05, 0.0),
    ('unigram', 0.10, 0.0),
    ('following', 0.05, 0.0),
    ('following', 0.10, 0.0),
    ('following', 0.15, 0.0),
    ('following', 0.20, 0.0),
    ('following', 0.10, 64.0),
    ('bigram', 0.10, 0.0),
    ('bigram', 0.20, 0.0),
    ('bigram', 0.30, 0.0),
    ('bigram', 0.40, 0.0),
    ('bigram', 0.50, 0.0),
    ('bigram', 0.20, 64.0),
    ('bigram', 0.40, 64.0),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    device, _ = setup(args.device, 'fp32', 4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = dict(checkpoint['config'])
    config['backbone'] = 'baseline' if checkpoint['implementation'] == 'model' else 'student'
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint['model'])
    data = load_data()
    rows = []
    for mode, strength, tau in CANDIDATES:
        model.cache_mode = mode
        model.cache_strength = strength
        model.cache_tau = tau
        result = score(model, *data['validation'], device, 'fp32')
        result.pop('window_nll_nats')
        row = {'mode': mode, 'strength': strength, 'tau': tau,
               'bpb': result['bpb'], 'seconds': result['seconds']}
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        'split': 'validation', 'precision': 'fp32', 'device': str(device),
        'base_checkpoint_sha256': sha(args.checkpoint),
        'student_cache_sha256': sha(ROOT / 'student_cache.py'),
        'candidates': rows,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
