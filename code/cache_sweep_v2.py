"""Validation-only sweep of context length and adaptive cache strength."""

import argparse
import json
from pathlib import Path

import torch

from common import ROOT, load_data, setup, sha
from evaluate import score
from student_cache import build_model


CANDIDATES = [
    ('bigram', .30, 0, 0),
    ('trigram', .20, 0, 0),
    ('trigram', .40, 0, 0),
    ('trigram', .60, 0, 0),
    *[('variable', strength, bonus, 0)
      for bonus in (2, 4, 8) for strength in (.10, .20, .30)],
    ('bigram', .45, 0, .5),
    ('bigram', .60, 0, 1),
    ('bigram', .90, 0, 2),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    device, _ = setup('cuda', 'fp32', 4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = dict(checkpoint['config'])
    config['backbone'] = 'baseline' if checkpoint['implementation'] == 'model' else 'student'
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint['model'])
    data = load_data()
    rows = []
    for mode, strength, bonus, count_scale in CANDIDATES:
        model.cache_mode = mode
        model.cache_strength = strength
        model.cache_tau = 0.0
        model.cache_bonus = bonus
        model.cache_count_scale = count_scale
        result = score(model, *data['validation'], device, 'fp32')
        row = {'mode': mode, 'strength': strength, 'bonus': bonus,
               'count_scale': count_scale, 'bpb': result['bpb'],
               'seconds': result['seconds']}
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps({
        'split': 'validation', 'precision': 'fp32', 'device': str(device),
        'base_checkpoint_sha256': sha(args.checkpoint),
        'student_cache_sha256': sha(ROOT / 'student_cache.py'),
        'candidates': rows,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
