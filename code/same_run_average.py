"""Validation-select weight averaging between checkpoints of one training run."""

import argparse
import json
from pathlib import Path

import torch

from common import PROTOCOL, load_data, make_model, setup, sha
from evaluate import score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--earlier', type=Path, required=True)
    p.add_argument('--later', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if args.earlier.parent.resolve() != args.later.parent.resolve():
        p.error('Both checkpoints must be from the same run directory.')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error('Output directory already contains results.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    a = torch.load(args.earlier, map_location='cpu', weights_only=True)
    b = torch.load(args.later, map_location='cpu', weights_only=True)
    if (a['protocol'] != PROTOCOL or b['protocol'] != PROTOCOL or
            a['implementation'] != b['implementation'] or a['config'] != b['config'] or
            a['seed'] != b['seed'] or a['train_tokens'] >= b['train_tokens']):
        raise ValueError('Incompatible or misordered same-run checkpoints.')
    device, _ = setup('cuda', 'fp32', 4)
    model, _ = make_model(a['implementation'], a['config'], device)
    validation = load_data()['validation']
    rows = []
    for later_weight in (0., .25, .5, .75, 1.):
        averaged = {k: (1-later_weight)*v + later_weight*b['model'][k]
                    for k, v in a['model'].items()}
        model.load_state_dict(averaged)
        result = score(model, *validation, device, 'fp32')
        row = dict(later_weight=later_weight, validation_bpb=result['bpb'], seconds=result['seconds'])
        rows.append(row)
        print(json.dumps(row), flush=True)
    selected = min(rows, key=lambda row: row['validation_bpb'])
    weight = selected['later_weight']
    checkpoint = dict(b)
    checkpoint['model'] = {k: (1-weight)*v + weight*b['model'][k]
                           for k, v in a['model'].items()}
    checkpoint['parent_checkpoints'] = [sha(args.earlier), sha(args.later)]
    checkpoint['parent_weight'] = [1-weight, weight]
    # These are two points on a single path, so ancestry counts the later step once.
    checkpoint['train_tokens'] = b['train_tokens']
    output = args.output_dir/'checkpoint.pt'
    torch.save(checkpoint, output)
    metrics = dict(candidates=rows, selected=selected,
                   parent_checkpoints=checkpoint['parent_checkpoints'],
                   train_tokens=checkpoint['train_tokens'], checkpoint_sha256=sha(output))
    (args.output_dir/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps(dict(selected=selected, checkpoint_sha256=metrics['checkpoint_sha256']), indent=2), flush=True)


if __name__ == '__main__':
    main()
