"""CPU validation runtime screen with random weights; scores are not model candidates."""

import argparse
import json
from pathlib import Path

from common import load_data, setup
from evaluate import score
from model import GPT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--widths', nargs='+', type=int, default=[352, 384, 416])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    device, _ = setup('cpu', 'fp32', 4)
    validation = load_data()['validation']
    rows = []
    for width in args.widths:
        model = GPT(dict(vocab=2048, width=width, heads=4, depth=4, context=256))
        result = score(model, *validation, device, 'fp32')
        row = {'width': width, 'parameters': sum(p.numel() for p in model.parameters()),
               'parameter_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
               'validation_cpu_seconds': result['seconds'],
               'random_weights_bpb_ignored': result['bpb']}
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'purpose': 'runtime screen only',
                                       'device': 'cpu', 'precision': 'fp32',
                                       'threads': 4, 'rows': rows}, indent=2) + '\n')


if __name__ == '__main__':
    main()
