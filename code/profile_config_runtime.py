"""CPU validation-time screen with random weights; BPB is not a model result."""

import argparse
import json
from pathlib import Path

from common import load_data, make_model, setup
from evaluate import score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--implementation', required=True)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    device, _ = setup('cpu', 'fp32', 4)
    config = json.loads(args.config.read_text())
    model, implementation_sha256 = make_model(args.implementation, config, device)
    result = score(model, *load_data()['validation'], device, 'fp32')
    row = {'purpose': 'runtime screen with random weights only',
           'implementation': args.implementation, 'implementation_sha256': implementation_sha256,
           'config': config, 'parameters': sum(p.numel() for p in model.parameters()),
           'uncompressed_parameter_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
           'validation_cpu_seconds': result['seconds'],
           'random_weights_bpb_ignored': result['bpb']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row, indent=2) + '\n')
    print(json.dumps(row, indent=2))


if __name__ == '__main__':
    main()
