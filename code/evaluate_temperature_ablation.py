"""Score a frozen single branch at a specified logit temperature on validation."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import PROTOCOL, ROOT, device_metrics, load_data, make_model, setup, sha
from evaluate import score


class TemperatureScaled(nn.Module):
    def __init__(self, base, temperature):
        super().__init__()
        self.base = base
        self.temperature = temperature

    def predict_log_probs(self, ids):
        return F.log_softmax(self.base(ids).float() / self.temperature, dim=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--temperature', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.temperature <= 0:
        parser.error('Temperature must be positive.')
    device, precision = setup('cpu', 'fp32', 4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['protocol'] != PROTOCOL:
        raise ValueError('Checkpoint uses an incompatible course protocol.')
    base, implementation_sha = make_model(checkpoint['implementation'], checkpoint['config'], device)
    base.load_state_dict(checkpoint['model'])
    model = TemperatureScaled(base, args.temperature)
    result = score(model, *load_data()['validation'], device, precision)
    losses = result.pop('window_nll_nats')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_suffix('.window-nll.npy'), np.asarray(losses))
    result.update(protocol=PROTOCOL, split='validation', precision=precision,
                  checkpoint_sha256=sha(args.checkpoint),
                  branch_implementation_sha256=implementation_sha,
                  temperature_ablation_sha256=sha(Path(__file__)),
                  evaluator_sha256=sha(ROOT / 'evaluate.py'),
                  tokenizer_sha256=sha(ROOT / 'data/tokenizer.json'),
                  temperature=args.temperature, **device_metrics(device))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
