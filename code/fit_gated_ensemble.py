"""Fit a small causal mixture gate using only the supplied train split."""

import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from common import PROTOCOL, load_data, setup, sha
from evaluate import score
from mixed_ensemble_sweep import configuration
from student_gated_ensemble import build_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first', type=Path, required=True)
    p.add_argument('--second', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--steps', type=int, default=250)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seed', type=int, default=47)
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error('Output directory already contains results.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device, _ = setup('cuda', 'fp32', 4)
    a = torch.load(args.first, map_location='cpu', weights_only=True)
    b = torch.load(args.second, map_location='cpu', weights_only=True)
    if a['protocol'] != PROTOCOL or b['protocol'] != PROTOCOL:
        raise ValueError('Incompatible checkpoints.')
    config = configuration(a, b, .6, 1.075)
    config['gate_strength'] = 1.
    model = build_model(config).to(device).eval()
    model.first.load_state_dict(a['model'])
    model.second.load_state_dict(b['model'])
    model.first.requires_grad_(False)
    model.second.requires_grad_(False)
    data = load_data()
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    optimizer = torch.optim.Adam([model.gate_delta], lr=.01)
    history = []
    for step in range(args.steps):
        starts = torch.randint(len(tokens)-257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:, None] + torch.arange(257, device=device)]
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch[:, :-1]) / model.logit_temperature
        loss = F.cross_entropy(logits.flatten(0, 1), batch[:, 1:].flatten())
        loss.backward()
        optimizer.step()
        if (step + 1) % 25 == 0 or step + 1 == args.steps:
            row = dict(step=step+1, train_loss=float(loss.detach()), gate_delta=model.gate_delta.detach().cpu().tolist())
            history.append(row)
            print(json.dumps(row), flush=True)
    # Validation is used to select interpolation strength, never to update gate parameters.
    candidates = []
    for strength in (0., .25, .5, 1.):
        model.gate_strength = strength
        result = score(model, *data['validation'], device, 'fp32')
        candidates.append(dict(gate_strength=strength, validation_bpb=result['bpb'], seconds=result['seconds']))
        print(json.dumps(candidates[-1]), flush=True)
    best = min(candidates, key=lambda row: row['validation_bpb'])
    model.gate_strength = best['gate_strength']
    config['gate_strength'] = best['gate_strength']
    checkpoint_path = args.output_dir/'checkpoint.pt'
    model = model.cpu()
    torch.save(dict(protocol=PROTOCOL, implementation='student_gated_ensemble', config=config,
                    model=model.state_dict(), parent_checkpoints=[sha(args.first), sha(args.second)],
                    train_tokens=a['train_tokens']+b['train_tokens']+args.steps*args.batch_size*256,
                    gate_train_tokens=args.steps*args.batch_size*256,
                    seed=[a['seed'], b['seed'], args.seed]), checkpoint_path)
    metrics = dict(candidates=candidates, selected=best, train_history=history,
                   gate_train_tokens=args.steps*args.batch_size*256,
                   parent_checkpoints=[sha(args.first), sha(args.second)],
                   checkpoint_sha256=sha(checkpoint_path), elapsed_seconds=time.perf_counter()-started)
    (args.output_dir/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps(dict(selected=best, checkpoint_sha256=metrics['checkpoint_sha256']), indent=2), flush=True)


if __name__ == '__main__':
    main()
