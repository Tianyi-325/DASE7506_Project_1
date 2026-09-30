"""Append idempotent round-8 summary rows; detailed grids remain next to this file."""

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import sha

LOG = ROOT/'RUN_LOG.csv'


def metrics(run):
    return json.loads((ROOT/'runs'/run/'metrics.json').read_text())


def checkpoint(run):
    return ROOT/'runs'/run/'checkpoint.pt'


rope9 = metrics('rope_dropout10_coverage_w256_9000_s17_round8')
qknorm = metrics('rope_qknorm_dropout10_coverage_w256_6000_s17_round8')
rope12 = metrics('rope_dropout10_coverage_w256_12000_s17_round8')
gelu29 = metrics('gelu_w256_4800_s29')
gate = metrics('gated_rope_coverage_s17_gelu_s29_round8')
swa9 = metrics('rope_coverage_s17_swa_7500_9000_round8')
swa12 = metrics('rope_coverage_s17_swa_10000_12000_round8')
selected_val = json.loads((ROOT/'experiment_results/round8/selected_validation_cpu_fp32.resource.json').read_text())
selected_test = json.loads((ROOT/'experiment_results/round8/selected_test_cpu_fp32.resource.json').read_text())

common = dict(config_path='configs/rope_dropout10_width256.json', seed='17',
              batch_size=32, context=256, hardware='NVIDIA GeForce RTX 3060 Laptop GPU',
              threads=4, train_precision='fp32', selected_on='validation')

rows = [
    dict(run_id='round8_gate_train_only_screen', method='Three-parameter train-only causal mixture gate',
         config_path='embedded checkpoint config', seed='17;29;47',
         parent_checkpoints=';'.join(gate['parent_checkpoints']), steps=250,
         batch_size=16, context=256, processed_targets_including_ancestry=89497600,
         parameters=7433219, hardware='NVIDIA GeForce RTX 3060 Laptop GPU', threads=4,
         train_precision='fp32', train_seconds=gate['elapsed_seconds'],
         validation_bpb=1.5193594345714285, selected_on='validation',
         notes='Best nonzero gate strength 0.25; fixed gate strength zero was 1.51934833; rejected; elapsed includes validation grid'),
    dict(common, run_id='rope_dropout10_coverage_w256_9000_s17_round8',
         method='RoPE dropout 0.1 coverage 9000-step plan', steps=9000,
         processed_targets_including_ancestry=rope9['train_tokens'],
         parameters=rope9['parameters'], train_seconds=rope9['train_seconds'],
         validation_seconds=rope9['validation']['seconds'],
         peak_gpu_allocated_gb=rope9['peak_allocated_gb'],
         validation_bpb=rope9['validation']['bpb'],
         checkpoint_sha256=rope9['checkpoint_sha256'],
         notes='Snapshots at 1500-step intervals; GPU contention inflates measured training time'),
    dict(common, run_id='rope_qknorm_dropout10_coverage_w256_6000_s17_round8',
         method='Q/K-normalized RoPE dropout 0.1 coverage', steps=6000,
         processed_targets_including_ancestry=qknorm['train_tokens'],
         parameters=qknorm['parameters'], train_seconds=qknorm['train_seconds'],
         validation_seconds=qknorm['validation']['seconds'],
         peak_gpu_allocated_gb=qknorm['peak_allocated_gb'],
         validation_bpb=qknorm['validation']['bpb'],
         checkpoint_sha256=qknorm['checkpoint_sha256'],
         notes='Matched-target Q/K normalization screen; rejected; GPU contention inflates time'),
    dict(common, run_id='rope_dropout10_coverage_w256_12000_s17_round8',
         method='RoPE dropout 0.1 coverage 12000-step plan', steps=12000,
         processed_targets_including_ancestry=rope12['train_tokens'],
         parameters=rope12['parameters'], train_seconds=rope12['train_seconds'],
         validation_seconds=rope12['validation']['seconds'],
         peak_gpu_allocated_gb=rope12['peak_allocated_gb'],
         validation_bpb=rope12['validation']['bpb'],
         checkpoint_sha256=rope12['checkpoint_sha256'],
         notes='Snapshots at 2000-step intervals; cosine schedule spans 12000 steps'),
    dict(common, run_id='rope_coverage_s17_swa_7500_9000_round8',
         method='Same-run average of 7500 and 9000-step RoPE weights', steps='7500;9000',
         processed_targets_including_ancestry=rope9['train_tokens'],
         parameters=rope9['parameters'], train_seconds=rope9['train_seconds'],
         validation_seconds=swa9['selected']['seconds'],
         validation_bpb=swa9['selected']['validation_bpb'],
         checkpoint_sha256=swa9['checkpoint_sha256'],
         notes='Equal snapshot weight; one training ancestry counted once'),
    dict(common, run_id='rope_coverage_s17_swa_10000_12000_round8',
         method='Same-run average of 10000 and 12000-step RoPE weights', steps='10000;12000',
         processed_targets_including_ancestry=rope12['train_tokens'],
         parameters=rope12['parameters'], train_seconds=rope12['train_seconds'],
         validation_seconds=swa12['selected']['seconds'],
         validation_bpb=swa12['selected']['validation_bpb'],
         checkpoint_sha256=swa12['checkpoint_sha256'],
         notes='Equal snapshot weight; one training ancestry counted once'),
    dict(run_id='rope_swa_10000_12000_gelu29_w065_t109_round8',
         method='Same-run averaged RoPE plus GELU-29 logit ensemble',
         config_path='embedded checkpoint config', seed='17;29',
         parent_checkpoints=';'.join([swa12['checkpoint_sha256'], gelu29['checkpoint_sha256']]),
         steps='12000;4800', batch_size=32, context=256,
         processed_targets_including_ancestry=rope12['train_tokens']+gelu29['train_tokens'],
         parameters=7433216, hardware='CPU; parents trained on RTX 3060 Laptop GPU',
         threads=4, train_precision='fp32',
         train_seconds=rope12['train_seconds']+gelu29['train_seconds'],
         validation_seconds=selected_val['scorer_seconds'],
         test_seconds=selected_test['scorer_seconds'],
         peak_ram_gb=selected_test['sampled_peak_working_set_gib'],
         validation_bpb=selected_val['bpb'], test_bpb=selected_test['bpb'],
         checkpoint_sha256=sha(checkpoint('rope_swa_10000_12000_gelu29_w065_t109_round8')),
         selected_on='validation',
         notes='RoPE weight 0.65 temperature 1.09; frozen before test; same-period baseline CPU test 8.8359224 sec; ratio 4.5748; checkpoint 28.39 MiB'),
]

with LOG.open(newline='', encoding='utf-8') as handle:
    reader = csv.DictReader(handle)
    columns = reader.fieldnames
    old_rows = list(reader)
existing = {row['run_id'] for row in old_rows}
with LOG.open('a', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=columns)
    for row in rows:
        if row['run_id'] not in existing:
            writer.writerow(row)
            print(row['run_id'])
