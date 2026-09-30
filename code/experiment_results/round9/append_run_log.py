"""Idempotently add round-9 training and weight-average rows to RUN_LOG.csv."""

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import sha

LOG = ROOT/'RUN_LOG.csv'
rows = []


def read_metrics(run):
    path = ROOT/'runs'/run/'metrics.json'
    return json.loads(path.read_text()) if path.exists() else None


def add_training(run, method, config, notes):
    m = read_metrics(run)
    if not m:
        return
    rows.append(dict(run_id=run, method=method, config_path=config,
                     seed=m['seed'], parent_checkpoints=m.get('initial_checkpoint_sha256') or '',
                     steps=m['history'][-1]['step'], batch_size=32, context=256,
                     processed_targets_including_ancestry=m['train_tokens'],
                     parameters=m['parameters'],
                     hardware='NVIDIA GeForce RTX 3060 Laptop GPU', threads=4,
                     train_precision='fp32', train_seconds=m['train_seconds'],
                     validation_seconds=m['validation']['seconds'],
                     peak_gpu_allocated_gb=m['peak_allocated_gb'],
                     validation_bpb=m['validation']['bpb'],
                     checkpoint_sha256=m['checkpoint_sha256'],
                     selected_on='validation', notes=notes))


add_training('gelu_dropout10_w256_4800_s29_round9',
             'GELU seed29 dropout 0.1 random-window matched control',
             'configs/gelu_dropout10_width256.json',
             'Same targets as ordinary GELU29; pair screen worse than round8 selected')
add_training('gelu_dropout10_coverage_w256_4800_s29_round9',
             'GELU seed29 dropout 0.1 coverage-sampling matched control',
             'configs/gelu_dropout10_width256.json',
             'Same targets as random-window dropout29; better single but worse pair')
add_training('rope_dropout10_coverage_wsd12000_s17_round9',
             'RoPE dropout 0.1 coverage WSD schedule 12000 steps',
             'configs/rope_dropout10_width256.json',
             'Matched targets vs round8 cosine control; stable fraction 0.75')
add_training('rope_wsd12_finetune2k_s47_round9',
             'Train-only low-rate continuation of WSD12 with fresh AdamW',
             'configs/rope_dropout10_width256.json',
             '2000 own steps at LR 0.0001 to 0.00001; parent targets counted')
add_training('rope_dropout10_coverage_wsd16000_s17_round9',
             'RoPE dropout 0.1 coverage WSD schedule 16000 steps',
             'configs/rope_dropout10_width256.json',
             'Stable fraction 0.75; snapshots every 2000 steps')
add_training('rope_dropout15_coverage_wsd12000_s17_round9',
             'RoPE dropout 0.15 coverage WSD schedule 12000 steps',
             'configs/rope_dropout15_width256.json',
             'Matched targets and schedule vs dropout 0.1 WSD control')

for run, parent in (
    ('rope_wsd_swa_10000_12000_round9', 'rope_dropout10_coverage_wsd12000_s17_round9'),
    ('rope_wsd16_swa_14000_16000_round9', 'rope_dropout10_coverage_wsd16000_s17_round9'),
):
    m = read_metrics(run)
    p = read_metrics(parent)
    if m and p:
        rows.append(dict(run_id=run, method='Same-run RoPE WSD checkpoint weight average',
                         config_path='configs/rope_dropout10_width256.json', seed=17,
                         parent_checkpoints=';'.join(m['parent_checkpoints']),
                         steps=parent.split('wsd')[-1].split('_')[0], batch_size=32,
                         context=256, processed_targets_including_ancestry=p['train_tokens'],
                         parameters=p['parameters'], hardware='NVIDIA GeForce RTX 3060 Laptop GPU',
                         threads=4, train_precision='fp32', train_seconds=p['train_seconds'],
                         validation_seconds=m['selected']['seconds'],
                         validation_bpb=m['selected']['validation_bpb'],
                         checkpoint_sha256=m['checkpoint_sha256'], selected_on='validation',
                         notes='Same training trajectory; ancestry counted once; selected average weight recorded in metrics JSON'))

selected_run = 'rope_wsd16_swa_gelu29_w07_t1105_round9'
selected_path = ROOT/'runs'/selected_run/'checkpoint.pt'
validation_path = ROOT/'experiment_results/round9/selected_validation_cpu_fp32.resource.json'
test_path = ROOT/'experiment_results/round9/selected_test_cpu_fp32.resource.json'
if selected_path.exists() and validation_path.exists() and test_path.exists():
    v = json.loads(validation_path.read_text())
    t = json.loads(test_path.read_text())
    wsd = read_metrics('rope_dropout10_coverage_wsd16000_s17_round9')
    gelu = read_metrics('gelu_w256_4800_s29')
    swa = read_metrics('rope_wsd16_swa_14000_16000_round9')
    rows.append(dict(run_id=selected_run,
                     method='WSD16 same-run averaged RoPE plus GELU29 logit ensemble',
                     config_path='embedded checkpoint config', seed='17;29',
                     parent_checkpoints=';'.join([swa['checkpoint_sha256'], gelu['checkpoint_sha256']]),
                     steps='16000;4800', batch_size=32, context=256,
                     processed_targets_including_ancestry=wsd['train_tokens']+gelu['train_tokens'],
                     parameters=7433216, hardware='CPU; parents trained on RTX 3060 Laptop GPU',
                     threads=4, train_precision='fp32',
                     train_seconds=wsd['train_seconds']+gelu['train_seconds'],
                     validation_seconds=v['scorer_seconds'], test_seconds=t['scorer_seconds'],
                     peak_ram_gb=t['sampled_peak_working_set_gib'],
                     validation_bpb=v['bpb'], test_bpb=t['bpb'],
                     checkpoint_sha256=sha(selected_path), selected_on='validation',
                     notes='RoPE weight 0.70 temperature 1.105; frozen before complete test; CPU ratio 4.2887; 28.39 MiB checkpoint'))

with LOG.open(newline='', encoding='utf-8') as handle:
    reader = csv.DictReader(handle)
    columns = reader.fieldnames
    existing = {r['run_id'] for r in reader}
with LOG.open('a', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=columns)
    for row in rows:
        if row['run_id'] not in existing:
            writer.writerow(row)
            print(row['run_id'])
