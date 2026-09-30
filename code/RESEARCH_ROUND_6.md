# Research round 6: course-inspired position encoding and training sampling

All learned weights used only the supplied train split. The data, tokenizer,
evaluator, and independent causal 256-token windows were unchanged. Candidate
models and ensemble settings were chosen on validation; the selected predictor
was frozen before complete CPU FP32 test evaluation. Complete training curves,
all mixture-grid scores, CPU scorer JSON, timing, and per-window losses are in
`experiment_results/round6/`. Checkpoints are under `runs/` and are ignored by Git.

## Controlled model experiments

All three new runs used width 256, depth 4, four heads, batch 32, seed 17,
CUDA FP32, 6,000 updates, 49,152,000 processed targets, and the same AdamW
schedule. The prior dropout-0.1 random-window run has the same target count
and schedule. The earlier 4,800-step ordinary GELU result is not used as the
dropout control because its schedule and target count differ.

| Run | Change | Parameters | Validation BPB | GPU train seconds |
|---|---|---:|---:|---:|
| `gelu_w256_6000_s17_round6` | No dropout, random windows | 3,749,376 | 1.62411199 | 338.13 |
| Prior `gelu_dropout10_w256_6000_s17` | Dropout 0.1, random windows | 3,749,376 | 1.59078591 | 467.57 |
| `rope_dropout10_w256_6000_s17_round6` | Replace learned absolute positions with RoPE, same dropout | 3,683,840 | **1.58140673** | 481.67 |
| `gelu_dropout10_coverage_w256_6000_s17_round6` | Same GELU/dropout, reshuffled nonoverlapping training windows | 3,749,376 | 1.58761635 | 705.47* |

The matched no-dropout control supports a 0.03333 validation-BPB gain from the
dropout training recipe at this budget. RoPE improves on its matched
absolute-position/dropout control by 0.00938. Uniform-coverage sampling
improves by 0.00317. These are one-seed results, not estimates of average
effects. *The coverage run overlapped GPU mixture sweeps and CPU scoring, so
its wall time is not directly comparable to the isolated training runs.

RoPE rotates queries and keys inside each causal attention block and removes
the learned absolute-position table. Dropout remains training-only. The
coverage sampler reshuffles nonoverlapping 256-target training windows with a
new random offset each pass. It changes neither the training text nor the
evaluator. The sampler was added to `train.py` as an opt-in flag; its default
remains the previous random-window method.

## Combinations on validation

Both ensemble members were trained only on the supplied training text. We
combined logits before one softmax. All sampled mixture weights and
temperatures, including rejected settings, are retained in the round-6 sweep
JSON files.

| Pair | Best sampled setting | Validation BPB |
|---|---|---:|
| Prior dropout-17 + ordinary GELU-29 | weight 0.55 on dropout, T 1.075 | 1.53515087 |
| Coverage dropout-17 + ordinary GELU-29 | weight 0.55 on coverage, T 1.075 | 1.53531061 |
| RoPE dropout-17 + prior dropout-17 | weight 0.50 on RoPE, T 1.075 | 1.53094455 |
| RoPE dropout-17 + coverage dropout-17 | weight 0.55 on RoPE, T 1.075 | 1.52886384 |
| **RoPE dropout-17 + ordinary GELU-29** | **weight 0.55 on RoPE, T 1.075** | **1.52383589** |

The coverage-trained single model improved slightly, but its pairing with
seed-29 GELU did not improve on the prior combination. RoPE provided a larger
combination gain, consistent with complementary prediction errors. The
measurements establish the gains; error complementarity is an explanation to
test further, not a proven cause.

## Frozen predictor and resource check

The selected bundle is
`runs/rope_s17_gelu_s29_w055_t1075_round6/checkpoint.pt`, SHA-256
`ec8f909757e7156eb14601183be7de8605d36ccd338a9b0dab7278e0bb8d5794`.
Its parent hashes are in the bundle. It records 88,473,600 processed targets
including both training ancestries and 7,433,216 unique FP32 parameters.
Its checkpoint is 29,769,336 bytes (28.39 MiB), below the 64 MiB inference
asset limit; it has no separate data asset. The two parents' isolated training
times total 978.25 GPU seconds. All three new training runs cost 1,525.28
GPU seconds in measured training time, with some scoring performed concurrently.

CPU FP32 validation reproduced **1.52383583 BPB**. An initial CPU run while
training was active took 61.74 seconds; it was not used for resource
assessment. The idle run took 57.81 seconds versus 13.67 seconds for the
same-period baseline, a 4.23-times ratio. After validation selection and
freezing, complete CPU FP32 test scored **1.54311893 BPB**, compared with the
previous best 1.55109976. Full-test scorer time was **64.63 seconds** versus
**15.50 seconds** for the baseline in the same period: 4.17 times, under the
5-times limit. Sampled Windows peak process working set was 1.859 GiB, under
4 GiB. Absolute CPU times were higher than earlier rounds for both baseline
and candidate, so the contemporaneous ratio is the relevant comparison.
This round improved the score but did not reach the aspirational 1.5 BPB.
No website submission was made.

The project contract suite, including a new RoPE/mixed-ensemble causality,
normalization and window-independence check, passed 10 tests. A further
independent-machine CPU timing repeat would strengthen the portability claim.

## Reproduction

Use fresh run directory names if those listed below already exist.

```text
python train.py --implementation model --config configs/baseline_width256.json --device cuda --precision fp32 --threads 4 --seed 17 --steps 6000 --eval-every 1200 --run-dir runs/gelu_w256_6000_s17_round6
python train.py --implementation student_rope --config configs/rope_dropout10_width256.json --device cuda --precision fp32 --threads 4 --seed 17 --steps 6000 --eval-every 1200 --run-dir runs/rope_dropout10_w256_6000_s17_round6
python train.py --implementation student_dropout --config configs/gelu_dropout10_width256.json --device cuda --precision fp32 --threads 4 --seed 17 --steps 6000 --eval-every 1200 --sampler coverage --run-dir runs/gelu_dropout10_coverage_w256_6000_s17_round6
python mixed_ensemble_sweep.py --first runs/rope_dropout10_w256_6000_s17_round6/checkpoint.pt --second runs/gelu_w256_4800_s29/checkpoint.pt --weights 0.55 0.6 0.65 0.7 --temperatures 1.05 1.075 1.1 --output experiment_results/round6/rope_s17_gelu_s29_fine_sweep.json
python make_mixed_ensemble_checkpoint.py --first runs/rope_dropout10_w256_6000_s17_round6/checkpoint.pt --second runs/gelu_w256_4800_s29/checkpoint.pt --first-weight 0.55 --temperature 1.075 --output runs/rope_s17_gelu_s29_w055_t1075_round6/checkpoint.pt
python evaluate.py --checkpoint runs/rope_s17_gelu_s29_w055_t1075_round6/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation
python evaluate.py --checkpoint runs/rope_s17_gelu_s29_w055_t1075_round6/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test
```
