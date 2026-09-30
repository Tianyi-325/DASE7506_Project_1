# Research round 7: joint RoPE/coverage, seed diversity, distillation, and MoE screen

All learning uses only the supplied train split. The fixed data, tokenizer,
evaluator, and independent causal 256-token windows remain unchanged. Model
and mixture choices use validation only. Complete training curves and sampled
mixture grids are in `experiment_results/round7/`; checkpoints are in `runs/`.

## Joint position encoding and training-window coverage

We trained the same width-256, four-layer, four-head, dropout-0.1 RoPE model
as round 6 with reshuffled nonoverlapping training windows instead of random
windows sampled with replacement. Both seed-17 runs used 6,000 updates,
batch 32, 49,152,000 processed targets, and the same optimizer schedule.

| Model | Sampler | Validation BPB | Measured GPU train seconds |
|---|---|---:|---:|
| Round-6 RoPE/dropout, seed 17 | Random | 1.58140673 | 481.67 |
| Round-7 RoPE/dropout, seed 17 | Coverage | **1.57062204** | 559.90 |

The joint model improved validation by 0.01078 BPB at fixed target count.
This is one seed, so interaction strength and generality remain uncertain.
The new checkpoint hash is
`22f616a1ccea0f2c3c0aebb8f06a3682a10814ed14eba47eef1957342ea70e84`.

Against the previously trained ordinary GELU seed-29 model, its best sampled
logit ensemble used RoPE weight 0.60 and temperature 1.075, reaching
**1.51934833 validation BPB**, compared with round 6's 1.52383589. Both the
coarse and fine grids are saved. The bundle is
`runs/rope_coverage_s17_gelu_s29_w06_t1075_round7/checkpoint.pt`, SHA-256
`49f2f47f2cf7a7432e473713567b3aeb66b5156a744bc4d4d3c9d7378bf418b7`.

## Independent RoPE seed and combinations

An independently initialized seed-29 RoPE/dropout model was trained for the
same 6,000 updates and 49,152,000 targets with random windows. Its validation
BPB was **1.58876057**, versus 1.58140673 for seed 17. Measured GPU train
time was 748.11 seconds, inflated by concurrent scoring workloads. Checkpoint
hash: `fd8440332aa402724d984a1c41b9cdc27c7c53b80c9893133cf1abd4d0fc6cd6`.

| Pair | Best sampled weight, temperature | Validation BPB |
|---|---|---:|
| Coverage RoPE-17 + RoPE-29 | 0.60 on first, 1.075 | 1.52755633 |
| Random RoPE-17 + RoPE-29 | 0.50 on first, 1.075 | 1.53110115 |
| Coverage RoPE-17 + ordinary GELU-29 | 0.60 on first, 1.075 | **1.51934833** |

The second RoPE seed improved neither sampled two-RoPE pair over the
heterogeneous RoPE/GELU pair. These scores are validation-only; rejected
combinations were not tested.

## Small MoE feasibility screen

We implemented a train-from-scratch FFN with one always-on shared branch and
one of four token-routed branches, each with a sigmoid gate. Its width-256,
four-layer candidate has 6,384,144 parameters (25,536,576 FP32 bytes). A
random-weight complete-validation CPU FP32 screen took **33.87 seconds**. Its
random-weight BPB is irrelevant to model quality and was not used for
selection. The screen ran concurrently with GPU training, so an idle CPU
comparison would be required before claiming resource compliance. A small
forward/backward smoke check showed finite gradients reaching the router and
causal predictions. Full training was not completed in this time-limited
round, so MoE quality remains unknown.

## Train-only distillation and final selection

We trained a fresh width-256 GELU/dropout student, seed 17, for 6,000 updates
on the supplied train split. Its frozen teacher was the train-only
RoPE-coverage-17 + GELU-29 ensemble at weight 0.60 and temperature 1.075.
For each training target, the student minimized an equal mix of ordinary
next-token cross entropy and cross entropy against the teacher's soft
distribution. The teacher saw only the current training input window; it
never saw validation or test tokens during student training. The student
uses no teacher at inference. Its checkpoint records the teacher hash and
137,625,600 processed targets including teacher ancestry, of which
49,152,000 belong to the student run itself.

| Model | Own training targets | Validation BPB | Measured GPU train seconds |
|---|---:|---:|---:|
| Prior GELU/dropout-17, ordinary next-token loss | 49,152,000 | 1.59078591 | 467.57 |
| GELU/dropout-17, 50% train-only distillation | 49,152,000 | **1.56284386** | 1,196.01 |

The matched validation gain is 0.02794 BPB for this one chosen distillation
weight. The student checkpoint hash is
`d610a8129b515e0f0457cc91077e18cc62eb3161bba12c3a37ca58bccf6005f3`.
Distillation required about 2.6 times the GPU training time of the prior
student run, although machine conditions were not identical. Its training
loss is a combined objective, so only validation BPB is directly comparable.

Distillation improved the single model but did not yield a stronger sampled
two-model predictor:

| Pair or interpolation | Best sampled validation BPB |
|---|---:|
| Distilled GELU-17 + coverage RoPE-17 | 1.52628893 |
| Distilled GELU-17 + RoPE-29 | 1.52805810 |
| Coverage RoPE-17 + GELU-29 | **1.51934833** |
| Weight interpolation of prior dropout-17 and distilled dropout-17 | 1.56284386 at 100% distilled; intermediate weights worsened |

The teacher and student likely share prediction errors, which may explain
why the improved student adds little to the ensemble. That is an inference;
the validation results establish only the measured combination scores.
All sampled weights and temperatures, including rejected candidates, are
preserved in the round-7 JSON files. No rejected candidate was evaluated on
test.

The selected bundle remained the validation-best coverage RoPE-17 +
GELU-29 ensemble above. It has 7,433,216 unique FP32 parameters, a
29,769,336-byte checkpoint (28.39 MiB), and no separate inference asset.
Its two parents used 88,473,600 processed training targets and 1,056.48
measured GPU training seconds in total. The three new training runs this
round used 2,504.02 measured GPU training seconds in aggregate; some
scoring overlapped training, affecting wall-time comparisons.

CPU FP32 validation reproduced **1.51934827 BPB** in 32.00 seconds,
versus 6.97 seconds for the baseline in the same period (4.59 times).
Sampled peak process working set was 1.858 GiB. After validation selection
and freezing, complete CPU FP32 test scored **1.54159969 BPB**, versus
the previous best 1.54311893. Full-test scorer time was **34.96 seconds**
versus **8.20 seconds** for the same-period baseline (4.26 times). Peak
working set was again 1.858 GiB. The selected predictor meets the 5-times
CPU, 4 GiB RAM, and 64 MiB inference-asset limits. The result is a small
improvement and remains above the aspirational 1.5 BPB. No website
submission was made.

The project contract suite passed 11 tests, including a new routed-FFN
causality and router-gradient check. MoE remains a screened but
untrained candidate; neither its BPB nor its final resource ratio is known.

## Reproduction

Use fresh output run directories if the listed ones already exist.

```text
python train.py --implementation student_rope --config configs/rope_dropout10_width256.json --device cuda --precision fp32 --threads 4 --seed 17 --steps 6000 --eval-every 1200 --sampler coverage --run-dir runs/rope_dropout10_coverage_w256_6000_s17_round7
python train.py --implementation student_rope --config configs/rope_dropout10_width256.json --device cuda --precision fp32 --threads 4 --seed 29 --steps 6000 --eval-every 1200 --run-dir runs/rope_dropout10_w256_6000_s29_round7
python mixed_ensemble_sweep.py --first runs/rope_dropout10_coverage_w256_6000_s17_round7/checkpoint.pt --second runs/gelu_w256_4800_s29/checkpoint.pt --weights 0.5 0.55 0.6 --temperatures 1.06 1.075 1.09 --output experiment_results/round7/rope_coverage_s17_gelu_s29_fine_sweep.json
python make_mixed_ensemble_checkpoint.py --first runs/rope_dropout10_coverage_w256_6000_s17_round7/checkpoint.pt --second runs/gelu_w256_4800_s29/checkpoint.pt --first-weight 0.6 --temperature 1.075 --output runs/rope_coverage_s17_gelu_s29_w06_t1075_round7/checkpoint.pt
python train.py --implementation student_dropout --config configs/gelu_dropout10_width256.json --device cuda --precision fp32 --threads 4 --seed 17 --steps 6000 --eval-every 1200 --teacher runs/rope_coverage_s17_gelu_s29_w06_t1075_round7/checkpoint.pt --distill-weight 0.5 --run-dir runs/gelu_dropout10_distilled_w256_6000_s17_round7
python evaluate.py --checkpoint runs/rope_coverage_s17_gelu_s29_w06_t1075_round7/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation
python evaluate.py --checkpoint runs/rope_coverage_s17_gelu_s29_w06_t1075_round7/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test
```
