# Research round 5: capacity, train-time regularization, and resource screening

All learning used the supplied train split. The benchmark text, tokenizer,
fixed evaluator, and independent causal windows were unchanged. Candidate
selection used validation only. The chosen predictor was frozen before one
complete CPU FP32 test evaluation. Complete numeric candidate grids,
training curves, scorer outputs, timing repeats, and per-window losses are
in `experiment_results/round5/`; run-level provenance is in `RUN_LOG.csv`.

## Capacity screens and matched-target experiment

We first measured CPU FP32 validation scoring time for random-weight
4-layer GELU GPTs. The random-weight BPB values in the screen are **not**
candidate model scores and were never used to select model quality.

| Width | Parameters | Uncompressed FP32 parameters | CPU validation seconds |
|---:|---:|---:|---:|
| 352 | 6,777,408 | 25.85 MiB | 42.99 |
| 384 | 7,983,360 | 30.45 MiB | 51.42 |
| 416 | 9,287,616 | 35.43 MiB | 58.70 |

The original width-128 baseline validation scorer took 7.93 seconds on
this machine, so its 5-times comparison is about 39.66 seconds. These
widths failed the runtime screen, despite satisfying the asset-size limit.
An 8-layer width-192 model would use 4,001,664 parameters and took 37.00
seconds as a single random-weight predictor. It was not trained: alone,
it had little time headroom; adding it to the current two-model predictor
would exceed the intended CPU budget.

Keeping attention width at 256 and enlarging each GELU feed-forward layer
from 4x to 8x yielded 5,850,624 parameters (22.32 MiB) and 37.18 CPU
validation seconds in the random-weight screen. We trained it for 4,800
updates, batch 32, seed 17, matching the 39,321,600 processed targets of
the ordinary width-256 GELU control. It scored **1.64877854 validation
BPB**, worse than the control's **1.62358452**, despite 2.10 million more
parameters. GPU FP32 training took 507.54 seconds. Its full curve and
checkpoint hash are in the metrics JSON; it was not tested.

## Train-derived bigram correction

We counted bigram transitions in the supplied train tokens only, then
smoothed the conditional frequencies toward the train unigram distribution.
The resulting 2,048 x 2,048 FP32 log-ratio matrix is a compact 16 MiB
train-derived asset. It is a fixed token-transition statistic, not a
retrieval database of documents or evaluation windows. At inference it adds
a causal, current-token-dependent logit correction to the frozen round-4
dual GELU. We swept smoothing alpha 100 and 1,000 and strengths 0, .05,
.1, .2, and .3 on validation. The best result, alpha 100 and strength .05,
was **1.53832192 BPB**, only 0.000564 below the unchanged model.
Concurrent CPU validation took 65.57 seconds while another model trained.
Because the gain was tiny and an idle resource measurement was not made,
this candidate was not selected or tested. All 10 scores are in the sweep
JSON. Its checkpoint records the extra 3,613,342 train-token transitions
used to build the matrix.

## Training-only dropout

The larger width and FFN experiments showed that available asset bytes did
not translate directly into better quality. We then kept the standard
width-256, 4-layer GELU inference architecture and applied dropout 0.1
during training in causal attention and after the attention/MLP residual
outputs. Dropout is disabled at evaluation. An exact-output check confirmed
that the evaluation model gives identical logits to an ordinary GELU GPT
loaded with the same weights (maximum absolute difference 0).

The new run used seed 17, batch 32, CUDA FP32, and 6,000 updates, totaling
49,152,000 processed train targets and 467.57 GPU training seconds. Its
validation curve was:

| Step | Validation BPB |
|---:|---:|
| 1,200 | 1.86364335 |
| 2,400 | 1.70095151 |
| 3,600 | 1.63753145 |
| 4,800 | 1.60578330 |
| 6,000 | **1.59078591** |

This is better than the earlier 4,800-step ordinary GELU (1.62358452).
Because the runs differ in both training duration and learning-rate
trajectory, this comparison does not isolate the dropout effect. It does
show that this combined regularization/longer-training recipe yielded a
stronger single model. No standalone test evaluation was run.

The dropout-trained weights were repackaged as an ordinary GELU GPT for
inference; the repackaging adds no learning. We combined it in logit space
with each prior width-256 GELU seed. At temperature 1.075, the best
dropout-17 + ordinary-17 sampled pair scored 1.54539473 validation BPB;
the dropout-17 + ordinary-29 pair reached **1.53515087** at dropout-model
weight 0.55. The full weight/temperature grids are preserved as JSON.

## Frozen selected predictor

The selected checkpoint is
`runs/dual_dropout10_s17_gelu_s29_w055_t1075/checkpoint.pt`, SHA-256
`08044b228f885d4a4013f24b493301bd2ef8c4780504397986a0a34fb05903fc`.
It records both parent checkpoint hashes and 88,473,600 processed train
targets including ancestry. The two parent training runs took 964.15 GPU
seconds together. The predictor has 7,498,752 FP32 parameters, or 28.61
MiB uncompressed, and its checkpoint is 30,031,481 bytes (28.64 MiB).

CPU FP32 validation reproduced **1.53515081 BPB**. The first timing was
an outlier at 53.23 seconds. A quiet rerun took **28.89 seconds**; the
original baseline remeasured in the same period took **7.27 seconds**, a
3.98-times ratio. After validation selection and freezing, complete CPU
FP32 test scored **1.55109976 BPB**, improving on the round-4 selected
model's 1.55541301. Full-test scorer time was **32.78 seconds**, or 3.64
times the original baseline full-test time of 9.02 seconds. Sampled peak
process working set was **1.855 GiB**. The selected predictor satisfies
the 5-times CPU time, 4 GiB RAM, and 64 MiB inference-asset limits. No
website submission was made.

Across both new training runs in this research round, GPU training cost
was 975.11 seconds and 88,473,600 processed targets. That search cost is
distinct from the selected checkpoint's ancestry; the FFN8 run is not
part of its inference or training lineage. Validation sweep and CPU
screening durations are retained per candidate in the result files.

## Reproduction

Use new run directory names if these already exist. The seed-29 ordinary
GELU parent is documented in round 4.

```text
python train.py --implementation student_dropout --config configs/gelu_dropout10_width256.json --device cuda --precision fp32 --threads 4 --seed 17 --steps 6000 --eval-every 1200 --run-dir runs/gelu_dropout10_w256_6000_s17
python make_dropout_inference_checkpoint.py --base runs/gelu_dropout10_w256_6000_s17/checkpoint.pt --output runs/gelu_dropout10_w256_6000_s17_inference/checkpoint.pt
python dual_gelu_sweep.py --first runs/gelu_dropout10_w256_6000_s17_inference/checkpoint.pt --second runs/gelu_w256_4800_s29/checkpoint.pt --weights 0.45 0.5 0.55 --temperatures 1.05 1.075 1.1 --output runs/dropout10_s17_gelu_s29_fine_sweep.json
python make_dual_gelu_checkpoint.py --first runs/gelu_dropout10_w256_6000_s17_inference/checkpoint.pt --second runs/gelu_w256_4800_s29/checkpoint.pt --first-weight 0.55 --temperature 1.075 --output runs/dual_dropout10_s17_gelu_s29_w055_t1075/checkpoint.pt
python evaluate.py --checkpoint runs/dual_dropout10_s17_gelu_s29_w055_t1075/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation
python evaluate.py --checkpoint runs/dual_dropout10_s17_gelu_s29_w055_t1075/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test
```
