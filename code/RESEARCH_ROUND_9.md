# Research round 9: complementary GELU branch and learning-rate schedule

All learned weights use the supplied training split. We select models and
mixtures using validation only, retain independent causal 256-token windows,
and leave the fixed data, tokenizer, and evaluator unchanged. Complete curves,
validation grids, and CPU resource records are in `runs/` and
`experiment_results/round9/`. No test result is used for tuning.

## Existing second-branch screens

We paired the round-8 selected RoPE average (10,000/12,000 steps) with three
already trained GELU models. Each screen used first-model weights 0.55, 0.65,
0.75 at temperature 1.09 on the full validation split.

| Second model | Best sampled validation BPB | RoPE weight |
|---|---:|---:|
| Previous GELU-29, 4,800 steps | **1.48962402** | 0.65 |
| GELU/dropout-17, 6,000 steps | 1.49330290 | 0.65 |
| GELU/dropout/coverage-17, 6,000 steps | 1.49410048 | 0.65 |
| Distilled GELU/dropout-17, 6,000 steps | 1.49577001 | 0.65 |

Better standalone validation BPB did not guarantee a better pair. The
older GELU-29 branch remains the control. All grids are retained as JSON.
We also screened direct *probability* averaging of the previous best pair at
weights 0.35, 0.50, 0.65 and temperatures 1.0, 1.09. The best sampled result
was **1.49680886** at RoPE weight 0.65 and temperature 1.09, worse than
the existing logit average at 1.48962402. The probability method is rejected;
its grid is retained in `rope_swa12_gelu29_probability_screen.json`.

## Matched GELU-29 experiments

The ordinary GELU-29 control trained for 4,800 steps on 39,321,600 targets
and reached 1.61914559 validation BPB. A same-seed, same-step dropout-0.1
run with random-window sampling isolates dropout. It reached 1.61857126
standalone BPB, only 0.00057 better, and its best sampled pair with the
frozen RoPE branch (weight 0.70, temperature 1.09) reached **1.49923182**,
worse than the control pair at 1.48962402. The dropout pair is rejected.
A coverage-sampling run with the same dropout recipe isolates the sampling
change at fixed training targets. It reached **1.60329258** standalone BPB,
improving by 0.01528 over the random-window dropout control. Yet its best
sampled pair with the frozen RoPE branch (weight 0.65, temperature 1.09)
reached **1.49588261**, also worse than the ordinary GELU-29 pair.
Standalone improvement thus did not yield complementary ensemble errors in
this setting; both new GELU/dropout candidates are rejected as replacements.
This is consistent with stronger correlation between their errors and the
RoPE branch, but we have not measured error correlation directly. Inference
cost would remain comparable because the candidate still uses two width-256
models; these worse validation pairs were not run on the test split.

## RoPE learning-rate schedule

A 12,000-step WSD candidate holds peak learning rate through 75% of the
updates, then decays to the same 10% floor as the prior cosine recipe. The
model, seed, sampler, batch size, training targets, and inference architecture
match the round-8 12,000-step RoPE run. We compare validation curves and
mixtures; the schedule is an experimental hypothesis, not a guaranteed gain.
The schedule idea is inspired by the primary [MiniCPM WSD study](https://arxiv.org/abs/2404.06395),
whose setting is much larger than this course project.

| Step | WSD standalone validation BPB | Cosine control BPB |
|---:|---:|---:|
| 2,000 | 1.69196090 | 1.69034287 |
| 4,000 | 1.60347508 | 1.59968981 |
| 6,000 | 1.57612890 | 1.56720351 |
| 8,000 | 1.56624927 | 1.55072771 |
| 10,000 | 1.54392328 | 1.53845604 |
| 12,000 | **1.51946018** | 1.53668045 |

The WSD model trailed until late decay, then improved the matched-budget
cosine control by 0.01722 BPB at 12,000 steps. Its pair with the original
GELU-29 branch reached **1.47815950** validation BPB at RoPE weight 0.65
and temperature 1.09, versus 1.48962402 for the previous selected pair.
The grid and training curve are saved. A 10,000/12,000 WSD same-run average
improved standalone validation to 1.51686743 but its best sampled pair was
1.47857030, slightly worse than the raw WSD endpoint; it was rejected.

## Low-rate continuation and longer WSD plan

Starting from the self-trained 12,000-step WSD endpoint, a fresh AdamW
optimizer trained for 2,000 more updates on the supplied train split with
coverage sampling, base learning rate 0.0001, no warmup, and a cosine decay
to 0.00001. The continuation used seed 47 and processed 16,384,000 further
targets; checkpoint ancestry includes the original 98,304,000 targets.
Standalone validation improved from 1.51946018 to **1.51723051**. At RoPE
weight 0.65 and temperature 1.09, its pair with GELU-29 reached
**1.47570018** validation BPB. It was the strongest candidate at this point;
the longer WSD schedule was evaluated next.
A separate 16,000-step WSD run from random initialization tested whether a
longer full schedule improves over the fresh-optimizer continuation.

The 16,000-step WSD run has now completed 131,072,000 train targets. At
10,000 and 12,000 steps, before its decay, standalone validation was
1.55255393 and 1.55309426, respectively. After decaying, it reached
1.52801914 at 14,000 steps and **1.51628848** at 16,000 steps. Its best
sampled pair with GELU-29 was **1.47169214** at RoPE weight 0.65 and
temperature 1.105. Averaging 14,000 and 16,000-step weights with 0.75 weight
on the latter improved standalone validation to **1.51293294**; the averaged
pair reached **1.47144813** at RoPE weight 0.70 and temperature 1.105.
The pair gain from averaging is only 0.00024 BPB, so its practical value is
small even though the validation selection favors it.

One additional RoPE dropout-0.15 WSD run used the same seed, architecture,
coverage sampler, and 12,000-step target count as the dropout-0.1 WSD control.
It reached **1.52249178** standalone BPB versus 1.51946018 for the control.
Its best sampled pair with GELU-29 reached **1.48630232** (RoPE weight 0.70,
temperature 1.09), considerably worse than the dropout-0.1 WSD16 average
pair. Stronger dropout was rejected. The complete curve and six-point pair
grid are retained.

## Frozen final predictor and resource check

After comparing the above candidates on validation, we froze
`runs/rope_wsd16_swa_gelu29_w07_t1105_round9/checkpoint.pt`, SHA-256
`e7df9537318e07a6019e0eaa4f1ea424351af9e2ea85f15ff1a9edd7d997383e`.
Its RoPE branch averages the 14,000 and 16,000-step WSD checkpoints with
weights 0.25 and 0.75. It mixes that branch's logits with the existing
GELU-29 model at RoPE weight 0.70 and temperature 1.105. Same-run averaging
does not double-count training ancestry: 131,072,000 RoPE targets plus
39,321,600 GELU targets = **170,393,600** processed targets.

CPU FP32 validation reproduced **1.47144806 BPB**. Candidate scoring took
34.56 seconds versus 8.13 for the same-period baseline (4.25 times).
The frozen candidate then scored **1.49135495 BPB** on the complete CPU FP32
test, improving the round-8 selected 1.50912712 by **0.01777217 BPB** and
crossing the 1.5 target. Full-test scoring took 38.91 seconds versus 9.07
for the same-period baseline (**4.289 times**). Peak test working set was
1.857 GiB. The model has 7,433,216 unique parameters; its checkpoint is
29,769,336 bytes (28.39 MiB), with no separate inference asset. It meets
the five-times CPU, 4 GiB RAM, and 64 MiB asset limits. The test result was
measured only after the validation-based choice was frozen. No website
submission was made. All four CPU resource/score JSON files are in
`experiment_results/round9/`.

The six new model trainings in this round used about 5,160.7 aggregate
measured GPU-process seconds, including rejected candidates and continuation;
the continuation's own training time was 134.78 seconds and its inherited
WSD training cost is recorded separately. Validation sweep logs and the
same-run average grids are retained. Some screens overlapped GPU training,
so these summed process seconds are not elapsed wall-clock time. The course
contract suite passed 11 tests.

## Reproduction commands

Run from `code/` with fresh output directories if the recorded names exist.
All model fitting uses the supplied train split. The rejected mixture grids
and intermediate validation scores are under `experiment_results/round9/`.

```text
python train.py --implementation student_dropout --config configs/gelu_dropout10_width256.json --device cuda --precision fp32 --threads 4 --seed 29 --steps 4800 --eval-every 1200 --run-dir runs/gelu_dropout10_w256_4800_s29_round9
python train.py --implementation student_dropout --config configs/gelu_dropout10_width256.json --device cuda --precision fp32 --threads 4 --seed 29 --steps 4800 --sampler coverage --eval-every 1200 --run-dir runs/gelu_dropout10_coverage_w256_4800_s29_round9
python train.py --implementation student_rope --config configs/rope_dropout10_width256.json --device cuda --precision fp32 --threads 4 --seed 17 --steps 12000 --sampler coverage --schedule wsd --stable-fraction 0.75 --eval-every 2000 --save-every 2000 --run-dir runs/rope_dropout10_coverage_wsd12000_s17_round9
python train.py --implementation student_rope --config configs/rope_dropout10_width256.json --device cuda --precision fp32 --threads 4 --seed 47 --steps 2000 --sampler coverage --init-checkpoint runs/rope_dropout10_coverage_wsd12000_s17_round9/checkpoint.pt --learning-rate 0.0001 --lr-floor 0.1 --warmup-steps 0 --eval-every 500 --save-every 500 --run-dir runs/rope_wsd12_finetune2k_s47_round9
python train.py --implementation student_rope --config configs/rope_dropout10_width256.json --device cuda --precision fp32 --threads 4 --seed 17 --steps 16000 --sampler coverage --schedule wsd --stable-fraction 0.75 --eval-every 2000 --save-every 2000 --run-dir runs/rope_dropout10_coverage_wsd16000_s17_round9
python same_run_average.py --earlier runs/rope_dropout10_coverage_wsd16000_s17_round9/checkpoint_step_14000.pt --later runs/rope_dropout10_coverage_wsd16000_s17_round9/checkpoint.pt --output-dir runs/rope_wsd16_swa_14000_16000_round9
python train.py --implementation student_rope --config configs/rope_dropout15_width256.json --device cuda --precision fp32 --threads 4 --seed 17 --steps 12000 --sampler coverage --schedule wsd --stable-fraction 0.75 --eval-every 2000 --save-every 2000 --run-dir runs/rope_dropout15_coverage_wsd12000_s17_round9
python mixed_ensemble_sweep.py --first runs/rope_wsd16_swa_14000_16000_round9/checkpoint.pt --second runs/gelu_w256_4800_s29/checkpoint.pt --weights 0.6 0.65 0.7 --temperatures 1.09 1.105 1.12 --output experiment_results/round9/rope_wsd16_swa_gelu29_sweep.json
python make_mixed_ensemble_checkpoint.py --first runs/rope_wsd16_swa_14000_16000_round9/checkpoint.pt --second runs/gelu_w256_4800_s29/checkpoint.pt --first-weight 0.7 --temperature 1.105 --output runs/rope_wsd16_swa_gelu29_w07_t1105_round9/checkpoint.pt
python evaluate.py --checkpoint runs/rope_wsd16_swa_gelu29_w07_t1105_round9/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation
python evaluate.py --checkpoint runs/rope_wsd16_swa_gelu29_w07_t1105_round9/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test
```
