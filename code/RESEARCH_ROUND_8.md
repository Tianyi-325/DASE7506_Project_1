# Research round 8: trained mixture gate and further RoPE training

All model fitting in this round uses the supplied train text only. Validation
is used for model and checkpoint selection. The tokenizer, data, and fixed
evaluator are unchanged; prediction remains causal within each independent
256-token window. Full logs and metrics are under `experiment_results/round8/`
and `runs/`.

## Causal train-only mixture gate

We froze the prior selected RoPE-coverage-17 and GELU-29 models. A three-parameter
logistic gate learned from 1,024,000 processed train targets at random windows.
At each token it may use only the two current-prefix top-logit margins and the
position within the current window. Its coefficient initialization reproduces
the prior weight-0.60 logit mixture; the shared logit temperature remains 1.075.
Gate strength was selected on validation, without fitting its parameters on
validation text.

| Gate strength | Validation BPB |
|---:|---:|
| 0 (prior fixed mix) | **1.51934833** |
| 0.25 | 1.51935943 |
| 0.5 | 1.52004001 |
| 1 | 1.52333675 |

The trained gate did not improve validation. Its learned intercept lowers
the RoPE weight on train text, but this did not transfer to validation.
Gate fitting and validation screening took 59.11 wall-clock seconds, including
loading and validation. Its complete curve and selected checkpoint are in
`runs/gated_rope_coverage_s17_gelu_s29_round8/`.
The gate is rejected and is not used in the final predictor.
Because the frozen parent models had already learned from the train text,
training-only gate loss could reward choices that do not generalize; the
validation grid is the evidence for rejecting it, not proof of that cause.

## Longer RoPE with coverage sampling

The prior best single RoPE model used 6,000 updates. We trained a 9,000-update
seed-17 run with the same architecture, batch size, coverage sampler, and
optimizer formula. Snapshots were saved every 1,500 updates, along with
validation measurements. The cosine schedule is defined over the *planned*
number of updates, so intermediate 9,000-run snapshots are not exact
continuations or matched controls of the previous 6,000-run schedule.

| Step on the 9,000-step plan | Processed targets | Single-model validation BPB | With GELU-29 at weight 0.60, T=1.075 |
|---:|---:|---:|---:|
| 6,000 | 49,152,000 | 1.56274733 | 1.51269123 |
| 7,500 | 61,440,000 | 1.54759383 | 1.50319319 |
| 9,000 | 73,728,000 | 1.54673325 | 1.50044339 |

The previous separate 6,000-step run reached 1.57062204 single-model and
1.51934833 combined validation BPB. The 9,000-step run took 1,701.91 measured
GPU training seconds while another GPU training job overlapped for much of it;
this is not a controlled training-speed comparison. Complete curves and
snapshot hashes are in `runs/rope_dropout10_coverage_w256_9000_s17_round8/`.

We also validation-screened weight averages of checkpoints from this *same*
run, which share one training ancestry. Averaging the 7,500 and 9,000-step
weights equally improved single-model validation BPB to **1.54343562**.
Averaging 6,000 and 9,000 steps was less effective: its best sampled weight
(0.75 on 9,000) scored 1.54534878. The full grids are saved in the round-8
logs and run metrics. Pairing the 7,500/9,000 average with GELU-29 gave
**1.50000340 validation BPB** at RoPE weight 0.65, temperature 1.09. This is
the best validation candidate so far, but its CPU resource use and full-test
BPB have not yet been measured. The corresponding bundle is
`runs/rope_swa_7500_9000_gelu29_w065_t109_round8/checkpoint.pt`.

## Q/K-normalized RoPE screen

A separate RoPE model normalizes each query and key per head to unit length,
then scales by the square root of head width before rotary rotation and
scaled-dot-product attention. This changes attention geometry without adding
parameters. It uses the same seed, dropout, coverage sampler, 6,000 updates,
and processed-target count as the previous ordinary RoPE run. Random-weight
causality and probability-normalization checks passed.

The Q/K-normalized model reached **1.57604705** validation BPB after 6,000
updates and 49,152,000 processed targets, versus 1.57062204 for the ordinary
RoPE control. It has the same 3,683,840 parameters and is rejected. Its
measured training time was 1,589.28 seconds under GPU contention. Complete
metrics are in `runs/rope_qknorm_dropout10_coverage_w256_6000_s17_round8/`.

## Additional 12,000-step training plan

A second from-scratch RoPE/coverage run used a 12,000-step cosine plan with
the same model, seed, batch size, and sampler. It processed 98,304,000 targets
and took 1,128.28 measured GPU training seconds. Snapshots and validation
curves are in `runs/rope_dropout10_coverage_w256_12000_s17_round8/`.

| Step | Single-model validation BPB |
|---:|---:|
| 2,000 | 1.69034287 |
| 4,000 | 1.59968981 |
| 6,000 | 1.56720351 |
| 8,000 | 1.55072771 |
| 10,000 | 1.53845604 |
| 12,000 | 1.53668045 |

At 10,000 steps, pairing with GELU-29 at RoPE weight 0.65 and temperature
1.09 reached 1.49317949 validation BPB. The raw 12,000-step model at those
mixture settings reached 1.48968424. Equal weight averaging of the 10,000
and 12,000-step checkpoints improved the single model from 1.53668045 to
**1.53379058**. The averaged model's best sampled mixture used the same
weight 0.65 and temperature 1.09, reaching **1.48962402 validation BPB**.
The full nine-point ensemble grid and five-point average grid are saved in
`experiment_results/round8/` and the average run's `metrics.json`.

The selected average and raw-12,000 mixtures differ by only 0.00006021 BPB
on validation. Thus, the main improvement this round comes from the longer
RoPE training plan, with a smaller additional effect from same-run averaging.
This is a single seed and validation-selected trajectory, not evidence that
the same gain must recur at every seed or benchmark.
More training targets and a longer cosine schedule plausibly improve the
RoPE model's estimated next-token distribution. Same-run averaging plausibly
reduces sensitivity to its final optimizer steps without adding inference
work. These explanations are consistent with the observed validation curves;
the experiment does not isolate training duration from schedule length.

## Frozen final predictor and resource check

The selected bundle is
`runs/rope_swa_10000_12000_gelu29_w065_t109_round8/checkpoint.pt`, SHA-256
`b2b4a3a9b553c08c27788c139266f0dc97018442dced4e11e66c25d583178670`.
It has 7,433,216 unique parameters and 29,769,336 checkpoint bytes
(28.39 MiB), with no separate inference asset. Its two independent training
ancestors used 98,304,000 + 39,321,600 = **137,625,600** processed training
targets; the same-run average does not double-count its two snapshots.

CPU FP32 validation reproduced **1.48962396 BPB**. Candidate scoring took
34.83 seconds versus 8.04 seconds for the same-period baseline (4.33 times),
with a 1.859 GiB sampled peak working set. After this validation-based
selection and freeze, complete CPU FP32 test gave **1.50912712 BPB**, versus
the previous selected 1.54159969. Candidate test scoring took 40.42 seconds
versus 8.84 seconds for the same-period baseline (**4.575 times**), and peak
working set was 1.857 GiB. The frozen candidate satisfies all three limits:
five-times CPU scoring, 4 GiB peak RAM, and 64 MiB uncompressed assets.
The target of below 1.5 *test* BPB was not reached. No website submission
was made. CPU result and resource JSON files are under
`experiment_results/round8/`.

The three new full model trainings took 1,701.91 + 1,589.28 + 1,128.28 =
4,419.47 measured GPU-process seconds in aggregate. The gate fitting and
screen added 59.11 elapsed seconds, and the separate validation grids and
CPU checks are retained in the round-8 logs. The first two model trainings
overlapped, so summing their measured durations is not elapsed wall time.

## Reproduction

Commands run from `code/` with the project environment. Use new output
directories if the recorded paths already exist. Only supplied train text
is used for fitting; the validation split is used for selection.

```text
python train.py --implementation student_rope --config configs/rope_dropout10_width256.json --device cuda --precision fp32 --threads 4 --seed 17 --steps 9000 --sampler coverage --eval-every 1500 --save-every 1500 --run-dir runs/rope_dropout10_coverage_w256_9000_s17_round8
python train.py --implementation student_rope_qknorm --config configs/rope_dropout10_width256.json --device cuda --precision fp32 --threads 4 --seed 17 --steps 6000 --sampler coverage --eval-every 1200 --run-dir runs/rope_qknorm_dropout10_coverage_w256_6000_s17_round8
python train.py --implementation student_rope --config configs/rope_dropout10_width256.json --device cuda --precision fp32 --threads 4 --seed 17 --steps 12000 --sampler coverage --eval-every 2000 --save-every 2000 --run-dir runs/rope_dropout10_coverage_w256_12000_s17_round8
python same_run_average.py --earlier runs/rope_dropout10_coverage_w256_12000_s17_round8/checkpoint_step_10000.pt --later runs/rope_dropout10_coverage_w256_12000_s17_round8/checkpoint.pt --output-dir runs/rope_coverage_s17_swa_10000_12000_round8
python mixed_ensemble_sweep.py --first runs/rope_coverage_s17_swa_10000_12000_round8/checkpoint.pt --second runs/gelu_w256_4800_s29/checkpoint.pt --weights 0.6 0.65 0.7 --temperatures 1.075 1.09 1.105 --output experiment_results/round8/swa_10000_12000_gelu29_sweep.json
python make_mixed_ensemble_checkpoint.py --first runs/rope_coverage_s17_swa_10000_12000_round8/checkpoint.pt --second runs/gelu_w256_4800_s29/checkpoint.pt --first-weight 0.65 --temperature 1.09 --output runs/rope_swa_10000_12000_gelu29_w065_t109_round8/checkpoint.pt
python evaluate.py --checkpoint runs/rope_swa_10000_12000_gelu29_w065_t109_round8/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation
python evaluate.py --checkpoint runs/rope_swa_10000_12000_gelu29_w065_t109_round8/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test
```
