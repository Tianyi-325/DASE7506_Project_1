# Research round 11: matched-temperature branch and calibration ablations

This round completes validation-only controls for the student-chosen simple
predictor: the raw 16,000-step RoPE WSD branch and the ordinary GELU-29
branch with equal logit weights and temperature 1.105. Both parents were
frozen; no model was trained or selected using test text in this round.
Evaluation used the fixed course scorer on CPU FP32 with four threads and
the full validation split. The single-branch temperature adapter only
applies `log_softmax(branch_logits / T)` and does not change the branch.

| Predictor | RoPE weight | Temperature | Validation BPB | Scorer seconds |
|---|---:|---:|---:|---:|
| Raw 16k RoPE alone | 1 | 1.105 | 1.50005046 | 18.80 |
| GELU-29 alone | 0 | 1.105 | 1.60930402 | 16.79 |
| Equal-logit mixture | 0.5 | 1.000 | 1.48649465 | 33.57 |
| Equal-logit mixture (student-chosen simple variant) | 0.5 | 1.105 | **1.47970454** | 42.76 (earlier measurement) |

At the same temperature, the equal mixture improves validation BPB by
0.02034592 relative to RoPE alone and 0.12959948 relative to GELU alone.
This verifies a mixture benefit in predictive quality, but it does not
isolate the effect of ensembling at a fixed total training-target budget:
the mixture includes the extra GELU branch's 39,321,600 train targets and
model parameters. With the same two frozen branches and equal weight,
temperature 1.105 improves validation BPB by 0.00679010 over temperature
1.0. Timing varies with background CPU load and is not a controlled
speed comparison between these settings.

The temperature-1.0 mixture bundle is
`runs/rope_raw16_gelu29_w05_t1000_round11/checkpoint.pt`, SHA-256
`8dd4e99bebba3fb88ff45cd5c946e534e1b494c32c18c17aa472909b923e9b69`.
The temperature-1.105 simple predictor remains
`runs/rope_raw16_gelu29_w05_t1105_round10/checkpoint.pt`, SHA-256
`c162aa90c1e5cd445128a8b105d5346d8b3dc156917d1116b49920e7fe9a3074`.
Its previously measured full-test CPU FP32 BPB is 1.49895557. No test
evaluation was performed in this round.

The three new scorer JSON files and per-window NLL arrays are in
`experiment_results/round11/`; the existing equal-mixture temperature-1.105
CPU record is `experiment_results/round10/raw_equal_validation_cpu_fp32.json`.
The single-branch evaluations use `evaluate_temperature_ablation.py`, which
calls the unchanged fixed `evaluate.score` function. The mixture uses the
unchanged `student_mixed_ensemble` implementation and `evaluate.py`.

Reproduce from `code/`:

```text
python evaluate_temperature_ablation.py --checkpoint runs/rope_dropout10_coverage_wsd16000_s17_round9/checkpoint.pt --temperature 1.105 --output experiment_results/round11/rope_raw16_t1105_validation_cpu_fp32.json
python evaluate_temperature_ablation.py --checkpoint runs/gelu_w256_4800_s29/checkpoint.pt --temperature 1.105 --output experiment_results/round11/gelu29_t1105_validation_cpu_fp32.json
python make_mixed_ensemble_checkpoint.py --first runs/rope_dropout10_coverage_wsd16000_s17_round9/checkpoint.pt --second runs/gelu_w256_4800_s29/checkpoint.pt --first-weight 0.5 --temperature 1.0 --output runs/rope_raw16_gelu29_w05_t1000_round11/checkpoint.pt
python evaluate.py --checkpoint runs/rope_raw16_gelu29_w05_t1000_round11/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation --output experiment_results/round11/equal_t1000_validation_cpu_fp32.json
```
