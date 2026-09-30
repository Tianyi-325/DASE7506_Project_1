# Research round 1: training budget and model width

All development decisions in this round used the supplied validation split.
Training used CUDA FP32, seed 17, batch size 32, context 256, and the same
optimizer and learning-rate recipe. Each 4,800-step run processed 39,321,600
training targets. The cosine schedule in `train.py` depends on the total step
count, so the 1,200-step and 4,800-step runs differ in both duration and
learning-rate trajectory. Within each 4,800-step comparison, the schedule and
number of targets are matched.

| Run | Change from preceding controlled run | Parameters | Validation BPB |
|---|---|---:|---:|
| `baseline_cuda_fp32_s17` | Initial 1,200-step baseline | 1,088,256 | 2.07108 |
| `baseline_4800_s17` | 4,800-step baseline recipe | 1,088,256 | 1.75243 |
| `swiglu_4800_s17` | Replace GELU MLP with near-parameter-matched SwiGLU | 1,093,056 | 1.74610 |
| `swiglu_w192_4800_s17` | Increase width from 128 to 192 | 2,223,232 | 1.69530 |
| `swiglu_w256_4800_s17` | Increase width from 192 to 256 | 3,767,168 | 1.64800 |

The 4,800-step baseline and SwiGLU pair is the mechanism ablation at equal
training targets. SwiGLU improves validation BPB by 0.00633 at this budget.
Width scaling yields larger additional gains, but also increases parameters,
training time, and CPU scoring time. The final width-256 run trained for
456.55 seconds on the RTX 3060 Laptop GPU. Its checkpoint is 15,086,645 bytes
(14.39 MiB). CPU FP32 validation scoring took 17.81 seconds versus 7.93
seconds for the original width-128 baseline on this machine. Windows
`PeakWorkingSet64`, sampled every 100 ms during width-256 CPU validation,
reached 1,962,831,872 bytes (1.828 GiB). A repeat full-test run reached
1,963,315,200 bytes (also 1.828 GiB) and reproduced the same BPB. See
`runs/swiglu_w256_4800_s17/resource_check.json` and
`resource_check_test.json` for the measurement method.

After selecting the width-256 checkpoint on validation, it was frozen for one
full-test CPU FP32 evaluation. The resulting test BPB was **1.6736554152**,
with 19.22 seconds of scorer time. The original 1,200-step baseline test took
9.02 seconds on this machine, so the measured ratio is about 2.13 times,
below the 5-times limit. No other new candidate was scored on test in this
round. Future model selection must continue to use validation only.

The largest observed improvement came from a longer training recipe; this is
not evidence that SwiGLU alone accounts for the full gain. The validation
curve still declined at 4,800 steps, though more slowly, so further training
could help or could eventually overfit. These are single-seed observations.
The six full runs recorded so far consumed about 1,025 seconds of GPU training
in total, excluding preparation, validation, and the earlier one-step smoke
check. A width-256 GELU run with the same 4,800-step budget would be needed to
isolate SwiGLU's contribution at the final model size.

Commands and run-specific metrics are retained in `RUN_LOG.csv`, each run's
`metrics.json`, and `runs/logs/`. Reproducing any training run requires a new
output directory because `train.py` refuses to overwrite existing results.
