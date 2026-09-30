# Research round 3: training budget and compact ensemble

All model and mixing choices in this round were made on validation. The
selected predictor was frozen before one complete CPU FP32 test evaluation.
The supplied train split alone was used for learning; the data, tokenizer,
evaluator, and independent 256-token scoring windows remain unchanged.
The full per-candidate results and timings are retained as JSON in
`experiment_results/round3/` (also in `runs/` locally); run-level results
and checkpoint hashes are in `RUN_LOG.csv`.

## More training at width 256

A new GELU GPT (seed 17, batch 32, width 256, 4 layers, 4 heads) was trained
for 8,000 steps: 65,536,000 processed training targets and 898.98 GPU FP32
training seconds. Its validation curve was:

| Step | Validation BPB |
|---:|---:|
| 2,000 | 1.70694757 |
| 4,000 | 1.64216383 |
| 6,000 | 1.63407878 |
| 8,000 | 1.64893094 |

The earlier 4,800-step GELU run scored 1.62358452. Simply extending the
training recipe did not improve validation; the final model was worse even
though its training loss continued falling. The full history is in
`runs/gelu_w256_8000_s17/metrics.json`. No test evaluation was run.

We also interpolated the 4,800- and 8,000-step model weights. Weights of
0, 0.25, 0.5, 0.75, and 1 on the 8,000-step model gave validation BPB
1.62358452, **1.62196422**, 1.62473150, 1.62606121, and 1.64893094.
The selected 0.25 interpolation incorporates both parent training costs:
104,857,600 processed targets and 1,379.91 GPU training seconds in total.
The sweep is in `runs/weight_soup_gelu_w256_s17.json`. A subsequent 15-way
cache sweep on this interpolated model reached 1.59490205 with bigram cache
strength 0.3, only 0.00078 BPB below the previous cached model's validation
score. The full grid is in `runs/cache_sweep_gelu_soup_w256_s17.json`. No test
evaluation was run for these candidates.

## Cache variants

On the original 4,800-step GELU model, we checked within-window trigram
matching, variable-order matching, and cache strength adjusted by the number
of matches. All are causal and reset between scoring windows. The 16
candidate results are in `runs/cache_sweep_v2_gelu_w256_4800_s17.json`.
The former fixed bigram cache at strength 0.3 scored 1.59568084. The best
variable-order setting scored 1.59567872, a difference of only 0.00000212
BPB. Trigram alone and count-adaptive cache were worse. This difference is
too small to justify replacing the simpler cache. These cache experiments
used validation only.

## Logit ensemble selected

The previous probability ensemble of the independently trained width-256
GELU and SwiGLU models exceeded the CPU time limit. We instead combined
their logits before one softmax, which also changes how disagreements affect
the final distribution. GELU weights 0.25, 0.50, and 0.75 gave validation
BPB **1.58089086, 1.55450645, and 1.56800386**. The complete sweep is in
`runs/ensemble_logit_sweep_w256_s17.json`. The selected 0.50 mix scored
1.55450638 on CPU FP32 validation in 33.40 seconds. Its checkpoint records
both parent hashes and their combined 78,643,200 processed targets; the
parent training runs took 937.48 GPU seconds together.

After selection, the frozen ensemble scored **1.57292078 BPB** on complete
CPU FP32 test. Scorer time was **37.40 seconds**, versus the original
baseline's 9.02 seconds on the same machine, a **4.15×** ratio below the
5× limit. Sampled Windows peak process working set was **1.86 GiB** below
4 GiB. The checkpoint is **30,102,585 bytes (28.71 MiB)**, below 64 MiB;
it has no separate inference asset. These measurements and scorer outputs
are in `runs/ensemble_logit_w256_mix05_s17/`. This is the best compliant
predictor established so far; no submission has been made.

The logit ensemble's gain is consistent with the two trained models making
partly different errors. It reduces validation BPB by 0.04117 compared with
the previous cached model, at about the same full-test CPU time (37.40 versus
37.13 seconds). This explanation is a mechanism hypothesis; the validation
and CPU measurements establish the observed effect, not its cause.

## Reproduction

From `code/`, in the documented Python environment, with fresh output
directory names if these exist already:

```text
python train.py --implementation model --config configs/baseline_width256.json --device cuda --precision fp32 --threads 4 --seed 17 --steps 4800 --eval-every 1200 --run-dir runs/gelu_w256_4800_s17
python train.py --implementation student --config configs/swiglu_width256.json --device cuda --precision fp32 --threads 4 --seed 17 --steps 4800 --eval-every 1200 --run-dir runs/swiglu_w256_4800_s17
python ensemble_sweep.py --gelu runs/gelu_w256_4800_s17/checkpoint.pt --swiglu runs/swiglu_w256_4800_s17/checkpoint.pt --method logit --output runs/ensemble_logit_sweep_w256_s17.json
python make_ensemble_checkpoint.py --gelu runs/gelu_w256_4800_s17/checkpoint.pt --swiglu runs/swiglu_w256_4800_s17/checkpoint.pt --gelu-weight 0.5 --method logit --output runs/ensemble_logit_w256_mix05_s17/checkpoint.pt
python evaluate.py --checkpoint runs/ensemble_logit_w256_mix05_s17/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation
python evaluate.py --checkpoint runs/ensemble_logit_w256_mix05_s17/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test
```
