# Research round 2: architecture ablation and window-local memory

All candidate choices used validation. The only new training run used the
supplied train text, CUDA FP32, seed 17, batch size 32, context 256, and 4,800
updates (39,321,600 processed targets). All models use the fixed tokenizer and
evaluator. The cache uses only tokens already observed in the current causal
window and is rebuilt on every call; it has no cross-window state.

## Width-256 mechanism ablation

| Feed-forward layer | Parameters | Validation BPB | GPU train seconds |
|---|---:|---:|---:|
| GELU | 3,749,376 | 1.62358452 | 480.93 |
| SwiGLU | 3,767,168 | 1.64799671 | 456.55 |

The two runs have the same seed, target count, depth, width, heads, optimizer,
and training schedule. At width 256, SwiGLU is worse by 0.02441 BPB. This
reverses its small advantage at width 128, so the GELU checkpoint became the
backbone for memory experiments. The checkpoint for the GELU run is
`runs/gelu_w256_4800_s17/checkpoint.pt` (SHA-256 in `RUN_LOG.csv`).

## Causal cache sweep

The cache mixes the neural next-token distribution with a distribution of
previously observed successors to the current context. For the selected
bigram method, position `t` searches earlier positions `s < t` with the same
two-token context and uses the already-observed `ids[s+1]` as a candidate
next token. When there is no match, prediction remains purely neural.

The full 15-candidate search is preserved in
`runs/cache_sweep_gelu_w256_s17_extended.json` and its console log. Key
validation FP32 results are:

| Cache mode | Weight | Validation BPB |
|---|---:|---:|
| None | 0 | 1.62358452 |
| Recent token frequency | 0.05 | 1.62451456 |
| Same-token successor | 0.10 | 1.59700731 |
| Same-bigram successor | 0.20 | 1.59626111 |
| Same-bigram successor | **0.30** | **1.59568084** |
| Same-bigram successor | 0.40 | 1.59671537 |

Distance decay with scale 64 did not improve the best score. An initial decay
sweep exposed a probability-normalization bug; that sweep failed, the code
was fixed, and the complete extended sweep was rerun successfully. Seven
contract tests now include decay normalization, causality, and independence.
The inference implementation was also simplified to scatter cache mass
directly into the dense output probabilities. It reproduced validation BPB
to within 1e-10 and reduced CPU validation scoring from 34.45 to 31.66
seconds on this machine.

The selected cache checkpoint was constructed from the trained GELU weights
without further training. It records the parent checkpoint SHA-256. Its
validation BPB is **1.5956808052** and frozen full-test CPU FP32 BPB is
**1.6096751344**. Full-test scorer time was 37.13 seconds, versus 9.02
seconds for the original width-128 baseline on the same computer: about
4.12 times, within the 5-times limit. Windows `PeakWorkingSet64`, sampled
every 100 ms during the full-test process, was 1.824 GiB. The checkpoint is
15,015,669 bytes (14.32 MiB). Resource measurements are also stored in
`runs/gelu_cache_bigram03_s17/resource_check_test.json` and the scorer JSON.

## Probability ensemble, rejected for resource cost

The independently trained width-256 GELU and SwiGLU models were mixed at
GELU weights 0.25, 0.50, and 0.75. Validation BPB was 1.57735, 1.56207,
and 1.57048, respectively; all candidates are recorded in
`runs/ensemble_sweep_w256_s17.json`. The 0.50 mixture had the best validation
score, but CPU FP32 validation scoring took **62.18 seconds**, versus 7.93
seconds for the initial baseline's CPU validation score. This exceeds the
5-times CPU limit, so the ensemble was rejected and was not evaluated on
test. Its 1.861 GiB process working set and 30,102,521-byte checkpoint
would satisfy the other two resource limits. Its checkpoint records both
parent hashes and 78,643,200 processed training targets including ancestry.

The selected cache model is the strongest compliant predictor established in
this round. Future model selection must continue to use validation only.

## Reproduction

Run from `code/` in the environment described in `README.md`, using new run
directory names if the listed directories already exist:

```text
python train.py --implementation model --config configs/baseline_width256.json --device cuda --precision fp32 --threads 4 --seed 17 --steps 4800 --eval-every 1200 --run-dir runs/gelu_w256_4800_s17
python make_cache_checkpoint.py --base runs/gelu_w256_4800_s17/checkpoint.pt --output runs/gelu_cache_bigram03_s17/checkpoint.pt --mode bigram --strength 0.3
python evaluate.py --checkpoint runs/gelu_cache_bigram03_s17/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation
python evaluate.py --checkpoint runs/gelu_cache_bigram03_s17/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test
```
