# Research round 10: checkpoint averaging and equal branch weights

This is a prespecified 2-by-2 ablation of the selected round-9 RoPE + GELU
predictor. The RoPE branch is either its raw 16,000-step WSD endpoint or the
same-run 14,000/16,000-step weight average (0.25/0.75). The frozen GELU-29
branch is identical in every cell. Their logits are combined at RoPE weight
0.50 or 0.70, always with temperature 1.105. No model was retrained, and no
temperature or weight was retuned after seeing the test results. All four
predictors were fully specified before testing. The earlier round-9 selection
remains the selected final model; the three new test evaluations are diagnostic
ablations, not a test-based model search.

| RoPE weights | RoPE logit weight | CPU FP32 validation BPB | Full-test BPB | Test delta vs selected |
|---|---:|---:|---:|---:|
| Raw 16k | 0.50 | 1.47970454 | 1.49895557 | +0.00760062 |
| Raw 16k | 0.70 | 1.47178198 | 1.49183713 | +0.00048218 |
| 14k/16k average | 0.50 | 1.48013037 | 1.49928177 | +0.00792682 |
| 14k/16k average | 0.70 | **1.47144806** | **1.49135495** | reference |

At fixed 0.70 branch weight, removing checkpoint averaging increases test
BPB by 0.00048218. At fixed equal branch weights, however, the average is
0.00032620 worse on test than the raw endpoint. This interaction cautions
against claiming a general benefit from averaging based on this one run.
At fixed raw RoPE weights, equal mixing is 0.00711844 worse on test than
0.70 RoPE weight; with averaged RoPE weights it is 0.00792682 worse. Both
equal-weight variants still achieve test BPB below 1.5. The validation
ordering agrees with the test ordering, but these test ablations are not used
for further tuning. A wider weight or temperature search was intentionally
not part of this controlled comparison.

The equal mix is simple and valid, but this evidence does not support replacing
the selected 0.70 mixture. A more capable RoPE branch plausibly deserves more
weight, yet the exact 0.70 value remains validation-selected, not theoretically
derived. Its advantage over the neighboring 0.65 setting on validation was
only 0.00014 BPB in the previous round, so the precise optimum is uncertain.

## CPU resource measurements

All new evaluations used the unchanged fixed scorer with CPU FP32 and four
threads. The same-period baseline full-test scorer took 10.76866 seconds.
The selected round-9 model's earlier full-test scorer took 38.90959 seconds
against its own same-period 9.07267-second baseline (4.289 times). Windows
timings vary across periods; compare each candidate against the baseline of
its measurement period. Every candidate has 7,433,216 unique parameters,
170,393,600 processed training targets in its two-branch ancestry, and one
29,769,336-byte checkpoint (28.39 MiB) with no separate inference asset.

| RoPE weights | RoPE logit weight | Validation scorer seconds | Test scorer seconds | Same-period test ratio | Peak test working set GiB |
|---|---:|---:|---:|---:|---:|
| Raw 16k | 0.50 | 42.76217 | 46.75966 | 4.342 | 1.8584 |
| Raw 16k | 0.70 | 43.93945 | 49.47600 | 4.594 | 1.8580 |
| 14k/16k average | 0.50 | 44.45428 | 41.42312 | 3.847 | 1.8572 |
| 14k/16k average | 0.70 | 34.56178 (round 9) | 38.90959 (round 9) | 4.289 (round 9) | 1.8573 (round 9) |

All measured variants meet the 5-times CPU, 4 GiB RAM, and 64 MiB inference
asset limits. Timing is sensitive to background load; the model architecture
and parameter count are identical across cells, so scorer timing differences
should not be interpreted as a speed effect of averaging or the scalar weight.

## Artifacts and reproduction

The three new bundles are `runs/rope_raw16_gelu29_w05_t1105_round10/`,
`runs/rope_raw16_gelu29_w07_t1105_round10/`, and
`runs/rope_swa16_gelu29_w05_t1105_round10/`. Their SHA-256 hashes are,
respectively, `c162aa90c1e5cd445128a8b105d5346d8b3dc156917d1116b49920e7fe9a3074`,
`d5159b3b869bc1c41d2c2147a9bdf401f4d10c04f809364a6c0818486e138369`,
and `455225a65bd7e6c07d9056a1c61ba078907fef25a116e04519b786e81a37434d`.
The selected 0.70 averaged bundle is the round-9 checkpoint with SHA-256
`e7df9537318e07a6019e0eaa4f1ea424351af9e2ea85f15ff1a9edd7d997383e`.
Every new validation/test scorer JSON, per-window NLL array, stdout/stderr log,
and sampled peak-working-set JSON is saved under `experiment_results/round10/`.

From `code/`, create the three bundles with `make_mixed_ensemble_checkpoint.py`
using the frozen RoPE raw/average and GELU-29 parents, the corresponding
`--first-weight` value above, and `--temperature 1.105`. Then run
`evaluate.py --checkpoint <bundle>/checkpoint.pt --device cpu --precision fp32
--threads 4 --split validation` and likewise for `test`. For peak working set
and fixed output filenames, use `measure_eval.ps1`. The inputs, evaluator,
tokenizer, and inference implementation were not changed in this round.
