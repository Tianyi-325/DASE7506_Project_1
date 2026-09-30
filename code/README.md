# MP1 — final model and reproduction

The student-selected submission method is an **equal-logit ensemble** of a raw
16,000-step RoPE GPT and an independently trained GELU GPT. Both have four
layers, width 256, four heads, and causal 256-token windows. The RoPE branch
uses training-only dropout 0.1, coverage sampling, and a WSD learning-rate
schedule; the GELU branch uses learned absolute positions. Their logits are
weighted **0.5:0.5** and divided by temperature **1.105** before softmax.
There is **no checkpoint weight averaging**. See the [project guide](../GUIDE.md)
for the assignment rules.

| Frozen-result item | Value |
|---|---:|
| Complete-test CPU FP32 BPB | **1.4989555702583726** |
| Full-validation CPU FP32 BPB | 1.4797045436795053 |
| Checkpoint SHA-256 | `c162aa90c1e5cd445128a8b105d5346d8b3dc156917d1116b49920e7fe9a3074` |
| CPU scoring time vs same-period baseline | 4.125 times (37.93 / 9.19 seconds) |
| Peak test working set | 1.858 GiB |
| Uncompressed inference assets | One 29,769,336-byte checkpoint (28.39 MiB) |
| Unique model parameters | 7,433,216 |
| Processed training targets, both branches | 170,393,600 |

The checkpoint is frozen locally at `../final_artifact/checkpoint.pt`; the
adjacent `FREEZE_MANIFEST.json` records its parent and inference-code hashes
and repeated full-test result. These local paths will be replaced by the
public checkpoint download and immutable code links when the release is
published. The higher-scoring 1.49135495 research variant used
checkpoint averaging and a 0.70 RoPE logit weight; it is **not** the selected
submission method.

## 1. Install

Run commands from **code/** with Python 3.12. The supplied train, validation,
and test text and BPE-2048 tokenizer are included; evaluation needs no network
access, API key, or retraining. Create and activate a virtual environment or
Conda environment, then install PyTorch for one device:

```bash
# CPU-only PyTorch; sufficient to reproduce the reported score
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
# Or, on a compatible NVIDIA system, use the CUDA wheel instead of the CPU wheel:
# python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

For macOS, install `torch==2.7.1` from the default PyPI index. The final
score was reproduced on Windows with Python 3.12.14, PyTorch 2.7.1+cu126
(CPU inference), `tokenizers` 0.21.4, and four CPU threads. All 11 project
tests passed. CPU time ratios depend on machine load and should be measured
against the baseline in the same period.

## 2. Reproduce the frozen score without training

Obtain the released `checkpoint.pt`, verify its SHA-256 against the value
above, and run the unchanged scorer from **code/**:

```bash
python evaluate.py --checkpoint /path/to/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test --output reproduced_test_cpu_fp32.json
```

On Windows PowerShell, verify the downloaded file with
`Get-FileHash -Algorithm SHA256 -LiteralPath C:\path\to\checkpoint.pt`.
The local frozen copy can be scored using `../final_artifact/checkpoint.pt`.
The expected `bpb` is **1.4989555702583726** on the complete test split.
The evaluator also writes per-window NLL to a `.window-nll.npy` file. The
checkpoint embeds both branches, so the original training checkpoints are
not needed for inference. Test results were reproduced from the frozen copy;
the code, evaluator, and tokenizer hashes are in its freeze manifest.

## 3. Reproduce training from the supplied train split

These commands describe the two parent training recipes and the packaging
step. They require a compatible NVIDIA GPU; the saved checkpoint above is
the authoritative inference artifact, because independent GPU training may
not reproduce identical checkpoint bytes. Run each command in a fresh
output directory. Training uses only the supplied train text; validation
is used for selection and the test split is not used for tuning.

```bash
python train.py --implementation model --config configs/baseline_width256.json --device cuda --precision fp32 --threads 4 --seed 29 --steps 4800 --eval-every 1200 --run-dir runs/reproduce_gelu29
python train.py --implementation student_rope --config configs/rope_dropout10_width256.json --device cuda --precision fp32 --threads 4 --seed 17 --steps 16000 --sampler coverage --schedule wsd --stable-fraction 0.75 --eval-every 2000 --save-every 2000 --run-dir runs/reproduce_rope16k
python make_mixed_ensemble_checkpoint.py --first runs/reproduce_rope16k/checkpoint.pt --second runs/reproduce_gelu29/checkpoint.pt --first-weight 0.5 --temperature 1.105 --output runs/reproduce_equal/checkpoint.pt
```

The RoPE branch processed 131,072,000 targets and the GELU branch
39,321,600. On the recorded RTX 3060 Laptop GPU runs, their measured
GPU-process training times summed to about 2,233.74 seconds (37.23 minutes),
excluding validation and preparation. The much larger search across
candidate models is documented in `RUN_LOG.csv` and the research-round logs;
parent training is counted once even when a checkpoint is reused.

For the original classroom baseline, use `--implementation model`,
`--config configs/baseline.json`, `--steps 1200`, and `--seed 17` with
`train.py`. Its measured full-test CPU FP32 BPB was **2.10125672**.

Training writes `checkpoint.pt` and `metrics.json`; evaluation writes a
scorer JSON and per-window NLL. The reported score is `bpb` from the complete
CPU FP32 test, not token perplexity or validation BPB.

## 4. Files and model interface

| Files | Use |
|---|---|
| `model.py`, `configs/baseline.json` | Runnable baseline; preserve for comparisons. |
| `student_rope.py`, `student_mixed_ensemble.py` | Final RoPE branch and equal-logit ensemble implementation. |
| `train.py`, `make_mixed_ensemble_checkpoint.py` | Parent training and final bundle construction. |
| `common.py`, `evaluate.py` | Fixed data checks, windows and scorer; keep unchanged. |
| `data/` | Supplied splits, tokenizer and dataset hashes; keep unchanged. |
| `tests/` | Contract checks for causality, normalization, independence, and gradients. |
| `RUN_LOG.csv`, `experiment_results/` | Search log and recorded comparison results. |
| `PACKAGE_MANIFEST.json` | Original starter-package hashes, **not** final-release hashes; use the frozen-model manifest for the submitted predictor. |

- `build_model(config)` returns a PyTorch model with `context=256`.
- The supplied trainer calls `forward(ids)` for unnormalized logits; the scorer calls `predict_log_probs(ids)` for finite, normalized natural-log probabilities. Both outputs have shape `[batch, time, 2048]`.
- A prediction at position t may use only the observed prefix through t. Reset temporary state between independent windows, examples and scoring passes. Compact training-derived assets may be reused across windows; evaluation-prefix state may not.
- Checkpoints record the implementation module and configuration. Include that module and every required asset so the evaluator can reconstruct the submitted predictor. No optimizer state is required for direct evaluation.
- Training length, architecture, optimizer, regularization, self-trained weight averaging and ensembles may change within the guide's constraints. Log all seeds, processed training targets, checkpoint ancestry and search costs; reusing a checkpoint does not erase its training cost. No particular seed or score improvement is mandated.

## 5. Benchmark and resource measurements

**Fixed score.** Protocol `7506-mp1-wt2-v2`: WikiText-2 raw text, train-fitted BPE-2048, independent windows of 256 targets, including the final short window. Every target except the first token of each split is scored once. Input windows share a boundary token but carry no state. BPB is summed negative log-base-2 next-token probability divided by the split's entire raw UTF-8 byte length, including the first token's bytes.

| Split | Scored targets | UTF-8 bytes |
|---|---:|---:|
| Validation | 376,599 | 1,148,007 |
| Test | 428,405 | 1,292,013 |

Use validation for all development and checkpoint/mixture selection. Weights, statistics and retrieval entries must derive only from training text. The public test text enables reproduction; it must not be used to tune the method. Once frozen, the same predictor may be evaluated repeatedly for timing or reproduction. Token perplexity is not directly comparable with published word-level perplexity.

Measure all three limits for the same frozen predictor:

- **CPU time ≤5× baseline:** run the supplied scorer with CPU FP32 and four
  threads for both checkpoints on the same machine in the same period. The
  frozen ensemble took 37.93 seconds versus 9.19 seconds for the classroom
  baseline, a 4.125× ratio. The baseline reproduction recipe is above.
- **Peak RAM ≤4 GiB:** on Windows PowerShell, run
  `./measure_eval.ps1 -Checkpoint ../final_artifact/checkpoint.pt -Split test -OutputPrefix final_resource`.
  The script uses the active environment's `python` by default and writes
  `final_resource.resource.json`; the recorded peak working set was 1.858 GiB.
- **Inference assets ≤64 MiB uncompressed:** the sole inference asset is
  `checkpoint.pt` at 29,769,336 bytes (28.39 MiB). Verify a downloaded copy
  with `(Get-Item -LiteralPath C:\path\to\checkpoint.pt).Length`.

## 6. Prepare your submission and reproduce a peer

The [guide](../GUIDE.md) specifies the deadline and website workflow. Include the following in your immutable code repository:

- **Report, at most 10 pages including figures, tables and references** 
- **Reproduction instructions**

Your final website submission must link to this code and the matching complete checkpoint bundle. The website generates the Issue JSON automatically. Keep all inference assets downloadable for verification.

To check a peer, obtain their exact code version and checkpoint, follow their installation instructions, and run their frozen model with the supplied evaluator:

```bash
python evaluate.py --checkpoint /path/to/peer-checkpoint.pt --device cpu --precision fp32 --split test --output peer-test.json
```

Compare reproduced BPB with the reported score. Submit **Peer Review Report** with the reproduced score; optionally include the command, environment, difference and evidence/log link.  The instructor adjudicates discrepancies. Confirmed discrepancies during the seven-day review earn bonus credit under the announced marking policy.

## 7. Experimental evidence and AI assistance

The final report should distinguish matched-target comparisons from changes
in training budget. The most relevant existing validation controls are:

| Comparison | Validation BPB |
|---|---:|
| Classroom baseline, width 128, 1,200 steps | 2.07108 |
| Width 128 vs width 256 GELU, both 4,800 steps | 1.75243 vs 1.62358 |
| No dropout vs dropout 0.1, both width 256 and 6,000 steps | 1.62411 vs 1.59079 |
| Learned absolute positions vs RoPE, both with dropout and 6,000 steps | 1.59079 vs 1.58141 |
| Random vs coverage sampling, both RoPE/dropout and 6,000 steps | 1.58141 vs 1.57062 |
| Cosine vs WSD schedule, both RoPE/dropout/coverage and 12,000 steps | 1.53668 vs 1.51946 |
| Final raw 16k RoPE alone vs GELU-29 alone vs equal fusion, all at T=1.105 | 1.50005 vs 1.60930 vs **1.47970** |
| Final equal fusion at T=1.0 vs T=1.105 | 1.48649 vs **1.47970** |

The two-branch comparison increases total processed training targets and
parameters, so its BPB difference is not a same-budget estimate of the
ensemble effect. Each matched pair above was evaluated on validation. The
full-test BPB of the submitted simple method is **1.49895557**. The simpler
method was chosen after the predefined round-10 diagnostic test ablations,
for interpretability; it must not be described as chosen before those tests.
The highest-scoring research variant at 1.49135495 is not submitted.

Methods and limitations are documented in
[`RESEARCH_ROUND_1.md`](RESEARCH_ROUND_1.md),
[`RESEARCH_ROUND_6.md`](RESEARCH_ROUND_6.md),
[`RESEARCH_ROUND_7.md`](RESEARCH_ROUND_7.md),
[`RESEARCH_ROUND_9.md`](RESEARCH_ROUND_9.md),
[`RESEARCH_ROUND_10.md`](RESEARCH_ROUND_10.md), and
[`RESEARCH_ROUND_11.md`](RESEARCH_ROUND_11.md). The complete search record,
including rejected directions and costs, is in [`RUN_LOG.csv`](RUN_LOG.csv),
[`experiment_results/`](experiment_results/), and local `runs/` artifacts.
The `runs/` directory, `*.pt` files under `code/`, and the local
`final_artifact/` directory are Git-ignored; the frozen submission
checkpoint must therefore be distributed separately.

Substantive AI assistance: OpenAI Codex helped analyze the starter model and
propose the optimization directions used in the final method: RoPE positional
encoding, training-only dropout, coverage sampling, the WSD learning-rate
schedule, and RoPE/GELU logit fusion. Codex contributed model,
ensemble-packaging, and evaluation-helper scripts, and helped summarize
ablation results and draft their descriptions. The student chose the final
equal-weight fusion without checkpoint averaging and is responsible for
reviewing, training, testing, analyzing, understanding, and explaining the submitted method. 
The baseline code was supplied by the course and reused as the starting point.

## 8. Data attribution

WikiText-2 was introduced by Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher in [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843). The text is by Wikipedia contributors. The [upstream dataset](https://huggingface.co/datasets/Salesforce/wikitext) identifies [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and the [GNU Free Documentation License](https://www.gnu.org/licenses/fdl-1.3.html); retain these notices when redistributing the data.

The supplied `wikitext-2-raw-v1` splits preserve revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Rows are joined with newlines and encoded as UTF-8; the tokenizer is fitted only to training text. Dataset hashes are in `data/manifest.json`. These dataset notices do not assign a new license to the surrounding classroom code.
