# DASE7506 Project 1: Small Language Model Challenge

This repository contains the course starter, the final model implementation, the
fixed data and evaluator, experimental records, and the report source. The
selected predictor is an equal-logit ensemble of a 16,000-step RoPE branch and
an independently trained GELU branch, with temperature 1.105 and no checkpoint
parameter averaging.

| Complete-test CPU FP32 result | Value |
|---|---:|
| BPB (lower is better) | **1.4989555702583726** |
| Classroom baseline BPB | 2.1012567154053494 |
| CPU scoring time / same-period baseline | 4.125× |
| Sampled peak working set | 1.858 GiB |
| Final checkpoint size | 29,769,336 bytes (28.39 MiB) |

The final checkpoint has SHA-256
`c162aa90c1e5cd445128a8b105d5346d8b3dc156917d1116b49920e7fe9a3074`.
It is distributed separately from the source repository because
`final_artifact/` is Git-ignored. The public checkpoint download and immutable
code links will be added after publication.

## Repository guide

- [GUIDE.md](GUIDE.md): assignment requirements and submission rules.
- [code/README.md](code/README.md): exact environment, training, packaging,
  evaluation, resource-measurement, and peer-reproduction instructions.
- [report/main.tex](report/main.tex): English report source. The compiled report
  will be added to `report/` before final submission; the course limit is 10
  pages.
- [code/RUN_LOG.csv](code/RUN_LOG.csv) and
  [code/experiment_results/](code/experiment_results/): experimental records
  and ablation evidence.

To reproduce the frozen score after downloading `checkpoint.pt`, work from
`code/`, install the dependencies as described in its README, verify the file
hash above, and run:

```bash
python evaluate.py --checkpoint /path/to/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test --output reproduced_test_cpu_fp32.json
```

The checkpoint contains both branches and needs no retraining or additional
inference asset. The supplied data and tokenizer are included under `code/data/`.

## Substantive AI assistance

Substantive AI assistance: OpenAI Codex helped analyze the starter model and
propose the optimization directions used in the final method: RoPE positional
encoding, training-only dropout, coverage sampling, the WSD learning-rate
schedule, and RoPE/GELU logit fusion. Codex contributed model,
ensemble-packaging, and evaluation-helper scripts, and helped summarize
ablation results and draft their descriptions. The student chose the final
equal-weight fusion without checkpoint averaging and is responsible for
reviewing, training, testing, analyzing, understanding, and explaining the submitted method. 
The baseline code was supplied by the course and reused as the starting point.

## Data attribution

The data attribution and redistribution notices are in
[code/README.md](code/README.md#8-data-attribution).
