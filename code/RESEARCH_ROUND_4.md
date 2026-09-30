# Research round 4: calibration and independent-seed diversity

Model training used only the supplied train split. The data, tokenizer,
evaluator, and independent causal windows were unchanged. All model,
mixture-weight, and temperature choices used validation only. The best
predictor was frozen before one complete CPU FP32 test evaluation. The full
candidate grids, training histories, CPU scorer outputs, per-window losses, and resource
measurement are preserved in `experiment_results/round4/`; checkpoint hashes
and run-level costs are in `RUN_LOG.csv`.

## Calibrating the previous GELU+SwiGLU ensemble

The previous two-model logit ensemble scored 1.55450645 validation BPB at
weight 0.5 and temperature 1.0. Keeping those trained weights fixed, we
swept temperature 0.7 through 1.3. Temperature 1.1 improved validation to
1.54979404 without adding parameters or a model forward pass. At that
temperature, GELU weight 0.55 improved it to 1.54942902. A finer
temperature sweep gave **1.54896729** at weight 0.55 and temperature 1.075.
This candidate was bundled as
`runs/ensemble_calibrated_w055_t1075_s17/checkpoint.pt` but was not tested,
because the independent-seed ensemble below validated better. The 7-way
temperature, 7-way weight, and 5-way fine-temperature sweeps are available
in the round-4 results directory.

## Independent-seed experiments

We trained width-256 GELU and SwiGLU GPTs at seed 29, each for 4,800 updates,
batch 32, context 256, and 39,321,600 processed targets. They used the same
recipes as the corresponding seed-17 models. Their complete loss and
validation histories are in the metrics JSON files.

| Model | Seed | Parameters | Validation BPB | GPU train seconds |
|---|---:|---:|---:|---:|
| GELU | 17 | 3,749,376 | 1.62358452 | 480.93 |
| GELU | 29 | 3,749,376 | **1.61914559** | 496.58 |
| SwiGLU | 17 | 3,767,168 | 1.64799668 | 456.55 |
| SwiGLU | 29 | 3,767,168 | 1.64839401 | 464.58 |

The extra training in this round cost 961.15 GPU seconds and 78,643,200
processed training targets. Model-search scoring seconds are recorded for
every candidate in the sweep JSON files. The final dual-GELU predictor
includes the ancestry of both GELU training runs: 78,643,200 targets and
977.51 GPU training seconds.

At logit temperature 1.075, the best validation results for each two-model
pair were:

| Pair | Best sampled weight | Validation BPB |
|---|---:|---:|
| GELU-17 + SwiGLU-17 | 0.55 on GELU | 1.54896729 |
| GELU-29 + SwiGLU-17 | 0.50 on GELU | 1.54272792 |
| GELU-17 + SwiGLU-29 | 0.50 on GELU | 1.54379430 |
| GELU-29 + SwiGLU-29 | 0.50 on GELU | 1.54562905 |
| **GELU-17 + GELU-29** | **0.50 on first** | **1.53888597** |

The mixed-architecture pairs were swept at GELU weights 0.4, 0.5, and 0.6.
The two-GELU pair was swept at weights 0.25, 0.5, and 0.75, then at weights
0.45, 0.5, and 0.55 and temperatures 1.05, 1.075, and 1.1. The best setting
was 0.5 and 1.075. The two independently initialized models likely reduce
one another's prediction errors; the validation comparisons establish the
benefit, while the exact causal explanation remains a hypothesis.

## Selected predictor and resource check

The selected dual-GELU checkpoint is
`runs/dual_gelu_s17_s29_w05_t1075/checkpoint.pt`, SHA-256
`1bf6995bda3d2006485c87557d295086057a4e5d6af7a8dc53feb7b1ef94f070`.
It has 7,498,752 unique trainable parameters and no separate inference
asset. CPU FP32 validation reproduced **1.53888592 BPB**. One concurrent
CPU measurement took 41.11 seconds while another model was training; after
training stopped, idle CPU validation took **33.17 seconds**. The idle
measurement was used for resource assessment.

After freezing this predictor, complete CPU FP32 test scored
**1.55541301 BPB**, versus 1.57292078 for the previous selected ensemble. The full-test
scorer took **36.49 seconds**, compared with **9.02 seconds** for the original
baseline on the same machine: **4.05 times**, under the 5-times limit.
Sampled Windows peak process working set was **1.853 GiB** (limit 4 GiB).
The checkpoint is **30,031,481 bytes (28.64 MiB)** (limit 64 MiB).
The unique FP32 parameters occupy **29,995,008 uncompressed bytes
(28.61 MiB)**, also below the asset limit.
The scorer JSON and resource details are in `experiment_results/round4/`.
No website submission was made.

## Reproduction

From `code/`, use new run directory names if the listed directories already
exist. The seed-17 GELU run is documented in round 2.

```text
python train.py --implementation model --config configs/baseline_width256.json --device cuda --precision fp32 --threads 4 --seed 29 --steps 4800 --eval-every 1200 --run-dir runs/gelu_w256_4800_s29
python train.py --implementation student --config configs/swiglu_width256.json --device cuda --precision fp32 --threads 4 --seed 29 --steps 4800 --eval-every 1200 --run-dir runs/swiglu_w256_4800_s29
python dual_gelu_sweep.py --first runs/gelu_w256_4800_s17/checkpoint.pt --second runs/gelu_w256_4800_s29/checkpoint.pt --weights 0.45 0.5 0.55 --temperatures 1.05 1.075 1.1 --output runs/dual_gelu_s17_s29_fine_sweep.json
python make_dual_gelu_checkpoint.py --first runs/gelu_w256_4800_s17/checkpoint.pt --second runs/gelu_w256_4800_s29/checkpoint.pt --first-weight 0.5 --temperature 1.075 --output runs/dual_gelu_s17_s29_w05_t1075/checkpoint.pt
python evaluate.py --checkpoint runs/dual_gelu_s17_s29_w05_t1075/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation
python evaluate.py --checkpoint runs/dual_gelu_s17_s29_w05_t1075/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test
```
