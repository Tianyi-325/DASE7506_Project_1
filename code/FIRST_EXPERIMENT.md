# Experiment 1: parameter-matched SwiGLU feed-forward layers

## Hypothesis and change

The baseline's feed-forward sublayers use a GELU MLP. A gated SwiGLU MLP may
represent token-dependent feature interactions more effectively at similar
parameter and inference cost. `student.py` replaces only those sublayers; it
retains the baseline's attention, absolute position embeddings, width, depth,
vocabulary, and causal prediction interface. The baseline MLP hidden width is
512, while the student SwiGLU hidden width is 344, giving 1,088,256 versus
1,093,056 total parameters (+0.44%).

## Controlled comparison

Both runs used the supplied training text, seed 17, 1,200 updates, batch size
32, context 256, CUDA FP32 training, and 9,830,400 processed training targets.
The baseline is the mechanism-off control. Configuration and code hashes are
recorded in the run artifacts.

| Predictor | Validation BPB, CPU FP32 | GPU training seconds | Validation CPU scoring seconds |
|---|---:|---:|---:|
| Baseline GELU | 2.0710807966 | 29.063 | 7.931 |
| Student SwiGLU | 2.0172391377 | 31.693 | 8.233 |

SwiGLU reduced validation BPB by 0.0538416589 (about 2.6%). Of 1,472
validation windows, 97.9% had lower negative log likelihood than the baseline.
This is consistent with a broadly useful change, but it is one seed and does
not isolate the effects of gating from every difference in optimization.

The student's checkpoint is 4,390,069 bytes. During a repeated CPU FP32
validation run, Windows `PeakWorkingSet64`, sampled every 100 ms, reached
1,937,182,720 bytes (1.804 GiB). This is a process working-set measurement,
not a direct measure of the scorer's internal tensor allocation. The repeated
validation score was identical to the first CPU score.

No student test-set score was used in this development experiment. Freeze the
method before running `evaluate.py --split test` on the selected checkpoint.

## Reproduction

Run from `code/` with the versions in `requirements.txt`:

```text
python train.py --implementation model --device cuda --precision fp32 --threads 4 --seed 17 --steps 1200 --eval-every 300 --run-dir runs/baseline_cuda_fp32_s17
python train.py --implementation student --config configs/swiglu.json --device cuda --precision fp32 --threads 4 --seed 17 --steps 1200 --eval-every 300 --run-dir runs/swiglu_cuda_fp32_s17
python evaluate.py --checkpoint runs/baseline_cuda_fp32_s17/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation
python evaluate.py --checkpoint runs/swiglu_cuda_fp32_s17/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation
```

The run directories already exist locally; use new names when reproducing.
