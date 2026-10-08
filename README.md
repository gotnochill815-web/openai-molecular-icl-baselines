# OpenAI ICL Baselines for Molecular Generation

Out-of-the-box OpenAI baselines for conditional molecular generation using
in-context learning (ICL).

## Model

- Model: `gpt-5-mini`
- Task: target-conditioned molecular generation
- Property evaluated: LogP
- Benchmark size: 20 generations
- Target values: `-1.8906`, `2.5206`, `5.1774`, `7.5889`

## Methods

### Random 5-shot

Five training examples are randomly sampled for each generation and supplied
as SMILES + LogP pairs.

### Nearest 5-shot

Five training examples with LogP values closest to the requested target are
selected for each generation.

### Zero-shot

No training examples are provided. The model receives only the target LogP
condition.

## Results

| Method | Validity | Uniqueness | Novelty | Copy Rate | LogP MAD |
|---|---:|---:|---:|---:|---:|
| Random 5-shot | 1.000 | 0.900 | 0.750 | 0.000 | 0.855 |
| Nearest 5-shot | 0.950 | 0.737 | 0.474 | 0.474 | 0.533 |
| Zero-shot | 1.000 | 0.600 | 0.650 | N/A | 1.136 |

## Token Usage

| Method | Input Tokens | Output Tokens | Total Tokens |
|---|---:|---:|---:|
| Random 5-shot | 7,174 | 16,639 | 23,813 |
| Nearest 5-shot | 7,690 | 17,087 | 24,777 |
| Zero-shot | 2,020 | 16,407 | 18,427 |

## Interpretation

Nearest-neighbor ICL achieves the lowest LogP MAD, but this comes with a
substantial increase in copying from the prompt and lower novelty.

Random 5-shot ICL provides the best overall balance between validity,
uniqueness, novelty, and property control.

Zero-shot generation uses substantially fewer input tokens but shows weaker
LogP control.

These results are from a 20-generation smoke benchmark and are intended as an
initial baseline comparison rather than a final large-scale evaluation.

## Reproducibility

The three scripts use the same target set and evaluation setup:

- `random_5shot.py`
- `nearest_5shot.py`
- `zero_shot.py`

API credentials must be supplied through the environment and must never be
committed to the repository.

## Results Files

Per-generation outputs are provided under `results/`:

- `random_5shot_20_results.csv`
- `nearest_5shot_20_results.csv`
- `zero_shot_20_results.csv`
<img width="768" height="498" alt="image" src="https://github.com/user-attachments/assets/595f2ea7-38d4-477c-a23d-088f0b63aa8f" />
