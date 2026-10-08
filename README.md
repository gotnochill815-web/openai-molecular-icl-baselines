# OpenAI ICL Baselines for Molecular Generation

Out-of-the-box OpenAI baselines for target-conditioned molecular generation using in-context learning (ICL).

This repository evaluates three prompting strategies with `gpt-5-mini` for molecular generation under a target LogP condition:

- Random 5-shot ICL
- Nearest 5-shot ICL
- Zero-shot generation

---

## Model

- **Model:** `gpt-5-mini`
- **Task:** Target-conditioned molecular generation
- **Property:** LogP
- **Benchmark:** 20 generations
- **Shots:** 5 for the two ICL settings
- **Targets:** `-1.8906`, `2.5206`, `5.1774`, `7.5889`
- **Random seed:** `42`

The benchmark uses the same fixed target values across all three methods.

---

## Methods

### 1. Random 5-shot ICL

For every generation, five molecules are randomly sampled from the GuacaMol 500K training split.

Each example is provided to the model as:

```text
Example 1
Target LogP: <logp>
SMILES: <smiles>

...

Example 5
Target LogP: <logp>
SMILES: <smiles>
```

The model is then asked to generate one molecule for the requested target:

```text
Now generate one molecule for:

Target LogP: <target>

Return ONLY the SMILES.
```

The five examples are sampled independently for each generation.

### 2. Nearest 5-shot ICL

For every target, the five training molecules whose LogP values are closest to the requested target are selected.

For each training molecule:

```text
distance = |training_logp - target_logp|
```

The five examples with the smallest distance are inserted into the same 5-shot prompt format used by the random baseline.

### 3. Zero-shot

No training examples are provided. The model receives only the molecular-generation instructions and the target LogP:

```text
Target LogP: <target>

Return ONLY the SMILES.
```

---

## Prompt Structure

The ICL prompt follows this structure:

```text
You are a molecular generation model.

Generate exactly ONE valid molecular SMILES string whose
calculated octanol/water partition coefficient (LogP) is as
close as possible to the requested target.

Rules:
- Return exactly one SMILES.
- Do not return explanations.
- Do not return markdown.
- Do not return multiple molecules.
- The output must be a valid SMILES string.

Here are examples from the training dataset:

Example 1
Target LogP: <example_logp_1>
SMILES: <example_smiles_1>

Example 2
Target LogP: <example_logp_2>
SMILES: <example_smiles_2>

Example 3
Target LogP: <example_logp_3>
SMILES: <example_smiles_3>

Example 4
Target LogP: <example_logp_4>
SMILES: <example_smiles_4>

Example 5
Target LogP: <example_logp_5>
SMILES: <example_smiles_5>

Now generate one molecule for:

Target LogP: <target_logp>

Return ONLY the SMILES.
```

For the nearest 5-shot baseline, the five examples are selected by minimum absolute LogP distance. For the random 5-shot baseline, they are randomly sampled from the training set.

---

## Dataset

Expected files:

```text
data/
└── guacamol_500k/
    ├── train_500k.csv
    └── test_500k.csv
```

The training CSV is used as the source of ICL demonstrations.

Relevant columns:

```text
smiles
qed
logp
tpsa
sas
```

The scripts require the training split to contain at least:

```text
smiles
logp
```

---

## Results

### Overall benchmark

| Method | Validity | Uniqueness | Novelty | Copy Rate | LogP MAD |
|---|---:|---:|---:|---:|---:|
| Random 5-shot | 1.000 | 0.900 | 0.750 | 0.000 | 0.855 |
| Nearest 5-shot | 0.950 | 0.737 | 0.474 | 0.474 | 0.533 |
| Zero-shot | 1.000 | 0.600 | 0.650 | N/A | 1.136 |

### Interpretation

Nearest-neighbor ICL achieves the lowest LogP MAD in this smoke benchmark. However, this comes with substantially higher copying from the prompt and lower novelty.

Random 5-shot ICL provides the strongest overall balance across validity, uniqueness, novelty, copy rate, and LogP error.

Zero-shot generation avoids prompt examples entirely and uses fewer input tokens, but shows weaker LogP control.

Because this evaluation contains only 20 generations, these results should be treated as an initial baseline comparison rather than a final large-scale evaluation.

---

## Token Usage

Measured token usage for the 20-generation benchmark:

| Method | Input Tokens | Output Tokens | Total Tokens |
|---|---:|---:|---:|
| Random 5-shot | 7,174 | 16,639 | 23,813 |
| Nearest 5-shot | 7,690 | 17,087 | 24,777 |
| Zero-shot | 2,020 | 16,407 | 18,427 |

5-shot prompting increases input-token usage because each request contains five training demonstrations.

---

## Evaluation

Each generated response is evaluated using RDKit.

### Validity

A generation is considered valid if the returned string can be parsed as a valid RDKit molecular SMILES.

### Canonicalization

Valid molecules are converted to canonical SMILES before computing uniqueness, novelty, and prompt-copy statistics.

### LogP error

For each valid molecule:

```text
absolute error = |generated_logp - target_logp|
```

The reported metric is:

```text
LogP MAD = mean(|generated_logp - target_logp|)
```

### Novelty

A valid generated molecule is novel if its canonical SMILES does not occur in the canonicalized training set.

### Copy rate

For the two ICL methods, a generated molecule is marked as copied when its canonical SMILES exactly matches one of the canonicalized molecules included in that generation's prompt.

For zero-shot generation, copy rate is not applicable because there are no prompt examples.

### Uniqueness

Uniqueness is computed among valid generated molecules using canonical SMILES.

---

## Repository Structure

```text
.
├── README.md
├── random_5shot.py
├── nearest_5shot.py
├── zero_shot.py
├── data/
│   └── guacamol_500k/
│       ├── train_500k.csv
│       └── test_500k.csv
└── results/
    ├── random_5shot_20_results.csv
    ├── nearest_5shot_20_results.csv
    └── zero_shot_20_results.csv
```

---

## Reproducibility

The scripts use the same fixed target set and evaluation protocol.

### Dataset path

By default, the scripts look for:

```text
data/guacamol_500k/
```

A different dataset location can be supplied through:

```bash
export GUACAMOL_500K_DIR=/path/to/guacamol_500k
```

For example, on the original cluster:

```bash
export GUACAMOL_500K_DIR=/home/intern1/CFG_test/data/guacamol_500k
```

### OpenAI API key

Set the API key through the environment:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
```

The API key must never be committed to GitHub.

### Run the baselines

```bash
python random_5shot.py
python nearest_5shot.py
python zero_shot.py
```

The scripts use:

```text
N_TARGETS = 4
N_PER_TARGET = 5
```

for a total of:

```text
4 × 5 = 20 generations
```

---

## Results Files

Per-generation results are provided under `results/`:

```text
results/random_5shot_20_results.csv
results/nearest_5shot_20_results.csv
results/zero_shot_20_results.csv
```

Each result file records the generated SMILES together with:

- target LogP
- generated canonical SMILES
- validity
- novelty
- whether the molecule was copied from the prompt
- actual calculated LogP
- absolute LogP error
- API error status

---

## Files

### `random_5shot.py`

Randomly samples five training examples for every generation and uses them as ICL demonstrations.

### `nearest_5shot.py`

Selects the five training examples with LogP values closest to the requested target and uses them as ICL demonstrations.

### `zero_shot.py`

Generates molecules without any training examples in the prompt.

---

## Benchmark Scope

This repository is intended as an initial out-of-the-box LLM baseline for conditional molecular generation.

The current experiment is a small smoke benchmark designed to compare:

```text
Zero-shot
    vs.
Random 5-shot ICL
    vs.
Nearest 5-shot ICL
```

A larger evaluation should be performed before drawing broad conclusions about the relative performance of these prompting strategies.

---

## Citation / Attribution

This benchmark uses:

- OpenAI `gpt-5-mini` for molecular generation
- RDKit for molecular parsing, canonicalization, and LogP calculation
- The GuacaMol-derived molecular dataset used by the experiment

Please follow the licensing and attribution requirements of the underlying dataset and software dependencies when redistributing or extending this benchmark.
