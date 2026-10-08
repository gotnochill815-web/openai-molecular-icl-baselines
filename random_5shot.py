import os
import re
import random
import numpy as np
import pandas as pd

from openai import OpenAI
from rdkit import Chem
from rdkit.Chem import Crippen


# ============================================================
# PATHS
# ============================================================

BASE = "/home/intern1/CFG_test"

TRAIN_PATH = (
    f"{BASE}/data/guacamol_500k/"
    "train_500k.csv"
)

TEST_PATH = (
    f"{BASE}/data/guacamol_500k/"
    "test_500k.csv"
)

OUTPUT_DIR = (
    f"{BASE}/outputs/baselines/openai/"
    "gpt-5-mini/logp"
)


# ============================================================
# EXPERIMENT
# ============================================================

N_ICL = 5
N_TARGETS = 4
N_PER_TARGET = 5
SEED = 42
MODEL = "gpt-5-mini"

TARGETS = [
    -1.8906,
    2.5206,
    5.1774,
    7.5889,
]

# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# RDKit HELPERS
# ============================================================

def canonicalize(smiles):

    if not isinstance(smiles, str):
        return None

    smiles = smiles.strip()

    if not smiles:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return None

    if mol is None:
        return None

    try:
        return Chem.MolToSmiles(
            mol,
            canonical=True,
        )
    except Exception:
        return None


def calc_logp(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return float(Crippen.MolLogP(mol))


# ============================================================
# SMILES EXTRACTION
# ============================================================

def extract_smiles(text):

    if not isinstance(text, str):
        return None

    text = text.strip()

    fenced = re.findall(
        r"```(?:smiles)?\s*([^`]+?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    candidates = fenced + [text]

    for candidate in candidates:

        lines = candidate.strip().splitlines()

        for line in lines:

            line = line.strip()

            if not line:
                continue

            line = re.sub(
                r"^(SMILES\s*[:=]\s*)",
                "",
                line,
                flags=re.IGNORECASE,
            ).strip()

            if len(line) < 4:
                continue

            canonical = canonicalize(line)

            if canonical is not None:
                return canonical

    return None


# ============================================================
# PROMPT
# ============================================================

def build_prompt(target_logp, examples):

    example_text = []

    for i, row in enumerate(
        examples.to_dict("records"),
        start=1,
    ):

        example_text.append(
            f"Example {i}\n"
            f"Target LogP: {row['logp']:.4f}\n"
            f"SMILES: {row['smiles']}\n"
        )

    examples_block = "\n".join(example_text)

    return f"""
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

{examples_block}

Now generate one molecule for:

Target LogP: {target_logp:.4f}

Return ONLY the SMILES.
""".strip()


# ============================================================
# MAIN
# ============================================================

def main():

    if not os.getenv("OPENAI_API_KEY"):

        raise RuntimeError(
            "OPENAI_API_KEY is not set."
        )

    seed_everything(SEED)

    client = OpenAI()

    # ========================================================
    # LOAD DATA
    # ========================================================

    test_df = pd.read_csv(TEST_PATH)
    train_df = pd.read_csv(TRAIN_PATH)

    train_examples = train_df[["smiles", "logp"]].dropna().reset_index(drop=True)

    print("=" * 80)
    print("OPENAI GPT-5 MINI LOGP ICL BASELINE")
    print("=" * 80)

    print("Model       :", MODEL)
    print("Train rows  :", len(train_df))
    print("Test rows   :", len(test_df))
    print("Samples     :", N_TARGETS * N_PER_TARGET)
    print("ICL examples:", N_ICL)

    # ========================================================
    # FIXED LOGP TARGETS
    # SAME TARGETS AS THE EXISTING 500K CFG SWEEP
    # ========================================================


    print("\nFixed LogP targets:")
    for i, value in enumerate(TARGETS):
        print(
            f"{i + 1:02d}/{len(TARGETS)}: {float(value):.4f}"
        )

    print("\nBuilding canonical training set...")

    train_canonical = set()

    for smi in train_df["smiles"].dropna():

        canonical = canonicalize(smi)

        if canonical is not None:
            train_canonical.add(canonical)

    print(
        "Canonical train molecules:",
        len(train_canonical),
    )

    # ========================================================
    # GENERATION
    # ========================================================

    results = []

    sample_counter = 0

    for target_idx, target_logp in enumerate(TARGETS):

        target_logp = float(target_logp)

        for repeat_idx in range(N_PER_TARGET):

            generation_seed = (
                SEED
                + target_idx * N_PER_TARGET
                + repeat_idx
            )

            seed_everything(
                generation_seed
            )

            # Random 5-shot ICL examples
            examples = (
                train_examples
                .sample(
                    n=N_ICL,
                    random_state=generation_seed,
                )
                .reset_index(drop=True)
            )

            # Store canonical SMILES from this prompt
            prompt_example_smiles = set()

            for example_smiles in examples["smiles"]:
                example_canonical = canonicalize(
                    example_smiles
                )

                if example_canonical is not None:
                    prompt_example_smiles.add(
                        example_canonical
                    )

            prompt = build_prompt(
                target_logp,
                examples,
            )

            sample_counter += 1

            print(
                f"{sample_counter:03d}/20 "
                f"| target={target_logp:.3f} "
                f"| repeat={repeat_idx + 1:02d}/{N_PER_TARGET}",
                flush=True,
            )

            api_error = False
            raw_text = ""
            canonical = None
            copied_from_prompt = False
            input_tokens = None
            output_tokens = None
            total_tokens = None

            try:

                response = client.responses.create(
                    model=MODEL,
                    input=prompt,
                )

                usage = getattr(response, "usage", None)

                if usage is not None:
                    input_tokens = getattr(
                        usage, "input_tokens", None
                    )
                    output_tokens = getattr(
                        usage, "output_tokens", None
                    )
                    total_tokens = getattr(
                        usage, "total_tokens", None
                    )

                    print(
                        f"  TOKENS: "
                        f"input={input_tokens} "
                        f"output={output_tokens} "
                        f"total={total_tokens}",
                        flush=True,
                    )

                raw_text = (
                    response.output_text.strip()
                )

                canonical = extract_smiles(
                    raw_text
                )

                copied_from_prompt = (
                    canonical is not None
                    and canonical in prompt_example_smiles
                )

            except Exception as exc:

                api_error = True

                print(
                    f"  ERROR: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

            # ====================================================
            # EVALUATION
            # ====================================================

            valid = (
                canonical is not None
            )

            actual_logp = (
                calc_logp(canonical)
                if valid
                else None
            )

            absolute_error = (
                abs(
                    actual_logp
                    - target_logp
                )
                if actual_logp is not None
                else None
            )

            novel = (
                valid
                and canonical not in train_canonical
            )

            results.append(
                {
                    "model": MODEL,
                    "sample": sample_counter,
                    "target_index": target_idx,
                    "repeat": repeat_idx + 1,
                    "seed": generation_seed,
                    "target_logp": target_logp,
                    "raw_output": raw_text,
                    "canonical_smiles": canonical,
                    "valid": valid,
                    "novel": novel,
                    "copied_from_prompt": copied_from_prompt,
                    "actual_logp": actual_logp,
                    "abs_logp_error": absolute_error,
                    "api_error": api_error,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                }
            )

            if api_error:

                print(
                    "  -> API ERROR"
                )

            elif valid:

                print(
                    f"  -> VALID "
                    f"| actual={actual_logp:.3f} "
                    f"| error={absolute_error:.3f} "
                    f"| copied={copied_from_prompt}"
                )

            else:

                print(
                    "  -> INVALID"
                )

    # ========================================================
    # SAVE RAW RESULTS
    # ========================================================

    results_df = pd.DataFrame(results)

    # Real per-target uniqueness and copy rate
    target_summary_rows = []

    for target_logp, group in results_df.groupby("target_logp"):
        valid_group = group[group["valid"] == True].copy()

        if len(valid_group) == 0:
            uniqueness = 0.0
            novelty = 0.0
            copy_rate = 0.0
            mad = float("nan")
        else:
            uniqueness = (
                valid_group["canonical_smiles"].nunique()
                / len(valid_group)
            )
            novelty = valid_group["novel"].mean()
            copy_rate = valid_group["copied_from_prompt"].mean()
            mad = valid_group["abs_logp_error"].mean()

        target_summary_rows.append({
            "target_logp": target_logp,
            "n": len(group),
            "validity": group["valid"].mean(),
            "uniqueness": uniqueness,
            "novelty": novelty,
            "copy_rate": copy_rate,
            "logp_mad": mad,
        })

    target_summary_df = pd.DataFrame(target_summary_rows)

    print("\nPER-TARGET SUMMARY")
    print(target_summary_df.to_string(index=False))

    valid_results = results_df[results_df["valid"] == True].copy()

    if len(valid_results) > 0:
        overall_uniqueness = (
            valid_results["canonical_smiles"].nunique()
            / len(valid_results)
        )
        overall_novelty = valid_results["novel"].mean()
        overall_copy_rate = valid_results["copied_from_prompt"].mean()
        overall_mad = valid_results["abs_logp_error"].mean()
    else:
        overall_uniqueness = 0.0
        overall_novelty = 0.0
        overall_copy_rate = 0.0
        overall_mad = float("nan")

    overall_summary = pd.DataFrame([{
        "n_generations": len(results_df),
        "validity": results_df["valid"].mean(),
        "uniqueness": overall_uniqueness,
        "novelty": overall_novelty,
        "copy_rate": overall_copy_rate,
        "logp_mad": overall_mad,
    }])

    print("\nOVERALL SUMMARY")
    print(overall_summary.to_string(index=False))

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    raw_path = (
        f"{OUTPUT_DIR}/"
        "openai_gpt5_mini_logp_icl_random_20_results.csv"
    )

    target_summary_df.to_csv(
        f"{OUTPUT_DIR}/openai_gpt5_mini_logp_icl_random_20_target_summary.csv",
        index=False,
    )

    overall_summary.to_csv(
        f"{OUTPUT_DIR}/openai_gpt5_mini_logp_icl_random_20_overall_summary.csv",
        index=False,
    )

    results_df.to_csv(
        raw_path,
        index=False,
    )

    print("\nRaw results saved to:")
    print(raw_path)

if __name__ == "__main__":
    main()
