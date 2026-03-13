"""
Generate Synthetic METU Risk Persona Dataset
=============================================

This script generates a synthetic dataset that mimics the structure of real
participant data collected in the METU Risk Persona study. The output is
intended exclusively for demonstration, testing, and development purposes.

**This data does not represent real participants.** All values are drawn from
parameterised random distributions chosen to approximate realistic ranges
observed in behavioural research, but no actual participant data was used
to fit these distributions.

Usage:
    python scripts/generate_synthetic.py
    python scripts/generate_synthetic.py --n 120
    python scripts/generate_synthetic.py --n 60 --seed 99
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────

FACULTIES = [
    "Engineering",
    "Arts and Sciences",
    "Economic and Administrative Sciences",
    "Education",
    "Architecture",
]

DEGREES = ["Undergraduate", "Graduate"]

DOSPERT_SUBSCALES = [
    "dospert_financial",
    "dospert_health_safety",
    "dospert_recreational",
    "dospert_ethical",
    "dospert_social",
]


# ── Helpers ───────────────────────────────────────────────────────────────

def _clip(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(arr, lo, hi)


def generate_participants(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate *n* synthetic participant records."""

    records: dict[str, list] = {
        "participant_id": [f"synth_{i:04d}" for i in range(1, n + 1)],
        "age": rng.integers(20, 29, size=n).tolist(),
        "gender": rng.choice(["Male", "Female"], size=n).tolist(),
        "faculty": rng.choice(FACULTIES, size=n).tolist(),
        "degree": rng.choice(DEGREES, size=n, p=[0.75, 0.25]).tolist(),
        "dorm": rng.choice([True, False], size=n, p=[0.6, 0.4]).tolist(),
        "grad_months": rng.integers(0, 25, size=n).tolist(),
        "employed": rng.choice([True, False], size=n, p=[0.3, 0.7]).tolist(),
        "prior_task": rng.choice([True, False], size=n, p=[0.15, 0.85]).tolist(),
    }

    # DOSPERT-30 subscales (1.0 -- 7.0 Likert means)
    for subscale in DOSPERT_SUBSCALES:
        raw = rng.normal(loc=3.8, scale=1.1, size=n)
        records[subscale] = _clip(raw, 1.0, 7.0).round(2).tolist()

    # ── BART metrics ──────────────────────────────────────────────────

    # rng_normalized_pumps ~ Beta(2, 5) -> range [0, 1], mean ~0.29
    records["bart_rng_normalized_pumps"] = (
        rng.beta(2, 5, size=n).round(4).tolist()
    )

    # impulsivity_index ~ Beta(2, 3) -> range [0, 1], mean ~0.40
    records["bart_impulsivity_index"] = (
        rng.beta(2, 3, size=n).round(4).tolist()
    )

    # patience_index_normalized ~ Beta(1.5, 8) -> range [0, 1], mean ~0.16
    records["bart_patience_normalized"] = (
        rng.beta(1.5, 8, size=n).round(4).tolist()
    )

    # mean_latency_between_pumps ~ Normal(450, 150), clipped [150, 1200] ms
    latency = rng.normal(450, 150, size=n)
    records["bart_mean_latency"] = (
        _clip(latency, 150, 1200).round(1).tolist()
    )

    # between_balloon_consistency ~ Beta(3, 3) -> range [0, 1], mean ~0.50
    records["bart_between_consistency"] = (
        rng.beta(3, 3, size=n).round(4).tolist()
    )

    # adaptive_strategy_score ~ Beta(2, 4) -> range [0, 1], mean ~0.33
    records["bart_adaptive_strategy"] = (
        rng.beta(2, 4, size=n).round(4).tolist()
    )

    # risk_sensitivity ~ Normal(0.5, 0.2), clipped [0, 1]
    risk_sens = rng.normal(0.5, 0.2, size=n)
    records["bart_risk_sensitivity"] = (
        _clip(risk_sens, 0.0, 1.0).round(4).tolist()
    )

    return pd.DataFrame(records)


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic METU Risk Persona dataset."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=60,
        help="Number of synthetic participants to generate (default: 60).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    df = generate_participants(args.n, rng)

    output_dir = Path(__file__).resolve().parent.parent / "data" / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"synthetic_metu_{args.n}.csv"

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} synthetic participants to {output_path}")


if __name__ == "__main__":
    main()
