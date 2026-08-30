#!/usr/bin/env python3
"""Paired attribution test: FCE versus reward-multiset-matched permutation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compute_comparison(
    fce: dict,
    control: dict,
    *,
    bootstrap_draws: int = 50_000,
) -> dict:
    import numpy as np

    fce_records = fce["records"]
    control_records = control["records"]
    fce_hashes = [record["prompt_hash"] for record in fce_records]
    control_hashes = [record["prompt_hash"] for record in control_records]
    if fce_hashes != control_hashes:
        raise ValueError("FCE and control evaluations are not paired")
    if (
        fce.get("offset") != 500
        or control.get("offset") != 500
        or fce.get("n") != 500
        or control.get("n") != 500
        or fce.get("decoding") != "greedy"
        or control.get("decoding") != "greedy"
    ):
        raise ValueError("matched control must use the frozen held-out protocol")

    fce_correct = np.asarray(
        [record["correct"] for record in fce_records],
        dtype=np.float64,
    )
    control_correct = np.asarray(
        [record["correct"] for record in control_records],
        dtype=np.float64,
    )
    delta = fce_correct - control_correct
    rng = np.random.default_rng(20260730)
    parts = []
    remaining = bootstrap_draws
    while remaining:
        draws = min(2_000, remaining)
        indices = rng.integers(0, len(delta), size=(draws, len(delta)))
        parts.append(delta[indices].mean(axis=1))
        remaining -= draws
    bootstrap = np.concatenate(parts)

    gain = float(delta.mean())
    p_nonpositive = float((bootstrap <= 0).mean())
    return {
        "method": "FCE versus trajectory-permuted FCE control",
        "n": len(delta),
        "test_offset": fce["offset"],
        "decoding": fce["decoding"],
        "fce_accuracy": float(fce_correct.mean()),
        "permuted_control_accuracy": float(control_correct.mean()),
        "fce_minus_control": gain,
        "gain_ci95": [
            float(value)
            for value in np.quantile(bootstrap, [0.025, 0.975])
        ],
        "p_gain_le_zero": p_nonpositive,
        "fce_only_correct": int(
            ((fce_correct == 1) & (control_correct == 0)).sum()
        ),
        "control_only_correct": int(
            ((fce_correct == 0) & (control_correct == 1)).sum()
        ),
        "attribution_gate": {
            "fce_better_than_control": bool(gain > 0),
            "paired_bootstrap_p_lt_0_05": bool(p_nonpositive < 0.05),
            "passed": bool(gain > 0 and p_nonpositive < 0.05),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fce", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=50_000)
    args = parser.parse_args()

    result = compute_comparison(
        json.loads(args.fce.read_text()),
        json.loads(args.control.read_text()),
        bootstrap_draws=args.bootstrap_draws,
    )
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
