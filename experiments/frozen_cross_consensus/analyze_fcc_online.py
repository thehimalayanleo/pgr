#!/usr/bin/env python3
"""Paired held-out comparison of base and frozen-evidence-trained policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=50_000)
    parser.add_argument("--min-gain", type=float, default=0.03)
    args = parser.parse_args()

    import numpy as np

    base = json.loads(args.base.read_text())
    trained = json.loads(args.trained.read_text())
    base_records = base["records"]
    fcc_records = trained["records"]
    base_hashes = [record["prompt_hash"] for record in base_records]
    fcc_hashes = [record["prompt_hash"] for record in fcc_records]
    if base_hashes != fcc_hashes:
        raise ValueError("base and FCC evaluations are not paired on identical prompts")

    base_correct = np.asarray(
        [record["correct"] for record in base_records],
        dtype=np.float64,
    )
    fcc_correct = np.asarray(
        [record["correct"] for record in fcc_records],
        dtype=np.float64,
    )
    delta = fcc_correct - base_correct
    rng = np.random.default_rng(20260729)
    boot_parts = []
    remaining = args.bootstrap_draws
    while remaining:
        draws = min(2_000, remaining)
        indices = rng.integers(0, len(delta), size=(draws, len(delta)))
        boot_parts.append(delta[indices].mean(axis=1))
        remaining -= draws
    boot = np.concatenate(boot_parts)

    gain = float(delta.mean())
    p_nonpositive = float((boot <= 0).mean())
    result = {
        "method": "Frozen Cross-Evidence GRPO",
        "n": len(delta),
        "test_offset": base["offset"],
        "decoding": base["decoding"],
        "base_accuracy": float(base_correct.mean()),
        "trained_accuracy": float(fcc_correct.mean()),
        "trained_minus_base": gain,
        "gain_ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "p_gain_le_zero": p_nonpositive,
        "trained_only_correct": int(((fcc_correct == 1) & (base_correct == 0)).sum()),
        "base_only_correct": int(((fcc_correct == 0) & (base_correct == 1)).sum()),
        "viability_gate": {
            "gain_at_least_minimum": bool(gain >= args.min_gain),
            "minimum_gain": args.min_gain,
            "paired_bootstrap_p_lt_0_05": bool(p_nonpositive < 0.05),
            "passed": bool(gain >= args.min_gain and p_nonpositive < 0.05),
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
