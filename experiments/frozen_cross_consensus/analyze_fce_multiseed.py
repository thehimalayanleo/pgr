#!/usr/bin/env python3
"""Hierarchical seed-and-prompt analysis for matched FCE replications."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_pair(fce_path: Path, control_path: Path) -> tuple[np.ndarray, dict]:
    fce = json.loads(fce_path.read_text())
    control = json.loads(control_path.read_text())
    fce_records = fce["records"]
    control_records = control["records"]
    fce_hashes = [record["prompt_hash"] for record in fce_records]
    control_hashes = [record["prompt_hash"] for record in control_records]
    if fce_hashes != control_hashes:
        raise ValueError(f"unpaired prompts: {fce_path} vs {control_path}")
    if len(fce_records) != 500:
        raise ValueError("each replication must contain exactly 500 prompts")
    for result in (fce, control):
        if (
            result.get("dataset") != "openai/gsm8k"
            or result.get("split") != "test"
            or result.get("offset") != 500
            or result.get("decoding") != "greedy"
        ):
            raise ValueError("evaluation does not match the frozen heldout protocol")
    delta = np.asarray(
        [
            float(fce_item["correct"]) - float(control_item["correct"])
            for fce_item, control_item in zip(fce_records, control_records)
        ],
        dtype=np.float64,
    )
    return delta, {
        "fce_accuracy": float(
            np.mean([record["correct"] for record in fce_records])
        ),
        "permuted_accuracy": float(
            np.mean([record["correct"] for record in control_records])
        ),
        "difference": float(delta.mean()),
        "fce_checkpoint": fce["checkpoint"],
        "permuted_checkpoint": control["checkpoint"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--fce", type=Path, action="append", required=True)
    parser.add_argument("--control", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=50_000)
    args = parser.parse_args()
    if not (len(args.seed) == len(args.fce) == len(args.control)):
        parser.error("--seed, --fce, and --control counts must match")
    if len(set(args.seed)) != len(args.seed):
        parser.error("training seeds must be unique")

    deltas = []
    per_seed = {}
    for seed, fce_path, control_path in zip(args.seed, args.fce, args.control):
        delta, summary = load_pair(fce_path, control_path)
        deltas.append(delta)
        per_seed[str(seed)] = summary
    matrix = np.stack(deltas)

    rng = np.random.default_rng(20260730)
    bootstrap_parts = []
    remaining = args.bootstrap_draws
    n_seeds, n_prompts = matrix.shape
    while remaining:
        draws = min(2_000, remaining)
        seed_indices = rng.integers(
            0,
            n_seeds,
            size=(draws, n_seeds),
        )
        prompt_indices = rng.integers(
            0,
            n_prompts,
            size=(draws, n_seeds, n_prompts),
        )
        sampled = matrix[
            seed_indices[:, :, None],
            prompt_indices,
        ]
        bootstrap_parts.append(sampled.mean(axis=(1, 2)))
        remaining -= draws
    bootstrap = np.concatenate(bootstrap_parts)

    seed_differences = matrix.mean(axis=1)
    mean_difference = float(seed_differences.mean())
    positive = int((seed_differences > 0).sum())
    ci95 = [
        float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
    ]
    gate = {
        "at_least_three_of_four_seed_differences_positive": bool(
            len(seed_differences) == 4 and positive >= 3
        ),
        "mean_difference_at_least_0_05": bool(mean_difference >= 0.05),
        "hierarchical_ci95_lower_bound_gt_zero": bool(ci95[0] > 0),
    }
    result = {
        "analysis": "matched FCE versus permuted multi-seed replication",
        "estimand": "mean paired accuracy difference over training seeds",
        "training_seeds": args.seed,
        "n_seeds": n_seeds,
        "prompts_per_seed": n_prompts,
        "per_seed": per_seed,
        "positive_seed_differences": positive,
        "mean_fce_minus_permuted": mean_difference,
        "hierarchical_seed_prompt_bootstrap_ci95": ci95,
        "p_hierarchical_difference_le_zero": float(
            (bootstrap <= 0).mean()
        ),
        "success_gate": gate,
        "passed": all(gate.values()),
    }
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
