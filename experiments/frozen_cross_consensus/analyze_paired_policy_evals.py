#!/usr/bin/env python3
"""Generic paired accuracy comparison for two frozen-protocol policy evals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--left-name", required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--right-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=50_000)
    args = parser.parse_args()

    left = json.loads(args.left.read_text())
    right = json.loads(args.right.read_text())
    left_records = left["records"]
    right_records = right["records"]
    if [row["prompt_hash"] for row in left_records] != [
        row["prompt_hash"] for row in right_records
    ]:
        raise ValueError("evaluations are not paired on identical prompts")
    for result in (left, right):
        if (
            result.get("offset") != 500
            or result.get("n") != 500
            or result.get("decoding") != "greedy"
        ):
            raise ValueError("evaluation does not match the frozen protocol")

    left_correct = np.asarray(
        [row["correct"] for row in left_records],
        dtype=np.float64,
    )
    right_correct = np.asarray(
        [row["correct"] for row in right_records],
        dtype=np.float64,
    )
    delta = left_correct - right_correct
    rng = np.random.default_rng(20260730)
    parts = []
    remaining = args.bootstrap_draws
    while remaining:
        draws = min(2_000, remaining)
        indices = rng.integers(0, len(delta), size=(draws, len(delta)))
        parts.append(delta[indices].mean(axis=1))
        remaining -= draws
    bootstrap = np.concatenate(parts)
    result = {
        "analysis": "paired policy accuracy comparison",
        "left_name": args.left_name,
        "right_name": args.right_name,
        "n": len(delta),
        "left_accuracy": float(left_correct.mean()),
        "right_accuracy": float(right_correct.mean()),
        "left_minus_right": float(delta.mean()),
        "difference_ci95": [
            float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
        ],
        "p_left_minus_right_le_zero": float((bootstrap <= 0).mean()),
        "left_only_correct": int(
            ((left_correct == 1) & (right_correct == 0)).sum()
        ),
        "right_only_correct": int(
            ((left_correct == 0) & (right_correct == 1)).sum()
        ),
        "left_checkpoint": left["checkpoint"],
        "right_checkpoint": right["checkpoint"],
    }
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
