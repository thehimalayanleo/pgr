#!/usr/bin/env python3
"""Descriptive behavior audit for base, FCE, and matched-control evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def validate_and_index(evaluation: dict) -> dict[str, dict]:
    records = evaluation.get("records", [])
    if len(records) != 500:
        raise ValueError("behavior audit requires exactly 500 records per arm")
    indexed = {record["prompt_hash"]: record for record in records}
    if len(indexed) != 500:
        raise ValueError("evaluation contains duplicate prompt hashes")
    return indexed


def arm_summary(evaluation: dict) -> dict:
    records = evaluation["records"]
    correct = sum(bool(record["correct"]) for record in records)
    parseable = sum(record.get("prediction") is not None for record in records)
    nonempty = sum(bool(record.get("completion", "").strip()) for record in records)
    character_lengths = [len(record.get("completion", "")) for record in records]
    word_lengths = [
        len(record.get("completion", "").split()) for record in records
    ]
    return {
        "checkpoint": evaluation["checkpoint"],
        "n": len(records),
        "correct": correct,
        "accuracy": correct / len(records),
        "parseable": parseable,
        "parseable_rate": parseable / len(records),
        "nonempty": nonempty,
        "nonempty_rate": nonempty / len(records),
        "correct_given_parseable": (
            correct / parseable if parseable else None
        ),
        "mean_completion_characters": statistics.mean(character_lengths),
        "median_completion_characters": statistics.median(character_lengths),
        "mean_completion_words": statistics.mean(word_lengths),
        "median_completion_words": statistics.median(word_lengths),
    }


def paired_summary(
    left: dict[str, dict],
    right: dict[str, dict],
    *,
    left_name: str,
    right_name: str,
) -> dict:
    if list(left) != list(right):
        raise ValueError(f"{left_name} and {right_name} are not exactly paired")
    pairs = [(left[key], right[key]) for key in left]
    left_only = sum(bool(a["correct"]) and not bool(b["correct"]) for a, b in pairs)
    right_only = sum(bool(b["correct"]) and not bool(a["correct"]) for a, b in pairs)
    both_correct = sum(bool(a["correct"]) and bool(b["correct"]) for a, b in pairs)
    both_wrong = len(pairs) - left_only - right_only - both_correct
    left_parseable = [
        a.get("prediction") is not None for a, _ in pairs
    ]
    right_parseable = [
        b.get("prediction") is not None for _, b in pairs
    ]
    both_parseable_indices = [
        index
        for index, (a, b) in enumerate(zip(left_parseable, right_parseable))
        if a and b
    ]
    left_both_parseable_correct = sum(
        bool(pairs[index][0]["correct"]) for index in both_parseable_indices
    )
    right_both_parseable_correct = sum(
        bool(pairs[index][1]["correct"]) for index in both_parseable_indices
    )
    right_correct_when_left_unparseable = sum(
        (not left_parseable[index]) and bool(pairs[index][1]["correct"])
        for index in range(len(pairs))
    )
    return {
        "left": left_name,
        "right": right_name,
        "n": len(pairs),
        "both_correct": both_correct,
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "both_wrong": both_wrong,
        "left_minus_right_accuracy": (left_only - right_only) / len(pairs),
        "left_parseable_rate": sum(left_parseable) / len(pairs),
        "right_parseable_rate": sum(right_parseable) / len(pairs),
        "left_minus_right_parseable_rate": (
            sum(left_parseable) - sum(right_parseable)
        ) / len(pairs),
        "both_parseable": len(both_parseable_indices),
        "left_accuracy_when_both_parseable": (
            left_both_parseable_correct / len(both_parseable_indices)
            if both_parseable_indices
            else None
        ),
        "right_accuracy_when_both_parseable": (
            right_both_parseable_correct / len(both_parseable_indices)
            if both_parseable_indices
            else None
        ),
        "right_correct_when_left_unparseable": right_correct_when_left_unparseable,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--fce", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evaluations = {
        "base": read_json(args.base),
        "fce": read_json(args.fce),
        "control": read_json(args.control),
    }
    indexed = {
        name: validate_and_index(evaluation)
        for name, evaluation in evaluations.items()
    }
    prompt_orders = [
        list(indexed[name]) for name in ("base", "fce", "control")
    ]
    if not prompt_orders[0] == prompt_orders[1] == prompt_orders[2]:
        raise ValueError("behavior audit inputs are not exactly paired")

    report = {
        "method": "Frozen Cross-Evidence GRPO behavior audit",
        "audit_version": 1,
        "descriptive_only": True,
        "inputs": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in {
                "base": args.base,
                "fce": args.fce,
                "control": args.control,
            }.items()
        },
        "arms": {
            name: arm_summary(evaluation)
            for name, evaluation in evaluations.items()
        },
        "paired": {
            "fce_vs_base": paired_summary(
                indexed["fce"],
                indexed["base"],
                left_name="fce",
                right_name="base",
            ),
            "fce_vs_control": paired_summary(
                indexed["fce"],
                indexed["control"],
                left_name="fce",
                right_name="control",
            ),
            "control_vs_base": paired_summary(
                indexed["control"],
                indexed["base"],
                left_name="control",
                right_name="base",
            ),
        },
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
