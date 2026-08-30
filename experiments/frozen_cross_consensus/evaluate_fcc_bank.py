#!/usr/bin/env python3
"""Gold-separated evaluation of an FCC bank.

Gold is loaded only here, after the frozen targets have already been written. This script
reports target precision, coverage, candidate selection gain, and paired bootstrap power.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fcc_reward import extract_answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=50_000)
    args = parser.parse_args()

    import numpy as np
    from datasets import load_dataset

    bank = json.loads(args.bank.read_text())
    if bank.get("partial"):
        raise ValueError("refusing to evaluate a partial FCC bank")
    if bank.get("candidate_panel_included") is not True:
        raise ValueError("selection evaluation requires an independent candidate panel")
    items = bank["items"]
    panel_size = int(bank["panel_size"])
    candidate_size = int(bank["candidate_panel_size"])
    if len({item["prompt_hash"] for item in items}) != len(items):
        raise ValueError("FCC bank contains duplicate prompt hashes")
    for item in items:
        if len(item["panel_a"]) != panel_size or len(item["panel_b"]) != panel_size:
            raise ValueError("FCC target panel length does not match bank metadata")
        if len(item["candidates"]) != candidate_size:
            raise ValueError("FCC candidate panel length does not match bank metadata")
    dataset = list(
        load_dataset("openai/gsm8k", "main", split=bank["split"])
    )[: len(items)]
    gold = [
        example["answer"].split("####")[-1].strip().replace(",", "")
        for example in dataset
    ]

    random_pick = []
    oracle = []
    fcc_pick = []
    target_correct = []
    accepted_mask = []
    wrong_reinforcement = []
    reinforcement_event = []
    informative_group = []
    for item, answer in zip(items, gold):
        candidate_answers = [
            extract_answer(completion) for completion in item["candidates"]
        ]
        candidate_correct = np.asarray(
            [float(prediction == answer) for prediction in candidate_answers]
        )
        random_pick.append(float(candidate_correct.mean()))
        oracle.append(float(candidate_correct.max()))

        frozen = item["frozen_target"]
        accepted = bool(frozen["accepted"])
        target = frozen["target"]
        accepted_mask.append(accepted)
        target_correct.append(float(accepted and target == answer))
        matching = [
            idx for idx, prediction in enumerate(candidate_answers)
            if accepted and prediction == target
        ]
        if matching:
            fcc_pick.append(float(candidate_correct[matching[0]]))
        else:
            # Expected accuracy of a random fallback, avoiding arbitrary first-item noise.
            fcc_pick.append(float(candidate_correct.mean()))
        reinforcement_event.append(float(bool(matching)))
        informative_group.append(
            float(0 < len(matching) < len(candidate_answers))
        )
        wrong_reinforcement.append(
            float(accepted and target != answer and bool(matching))
        )

    random_arr = np.asarray(random_pick)
    oracle_arr = np.asarray(oracle)
    fcc_arr = np.asarray(fcc_pick)
    accepted_arr = np.asarray(accepted_mask, dtype=bool)
    target_correct_arr = np.asarray(target_correct)
    wrong_arr = np.asarray(wrong_reinforcement)
    reinforcement_arr = np.asarray(reinforcement_event, dtype=bool)
    informative_arr = np.asarray(informative_group, dtype=bool)

    rng = np.random.default_rng(20260729)
    n = len(items)
    indices = rng.integers(0, n, size=(args.bootstrap_draws, n))
    random_boot = random_arr[indices].mean(axis=1)
    oracle_boot = oracle_arr[indices].mean(axis=1)
    fcc_boot = fcc_arr[indices].mean(axis=1)
    gain_boot = fcc_boot - random_boot
    denom = oracle_boot - random_boot
    captured_boot = np.divide(
        gain_boot,
        denom,
        out=np.full_like(denom, np.nan),
        where=np.abs(denom) > 1e-12,
    )
    captured_boot = captured_boot[np.isfinite(captured_boot)]

    oracle_gain = oracle_arr.mean() - random_arr.mean()
    result = {
        "method": "Frozen Cross-Consensus GRPO",
        "model": bank["model"],
        "n": n,
        "panel_size": panel_size,
        "candidate_panel_size": candidate_size,
        "target_coverage": float(accepted_arr.mean()),
        "target_precision_when_accepted": (
            float(target_correct_arr[accepted_arr].mean())
            if accepted_arr.any()
            else None
        ),
        "random_candidate_accuracy": float(random_arr.mean()),
        "fcc_pick_accuracy": float(fcc_arr.mean()),
        "oracle_candidate_accuracy": float(oracle_arr.mean()),
        "fcc_minus_random": float(fcc_arr.mean() - random_arr.mean()),
        "gain_ci95": [float(x) for x in np.quantile(gain_boot, [0.025, 0.975])],
        "p_gain_le_zero": float((gain_boot <= 0).mean()),
        "oracle_gain_captured": float(
            (fcc_arr.mean() - random_arr.mean()) / max(oracle_gain, 1e-12)
        ),
        "captured_ci95": [
            float(x) for x in np.quantile(captured_boot, [0.025, 0.975])
        ],
        "wrong_target_reinforcement_rate": float(wrong_arr.mean()),
        "reinforcement_event_rate": float(reinforcement_arr.mean()),
        "informative_group_rate": float(informative_arr.mean()),
        "wrong_given_reinforcement": (
            float(wrong_arr[reinforcement_arr].mean())
            if reinforcement_arr.any()
            else None
        ),
        "viability_gate": {
            "captures_at_least_70pct_oracle_gain": bool(
                (fcc_arr.mean() - random_arr.mean()) / max(oracle_gain, 1e-12) >= 0.70
            ),
            "paired_bootstrap_p_lt_0_05": bool((gain_boot <= 0).mean() < 0.05),
            "informative_group_rate_at_least_0_30": bool(
                informative_arr.mean() >= 0.30
            ),
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
