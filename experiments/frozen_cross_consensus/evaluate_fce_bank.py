#!/usr/bin/env python3
"""Gold-separated selection evaluation for Frozen Cross-Evidence rewards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fcc_reward import extract_answer
from fce_tasks import gold_for_example, load_task_dataset, task_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=50_000)
    args = parser.parse_args()

    import numpy as np
    bank = json.loads(args.bank.read_text())
    if bank.get("partial") or bank.get("frozen_evidence_attached") is not True:
        raise ValueError("refusing to evaluate an incomplete FCE bank")
    if bank.get("candidate_panel_included") is not True:
        raise ValueError("FCE selection requires an independent candidate panel")
    items = bank["items"]
    panel_size = int(bank["panel_size"])
    candidate_size = int(bank["candidate_panel_size"])
    if len({item["prompt_hash"] for item in items}) != len(items):
        raise ValueError("FCE bank contains duplicate prompt hashes")
    for item in items:
        if len(item["panel_a"]) != panel_size or len(item["panel_b"]) != panel_size:
            raise ValueError("FCE target panel length does not match metadata")
        if len(item["candidates"]) != candidate_size:
            raise ValueError("FCE candidate panel length does not match metadata")
        if "frozen_evidence" not in item:
            raise ValueError("FCE item lacks frozen evidence")

    spec = task_spec(bank["dataset"])
    answer_mode = bank.get("answer_mode", "numeric")
    if answer_mode != spec.answer_mode:
        raise ValueError("bank answer_mode disagrees with task adapter")
    dataset = list(load_task_dataset(bank["dataset"], bank["split"]))[: len(items)]
    gold = [gold_for_example(example, bank["dataset"]) for example in dataset]

    random_pick = []
    oracle = []
    fce_pick = []
    evidence_coverage = []
    top_evidence_correct = []
    reinforcement_event = []
    informative_group = []
    wrong_reinforcement = []
    for item, answer in zip(items, gold):
        predictions = [
            extract_answer(completion, answer_mode)
            for completion in item["candidates"]
        ]
        correct = np.asarray(
            [float(prediction == answer) for prediction in predictions]
        )
        random_pick.append(float(correct.mean()))
        oracle.append(float(correct.max()))

        evidence = item["frozen_evidence"]
        scores = evidence["scores"]
        evidence_coverage.append(float(bool(scores)))
        if scores:
            best = max(scores.values())
            top_answers = [
                candidate for candidate, score in scores.items()
                if abs(score - best) <= 1e-15
            ]
            top_evidence_correct.append(
                float(np.mean([candidate == answer for candidate in top_answers]))
            )
        else:
            top_evidence_correct.append(0.0)

        candidate_scores = np.asarray(
            [float(scores.get(prediction, 0.0)) for prediction in predictions]
        )
        informative = bool(
            len(candidate_scores)
            and candidate_scores.max() - candidate_scores.min() > 1e-12
        )
        informative_group.append(float(informative))
        reinforced = bool(len(candidate_scores) and candidate_scores.max() > 0)
        reinforcement_event.append(float(reinforced))
        if reinforced:
            selected = np.flatnonzero(
                np.isclose(candidate_scores, candidate_scores.max())
            )
            selected_accuracy = float(correct[selected].mean())
            fce_pick.append(selected_accuracy)
            wrong_reinforcement.append(1.0 - selected_accuracy)
        else:
            fce_pick.append(float(correct.mean()))
            wrong_reinforcement.append(0.0)

    random_arr = np.asarray(random_pick)
    oracle_arr = np.asarray(oracle)
    fce_arr = np.asarray(fce_pick)
    coverage_arr = np.asarray(evidence_coverage, dtype=bool)
    top_correct_arr = np.asarray(top_evidence_correct)
    reinforcement_arr = np.asarray(reinforcement_event, dtype=bool)
    informative_arr = np.asarray(informative_group, dtype=bool)
    wrong_arr = np.asarray(wrong_reinforcement)

    rng = np.random.default_rng(20260729)
    n = len(items)
    indices = rng.integers(0, n, size=(args.bootstrap_draws, n))
    random_boot = random_arr[indices].mean(axis=1)
    oracle_boot = oracle_arr[indices].mean(axis=1)
    fce_boot = fce_arr[indices].mean(axis=1)
    gain_boot = fce_boot - random_boot
    denominator = oracle_boot - random_boot
    captured_boot = np.divide(
        gain_boot,
        denominator,
        out=np.full_like(denominator, np.nan),
        where=np.abs(denominator) > 1e-12,
    )
    captured_boot = captured_boot[np.isfinite(captured_boot)]
    oracle_gain = oracle_arr.mean() - random_arr.mean()
    captured = (
        (fce_arr.mean() - random_arr.mean()) / max(oracle_gain, 1e-12)
    )
    result = {
        "method": "Frozen Cross-Evidence GRPO",
        "model": bank["model"],
        "dataset": bank["dataset"],
        "split": bank["split"],
        "answer_mode": answer_mode,
        "n": n,
        "panel_size": panel_size,
        "candidate_panel_size": candidate_size,
        "evidence_coverage": float(coverage_arr.mean()),
        "top_evidence_precision_when_covered": (
            float(top_correct_arr[coverage_arr].mean())
            if coverage_arr.any()
            else None
        ),
        "random_candidate_accuracy": float(random_arr.mean()),
        "fce_pick_accuracy": float(fce_arr.mean()),
        "oracle_candidate_accuracy": float(oracle_arr.mean()),
        "fce_minus_random": float(fce_arr.mean() - random_arr.mean()),
        "gain_ci95": [float(x) for x in np.quantile(gain_boot, [0.025, 0.975])],
        "p_gain_le_zero": float((gain_boot <= 0).mean()),
        "oracle_gain_captured": float(captured),
        "captured_ci95": [
            float(x) for x in np.quantile(captured_boot, [0.025, 0.975])
        ],
        "reinforcement_event_rate": float(reinforcement_arr.mean()),
        "informative_group_rate": float(informative_arr.mean()),
        "wrong_target_reinforcement_rate": float(wrong_arr.mean()),
        "wrong_given_reinforcement": (
            float(wrong_arr[reinforcement_arr].mean())
            if reinforcement_arr.any()
            else None
        ),
        "viability_gate": {
            "captures_at_least_70pct_oracle_gain": bool(captured >= 0.70),
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
