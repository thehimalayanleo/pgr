#!/usr/bin/env python3
"""Gold-separated stress test for systematic wrong FCE consensus.

The safeguards use only frozen panel statistics. Gold is loaded after all
scores have been constructed and is used solely to evaluate candidate picks.
Run SmolLM2 with ``--phase development`` to lock one safeguard, then pass that
name to the Qwen run via ``--locked-safeguard``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from fcc_reward import extract_answer
from fce_tasks import gold_for_example, load_task_dataset, task_spec


VARIANTS = (
    "baseline_fce",
    "minimum_two_occurrences_in_each_panel",
    "minimum_three_independent_paths_total",
    "cross_panel_support_times_panel_margin",
    "cross_panel_support_times_one_minus_normalized_panel_entropy",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def answer_counts(completions: list[str], answer_mode: str) -> Counter:
    return Counter(
        answer
        for text in completions
        if (answer := extract_answer(text, answer_mode)) is not None
    )


def normalized_entropy(counts: Counter, panel_size: int) -> float:
    if not counts or panel_size <= 1:
        return 0.0
    probabilities = np.asarray(list(counts.values()), dtype=np.float64)
    probabilities /= probabilities.sum()
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    return entropy / math.log(panel_size)


def panel_margin(counts: Counter, answer: str, panel_size: int) -> float:
    other = max(
        (count for candidate, count in counts.items() if candidate != answer),
        default=0,
    )
    return max(counts.get(answer, 0) - other, 0) / panel_size


def variant_scores(item: dict, answer_mode: str, panel_size: int) -> dict[str, dict]:
    baseline = {
        str(answer): float(score)
        for answer, score in item["frozen_evidence"]["scores"].items()
    }
    details = item["frozen_evidence"]["details"]
    counts_a = answer_counts(item["panel_a"], answer_mode)
    counts_b = answer_counts(item["panel_b"], answer_mode)
    entropy_factor = max(
        0.0,
        1.0
        - 0.5
        * (
            normalized_entropy(counts_a, panel_size)
            + normalized_entropy(counts_b, panel_size)
        ),
    )

    return {
        "baseline_fce": baseline,
        "minimum_two_occurrences_in_each_panel": {
            answer: score
            for answer, score in baseline.items()
            if details[answer]["support_a"] >= 2
            and details[answer]["support_b"] >= 2
        },
        "minimum_three_independent_paths_total": {
            answer: score
            for answer, score in baseline.items()
            if details[answer]["independent_paths"] >= 3
        },
        "cross_panel_support_times_panel_margin": {
            answer: score
            * math.sqrt(
                panel_margin(counts_a, answer, panel_size)
                * panel_margin(counts_b, answer, panel_size)
            )
            for answer, score in baseline.items()
            if panel_margin(counts_a, answer, panel_size) > 0
            and panel_margin(counts_b, answer, panel_size) > 0
        },
        "cross_panel_support_times_one_minus_normalized_panel_entropy": {
            answer: score * entropy_factor
            for answer, score in baseline.items()
            if entropy_factor > 0
        },
    }


def summarize_variant(records: list[dict], bootstrap_indices: np.ndarray) -> dict:
    random_accuracy = np.asarray(
        [record["random_accuracy"] for record in records],
        dtype=np.float64,
    )
    pick_accuracy = np.asarray(
        [record["pick_accuracy"] for record in records],
        dtype=np.float64,
    )
    reinforced = np.asarray(
        [record["reinforced"] for record in records],
        dtype=np.float64,
    )
    informative = np.asarray(
        [record["informative"] for record in records],
        dtype=np.float64,
    )
    wrong = np.asarray(
        [record["wrong_reinforcement"] for record in records],
        dtype=np.float64,
    )
    gain = pick_accuracy - random_accuracy
    wrong_given = (
        float(wrong.sum() / reinforced.sum()) if reinforced.sum() else None
    )

    gain_boot = gain[bootstrap_indices].mean(axis=1)
    reinforced_boot = reinforced[bootstrap_indices].sum(axis=1)
    wrong_boot = wrong[bootstrap_indices].sum(axis=1)
    wrong_given_boot = np.divide(
        wrong_boot,
        reinforced_boot,
        out=np.full_like(wrong_boot, np.nan),
        where=reinforced_boot > 0,
    )
    wrong_given_boot = wrong_given_boot[np.isfinite(wrong_given_boot)]
    return {
        "random_candidate_accuracy": float(random_accuracy.mean()),
        "fce_pick_accuracy": float(pick_accuracy.mean()),
        "selection_gain": float(gain.mean()),
        "selection_gain_ci95": [
            float(value) for value in np.quantile(gain_boot, [0.025, 0.975])
        ],
        "reinforcement_event_rate": float(reinforced.mean()),
        "informative_group_rate": float(informative.mean()),
        "wrong_given_reinforcement": wrong_given,
        "wrong_given_reinforcement_ci95": (
            [
                float(value)
                for value in np.quantile(wrong_given_boot, [0.025, 0.975])
            ]
            if len(wrong_given_boot)
            else None
        ),
        "_gain_boot": gain_boot,
        "_wrong_given_boot": wrong_given_boot,
    }


def strip_arrays(summary: dict) -> dict:
    return {
        key: value
        for key, value in summary.items()
        if not key.startswith("_")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=["development", "locked_replication"],
        required=True,
    )
    parser.add_argument("--locked-safeguard", choices=VARIANTS[1:])
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    args = parser.parse_args()
    if args.phase == "locked_replication" and not args.locked_safeguard:
        parser.error("--locked-safeguard is required for locked replication")

    bank = json.loads(args.bank.read_text())
    if bank.get("partial") or bank.get("frozen_evidence_attached") is not True:
        raise ValueError("bank must be complete with frozen evidence")
    if bank.get("candidate_panel_included") is not True:
        raise ValueError("wrong-consensus analysis needs independent candidates")
    if bank.get("gold_stored") is not False:
        raise ValueError("bank must explicitly be gold-free")

    dataset_name = bank["dataset"]
    spec = task_spec(dataset_name)
    answer_mode = bank.get("answer_mode", "numeric")
    if answer_mode != spec.answer_mode:
        raise ValueError("bank answer mode disagrees with task adapter")
    items = bank["items"]
    dataset = list(load_task_dataset(dataset_name, bank["split"]))[: len(items)]
    gold = [gold_for_example(example, dataset_name) for example in dataset]
    if any(answer is None for answer in gold):
        raise ValueError("at least one evaluation gold answer could not be parsed")

    panel_size = int(bank["panel_size"])
    records = {variant: [] for variant in VARIANTS}
    for item, correct_answer in zip(items, gold):
        predictions = [
            extract_answer(text, answer_mode) for text in item["candidates"]
        ]
        candidate_correct = np.asarray(
            [float(prediction == correct_answer) for prediction in predictions]
        )
        random_accuracy = float(candidate_correct.mean())
        all_scores = variant_scores(item, answer_mode, panel_size)
        for variant, scores in all_scores.items():
            candidate_scores = np.asarray(
                [float(scores.get(prediction, 0.0)) for prediction in predictions]
            )
            reinforced = bool(len(candidate_scores) and candidate_scores.max() > 0)
            informative = bool(
                len(candidate_scores)
                and candidate_scores.max() - candidate_scores.min() > 1e-12
            )
            if reinforced:
                selected = np.flatnonzero(
                    np.isclose(candidate_scores, candidate_scores.max())
                )
                pick_accuracy = float(candidate_correct[selected].mean())
                wrong_reinforcement = 1.0 - pick_accuracy
            else:
                pick_accuracy = random_accuracy
                wrong_reinforcement = 0.0
            records[variant].append(
                {
                    "random_accuracy": random_accuracy,
                    "pick_accuracy": pick_accuracy,
                    "reinforced": float(reinforced),
                    "informative": float(informative),
                    "wrong_reinforcement": wrong_reinforcement,
                }
            )

    rng = np.random.default_rng(20260730)
    bootstrap_indices = rng.integers(
        0,
        len(items),
        size=(args.bootstrap_draws, len(items)),
    )
    raw_summaries = {
        variant: summarize_variant(variant_records, bootstrap_indices)
        for variant, variant_records in records.items()
    }
    baseline = raw_summaries["baseline_fce"]
    comparison = {}
    for variant in VARIANTS[1:]:
        current = raw_summaries[variant]
        baseline_wrong = baseline["wrong_given_reinforcement"]
        current_wrong = current["wrong_given_reinforcement"]
        wrong_reduction = (
            (baseline_wrong - current_wrong) / baseline_wrong
            if baseline_wrong and current_wrong is not None
            else None
        )
        gain_retention = (
            current["selection_gain"] / baseline["selection_gain"]
            if abs(baseline["selection_gain"]) > 1e-12
            else None
        )
        gate = {
            "wrong_relative_reduction_at_least_0_25": bool(
                wrong_reduction is not None and wrong_reduction >= 0.25
            ),
            "selection_gain_retention_at_least_0_80": bool(
                gain_retention is not None and gain_retention >= 0.80
            ),
            "informative_group_rate_at_least_0_30": bool(
                current["informative_group_rate"] >= 0.30
            ),
        }
        comparison[variant] = {
            "wrong_given_reinforcement_relative_reduction": wrong_reduction,
            "selection_gain_retention": gain_retention,
            "gate": gate,
            "passes_all": all(gate.values()),
        }

    chosen = None
    if args.phase == "development":
        eligible = [
            variant for variant in VARIANTS[1:]
            if comparison[variant]["passes_all"]
        ]
        if eligible:
            chosen = max(
                eligible,
                key=lambda variant: (
                    raw_summaries[variant]["selection_gain"],
                    -VARIANTS.index(variant),
                ),
            )
    else:
        chosen = args.locked_safeguard

    result = {
        "analysis": "systematic wrong frozen-consensus safeguards",
        "phase": args.phase,
        "bank_path": str(args.bank),
        "bank_sha256": sha256_file(args.bank),
        "model": bank["model"],
        "dataset": dataset_name,
        "split": bank["split"],
        "answer_mode": answer_mode,
        "n": len(items),
        "panel_size": panel_size,
        "candidate_panel_size": int(bank["candidate_panel_size"]),
        "gold_used_for_score_construction": False,
        "safeguard_definitions": {
            "minimum_two_occurrences_in_each_panel": "drop answers with support below two in either panel",
            "minimum_three_independent_paths_total": "drop answers with fewer than three lexically independent supporting paths across panels",
            "cross_panel_support_times_panel_margin": "multiply FCE by sqrt(clipped_margin_a*clipped_margin_b), where each margin is (answer_count-max_other_count)/panel_size",
            "cross_panel_support_times_one_minus_normalized_panel_entropy": "multiply FCE by one minus mean panel answer entropy normalized by log(panel_size)",
        },
        "variants": {
            variant: strip_arrays(summary)
            for variant, summary in raw_summaries.items()
        },
        "comparisons_to_baseline": comparison,
        "development_selected_safeguard": (
            chosen if args.phase == "development" else None
        ),
        "locked_safeguard": (
            chosen if args.phase == "locked_replication" else None
        ),
        "locked_safeguard_passes_all": (
            comparison[chosen]["passes_all"]
            if args.phase == "locked_replication"
            else None
        ),
        "impossibility_boundary": (
            "A label-free consensus method cannot distinguish a unanimous, "
            "diversely worded wrong answer from a correct one when every "
            "observable panel statistic is the same."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
