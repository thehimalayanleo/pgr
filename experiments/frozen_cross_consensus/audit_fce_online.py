#!/usr/bin/env python3
"""Fail-closed audit for FCE's bank, held-out evaluations, and final result."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from build_fcc_bank import panel_seed, prompt_hash
from fcc_reward import build_frozen_target
from fce_reward import attach_frozen_cross_evidence


MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
DATASET = "openai/gsm8k"
TRAIN_N = 1000
PANEL_SIZE = 12
BASE_SEED = 20260729
MIN_ACCEPTED = 200
HELDOUT_OFFSET = 500
HELDOUT_N = 500
BOOTSTRAP_DRAWS = 50_000
MIN_GAIN = 0.03


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def prompt_for(question: str) -> str:
    return f"Solve step by step:\n{question}\n\nSolution:"


def normalized_question(question: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", question.lower())


def audit_bank(bank_path: Path, train_rows: list[dict], test_rows: list[dict]) -> dict:
    bank = json.loads(bank_path.read_text())
    expected_top_level = {
        "method",
        "model",
        "dataset",
        "split",
        "panel_size",
        "candidate_panel_size",
        "seed",
        "candidate_panel_included",
        "frozen_evidence_attached",
        "gold_stored",
        "items",
        "partial",
    }
    require(set(bank) == expected_top_level, "bank top-level schema mismatch")
    require(bank["model"] == MODEL, "bank model mismatch")
    require(bank["dataset"] == DATASET, "bank dataset mismatch")
    require(bank["split"] == "train", "bank split mismatch")
    require(bank["panel_size"] == PANEL_SIZE, "bank panel size mismatch")
    require(bank["candidate_panel_size"] == 0, "training bank has candidates")
    require(
        bank["candidate_panel_included"] is False,
        "training bank declares candidate panel",
    )
    require(
        bank["frozen_evidence_attached"] is True,
        "bank evidence is not finalized",
    )
    require(bank["gold_stored"] is False, "bank does not declare gold-free state")
    require(bank["partial"] is False, "bank is partial")
    require(bank["seed"] == BASE_SEED, "bank seed mismatch")

    items = bank["items"]
    require(len(items) == TRAIN_N, "bank does not contain exactly 1000 items")
    require(len(train_rows) >= TRAIN_N, "training dataset is unexpectedly short")
    allowed_item_keys = {
        "index",
        "prompt_hash",
        "prompt",
        "panel_a",
        "panel_b",
        "candidates",
        "panel_seeds",
        "frozen_target",
        "frozen_evidence",
    }

    hashes = []
    for index, item in enumerate(items):
        require(set(item) == allowed_item_keys, f"bank item {index} schema mismatch")
        expected_prompt = prompt_for(train_rows[index]["question"])
        expected_hash = prompt_hash(expected_prompt)
        require(item["index"] == index, f"bank item {index} is out of order")
        require(item["prompt"] == expected_prompt, f"bank prompt {index} mismatch")
        require(item["prompt_hash"] == expected_hash, f"bank hash {index} mismatch")
        require(
            len(item["panel_a"]) == PANEL_SIZE
            and len(item["panel_b"]) == PANEL_SIZE,
            f"bank panel length mismatch at item {index}",
        )
        require(item["candidates"] == [], f"candidate leakage at item {index}")
        expected_seeds = {
            name: panel_seed(BASE_SEED, expected_hash, name)
            for name in ("a", "b")
        }
        require(
            item["panel_seeds"] == expected_seeds,
            f"panel seed mismatch at item {index}",
        )
        expected_target = build_frozen_target(
            item["panel_a"],
            item["panel_b"],
        ).to_dict()
        require(
            item["frozen_target"] == expected_target,
            f"frozen target mismatch at item {index}",
        )
        hashes.append(expected_hash)
    require(len(set(hashes)) == TRAIN_N, "bank prompt hashes are not unique")

    recomputed = attach_frozen_cross_evidence(
        items,
        panel_size=PANEL_SIZE,
    )
    for index, (stored, derived) in enumerate(zip(items, recomputed)):
        require(
            stored["frozen_evidence"] == derived["frozen_evidence"],
            f"frozen evidence mismatch at item {index}",
        )

    accepted = sum(
        bool(item["frozen_evidence"]["scores"])
        for item in items
    )
    require(
        accepted >= MIN_ACCEPTED,
        f"bank has fewer than {MIN_ACCEPTED} evidence-bearing prompts",
    )

    train_questions = {
        normalized_question(row["question"])
        for row in train_rows[:TRAIN_N]
    }
    heldout_rows = test_rows[HELDOUT_OFFSET : HELDOUT_OFFSET + HELDOUT_N]
    require(len(heldout_rows) == HELDOUT_N, "held-out dataset slice is incomplete")
    heldout_questions = {
        normalized_question(row["question"])
        for row in heldout_rows
    }
    require(
        not train_questions.intersection(heldout_questions),
        "normalized train/held-out question overlap detected",
    )
    heldout_hashes = {
        prompt_hash(prompt_for(row["question"]))
        for row in heldout_rows
    }
    require(
        not set(hashes).intersection(heldout_hashes),
        "exact train/held-out prompt overlap detected",
    )

    return {
        "sha256": sha256_file(bank_path),
        "items": len(items),
        "accepted_evidence_prompts": accepted,
        "accepted_fraction": accepted / len(items),
        "unique_ordered_prompt_hashes": True,
        "panels_exactly_12_each": True,
        "candidates_absent": True,
        "gold_stored": False,
        "frozen_evidence_recomputed_exactly": True,
        "normalized_train_heldout_overlap": 0,
        "exact_train_heldout_overlap": 0,
    }


def expected_eval_rows(test_rows: list[dict]) -> list[dict]:
    rows = test_rows[HELDOUT_OFFSET : HELDOUT_OFFSET + HELDOUT_N]
    return [
        {
            "prompt_hash": prompt_hash(prompt_for(row["question"])),
            "gold": row["answer"].split("####")[-1].strip().replace(",", ""),
        }
        for row in rows
    ]


def audit_evaluation(
    path: Path,
    *,
    checkpoint: str,
    expected_rows: list[dict],
) -> tuple[dict, list[float]]:
    result = json.loads(path.read_text())
    require(result["checkpoint"] == checkpoint, f"{path.name} checkpoint mismatch")
    require(result["dataset"] == DATASET, f"{path.name} dataset mismatch")
    require(result["split"] == "test", f"{path.name} split mismatch")
    require(result["offset"] == HELDOUT_OFFSET, f"{path.name} offset mismatch")
    require(result["n"] == HELDOUT_N, f"{path.name} n mismatch")
    require(result["decoding"] == "greedy", f"{path.name} decoding mismatch")
    records = result["records"]
    require(len(records) == HELDOUT_N, f"{path.name} record count mismatch")

    correctness = []
    for index, (record, expected) in enumerate(zip(records, expected_rows)):
        require(
            record["prompt_hash"] == expected["prompt_hash"],
            f"{path.name} prompt mismatch at {index}",
        )
        require(
            record["gold"] == expected["gold"],
            f"{path.name} gold mismatch at {index}",
        )
        correct = record.get("prediction") == expected["gold"]
        require(
            record["correct"] is correct,
            f"{path.name} correctness mismatch at {index}",
        )
        require(
            isinstance(record.get("completion"), str),
            f"{path.name} completion missing at {index}",
        )
        correctness.append(float(correct))
    correct_count = int(sum(correctness))
    require(result["correct"] == correct_count, f"{path.name} correct total mismatch")
    require(
        result["accuracy"] == correct_count / HELDOUT_N,
        f"{path.name} accuracy mismatch",
    )
    return {
        "sha256": sha256_file(path),
        "checkpoint": checkpoint,
        "correct": correct_count,
        "accuracy": correct_count / HELDOUT_N,
        "paired_prompt_order_verified": True,
    }, correctness


def recompute_analysis(base_correct: list[float], trained_correct: list[float]) -> dict:
    import numpy as np

    base = np.asarray(base_correct, dtype=np.float64)
    trained = np.asarray(trained_correct, dtype=np.float64)
    delta = trained - base
    rng = np.random.default_rng(20260729)
    parts = []
    remaining = BOOTSTRAP_DRAWS
    while remaining:
        draws = min(2000, remaining)
        indices = rng.integers(0, len(delta), size=(draws, len(delta)))
        parts.append(delta[indices].mean(axis=1))
        remaining -= draws
    bootstrap = np.concatenate(parts)
    gain = float(delta.mean())
    p_nonpositive = float((bootstrap <= 0).mean())
    return {
        "method": "Frozen Cross-Evidence GRPO",
        "n": len(delta),
        "test_offset": HELDOUT_OFFSET,
        "decoding": "greedy",
        "base_accuracy": float(base.mean()),
        "trained_accuracy": float(trained.mean()),
        "trained_minus_base": gain,
        "gain_ci95": [
            float(value)
            for value in np.quantile(bootstrap, [0.025, 0.975])
        ],
        "p_gain_le_zero": p_nonpositive,
        "trained_only_correct": int(((trained == 1) & (base == 0)).sum()),
        "base_only_correct": int(((trained == 0) & (base == 1)).sum()),
        "viability_gate": {
            "gain_at_least_minimum": bool(gain >= MIN_GAIN),
            "minimum_gain": MIN_GAIN,
            "paired_bootstrap_p_lt_0_05": bool(p_nonpositive < 0.05),
            "passed": bool(gain >= MIN_GAIN and p_nonpositive < 0.05),
        },
    }


def audit_final_model(path: Path) -> dict:
    require(path.is_dir(), "final model directory is missing")
    required = [path / "config.json", path / "tokenizer_config.json"]
    for candidate in required:
        require(candidate.is_file() and candidate.stat().st_size, f"missing {candidate}")
    tokenizer_candidates = [
        path / "tokenizer.json",
        path / "tokenizer.model",
        path / "spiece.model",
        path / "vocab.json",
    ]
    require(
        any(candidate.is_file() and candidate.stat().st_size for candidate in tokenizer_candidates),
        "final tokenizer payload is missing",
    )
    weight_files = sorted(path.glob("*.safetensors")) + sorted(
        path.glob("pytorch_model*.bin")
    )
    require(weight_files, "final model weights are missing")
    return {
        "path": str(path),
        "config_sha256": sha256_file(path / "config.json"),
        "weight_files": {
            weight.name: {
                "bytes": weight.stat().st_size,
                "sha256": sha256_file(weight),
            }
            for weight in weight_files
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--base-eval", type=Path)
    parser.add_argument("--trained-eval", type=Path)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--final-model", type=Path)
    parser.add_argument("--control-eval", type=Path)
    parser.add_argument("--control-analysis", type=Path)
    parser.add_argument("--control-model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    final_arguments = [
        args.base_eval,
        args.trained_eval,
        args.analysis,
        args.final_model,
    ]
    require(
        all(value is None for value in final_arguments)
        or all(value is not None for value in final_arguments),
        "supply either no final artifacts or all final artifacts",
    )
    control_arguments = [
        args.control_eval,
        args.control_analysis,
        args.control_model,
    ]
    require(
        all(value is None for value in control_arguments)
        or all(value is not None for value in control_arguments),
        "supply either no matched-control artifacts or all matched-control artifacts",
    )
    require(
        args.control_eval is None or args.base_eval is not None,
        "matched-control audit requires primary final artifacts",
    )

    from datasets import load_dataset

    train_rows = list(load_dataset(DATASET, "main", split="train"))
    test_rows = list(load_dataset(DATASET, "main", split="test"))
    report = {
        "method": "Frozen Cross-Evidence GRPO",
        "audit_version": 2,
        "bank": audit_bank(args.bank, train_rows, test_rows),
        "source_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in (
                "fce_reward.py",
                "fce_data.py",
                "fcc_reward.py",
                "build_fcc_bank.py",
                "fcc_step_pgr_trainer.py",
                "local_fcc_train.py",
                "evaluate_fcc_online.py",
                "analyze_fcc_online.py",
                "analyze_fce_control.py",
                "run_fcc_online_gate.sh",
                "audit_fce_online.py",
            )
        },
        "passed": True,
    }

    if args.base_eval is not None:
        expected_rows = expected_eval_rows(test_rows)
        base_summary, base_correct = audit_evaluation(
            args.base_eval,
            checkpoint=MODEL,
            expected_rows=expected_rows,
        )
        trained_checkpoint = str(args.final_model)
        trained_summary, trained_correct = audit_evaluation(
            args.trained_eval,
            checkpoint=trained_checkpoint,
            expected_rows=expected_rows,
        )
        derived_analysis = recompute_analysis(base_correct, trained_correct)
        stored_analysis = json.loads(args.analysis.read_text())
        require(stored_analysis == derived_analysis, "online analysis recomputation mismatch")
        report.update(
            {
                "base_evaluation": base_summary,
                "trained_evaluation": trained_summary,
                "analysis": {
                    **derived_analysis,
                    "sha256": sha256_file(args.analysis),
                    "recomputed_exactly": True,
                },
                "final_model": audit_final_model(args.final_model),
            }
        )

        if args.control_eval is not None:
            from analyze_fce_control import compute_comparison

            control_summary, _ = audit_evaluation(
                args.control_eval,
                checkpoint=str(args.control_model),
                expected_rows=expected_rows,
            )
            control_result = json.loads(args.control_analysis.read_text())
            derived_control = compute_comparison(
                json.loads(args.trained_eval.read_text()),
                json.loads(args.control_eval.read_text()),
            )
            require(
                control_result == derived_control,
                "matched-control analysis recomputation mismatch",
            )
            report.update(
                {
                    "matched_control_evaluation": control_summary,
                    "matched_control_analysis": {
                        **derived_control,
                        "sha256": sha256_file(args.control_analysis),
                        "recomputed_exactly": True,
                    },
                    "matched_control_model": audit_final_model(
                        args.control_model
                    ),
                }
            )

    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
