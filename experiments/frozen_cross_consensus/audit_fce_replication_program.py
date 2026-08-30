#!/usr/bin/env python3
"""Fail-closed completion audit for the 2026-07-30 FCE replication program."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def source_manifest_audit(exp: Path) -> dict:
    manifest = exp / "REPLICATION_SOURCE_MANIFEST_2026-07-30_V3.sha256"
    if not manifest.is_file():
        return {"complete": False, "passed": False, "reason": "manifest missing"}
    files = {}
    passed = True
    for line in manifest.read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = exp / name
        actual = sha256(path) if path.is_file() else None
        match = actual == expected
        passed &= match
        files[name] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "match": match,
        }
    return {"complete": True, "passed": passed, "files": files}


def bank_audit(path: Path, dataset: str, split: str) -> dict:
    if not path.is_file():
        return {"complete": False, "passed": False, "path": str(path)}
    bank = read_json(path)
    items = bank.get("items", [])
    hashes = [item.get("prompt_hash") for item in items]
    accepted = sum(
        bool(item.get("frozen_evidence", {}).get("scores")) for item in items
    )
    passed = bool(
        bank.get("dataset") == dataset
        and bank.get("split") == split
        and bank.get("gold_stored") is False
        and bank.get("candidate_panel_included") is False
        and bank.get("frozen_evidence_attached") is True
        and bank.get("partial") is False
        and bank.get("panel_size") == 12
        and len(items) == 1000
        and len(set(hashes)) == 1000
        and all(isinstance(value, str) and len(value) == 64 for value in hashes)
        and accepted >= 200
    )
    return {
        "complete": True,
        "passed": passed,
        "path": str(path),
        "sha256": sha256(path),
        "items": len(items),
        "unique_prompt_hashes": len(set(hashes)),
        "accepted": accepted,
        "gold_stored": bank.get("gold_stored"),
    }


def model_audit(checkpoint: str | None) -> dict:
    if not checkpoint:
        return {"complete": False, "passed": False}
    path = Path(checkpoint)
    weights = list(path.glob("*.safetensors")) + list(
        path.glob("pytorch_model*.bin")
    )
    passed = (
        path.is_dir()
        and (path / "config.json").is_file()
        and (path / "tokenizer_config.json").is_file()
        and bool(weights)
        and all(weight.stat().st_size > 0 for weight in weights)
    )
    return {
        "complete": path.is_dir(),
        "passed": passed,
        "path": str(path),
        "weights": {
            weight.name: {
                "bytes": weight.stat().st_size,
                "sha256": sha256(weight),
            }
            for weight in weights
        },
    }


def eval_audit(path: Path, dataset: str) -> dict:
    if not path.is_file():
        return {"complete": False, "passed": False, "path": str(path)}
    result = read_json(path)
    records = result.get("records", [])
    hashes = [record.get("prompt_hash") for record in records]
    correct = sum(bool(record.get("correct")) for record in records)
    model = model_audit(result.get("checkpoint"))
    passed = bool(
        result.get("dataset") == dataset
        and result.get("split") == "test"
        and result.get("offset") == 500
        and result.get("n") == 500
        and result.get("decoding") == "greedy"
        and len(records) == 500
        and len(set(hashes)) == 500
        and abs(result.get("accuracy", -1) - correct / 500) <= 1e-15
        and model["passed"]
    )
    return {
        "complete": True,
        "passed": passed,
        "path": str(path),
        "sha256": sha256(path),
        "checkpoint": result.get("checkpoint"),
        "correct": correct,
        "accuracy": result.get("accuracy"),
        "model": model,
    }


def analysis_audit(
    path: Path,
    *,
    difference_key: str,
    p_key: str,
    minimum_difference: float = 0.0,
) -> dict:
    if not path.is_file():
        return {"complete": False, "passed": False, "path": str(path)}
    result = read_json(path)
    difference = result.get(difference_key)
    p_value = result.get(p_key)
    passed = bool(
        isinstance(difference, (int, float))
        and difference >= minimum_difference
        and isinstance(p_value, (int, float))
        and p_value < 0.05
    )
    return {
        "complete": True,
        "passed": passed,
        "path": str(path),
        "sha256": sha256(path),
        "difference": difference,
        "minimum_difference": minimum_difference,
        "p_value": p_value,
    }


def selection_audit(path: Path) -> dict:
    if not path.is_file():
        return {"complete": False, "passed": False, "path": str(path)}
    result = read_json(path)
    gate = result.get("viability_gate", {})
    passed = bool(
        gate.get("captures_at_least_70pct_oracle_gain") is True
        and gate.get("paired_bootstrap_p_lt_0_05") is True
        and gate.get("informative_group_rate_at_least_0_30") is True
    )
    return {
        "complete": True,
        "passed": passed,
        "path": str(path),
        "sha256": sha256(path),
        "oracle_gain_captured": result.get("oracle_gain_captured"),
        "informative_group_rate": result.get("informative_group_rate"),
        "gain_ci95": result.get("gain_ci95"),
        "gate": gate,
    }


def multiseed_audit(path: Path) -> dict:
    if not path.is_file():
        return {"complete": False, "passed": False, "path": str(path)}
    result = read_json(path)
    seeds = result.get("training_seeds")
    gate = result.get("success_gate", {})
    per_seed = result.get("per_seed", {})
    checkpoint_audits = {
        str(seed): {
            "fce": model_audit(
                per_seed.get(str(seed), {}).get("fce_checkpoint")
            ),
            "permuted": model_audit(
                per_seed.get(str(seed), {}).get("permuted_checkpoint")
            ),
        }
        for seed in (seeds or [])
    }
    checkpoints_passed = bool(
        seeds
        and all(
            audit["passed"]
            for pair in checkpoint_audits.values()
            for audit in pair.values()
        )
    )
    passed = bool(
        seeds == [43, 44, 45, 46]
        and result.get("n_seeds") == 4
        and result.get("prompts_per_seed") == 500
        and gate.get(
            "at_least_three_of_four_seed_differences_positive"
        ) is True
        and gate.get("mean_difference_at_least_0_05") is True
        and gate.get("hierarchical_ci95_lower_bound_gt_zero") is True
        and result.get("passed") is True
        and checkpoints_passed
    )
    return {
        "complete": True,
        "passed": passed,
        "path": str(path),
        "sha256": sha256(path),
        "training_seeds": seeds,
        "per_seed": per_seed,
        "checkpoint_audits": checkpoint_audits,
        "mean_fce_minus_permuted": result.get("mean_fce_minus_permuted"),
        "ci95": result.get("hierarchical_seed_prompt_bootstrap_ci95"),
        "gate": gate,
    }


def descriptive_comparison(path: Path) -> dict:
    if not path.is_file():
        return {"complete": False, "passed": False, "path": str(path)}
    result = read_json(path)
    left_model = model_audit(result.get("left_checkpoint"))
    right_model = model_audit(result.get("right_checkpoint"))
    return {
        "complete": True,
        "passed": left_model["passed"] and right_model["passed"],
        "path": str(path),
        "sha256": sha256(path),
        "left_name": result.get("left_name"),
        "right_name": result.get("right_name"),
        "left_accuracy": result.get("left_accuracy"),
        "right_accuracy": result.get("right_accuracy"),
        "left_minus_right": result.get("left_minus_right"),
        "difference_ci95": result.get("difference_ci95"),
        "left_model": left_model,
        "right_model": right_model,
    }


def robustness_audit(exp: Path) -> dict:
    path = exp / "results/wrong_consensus_qwen_locked_replication.json"
    if not path.is_file():
        return {"complete": False, "passed": False, "path": str(path)}
    result = read_json(path)
    passed = bool(
        result.get("phase") == "locked_replication"
        and result.get("locked_safeguard")
        == "minimum_two_occurrences_in_each_panel"
        and result.get("locked_safeguard_passes_all") is True
    )
    return {
        "complete": True,
        "passed": passed,
        "path": str(path),
        "sha256": sha256(path),
        "comparison": result.get("comparisons_to_baseline", {}).get(
            "minimum_two_occurrences_in_each_panel"
        ),
    }


def domain_audit(
    tag: str,
    dataset: str,
    train_split: str,
    core: Path,
    domain: Path,
) -> dict:
    selection = selection_audit(core / f"{tag}_qwen_selection_result.json")
    if not selection["complete"]:
        return {
            "complete": False,
            "passed": False,
            "selection": selection,
            "stage": "selection_pending",
        }
    if not selection["passed"]:
        return {
            "complete": True,
            "passed": False,
            "selection": selection,
            "stage": "selection_failed_training_sealed",
        }

    bank = bank_audit(
        domain / f"{tag}_qwen_train_n1000_p12.json",
        dataset,
        train_split,
    )
    base_eval = eval_audit(domain / f"{tag}_qwen_base_heldout500.json", dataset)
    fce_eval = eval_audit(domain / f"{tag}_qwen_fce_heldout500.json", dataset)
    control_eval = eval_audit(
        domain / f"{tag}_qwen_permuted_heldout500.json",
        dataset,
    )
    versus_base = analysis_audit(
        domain / f"{tag}_fce_vs_base.json",
        difference_key="trained_minus_base",
        p_key="p_gain_le_zero",
        minimum_difference=0.03,
    )
    versus_control = analysis_audit(
        domain / f"{tag}_fce_vs_permuted.json",
        difference_key="fce_minus_control",
        p_key="p_gain_le_zero",
        minimum_difference=0.0,
    )
    components = {
        "bank": bank,
        "base_eval": base_eval,
        "fce_eval": fce_eval,
        "control_eval": control_eval,
        "versus_base": versus_base,
        "versus_control": versus_control,
    }
    complete = all(value["complete"] for value in components.values())
    passed = complete and all(value["passed"] for value in components.values())
    return {
        "complete": complete,
        "passed": passed,
        "selection": selection,
        "stage": "policy_training_complete" if complete else "policy_training_pending",
        **components,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/ajinkya/pgr"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args()

    exp = args.root / "experiments/frozen_cross_consensus"
    core = args.root / "checkpoints/fce_replication_20260730"
    domain = args.root / "checkpoints/fce_domain_policy_20260730"

    requirements = {
        "source_manifest": source_manifest_audit(exp),
        "locked_wrong_consensus_replication": robustness_audit(exp),
        "smollm_multiseed": multiseed_audit(
            core / "smollm_multiseed_hierarchical.json"
        ),
        "qwen_train_bank": bank_audit(
            args.root / "checkpoints/fcc_qwen_train_n1000_p12.json",
            "openai/gsm8k",
            "train",
        ),
        "qwen_base_eval": eval_audit(
            core / "qwen_base_heldout500.json",
            "openai/gsm8k",
        ),
        "qwen_fce_eval": eval_audit(
            core / "qwen_seed42_fce_heldout500.json",
            "openai/gsm8k",
        ),
        "qwen_permuted_eval": eval_audit(
            core / "qwen_seed42_permuted_heldout500.json",
            "openai/gsm8k",
        ),
        "qwen_fce_vs_base": analysis_audit(
            core / "qwen_fce_vs_base.json",
            difference_key="trained_minus_base",
            p_key="p_gain_le_zero",
            minimum_difference=0.03,
        ),
        "qwen_fce_vs_permuted": analysis_audit(
            core / "qwen_fce_vs_permuted.json",
            difference_key="fce_minus_control",
            p_key="p_gain_le_zero",
            minimum_difference=0.0,
        ),
        "fce_vs_majority": analysis_audit(
            core / "smollm_fce_vs_majority_seed42.json",
            difference_key="left_minus_right",
            p_key="p_left_minus_right_le_zero",
            minimum_difference=0.0,
        ),
        "smollm_gold_eval": eval_audit(
            core / "smollm_gold_seed42_heldout500.json",
            "openai/gsm8k",
        ),
        "smollm_majority_eval": eval_audit(
            core / "smollm_majority_seed42_heldout500.json",
            "openai/gsm8k",
        ),
        "fce_vs_gold_descriptive": descriptive_comparison(
            core / "smollm_fce_vs_gold_seed42.json"
        ),
        "math_hard_transfer": domain_audit(
            "math_hard",
            "lighteval/MATH-Hard",
            "train",
            core,
            domain,
        ),
        "mmlu_transfer": domain_audit(
            "mmlu",
            "cais/mmlu",
            "auxiliary_train",
            core,
            domain,
        ),
    }
    complete = all(value["complete"] for value in requirements.values())
    all_confirmatory_gates_passed = complete and all(
        value["passed"] for value in requirements.values()
    )
    failed = [
        name
        for name, value in requirements.items()
        if value["complete"] and not value["passed"]
    ]
    pending = [
        name for name, value in requirements.items() if not value["complete"]
    ]
    result = {
        "audit": "FCE 2026-07-30 replication program",
        "complete": complete,
        "all_confirmatory_gates_passed": all_confirmatory_gates_passed,
        "failed_requirements": failed,
        "pending_requirements": pending,
        "requirements": requirements,
        "claim_boundary": (
            "A completed failed gate remains a completed experiment and must "
            "not be relabeled as pending or omitted."
        ),
    }
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2))
    if not complete and not args.allow_pending:
        raise SystemExit(2)
    if complete and not all_confirmatory_gates_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
