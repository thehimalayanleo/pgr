#!/usr/bin/env python3
"""Certify an exact checkpoint-200 -> checkpoint-250 control replay.

The only allowed matched-control recovery starts from the archived checkpoint
200, where Transformers' floor-based dataloader skip is still identical to the
uninterrupted trajectory.  This auditor waits for the replayed checkpoint 250
and requires all state that can affect continuation to match the archived
uninterrupted checkpoint bit-for-bit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path


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


def model_weight(path: Path) -> Path:
    candidates = sorted(path.glob("*.safetensors"))
    candidates.extend(sorted(path.glob("pytorch_model*.bin")))
    if len(candidates) != 1:
        raise ValueError(f"expected one model weight in {path}, got {candidates}")
    return candidates[0]


def file_record(path: Path) -> dict:
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def checkpoint_ready(path: Path) -> bool:
    try:
        trainer_state = json.loads((path / "trainer_state.json").read_text())
        manifest = json.loads(
            (path / "lightweight_resume_state.json").read_text()
        )
        optimizer_dir = path / manifest["optimizer_state_dir"]
        return (
            trainer_state.get("global_step") == 250
            and manifest.get("format") == "fce-streamed-exact-resume-v2"
            and manifest.get("global_step") == 250
            and manifest.get("optimizer_state_saved") is True
            and manifest.get("scheduler_state_saved") is True
            and manifest.get("rng_state_saved") is True
            and optimizer_dir.is_dir()
            and (optimizer_dir / "metadata.pt").is_file()
            and len(list(optimizer_dir.glob("state-*.pt"))) == 218
            and (path / "scheduler.pt").is_file()
            and len(list(path.glob("rng_state*.pth"))) == 1
            and model_weight(path).stat().st_size > 0
        )
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return False


def optimizer_records(checkpoint: Path, manifest: dict) -> dict[str, dict]:
    optimizer_dir = checkpoint / manifest["optimizer_state_dir"]
    return {
        path.name: file_record(path)
        for path in sorted(optimizer_dir.iterdir())
        if path.is_file()
    }


def normalized_manifest(manifest: dict) -> dict:
    normalized = dict(manifest)
    normalized["optimizer_state_dir"] = "<normalized-random-directory>"
    return normalized


def compare(reference: Path, candidate: Path) -> dict:
    errors: list[str] = []
    reference_state_path = reference / "trainer_state.json"
    candidate_state_path = candidate / "trainer_state.json"
    reference_state = json.loads(reference_state_path.read_text())
    candidate_state = json.loads(candidate_state_path.read_text())
    if candidate_state != reference_state:
        errors.append("trainer_state.json differs")

    reference_manifest_path = reference / "lightweight_resume_state.json"
    candidate_manifest_path = candidate / "lightweight_resume_state.json"
    reference_manifest = json.loads(reference_manifest_path.read_text())
    candidate_manifest = json.loads(candidate_manifest_path.read_text())
    if normalized_manifest(candidate_manifest) != normalized_manifest(
        reference_manifest
    ):
        errors.append("normalized exact-resume manifest differs")

    reference_weight = model_weight(reference)
    candidate_weight = model_weight(candidate)
    reference_weight_record = file_record(reference_weight)
    candidate_weight_record = file_record(candidate_weight)
    if candidate_weight_record != reference_weight_record:
        errors.append("model weight differs")

    reference_scheduler = file_record(reference / "scheduler.pt")
    candidate_scheduler = file_record(candidate / "scheduler.pt")
    if candidate_scheduler != reference_scheduler:
        errors.append("scheduler state differs")

    reference_rng_paths = sorted(reference.glob("rng_state*.pth"))
    candidate_rng_paths = sorted(candidate.glob("rng_state*.pth"))
    reference_rng = {
        path.name: file_record(path) for path in reference_rng_paths
    }
    candidate_rng = {
        path.name: file_record(path) for path in candidate_rng_paths
    }
    if candidate_rng != reference_rng:
        errors.append("RNG state differs")

    reference_optimizer = optimizer_records(reference, reference_manifest)
    candidate_optimizer = optimizer_records(candidate, candidate_manifest)
    if candidate_optimizer != reference_optimizer:
        errors.append("streamed optimizer state differs")

    immutable_names = (
        "config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "training_args.bin",
    )
    reference_immutable = {
        name: file_record(reference / name) for name in immutable_names
    }
    candidate_immutable = {
        name: file_record(candidate / name) for name in immutable_names
    }
    if candidate_immutable != reference_immutable:
        errors.append("immutable model or training metadata differs")

    reference_logs = reference_state.get("log_history", [])
    candidate_logs = candidate_state.get("log_history", [])
    expected_log_steps = list(range(5, 251, 5))
    if [entry.get("step") for entry in reference_logs] != expected_log_steps:
        errors.append("reference log history is incomplete")
    if [entry.get("step") for entry in candidate_logs] != expected_log_steps:
        errors.append("candidate log history is incomplete")

    return {
        "audit_version": 1,
        "method": "FCE matched-control pre-boundary exact replay",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_checkpoint": str(reference),
        "candidate_checkpoint": str(candidate),
        "resume_checkpoint_step": 200,
        "replay_checkpoint_step": 250,
        "preboundary_skip_proof": {
            "accepted_prompts": 895,
            "gradient_accumulation_steps": 4,
            "transformers_floor_updates_per_epoch": 895 // 4,
            "resume_global_step": 200,
            "batches_skipped_by_transformers": (
                (200 % (895 // 4)) * 4
            ),
            "batches_consumed_uninterrupted": 200 * 4,
            "exact": (200 % (895 // 4)) * 4 == 200 * 4 == 800,
        },
        "trainer_state": {
            "reference": file_record(reference_state_path),
            "candidate": file_record(candidate_state_path),
            "json_equal": candidate_state == reference_state,
            "complete_log_history_equal": candidate_logs == reference_logs,
        },
        "normalized_resume_manifest_equal": (
            normalized_manifest(candidate_manifest)
            == normalized_manifest(reference_manifest)
        ),
        "model_weight": {
            "reference": reference_weight_record,
            "candidate": candidate_weight_record,
            "equal": candidate_weight_record == reference_weight_record,
        },
        "scheduler": {
            "reference": reference_scheduler,
            "candidate": candidate_scheduler,
            "equal": candidate_scheduler == reference_scheduler,
        },
        "rng_state": {
            "reference": reference_rng,
            "candidate": candidate_rng,
            "equal": candidate_rng == reference_rng,
        },
        "streamed_optimizer": {
            "files": len(reference_optimizer),
            "reference": reference_optimizer,
            "candidate": candidate_optimizer,
            "equal": candidate_optimizer == reference_optimizer,
        },
        "immutable_metadata_equal": (
            candidate_immutable == reference_immutable
        ),
        "errors": errors,
        "passed": not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=7200)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite existing audit: {args.output}")
    if not checkpoint_ready(args.reference):
        raise ValueError(f"reference checkpoint is incomplete: {args.reference}")

    deadline = time.monotonic() + args.wait_seconds
    while time.monotonic() < deadline and not checkpoint_ready(args.candidate):
        time.sleep(2)
    if not checkpoint_ready(args.candidate):
        raise TimeoutError(f"candidate checkpoint did not complete: {args.candidate}")

    report = compare(args.reference, args.candidate)
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
