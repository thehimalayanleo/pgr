#!/usr/bin/env python3
"""End-to-end execution audit for the preregistered FCE-GRPO experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED = {
    "max_steps": 1000,
    "num_generations": 4,
    "max_completion_length": 384,
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 5,
    "seed": 42,
    "save_steps": 50,
    "save_total_limit": 2,
    "bf16": True,
}
RESUME_FORMAT = "fce-streamed-exact-resume-v2"
ORIGINAL_TRAINER_SHA256 = (
    "d9676cea27710ddafd8988e202c8b7be3bfe341c72a3a6a3e1f7e38e136f8698"
)
CONTROL_TRAINER_SHA256 = (
    "619981e3a5b19bcd646bd41dded2212edaad1742a2f488064f69fe9c34d882c5"
)
CONTROL_RESERVATION_RUNNER_SHA256 = (
    "8f4f496a98e11f86e6387204450c977c9b0e0379132f19af066febd5cd2d2941"
)
CONTROL_CAPTURE_SHA256 = (
    "d36210e0cff97c1ca1dc2c6392dfb64d112dbf65074170b556ea7212188d76c6"
)
CONTROL_REPLAY_AUDITOR_SHA256 = (
    "94ccb1639c8c9a6fb0f8d26c4902394caeeff737a6170805a098dd28b0609b45"
)
BEHAVIOR_ANALYSIS_SHA256 = (
    "f305d768ea3ddff0d22de4ebcda4c035425b3848dfaecce4020518c59f2d53b9"
)
CONTROL_EXPECTED_FLAGS = {
    "--max-steps": "1000",
    "--seed": "42",
    "--k": "4",
    "--dataset": "openai/gsm8k",
    "--model": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "--max-completion": "384",
    "--lr": "2e-5",
    "--output-suffix": "fce_control",
    "--save-steps": "50",
    "--save-total-limit": "2",
    "--alpha": "0",
    "--step-advantage-mode": "group_mean",
    "--terminal-spread": "constant",
    "--reward-source": "fce_permuted",
    "--fcc-bank": "/home/ajinkya/pgr/checkpoints/fcc_smollm_train_n1000_p12.json",
}
EXPECTED_PROTOCOL_SHA256 = {
    "PREREGISTRATION_2026-07-29.json": (
        "f3ac802d4d1a05f9d264d58316ec54b01d81d4903481a726890d80d861b327b9"
    ),
    "PREREGISTRATION_2026-07-29_MATCHED_CONTROL.json": (
        "6d42fbd34a4bf4b2653d2ce6af87eac01a28baabfb18341554dc12f46e240173"
    ),
    "AMENDMENT_2026-07-29_UNCONDITIONAL_CONTROL.json": (
        "61e3e31f964c07b53b8b098160a308f14f9fdbd95862cbf054222a450cb8d062"
    ),
    "AMENDMENT_2026-07-29_UNIFORM_SHUFFLE_CONTROL.json": (
        "b06d95a055d7fb3f6fe484e0572827581084d58c889b2cb326038da88895ed63"
    ),
    "AMENDMENT_2026-07-29_EXACT_OPTIMIZER_RESUME.json": (
        "9aa30b2f66f913d4d54e20f14093c6ff60a183b1da7a7bc08c9c305da9a8f603"
    ),
    "AMENDMENT_2026-07-29_STREAMED_RESUME_PARENT_BYPASS.json": (
        "706f772c564dff676b781514b51f35fd12e6c2bc47a3e8f6ed273b2e186410d6"
    ),
    "AMENDMENT_2026-07-29_FAIL_CLOSED_STOCHASTIC_RESUME.json": (
        "93797735d43125b402b961326eee1f5f116c5b7a426fa396ed1dd7cc4b807b30"
    ),
    "AMENDMENT_2026-07-29_CONTROL_RESERVATION.json": (
        "165ceb47042ac7cf6bd75e2467ba8b3ae47212d4508bcb3814019efcb9f3854b"
    ),
    "AMENDMENT_2026-07-29_PREBOUNDARY_REPLAY.json": (
        "7179b7f2c61cf8d43efda3a132b5238c850827d8d925d60db82370faef49fa64"
    ),
    "AMENDMENT_2026-07-30_STARTUP_QUIESCENCE.json": (
        "85c503231d33e36a33d1666c9ec30055b7a5880065da5b54df6a879e30b38bfc"
    ),
    "AMENDMENT_2026-07-30_SHARED_TASK_COORDINATION.json": (
        "f97df41808a47046608e56c790a92b2079b2017bb544a54b3c218d8cd03aff5b"
    ),
    "AMENDMENT_2026-07-30_CLEAN_CONTROL_ONLY.json": (
        "8823695743b497f63a0e94255fa929b3e59d074a40fa2b553536ffab09d3397e"
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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


def model_weight(path: Path) -> Path:
    candidates = sorted(path.glob("*.safetensors"))
    candidates.extend(sorted(path.glob("pytorch_model*.bin")))
    require(len(candidates) == 1, f"expected one model weight file in {path}")
    require(candidates[0].stat().st_size > 0, f"empty model weight in {path}")
    return candidates[0]


def audit_training_log(
    *,
    path: Path,
    start_marker: str,
    bank_marker: str,
    attempt_marker: str,
    required_markers: tuple[str, ...] = (),
    expected_resume_marker: str | None = None,
) -> dict:
    text = path.read_text(errors="replace")
    require(
        text.count(start_marker) == 1,
        f"{path.name} does not contain exactly one clean training start",
    )
    require(
        text.count(bank_marker) == 1,
        f"{path.name} does not contain exactly one bank-coverage marker",
    )
    require(
        text.count(attempt_marker) == 1,
        f"{path.name} does not contain exactly one launch attempt",
    )
    if expected_resume_marker is None:
        require(
            "RESUMING from " not in text,
            f"{path.name} contains an unexpected resume event",
        )
    else:
        require(
            text.count(expected_resume_marker) == 1
            and text.count("RESUMING from ") == 1,
            f"{path.name} does not contain exactly the allowed resume event",
        )
    require(
        all(text.count(marker) == 1 for marker in required_markers),
        f"{path.name} is missing a required one-time execution marker",
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "one_clean_start": True,
        "one_launch_attempt": True,
        "resume_event_absent": expected_resume_marker is None,
        "allowed_resume_event": expected_resume_marker,
        "required_one_time_markers": list(required_markers),
    }


def audit_control_capture(
    *,
    capture_path: Path,
    capture_script: Path,
    bank_sha256: str,
) -> dict:
    capture = read_json(capture_path)
    require(
        sha256_file(capture_script) == CONTROL_CAPTURE_SHA256,
        "control invocation capture source mismatch",
    )
    require(
        capture.get("observed_flags") == CONTROL_EXPECTED_FLAGS
        and capture.get("expected_flags") == CONTROL_EXPECTED_FLAGS,
        "captured control flags do not match the preregistered invocation",
    )
    argv = capture.get("argv", [])
    require(
        capture.get("capture_version") == 1
        and capture.get("invocation_matches_preregistered_control") is True
        and capture.get("resume_flag_absent") is True
        and "--resume-from" not in argv,
        "captured clean step-0 control invocation was rejected",
    )
    require(
        capture.get("cwd") == "/home/ajinkya/pgr",
        "captured control working directory mismatch",
    )
    require(
        capture.get("executable")
        == "/home/ajinkya/miniconda3/envs/pgr/bin/python3.11",
        "captured control Python executable mismatch",
    )
    source = capture.get("source_sha256", {})
    require(
        source.get("local_fcc_train.py") == CONTROL_TRAINER_SHA256
        and source.get("fcc_step_pgr_trainer.py")
        == "620e894de2f3f455dde2a9af14a10d24352bccc4f80844e97b821b15d2237fd8"
        and source.get("fce_reward.py")
        == "6ab3f52d909c32370a9145a44a90d78edd191aad56c168eaf97fca243c642d02",
        "captured control source hashes mismatch",
    )
    require(
        capture.get("bank", {}).get("sha256") == bank_sha256,
        "captured control bank hash mismatch",
    )
    return {
        **capture,
        "capture_path": str(capture_path),
        "capture_sha256": sha256_file(capture_path),
        "capture_script_sha256": CONTROL_CAPTURE_SHA256,
    }


def audit_permutation_domain(bank: dict) -> dict:
    """Check that the fixed control hash produces an effectively uniform map.

    This evaluates the complete preregistered 895-prompt by 1,000-step domain,
    not the realized stochastic dataloader order. It catches positional bias in
    the deterministic permutation construction while keeping the runtime arm
    reproducible.
    """
    prompt_hashes = []
    for item in bank["items"]:
        if not item.get("frozen_evidence", {}).get("scores"):
            continue
        prompt = item["prompt"]
        text = (
            prompt
            if isinstance(prompt, str)
            else json.dumps(prompt, sort_keys=True)
        )
        prompt_hashes.append(hashlib.sha256(text.encode("utf-8")).hexdigest())
    require(len(prompt_hashes) == 895, "unexpected permutation prompt domain")

    counts = np.zeros((4, 4), dtype=np.int64)
    identity = 0
    total = 0
    expected_order = np.arange(4)
    for global_step in range(1000):
        for prompt_hash in prompt_hashes:
            material = (
                f"42:{global_step}:0:{prompt_hash}:fce-permuted"
            ).encode("utf-8")
            seed = int.from_bytes(
                hashlib.sha256(material).digest()[:8],
                "big",
            )
            permutation = np.random.default_rng(seed).permutation(4)
            require(
                sorted(permutation.tolist()) == [0, 1, 2, 3],
                "control transformation did not preserve the reward multiset",
            )
            identity += int(np.array_equal(permutation, expected_order))
            total += 1
            for destination, source in enumerate(permutation):
                counts[destination, source] += 1

    fractions = counts.astype(np.float64) / total
    identity_rate = identity / total
    max_mapping_deviation = float(np.abs(fractions - 0.25).max())
    require(
        abs(identity_rate - (1.0 / 24.0)) < 0.002,
        "control permutation identity rate is unexpectedly biased",
    )
    require(
        max_mapping_deviation < 0.002,
        "control permutation has an unexpectedly biased position map",
    )
    return {
        "prompts": len(prompt_hashes),
        "global_steps": 1000,
        "permutations_checked": total,
        "identity_rate": identity_rate,
        "expected_identity_rate": 1.0 / 24.0,
        "position_mapping_fractions": fractions.tolist(),
        "max_absolute_position_mapping_deviation": max_mapping_deviation,
        "reward_multiset_preserved_for_every_permutation": True,
        "scope": "complete prompt-by-step domain, not realized dataloader order",
    }


def audit_rejected_control_archive(root: Path) -> dict:
    checkpoint_root = (
        root
        / "checkpoints"
        / (
            "fce_control_trajectory_fce_permuted_seed42_steps1000_k4"
            "_INTERRUPTED_ARCHIVE_20260729_2258"
        )
    )
    checkpoint = checkpoint_root / "checkpoint-250"
    log = (
        root
        / "logs"
        / "fce_matched_control_train_INTERRUPTED_ARCHIVE_20260729_2258.log"
    )
    invocation = (
        root
        / "checkpoints"
        / (
            "fce_control_live_invocation"
            "_INTERRUPTED_ARCHIVE_20260729_2258.json"
        )
    )
    state = read_json(checkpoint / "trainer_state.json")
    log_text = log.read_text(errors="replace")
    require(
        state.get("global_step") == 250,
        "rejected control archive checkpoint mismatch",
    )
    require(
        "yielding PID 3649628 to foreign GPU work" in log_text,
        "rejected control archive lacks its interruption event",
    )
    require(
        not (checkpoint_root.parent / f"{checkpoint_root.name}_final").exists(),
        "rejected control archive unexpectedly has a final model",
    )
    capture = read_json(invocation)
    require(
        capture.get("invocation_matches_preregistered_control") is True
        and capture.get("resume_flag_absent") is True,
        "rejected control invocation archive mismatch",
    )
    return {
        "checkpoint": str(checkpoint),
        "latest_certified_step": 250,
        "live_step_before_yield": 268,
        "reason": "foreign CUDA admission race",
        "log": {"path": str(log), "sha256": sha256_file(log)},
        "invocation": {
            "path": str(invocation),
            "sha256": sha256_file(invocation),
        },
        "heldout_evaluation_performed": False,
        "use_for_attribution": False,
    }


def audit_rejected_adhoc_control_archive(root: Path) -> dict:
    checkpoint_root = (
        root
        / "checkpoints"
        / (
            "fce_control_trajectory_fce_permuted_seed42_steps1000_k4"
            "_INTERRUPTED_ADHOC_ARCHIVE_20260729_2313"
        )
    )
    log = (
        root
        / "logs"
        / (
            "fce_matched_control_train"
            "_INTERRUPTED_ADHOC_ARCHIVE_20260729_2313.log"
        )
    )
    invocation = (
        root
        / "checkpoints"
        / (
            "fce_control_live_invocation"
            "_INTERRUPTED_ADHOC_ARCHIVE_20260729_2313.json"
        )
    )
    require(checkpoint_root.is_dir(), "ad-hoc interruption archive is missing")
    require(
        not any(checkpoint_root.glob("checkpoint-*")),
        "ad-hoc interruption archive unexpectedly contains a checkpoint",
    )
    log_text = log.read_text(errors="replace")
    require(
        "yielding PID 3661813 to foreign GPU work" in log_text
        and "'step': 25" not in log_text
        and " 25/1000" in log_text,
        "ad-hoc interruption archive lacks its step-25 yield event",
    )
    require(
        not (checkpoint_root.parent / f"{checkpoint_root.name}_final").exists(),
        "ad-hoc interruption archive unexpectedly has a final model",
    )
    capture = read_json(invocation)
    require(
        capture.get("capture_version") == 1
        and capture.get("invocation_matches_preregistered_control") is True
        and capture.get("resume_flag_absent") is True,
        "ad-hoc interruption invocation archive mismatch",
    )
    return {
        "training_archive": str(checkpoint_root),
        "latest_live_step": 25,
        "certified_checkpoint_written": False,
        "reason": "foreign ad-hoc CUDA process",
        "log": {"path": str(log), "sha256": sha256_file(log)},
        "invocation": {
            "path": str(invocation),
            "sha256": sha256_file(invocation),
        },
        "heldout_evaluation_performed": False,
        "use_for_attribution": False,
    }


def audit_rejected_startup_race_archive(root: Path) -> dict:
    suffix = "STARTUP_RACE_ARCHIVE_20260730_0029"
    checkpoint_root = (
        root
        / "checkpoints"
        / (
            "fce_control_trajectory_fce_permuted_seed42_steps1000_k4_"
            f"{suffix}"
        )
    )
    log = (
        root
        / "logs"
        / f"fce_matched_control_train_{suffix}.log"
    )
    invocation = (
        root
        / "checkpoints"
        / f"fce_control_live_invocation_{suffix}.json"
    )
    supervisor_log = (
        root
        / "logs"
        / f"fce_gate_supervisor_{suffix}.log"
    )
    require(checkpoint_root.is_dir(), "startup-race archive is missing")
    require(
        not any(checkpoint_root.rglob("*")),
        "startup-race archive unexpectedly contains training state",
    )
    log_text = log.read_text(errors="replace")
    require(
        "yielding PID 3688093 to foreign GPU work" in log_text
        and "201/1000" in log_text,
        "startup-race archive lacks its step-201 yield event",
    )
    require(
        "refusing non-exact matched-control resume" in (
            supervisor_log.read_text(errors="replace")
        ),
        "startup-race supervisor did not fail closed",
    )
    require(
        not (checkpoint_root.parent / f"{checkpoint_root.name}_final").exists(),
        "startup-race archive unexpectedly has a final model",
    )
    capture = read_json(invocation)
    require(
        capture.get("capture_version") == 2
        and capture.get("invocation_matches_preregistered_control") is True
        and capture.get("resume_path_exact") is True
        and capture.get("preboundary_skip_exact") is True,
        "startup-race invocation archive mismatch",
    )
    return {
        "training_archive": str(checkpoint_root),
        "latest_live_step": 201,
        "certified_checkpoint_written": False,
        "reason": "direct Qwen CUDA startup race",
        "foreign_process": {
            "pid": 3688546,
            "script": "qwen_ruler_histogram_eval.py",
            "observed_cuda_memory_mib": 2664,
        },
        "log": {"path": str(log), "sha256": sha256_file(log)},
        "invocation": {
            "path": str(invocation),
            "sha256": sha256_file(invocation),
        },
        "supervisor_log": {
            "path": str(supervisor_log),
            "sha256": sha256_file(supervisor_log),
        },
        "heldout_evaluation_performed": False,
        "use_for_attribution": False,
    }


def audit_rejected_postquiescence_archive(root: Path) -> dict:
    suffix = "POSTQUIET_FOREIGN_ARCHIVE_20260730_0059"
    checkpoint_root = (
        root
        / "checkpoints"
        / (
            "fce_control_trajectory_fce_permuted_seed42_steps1000_k4_"
            f"{suffix}"
        )
    )
    log = root / "logs" / f"fce_matched_control_train_{suffix}.log"
    invocation = (
        root / "checkpoints" / f"fce_control_live_invocation_{suffix}.json"
    )
    supervisor_log = (
        root / "logs" / f"fce_gate_supervisor_{suffix}.log"
    )
    require(
        checkpoint_root.is_dir() and not any(checkpoint_root.rglob("*")),
        "post-quiescence interruption archive is missing or nonempty",
    )
    log_text = log.read_text(errors="replace")
    require(
        "PGR control reservation ready 6144 MiB" in log_text
        and "PGR control reservation quiescent 60 seconds" in log_text
        and "yielding PID 3698633 to foreign GPU work" in log_text
        and "205/1000" in log_text,
        "post-quiescence archive lacks its step-205 yield event",
    )
    capture = read_json(invocation)
    require(
        capture.get("capture_version") == 2
        and capture.get("invocation_matches_preregistered_control") is True
        and capture.get("resume_path_exact") is True
        and capture.get("preboundary_skip_exact") is True,
        "post-quiescence invocation archive mismatch",
    )
    require(
        not (checkpoint_root.parent / f"{checkpoint_root.name}_final").exists(),
        "post-quiescence archive unexpectedly has a final model",
    )
    return {
        "training_archive": str(checkpoint_root),
        "latest_live_step": 205,
        "certified_checkpoint_written": False,
        "reason": "foreign CUDA launch after quiescent control admission",
        "foreign_process": {
            "observed_pid": 3698915,
            "observed_cuda_memory_mib": 2494,
            "owner": "active SAE/log-linear-attention Codex task",
        },
        "log": {"path": str(log), "sha256": sha256_file(log)},
        "invocation": {
            "path": str(invocation),
            "sha256": sha256_file(invocation),
        },
        "supervisor_log": {
            "path": str(supervisor_log),
            "sha256": sha256_file(supervisor_log),
        },
        "heldout_evaluation_performed": False,
        "use_for_attribution": False,
    }


def audit_rejected_precoord_relaunch(root: Path) -> dict:
    suffix = "PRECOORD_RELAUNCH_ARCHIVE_20260730_0101"
    checkpoint_root = (
        root
        / "checkpoints"
        / (
            "fce_control_trajectory_fce_permuted_seed42_steps1000_k4_"
            f"{suffix}"
        )
    )
    invocation = (
        root / "checkpoints" / f"fce_control_live_invocation_{suffix}.json"
    )
    log = root / "logs" / f"fce_matched_control_train_{suffix}.log"
    supervisor_log = root / "logs" / f"fce_gate_supervisor_{suffix}.log"
    log_text = log.read_text(errors="replace")
    require(
        not checkpoint_root.exists() and not invocation.exists(),
        "pre-coordination relaunch unexpectedly contains training evidence",
    )
    require(
        "matched-control attempt 2 "
        "resume=certified-pre-boundary-checkpoint-200" in log_text
        and "yielding PID 3699766 to foreign GPU work" in log_text,
        "pre-coordination relaunch lacks its immediate yield event",
    )
    return {
        "training_archive": str(checkpoint_root),
        "trainer_process_reached_evidence_capture": False,
        "checkpoint_written": False,
        "reason": "foreign CUDA work was still active before coordination",
        "log": {"path": str(log), "sha256": sha256_file(log)},
        "supervisor_log": {
            "path": str(supervisor_log),
            "sha256": sha256_file(supervisor_log),
        },
        "heldout_evaluation_performed": False,
        "use_for_attribution": False,
    }


def audit_failed_preboundary_replay(
    *,
    report_path: Path,
    auditor_path: Path,
    root: Path,
) -> dict:
    require(
        sha256_file(auditor_path) == CONTROL_REPLAY_AUDITOR_SHA256,
        "pre-boundary replay auditor source mismatch",
    )
    report = read_json(report_path)
    reference = (
        root
        / "checkpoints"
        / (
            "fce_control_trajectory_fce_permuted_seed42_steps1000_k4"
            "_INTERRUPTED_ARCHIVE_20260729_2258"
        )
        / "checkpoint-250"
    )
    candidate_archive = (
        root
        / "checkpoints"
        / (
            "fce_control_trajectory_fce_permuted_seed42_steps1000_k4"
            "_REPLAY_FAILED_ARCHIVE_20260730_0109"
        )
        / "checkpoint-250"
    )
    require(reference.is_dir(), "failed replay reference is missing")
    require(candidate_archive.is_dir(), "failed replay candidate is missing")
    required_errors = {
        "trainer_state.json differs",
        "model weight differs",
        "RNG state differs",
        "streamed optimizer state differs",
        "immutable model or training metadata differs",
    }
    checks = (
        report.get("audit_version") == 1,
        report.get("passed") is False,
        report.get("reference_checkpoint") == str(reference),
        report.get("resume_checkpoint_step") == 200,
        report.get("replay_checkpoint_step") == 250,
        report.get("preboundary_skip_proof", {}).get("exact") is True,
        report.get("trainer_state", {}).get("json_equal") is False,
        report.get("trainer_state", {}).get(
            "complete_log_history_equal"
        ) is False,
        report.get("normalized_resume_manifest_equal") is True,
        report.get("model_weight", {}).get("equal") is False,
        report.get("scheduler", {}).get("equal") is True,
        report.get("rng_state", {}).get("equal") is False,
        report.get("streamed_optimizer", {}).get("files") == 219,
        report.get("streamed_optimizer", {}).get("equal") is False,
        report.get("immutable_metadata_equal") is False,
        set(report.get("errors", [])) == required_errors,
    )
    require(all(checks), "failed pre-boundary replay evidence mismatch")
    final_model = candidate_archive.parent / (
        "fce_control_trajectory_fce_permuted_seed42_steps1000_k4"
        "_REPLAY_FAILED_ARCHIVE_20260730_0109_final"
    )
    require(not final_model.exists(), "failed replay unexpectedly has a final model")
    return {
        "audit_version": report["audit_version"],
        "passed": report["passed"],
        "reference_checkpoint": str(reference),
        "candidate_checkpoint_archive": str(candidate_archive),
        "trainer_state_equal": False,
        "model_weight_equal": False,
        "streamed_optimizer_equal": False,
        "scheduler_equal": True,
        "rng_state_equal": False,
        "immutable_metadata_equal": False,
        "errors": report["errors"],
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "auditor_path": str(auditor_path),
        "auditor_sha256": CONTROL_REPLAY_AUDITOR_SHA256,
        "heldout_evaluation_performed": False,
        "use_for_attribution": False,
    }


def audit_training_arm(
    *,
    name: str,
    checkpoint_root: Path,
    final_model: Path,
    expected_output_dir: str,
    expected_run_name: str,
) -> dict:
    checkpoint = checkpoint_root / "checkpoint-1000"
    state = read_json(checkpoint / "trainer_state.json")
    manifest = read_json(checkpoint / "lightweight_resume_state.json")
    require(state.get("global_step") == 1000, f"{name} global step mismatch")
    require(state.get("max_steps") == 1000, f"{name} max steps mismatch")
    require(state.get("save_steps") == 50, f"{name} save cadence mismatch")
    history = state.get("log_history", [])
    steps = [entry.get("step") for entry in history]
    require(
        steps == list(range(5, 1001, 5)),
        f"{name} log history is not a complete 5-step sequence",
    )
    require(
        all(entry.get("fce_bank_coverage") == 1.0 for entry in history),
        f"{name} trained outside frozen-bank coverage",
    )
    max_within_rollout_variance = max(
        abs(float(entry["within_rollout_adv_var"])) for entry in history
    )
    require(
        max_within_rollout_variance < 1e-10,
        f"{name} did not use constant trajectory advantages",
    )
    require(
        all(0.0 <= float(entry["adv_token_coverage"]) <= 1.0 for entry in history),
        f"{name} has invalid advantage coverage",
    )
    require(
        manifest.get("format") == RESUME_FORMAT
        and manifest.get("global_step") == 1000
        and manifest.get("scheduler_state_saved") is True
        and manifest.get("rng_state_saved") is True
        and manifest.get("optimizer_state_saved") is True
        and manifest.get("optimizer_state_format")
        == "streamed-per-parameter-v1"
        and manifest.get("optimizer_reset_on_resume") is False,
        f"{name} exact-resume manifest mismatch",
    )
    optimizer_dir = checkpoint / manifest["optimizer_state_dir"]
    require(
        optimizer_dir.is_dir() and (optimizer_dir / "metadata.pt").is_file(),
        f"{name} streamed optimizer is incomplete",
    )
    require(
        (checkpoint / "scheduler.pt").is_file()
        and any(checkpoint.glob("rng_state*.pth")),
        f"{name} scheduler or RNG state is missing",
    )

    import torch

    arguments = torch.load(
        final_model / "training_args.bin",
        map_location="cpu",
        weights_only=False,
    )
    observed_arguments = {
        key: getattr(arguments, key, None)
        for key in EXPECTED
    }
    require(
        observed_arguments == EXPECTED,
        f"{name} training arguments mismatch: {observed_arguments}",
    )
    require(
        arguments.output_dir == expected_output_dir,
        f"{name} output directory mismatch",
    )
    require(arguments.run_name == expected_run_name, f"{name} run name mismatch")

    checkpoint_weight = model_weight(checkpoint)
    final_weight = model_weight(final_model)
    checkpoint_hash = sha256_file(checkpoint_weight)
    final_hash = sha256_file(final_weight)
    require(
        checkpoint_weight.stat().st_size == final_weight.stat().st_size
        and checkpoint_hash == final_hash,
        f"{name} final model does not equal checkpoint 1000",
    )
    return {
        "checkpoint": str(checkpoint),
        "global_step": state["global_step"],
        "history_records": len(history),
        "training_arguments": observed_arguments,
        "mean_advantage_token_coverage": (
            sum(float(entry["adv_token_coverage"]) for entry in history)
            / len(history)
        ),
        "max_within_rollout_advantage_variance": max_within_rollout_variance,
        "exact_resume_manifest": manifest,
        "final_weight": {
            "path": str(final_weight),
            "bytes": final_weight.stat().st_size,
            "sha256": final_hash,
            "equals_checkpoint_1000": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/ajinkya/pgr"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoints = args.root / "checkpoints"

    bank_audit = read_json(checkpoints / "fce_online_bank_audit.json")
    primary_audit = read_json(checkpoints / "fce_online_final_audit.json")
    control_audit = read_json(checkpoints / "fce_online_control_audit.json")
    primary_result = read_json(checkpoints / "fce_online_viability_result.json")
    control_result = read_json(checkpoints / "fce_matched_control_result.json")
    require(bank_audit.get("passed") is True, "bank audit did not pass")
    require(primary_audit.get("passed") is True, "primary audit did not pass")
    require(control_audit.get("passed") is True, "control audit did not pass")
    require(
        bank_audit["bank"]["items"] == 1000
        and bank_audit["bank"]["gold_stored"] is False
        and bank_audit["bank"]["frozen_evidence_recomputed_exactly"] is True,
        "bank contract mismatch",
    )

    base_eval = read_json(checkpoints / "fce_base_heldout500_greedy.json")
    primary_eval = read_json(checkpoints / "fce_online_heldout500_greedy.json")
    control_eval = read_json(checkpoints / "fce_permuted_heldout500_greedy.json")
    behavior = read_json(checkpoints / "fce_behavior_audit.json")
    prompt_orders = [
        [record["prompt_hash"] for record in evaluation["records"]]
        for evaluation in (base_eval, primary_eval, control_eval)
    ]
    require(
        prompt_orders[0] == prompt_orders[1] == prompt_orders[2],
        "held-out evaluations are not exactly paired",
    )
    require(
        len(prompt_orders[0]) == 500 and len(set(prompt_orders[0])) == 500,
        "held-out prompt set mismatch",
    )
    evaluation_paths = {
        "base": checkpoints / "fce_base_heldout500_greedy.json",
        "fce": checkpoints / "fce_online_heldout500_greedy.json",
        "control": checkpoints / "fce_permuted_heldout500_greedy.json",
    }
    require(
        behavior.get("descriptive_only") is True
        and all(
            behavior.get("inputs", {}).get(name, {}).get("sha256")
            == sha256_file(path)
            for name, path in evaluation_paths.items()
        ),
        "behavior audit does not match the held-out evaluations",
    )
    require(
        behavior["arms"]["base"]["correct"]
        == sum(record["correct"] for record in base_eval["records"])
        and behavior["arms"]["fce"]["correct"]
        == sum(record["correct"] for record in primary_eval["records"])
        and behavior["arms"]["control"]["correct"]
        == sum(record["correct"] for record in control_eval["records"]),
        "behavior audit correctness counts mismatch",
    )

    primary = audit_training_arm(
        name="primary",
        checkpoint_root=(
            checkpoints / "fce_online_trajectory_fce_seed42_steps1000_k4"
        ),
        final_model=(
            checkpoints / "fce_online_trajectory_fce_seed42_steps1000_k4_final"
        ),
        expected_output_dir=(
            "./checkpoints/fce_online_trajectory_fce_seed42_steps1000_k4"
        ),
        expected_run_name="fce_online_trajectory_fce_k4",
    )
    control = audit_training_arm(
        name="control",
        checkpoint_root=(
            checkpoints
            / "fce_control_trajectory_fce_permuted_seed42_steps1000_k4"
        ),
        final_model=(
            checkpoints
            / "fce_control_trajectory_fce_permuted_seed42_steps1000_k4_final"
        ),
        expected_output_dir=(
            "./checkpoints/"
            "fce_control_trajectory_fce_permuted_seed42_steps1000_k4"
        ),
        expected_run_name="fce_control_trajectory_fce_permuted_k4",
    )

    source_archive = checkpoints / "fce_executed_source_20260729"
    original_trainer = source_archive / "local_fcc_train.py"
    control_trainer = source_archive / "local_fcc_train.control_resume.py"
    behavior_analysis = source_archive / "analyze_fce_behavior.py"
    control_runner = source_archive / "run_fcc_online_gate.final_clean_start.sh"
    replay_auditor = source_archive / "audit_fce_preboundary_replay.py"
    require(
        sha256_file(original_trainer) == ORIGINAL_TRAINER_SHA256,
        "original executed trainer source mismatch",
    )
    require(
        sha256_file(control_trainer) == CONTROL_TRAINER_SHA256,
        "control trainer source mismatch",
    )
    require(
        sha256_file(behavior_analysis) == BEHAVIOR_ANALYSIS_SHA256,
        "behavior-analysis source mismatch",
    )
    require(
        sha256_file(control_runner) == CONTROL_RESERVATION_RUNNER_SHA256,
        "control-reservation runner source mismatch",
    )
    protocol_sha256 = {
        name: sha256_file(source_archive / name)
        for name in EXPECTED_PROTOCOL_SHA256
    }
    require(
        protocol_sha256 == EXPECTED_PROTOCOL_SHA256,
        "preregistration or amendment archive mismatch",
    )
    require(
        primary_audit["source_sha256"]["local_fcc_train.py"]
        == ORIGINAL_TRAINER_SHA256,
        "primary audit source does not match archived trainer",
    )
    require(
        control_audit["source_sha256"]["local_fcc_train.py"]
        == CONTROL_TRAINER_SHA256,
        "control audit source does not match archived trainer",
    )
    require(
        control_audit["source_sha256"]["run_fcc_online_gate.sh"]
        == CONTROL_RESERVATION_RUNNER_SHA256,
        "control audit source does not match reservation runner",
    )
    primary_log = audit_training_log(
        path=args.root / "logs" / "fcc_online_train.log",
        start_marker=(
            "=== LOCAL 5090 SMOKE [trajectory_fce | K=4 | steps=1000] ==="
        ),
        bank_marker="FCE bank coverage: training on 895 accepted prompts",
        attempt_marker="FCC train attempt 1",
    )
    control_log = audit_training_log(
        path=args.root / "logs" / "fce_matched_control_train.log",
        start_marker=(
            "=== LOCAL 5090 SMOKE "
            "[trajectory_fce_permuted | K=4 | steps=1000] ==="
        ),
        bank_marker="FCE_PERMUTED bank coverage: training on 895 accepted prompts",
        attempt_marker=(
            "matched-control attempt 1 resume=disabled"
        ),
        required_markers=(
            "PGR control reservation ready 6144 MiB",
            "PGR control reservation quiescent 60 seconds",
        ),
    )
    control_capture = audit_control_capture(
        capture_path=checkpoints / "fce_control_live_invocation.json",
        capture_script=(
            source_archive / "capture_fce_control_invocation.py"
        ),
        bank_sha256=bank_audit["bank"]["sha256"],
    )
    failed_preboundary_replay = audit_failed_preboundary_replay(
        report_path=(
            checkpoints
            / (
                "fce_control_preboundary_replay_audit"
                "_REPLAY_FAILED_ARCHIVE_20260730_0109.json"
            )
        ),
        auditor_path=replay_auditor,
        root=args.root,
    )
    permutation_domain = audit_permutation_domain(
        read_json(checkpoints / "fcc_smollm_train_n1000_p12.json")
    )
    rejected_control = audit_rejected_control_archive(args.root)
    rejected_adhoc_control = audit_rejected_adhoc_control_archive(args.root)
    rejected_startup_race = audit_rejected_startup_race_archive(args.root)
    rejected_postquiescence = audit_rejected_postquiescence_archive(args.root)
    rejected_precoord_relaunch = audit_rejected_precoord_relaunch(args.root)

    primary_gate = primary_result["viability_gate"]
    attribution_gate = control_result["attribution_gate"]
    viability_supported = bool(
        primary_gate.get("passed") and attribution_gate.get("passed")
    )
    report = {
        "method": "Frozen Cross-Evidence GRPO",
        "audit_version": 1,
        "bank": bank_audit["bank"],
        "primary_training": primary,
        "control_training": control,
        "primary_training_log": primary_log,
        "control_training_log": control_log,
        "control_live_invocation": control_capture,
        "failed_control_preboundary_replay": failed_preboundary_replay,
        "control_permutation_domain": permutation_domain,
        "rejected_interrupted_control": rejected_control,
        "rejected_adhoc_interrupted_control": rejected_adhoc_control,
        "rejected_startup_race_control": rejected_startup_race,
        "rejected_postquiescence_control": rejected_postquiescence,
        "rejected_precoord_relaunch": rejected_precoord_relaunch,
        "behavior_audit": behavior,
        "paired_heldout_prompts": 500,
        "primary_result": primary_result,
        "matched_control_result": control_result,
        "source_provenance": {
            "original_trainer_sha256": ORIGINAL_TRAINER_SHA256,
            "control_trainer_sha256": CONTROL_TRAINER_SHA256,
            "control_reservation_runner_sha256": (
                CONTROL_RESERVATION_RUNNER_SHA256
            ),
            "control_capture_script_sha256": CONTROL_CAPTURE_SHA256,
            "control_replay_auditor_sha256": (
                CONTROL_REPLAY_AUDITOR_SHA256
            ),
            "behavior_analysis_sha256": BEHAVIOR_ANALYSIS_SHA256,
            "protocol_and_amendment_sha256": protocol_sha256,
            "control_design_note": (
                "The initial nonzero cyclic-shift control was replaced before "
                "either online arm began by the archived uniform-shuffle "
                "amendment; later resume and reservation amendments changed "
                "execution safety, not the reward transformation or "
                "attribution gate. The final 6 GiB reservation and 60-second "
                "quiescence gate close the observed simultaneous-start race; "
                "cross-task coordination prevents direct launches after "
                "admission without changing any experimental parameter. The "
                "checkpoint-200 recovery assumption was empirically rejected "
                "because its checkpoint-250 replay diverged in trainer state, "
                "model weights, optimizer state, RNG state, and immutable "
                "metadata. The only attribution-eligible control is therefore "
                "the final fresh uninterrupted step-0 run."
            ),
        },
        "integrity_audit_passed": True,
        "viability_supported": viability_supported,
        "decision": (
            "supported: FCE-GRPO beats both base and matched permutation control"
            if viability_supported
            else "not supported: one or more preregistered gates failed"
        ),
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
