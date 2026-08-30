#!/usr/bin/env python3
"""Capture the only allowed exact matched-control resume invocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_FLAGS = {
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
    "--fcc-bank": (
        "/home/ajinkya/pgr/checkpoints/fcc_smollm_train_n1000_p12.json"
    ),
    "--resume-from": (
        "/home/ajinkya/pgr/checkpoints/"
        "fce_control_trajectory_fce_permuted_seed42_steps1000_k4"
        "_INTERRUPTED_ARCHIVE_20260729_2258/checkpoint-200"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def process_argv(pid: int) -> list[str]:
    raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    return [part.decode(errors="surrogateescape") for part in raw.split(b"\0") if part]


def matching_processes(script: Path) -> list[tuple[int, list[str]]]:
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = process_argv(int(entry.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if str(script) in argv:
            matches.append((int(entry.name), argv))
    return sorted(matches)


def flag_values(argv: list[str]) -> dict[str, str]:
    observed = {}
    for index, token in enumerate(argv[:-1]):
        if token in EXPECTED_FLAGS:
            observed[token] = argv[index + 1]
    return observed


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/ajinkya/pgr"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=14_400)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite existing capture: {args.output}")

    experiment = args.root / "experiments" / "frozen_cross_consensus"
    script = experiment / "local_fcc_train.py"
    trainer = experiment / "fcc_step_pgr_trainer.py"
    reward = experiment / "fce_reward.py"
    bank = args.root / "checkpoints" / "fcc_smollm_train_n1000_p12.json"
    resume = Path(EXPECTED_FLAGS["--resume-from"])
    deadline = time.monotonic() + args.timeout_seconds

    while time.monotonic() < deadline:
        matches = matching_processes(script)
        if not matches:
            time.sleep(1)
            continue
        if len(matches) != 1:
            raise ValueError(f"expected one FCE trainer process, observed {matches}")
        pid, argv = matches[0]
        observed = flag_values(argv)
        accepted = observed == EXPECTED_FLAGS
        proc = Path("/proc") / str(pid)
        state = json.loads((resume / "trainer_state.json").read_text())
        manifest = json.loads(
            (resume / "lightweight_resume_state.json").read_text()
        )
        preboundary = (
            state.get("global_step") == 200
            and 200 < (895 // 4)
            and (200 % (895 // 4)) * 4 == 200 * 4 == 800
            and manifest.get("global_step") == 200
            and manifest.get("optimizer_state_saved") is True
            and manifest.get("scheduler_state_saved") is True
            and manifest.get("rng_state_saved") is True
        )
        payload = {
            "capture_version": 2,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "pid": pid,
            "ppid": int(
                next(
                    line.split()[1]
                    for line in (proc / "status").read_text().splitlines()
                    if line.startswith("PPid:")
                )
            ),
            "argv": argv,
            "observed_flags": observed,
            "expected_flags": EXPECTED_FLAGS,
            "resume_path_exact": observed.get("--resume-from")
            == EXPECTED_FLAGS["--resume-from"],
            "preboundary_skip_exact": preboundary,
            "invocation_matches_preregistered_control": accepted and preboundary,
            "cwd": os.readlink(proc / "cwd"),
            "executable": os.readlink(proc / "exe"),
            "source_sha256": {
                "local_fcc_train.py": sha256_file(script),
                "fcc_step_pgr_trainer.py": sha256_file(trainer),
                "fce_reward.py": sha256_file(reward),
            },
            "bank": {
                "path": str(bank),
                "bytes": bank.stat().st_size,
                "sha256": sha256_file(bank),
            },
            "resume_checkpoint": {
                "path": str(resume),
                "global_step": state["global_step"],
                "model_sha256": sha256_file(next(resume.glob("*.safetensors"))),
            },
        }
        write_json_atomic(args.output, payload)
        print(json.dumps(payload, indent=2), flush=True)
        if not payload["invocation_matches_preregistered_control"]:
            raise ValueError("live exact-resume invocation does not match contract")
        return
    raise TimeoutError("timed out waiting for exact-resume control process")


if __name__ == "__main__":
    main()
