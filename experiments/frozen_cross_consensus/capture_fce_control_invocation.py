#!/usr/bin/env python3
"""Capture the live matched-control invocation before its first checkpoint.

This is deliberately independent of the training process. It observes `/proc`, records
the exact argv/cwd/executable and hashes the immutable inputs used by the process. The
result gives the final execution audit direct evidence that the control really launched
with `fce_permuted` and without a resume flag.
"""

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
    matches: list[tuple[int, list[str]]] = []
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
    observed: dict[str, str] = {}
    for index, token in enumerate(argv[:-1]):
        if token in EXPECTED_FLAGS or token == "--fcc-bank":
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
    expected_bank = str(bank)
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
        required_argv = (
            observed == {**EXPECTED_FLAGS, "--fcc-bank": expected_bank}
            and "--resume-from" not in argv
        )
        proc = Path("/proc") / str(pid)
        payload = {
            "capture_version": 1,
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
            "expected_flags": {**EXPECTED_FLAGS, "--fcc-bank": expected_bank},
            "resume_flag_absent": "--resume-from" not in argv,
            "invocation_matches_preregistered_control": required_argv,
            "cwd": os.readlink(proc / "cwd"),
            "executable": os.readlink(proc / "exe"),
            "source_sha256": {
                "local_fcc_train.py": sha256_file(script),
                "fcc_step_pgr_trainer.py": sha256_file(trainer),
                "fce_reward.py": sha256_file(reward),
            },
            "bank": {
                "path": expected_bank,
                "bytes": bank.stat().st_size,
                "sha256": sha256_file(bank),
            },
        }
        write_json_atomic(args.output, payload)
        print(json.dumps(payload, indent=2), flush=True)
        if not required_argv:
            raise ValueError("live control invocation does not match preregistration")
        return

    raise TimeoutError("timed out waiting for the matched-control process")


if __name__ == "__main__":
    main()
