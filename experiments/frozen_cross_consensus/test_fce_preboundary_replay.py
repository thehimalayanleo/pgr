from __future__ import annotations

import json
from pathlib import Path

from audit_fce_preboundary_replay import compare


def write_checkpoint(path: Path, *, optimizer_dir: str) -> None:
    path.mkdir(parents=True)
    log_history = [{"step": step, "loss": step / 1000} for step in range(5, 251, 5)]
    (path / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": 250,
                "max_steps": 1000,
                "log_history": log_history,
            }
        )
    )
    (path / "lightweight_resume_state.json").write_text(
        json.dumps(
            {
                "format": "fce-streamed-exact-resume-v2",
                "global_step": 250,
                "optimizer_state_saved": True,
                "scheduler_state_saved": True,
                "rng_state_saved": True,
                "optimizer_state_format": "streamed-per-parameter-v1",
                "optimizer_state_dir": optimizer_dir,
            }
        )
    )
    (path / "model.safetensors").write_bytes(b"model")
    (path / "scheduler.pt").write_bytes(b"scheduler")
    (path / "rng_state.pth").write_bytes(b"rng")
    for name in (
        "config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "training_args.bin",
    ):
        (path / name).write_bytes(name.encode())
    optimizer = path / optimizer_dir
    optimizer.mkdir()
    (optimizer / "metadata.pt").write_bytes(b"metadata")
    for index in range(218):
        (optimizer / f"state-{index:06d}.pt").write_bytes(
            f"state-{index}".encode()
        )


def test_exact_replay_ignores_only_random_optimizer_directory(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    write_checkpoint(reference, optimizer_dir="optimizer-stream-reference")
    write_checkpoint(candidate, optimizer_dir="optimizer-stream-candidate")

    report = compare(reference, candidate)

    assert report["passed"] is True
    assert report["errors"] == []
    assert report["normalized_resume_manifest_equal"] is True
    assert report["streamed_optimizer"]["files"] == 219
    assert report["streamed_optimizer"]["equal"] is True


def test_exact_replay_rejects_rng_divergence(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    write_checkpoint(reference, optimizer_dir="optimizer-stream-reference")
    write_checkpoint(candidate, optimizer_dir="optimizer-stream-candidate")
    (candidate / "rng_state.pth").write_bytes(b"rng-diverged")

    report = compare(reference, candidate)

    assert report["passed"] is False
    assert "RNG state differs" in report["errors"]
