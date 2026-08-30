#!/usr/bin/env python3
"""Analyze the repaired SmolLM2 group-normalized reward-source gate."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path("/home/ajinkya/pgr")
ARMS = {
    name: ROOT / "checkpoints" / f"fixsrc_gn_{name}_n500.json"
    for name in ("gold", "random", "majority")
}


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [center - half, center + half]


def two_prop(a: dict, b: dict) -> dict:
    k1, n1 = a["correct"], a["n"]
    k2, n2 = b["correct"], b["n"]
    p1, p2 = k1 / n1, k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se else 0.0
    return {
        "difference": p1 - p2,
        "z": z,
        "p_two_sided": math.erfc(abs(z) / math.sqrt(2)),
    }


def main() -> None:
    arms = {}
    for name, path in ARMS.items():
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        arms[name] = {
            "correct": data["correct"],
            "n": data["n"],
            "accuracy": data["accuracy"],
            "ci95": wilson(data["correct"], data["n"]),
            "avg_gen_tokens": data["avg_gen_tokens"],
        }

    result = {
        "experiment": "fixed reward-source GRPO gate",
        "trainer_sha256": "5fa28045079fe53448a4bfc7f589c841a488f923c750c212546b63265246b273",
        "config": {
            "model": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "dataset": "openai/gsm8k",
            "steps": 1000,
            "k": 4,
            "seed": 42,
            "learning_rate": 2e-5,
            "step_advantage_mode": "group_mean",
            "terminal_spread": "constant",
            "alpha": 0.0,
        },
        "arms": arms,
    }
    if "gold" in arms and "random" in arms:
        result["gold_minus_random"] = two_prop(arms["gold"], arms["random"])
        result["instrument_alive_3pp_gate"] = (
            arms["random"]["accuracy"] < arms["gold"]["accuracy"] - 0.03
        )
    if "majority" in arms and "random" in arms:
        result["majority_minus_random"] = two_prop(arms["majority"], arms["random"])
    if "majority" in arms and "gold" in arms:
        result["majority_minus_gold"] = two_prop(arms["majority"], arms["gold"])

    output = ROOT / "checkpoints" / "fixed_source_gate_analysis.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
