#!/usr/bin/env python3
"""Regression tests for FCE's reward-multiset-matched attribution control."""

from __future__ import annotations

from analyze_fce_control import compute_comparison


def evaluation(correct: list[bool], *, reverse_hashes: bool = False) -> dict:
    hashes = [f"{index:064x}" for index in range(500)]
    if reverse_hashes:
        hashes.reverse()
    return {
        "offset": 500,
        "n": 500,
        "decoding": "greedy",
        "records": [
            {"prompt_hash": prompt_hash, "correct": value}
            for prompt_hash, value in zip(hashes, correct)
        ],
    }


def main() -> None:
    # The first 50 paired prompts are the only difference.
    fce_correct = [True] * 300 + [False] * 200
    control_correct = [False] * 50 + [True] * 250 + [False] * 200
    result = compute_comparison(
        evaluation(fce_correct),
        evaluation(control_correct),
        bootstrap_draws=2_000,
    )
    assert result["fce_accuracy"] == 0.6
    assert result["permuted_control_accuracy"] == 0.5
    assert result["fce_minus_control"] == 0.1
    assert result["fce_only_correct"] == 50
    assert result["control_only_correct"] == 0
    assert result["attribution_gate"]["passed"] is True

    try:
        compute_comparison(
            evaluation(fce_correct),
            evaluation(control_correct, reverse_hashes=True),
            bootstrap_draws=10,
        )
    except ValueError as error:
        assert "not paired" in str(error)
    else:
        raise AssertionError("matched-control analysis accepted unpaired prompts")

    print("matched-control regression passed")


if __name__ == "__main__":
    main()
