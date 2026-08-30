#!/usr/bin/env python3
"""Prompt-only preprocessing for verifier-free FCE/FCC training."""

from __future__ import annotations

from fce_tasks import gold_for_example, prompt_for_example


FROZEN_REWARD_SOURCES = frozenset({"fcc", "fce", "fce_permuted"})


def prepare_training_example(
    example: dict,
    *,
    dataset_name: str,
    reward_source: str,
) -> dict[str, str]:
    """Build one training example without touching gold in frozen-reward modes."""
    verifier_free = reward_source != "gold"
    prompt = prompt_for_example(example, dataset_name)
    if verifier_free:
        return {"prompt": prompt, "answer": ""}
    gold = gold_for_example(example, dataset_name)
    return {"prompt": prompt, "answer": f"\\boxed{{{gold}}}"}
