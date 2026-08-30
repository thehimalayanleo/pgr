#!/usr/bin/env python3

from fce_data import prepare_training_example


class GoldAccessForbidden(dict):
    def __getitem__(self, key):
        if key in {"answer", "solution"}:
            raise AssertionError(f"verifier-free preprocessing accessed {key}")
        return super().__getitem__(key)


def test_frozen_reward_preprocessing_never_reads_gold() -> None:
    gsm8k = GoldAccessForbidden(
        question="A box has 2 red and 3 blue balls.",
        answer="This field must remain unread.",
    )
    expected_prompt = (
        "Solve step by step:\n"
        "A box has 2 red and 3 blue balls.\n\n"
        "Solution:"
    )
    for source in (
        "fcc",
        "fce",
        "fce_permuted",
        "majority",
        "random",
        "consensus",
    ):
        prepared = prepare_training_example(
            gsm8k,
            dataset_name="openai/gsm8k",
            reward_source=source,
        )
        assert prepared == {"prompt": expected_prompt, "answer": ""}

    math_hard = GoldAccessForbidden(
        problem="Compute 2+3.",
        solution="This field must remain unread.",
    )
    prepared = prepare_training_example(
        math_hard,
        dataset_name="lighteval/MATH-Hard",
        reward_source="fce",
    )
    assert prepared == {
        "prompt": "Solve step by step:\nCompute 2+3.\n\nSolution:",
        "answer": "",
    }

    mmlu = GoldAccessForbidden(
        question="Which planet is known as the Red Planet?",
        choices=["Venus", "Mars", "Jupiter", "Mercury"],
        answer=1,
    )
    prepared = prepare_training_example(
        mmlu,
        dataset_name="cais/mmlu",
        reward_source="fce",
    )
    assert prepared["answer"] == ""
    assert "A. Venus" in prepared["prompt"]
    assert "B. Mars" in prepared["prompt"]
    assert "Final answer:" in prepared["prompt"]
