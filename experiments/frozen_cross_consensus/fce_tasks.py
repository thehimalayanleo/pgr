#!/usr/bin/env python3
"""Task adapters for label-free FCE bank construction and gold-separated eval."""

from __future__ import annotations

from dataclasses import dataclass

from fcc_reward import extract_answer, normalize_answer


@dataclass(frozen=True)
class TaskSpec:
    dataset: str
    config: str | None
    train_split: str
    eval_split: str
    answer_mode: str


_TASKS = {
    "openai/gsm8k": TaskSpec(
        "openai/gsm8k", "main", "train", "test", "numeric"
    ),
    "lighteval/MATH-Hard": TaskSpec(
        "lighteval/MATH-Hard", None, "train", "test", "math"
    ),
    "cais/mmlu": TaskSpec(
        "cais/mmlu", "all", "auxiliary_train", "test", "choice"
    ),
}


def task_spec(dataset_name: str) -> TaskSpec:
    try:
        return _TASKS[dataset_name]
    except KeyError as exc:
        raise ValueError(f"unsupported FCE dataset: {dataset_name}") from exc


def load_task_dataset(dataset_name: str, split: str | None = None):
    from datasets import load_dataset

    spec = task_spec(dataset_name)
    selected_split = split or spec.train_split
    if spec.config is None:
        return load_dataset(spec.dataset, split=selected_split)
    return load_dataset(spec.dataset, spec.config, split=selected_split)


def prompt_for_example(example: dict, dataset_name: str) -> str:
    if dataset_name == "openai/gsm8k":
        # This byte-exact prompt is part of the original GSM8K bank contract.
        return f"Solve step by step:\n{example['question']}\n\nSolution:"
    if dataset_name == "lighteval/MATH-Hard":
        return f"Solve step by step:\n{example['problem']}\n\nSolution:"
    if dataset_name == "cais/mmlu":
        choices = example["choices"]
        rendered = "\n".join(
            f"{letter}. {choice}"
            for letter, choice in zip("ABCD", choices)
        )
        return (
            f"Answer the multiple-choice question. Explain your reasoning.\n"
            f"{example['question']}\n{rendered}\n\n"
            "End with Final answer: \\boxed{A}, \\boxed{B}, "
            "\\boxed{C}, or \\boxed{D}.\n\nSolution:"
        )
    raise ValueError(f"unsupported FCE dataset: {dataset_name}")


def gold_for_example(example: dict, dataset_name: str) -> str | None:
    """Read gold only in explicit supervised/evaluation paths."""
    if dataset_name == "openai/gsm8k":
        raw = example["answer"]
        if "####" not in raw:
            return extract_answer(raw, "numeric")
        return normalize_answer(raw.split("####")[-1])
    if dataset_name == "lighteval/MATH-Hard":
        return extract_answer(example["solution"], "math")
    if dataset_name == "cais/mmlu":
        answer = example["answer"]
        if isinstance(answer, int):
            return "ABCD"[answer]
        text = str(answer).strip().upper()
        if text.isdigit() and 0 <= int(text) < 4:
            return "ABCD"[int(text)]
        return text if text in set("ABCD") else None
    raise ValueError(f"unsupported FCE dataset: {dataset_name}")
