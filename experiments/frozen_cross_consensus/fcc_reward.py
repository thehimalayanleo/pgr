#!/usr/bin/env python3
"""Pure reward logic for Frozen Cross-Consensus GRPO (FCC-GRPO).

FCC builds pseudo-targets from two independent rollout panels sampled once from the
initial policy. A target exists only when both panels independently choose the same
answer and each panel contains enough independently worded supporting paths. The target
is then frozen for the whole GRPO run.

No gold answer, correctness label, or external verifier enters this module.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable


def normalize_answer(value: str) -> str:
    cleaned = (
        value.strip()
        .replace("\\$", "")
        .replace("$", "")
        .replace("\\,", "")
        .replace(",", "")
        .replace("\\%", "")
        .replace("%", "")
        .rstrip(".")
        .strip()
    )
    # Boxed GSM8K answers sometimes include a unit ("80 dollars"). If there is
    # exactly one numeric value, canonicalize to it. Preserve multi-number LaTeX
    # expressions rather than guessing (for example, \frac{1}{2}).
    numbers = re.findall(r"-?[0-9][0-9]*\.?[0-9]*", cleaned)
    if len(numbers) == 1:
        return numbers[0].rstrip(".")
    return cleaned


def _extract_boxed_values(text: str) -> list[str]:
    values: list[str] = []
    start = 0
    while (idx := text.find("\\boxed{", start)) != -1:
        cursor = idx + len("\\boxed{")
        depth = 1
        out: list[str] = []
        while cursor < len(text) and depth > 0:
            char = text[cursor]
            if char == "{":
                depth += 1
                out.append(char)
            elif char == "}":
                depth -= 1
                if depth > 0:
                    out.append(char)
            else:
                out.append(char)
            cursor += 1
        if depth == 0:
            values.append("".join(out))
            start = cursor
        else:
            break
    return values


def _extract_boxed(text: str) -> str | None:
    values = _extract_boxed_values(text)
    return values[0] if values else None


def normalize_choice(value: str) -> str | None:
    cleaned = value.strip()
    text_match = re.fullmatch(r"\\text\{\s*([A-Da-d])\s*\}", cleaned)
    if text_match:
        return text_match.group(1).upper()
    cleaned = cleaned.strip("()[]{}.$: ").upper()
    return cleaned if re.fullmatch(r"[A-D]", cleaned) else None


def extract_answer(text: str, answer_mode: str = "numeric") -> str | None:
    """Extract a canonical answer without consulting a verifier.

    ``numeric`` preserves the original GSM8K/MATH behavior. ``choice`` only
    accepts boxed A-D answers or an explicit answer cue, so incidental prose
    letters cannot silently become pseudo-labels.
    """
    if answer_mode in {"numeric", "math"}:
        boxed = _extract_boxed(text)
        if boxed is not None:
            return normalize_answer(boxed)
        matches = re.findall(r"####\s*(-?[0-9][0-9,]*\.?[0-9]*)", text)
        if matches:
            return normalize_answer(matches[-1])
        matches = re.findall(r"-?[0-9][0-9,]*\.?[0-9]*", text)
        return normalize_answer(matches[-1]) if matches else None

    if answer_mode == "choice":
        # Multiple-choice reasoning may mention a provisional boxed option
        # before its explicit final one. Prefer the last valid boxed choice.
        for boxed in reversed(_extract_boxed_values(text)):
            choice = normalize_choice(boxed)
            if choice is not None:
                return choice
        matches = re.findall(
            r"\b(?:final\s+answer|answer|option|choice)\s*"
            r"(?:is|:|=)?\s*(?:(?:choice|option)\s*)?\(?([A-D])\)?\b",
            text,
            flags=re.IGNORECASE,
        )
        return matches[-1].upper() if matches else None

    raise ValueError(f"unknown answer_mode: {answer_mode}")


def _word_ngrams(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {
        tuple(words[i : i + n])
        for i in range(max(0, len(words) - n + 1))
    }


def lexical_overlap(a: str, b: str, n: int = 3) -> float:
    left, right = _word_ngrams(a, n), _word_ngrams(b, n)
    if not left or not right:
        return 0.0
    return 0.5 * (
        len(left & right) / len(left)
        + len(left & right) / len(right)
    )


def independent_support_count(
    completions: list[str],
    overlap_ceiling: float,
) -> int:
    """Greedily count non-clone paths without using labels or embeddings."""
    accepted: list[str] = []
    # Longest-first avoids counting a short substring and then rejecting a fuller path.
    for completion in sorted(completions, key=len, reverse=True):
        if all(
            lexical_overlap(completion, previous) < overlap_ceiling
            for previous in accepted
        ):
            accepted.append(completion)
    return len(accepted)


@dataclass(frozen=True)
class PanelDecision:
    target: str | None
    support: int
    runner_up_support: int
    independent_paths: int
    parsed: int
    total: int
    accepted: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FrozenTarget:
    target: str | None
    panel_a: PanelDecision
    panel_b: PanelDecision
    accepted: bool
    reason: str

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "panel_a": self.panel_a.to_dict(),
            "panel_b": self.panel_b.to_dict(),
            "accepted": self.accepted,
            "reason": self.reason,
        }


def decide_panel(
    completions: Iterable[str],
    *,
    answer_mode: str = "numeric",
    min_support: int = 2,
    min_margin: int = 1,
    min_independent_paths: int = 2,
    overlap_ceiling: float = 0.90,
) -> PanelDecision:
    completion_list = list(completions)
    parsed_pairs = [
        (answer, completion)
        for completion in completion_list
        if (answer := extract_answer(completion, answer_mode)) is not None
    ]
    counts = Counter(answer for answer, _ in parsed_pairs)
    if not counts:
        return PanelDecision(
            target=None,
            support=0,
            runner_up_support=0,
            independent_paths=0,
            parsed=0,
            total=len(completion_list),
            accepted=False,
        )
    ranked = counts.most_common()
    target, support = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0
    supporting_paths = [
        completion for answer, completion in parsed_pairs if answer == target
    ]
    independent_paths = independent_support_count(
        supporting_paths,
        overlap_ceiling=overlap_ceiling,
    )
    accepted = (
        support >= min_support
        and support - runner_up >= min_margin
        and independent_paths >= min_independent_paths
    )
    return PanelDecision(
        target=target,
        support=support,
        runner_up_support=runner_up,
        independent_paths=independent_paths,
        parsed=len(parsed_pairs),
        total=len(completion_list),
        accepted=accepted,
    )


def build_frozen_target(
    panel_a: Iterable[str],
    panel_b: Iterable[str],
    **decision_kwargs,
) -> FrozenTarget:
    first = decide_panel(panel_a, **decision_kwargs)
    second = decide_panel(panel_b, **decision_kwargs)
    if not first.accepted:
        return FrozenTarget(None, first, second, False, "panel_a_weak")
    if not second.accepted:
        return FrozenTarget(None, first, second, False, "panel_b_weak")
    if first.target != second.target:
        return FrozenTarget(None, first, second, False, "panels_disagree")
    return FrozenTarget(first.target, first, second, True, "accepted")


def score_current_rollouts(
    target: FrozenTarget,
    current_completions: Iterable[str],
    *,
    answer_mode: str = "numeric",
) -> list[float]:
    """Return frozen binary rewards; current rollouts cannot modify `target`."""
    if not target.accepted or target.target is None:
        return [0.0 for _ in current_completions]
    return [
        float(extract_answer(completion, answer_mode) == target.target)
        for completion in current_completions
    ]
