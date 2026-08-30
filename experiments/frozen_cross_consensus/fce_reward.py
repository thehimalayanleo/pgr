#!/usr/bin/env python3
"""Pure Frozen Cross-Evidence reward construction.

FCE assigns a frozen score to every answer independently observed in both initial-policy
panels. Scores use geometric cross-panel support and an across-prompt inverse-frequency
penalty. Current-policy samples can query the scores but cannot modify them.
"""

from __future__ import annotations

import math
from collections import Counter
from copy import deepcopy
from typing import Iterable

from fcc_reward import extract_answer, independent_support_count


def _panel_support(
    completions: Iterable[str],
    *,
    answer_mode: str,
) -> tuple[Counter, dict[str, list[str]]]:
    counts: Counter = Counter()
    paths: dict[str, list[str]] = {}
    for completion in completions:
        answer = extract_answer(completion, answer_mode)
        if answer is None:
            continue
        counts[answer] += 1
        paths.setdefault(answer, []).append(completion)
    return counts, paths


def attach_frozen_cross_evidence(
    items: list[dict],
    *,
    panel_size: int,
    overlap_ceiling: float = 0.90,
    answer_mode: str = "numeric",
) -> list[dict]:
    """Return bank items enriched with label-free, globally calibrated FCE scores."""
    if panel_size <= 0:
        raise ValueError("panel_size must be positive")

    prepared: list[tuple[Counter, Counter, dict[str, list[str]], dict[str, list[str]]]] = []
    document_frequency: Counter = Counter()
    for item in items:
        counts_a, paths_a = _panel_support(
            item["panel_a"],
            answer_mode=answer_mode,
        )
        counts_b, paths_b = _panel_support(
            item["panel_b"],
            answer_mode=answer_mode,
        )
        cross_answers = set(counts_a) & set(counts_b)
        document_frequency.update(cross_answers)
        prepared.append((counts_a, counts_b, paths_a, paths_b))

    n_prompts = len(items)
    normalizer = math.log(n_prompts + 1.0) if n_prompts else 1.0
    enriched = deepcopy(items)
    for item, (counts_a, counts_b, paths_a, paths_b) in zip(enriched, prepared):
        scores: dict[str, float] = {}
        details: dict[str, dict] = {}
        for answer in sorted(set(counts_a) & set(counts_b)):
            supporting_paths = paths_a[answer] + paths_b[answer]
            independent_paths = independent_support_count(
                supporting_paths,
                overlap_ceiling=overlap_ceiling,
            )
            # A repeated clone across the two panels is not independent evidence.
            if independent_paths < 2:
                continue
            cross_support = math.sqrt(
                (counts_a[answer] / panel_size)
                * (counts_b[answer] / panel_size)
            )
            # IDF suppresses generic numeric collapse modes such as answering
            # "0" everywhere. It is invalid for a fixed multiple-choice
            # vocabulary: A-D are intentionally reused across every prompt, so
            # rarity would measure answer position rather than evidence.
            idf = (
                1.0
                if answer_mode == "choice"
                else (
                    math.log(
                        (n_prompts + 1.0)
                        / (document_frequency[answer] + 1.0)
                    )
                    / normalizer
                    if n_prompts
                    else 0.0
                )
            )
            score = cross_support * idf
            if score <= 0:
                continue
            scores[answer] = score
            details[answer] = {
                "support_a": counts_a[answer],
                "support_b": counts_b[answer],
                "independent_paths": independent_paths,
                "document_frequency": document_frequency[answer],
                "cross_support": cross_support,
                "idf": idf,
                "score": score,
            }
        item["frozen_evidence"] = {
            "scores": scores,
            "details": details,
            "accepted": bool(scores),
            "formula": (
                "sqrt(freq_a*freq_b)"
                if answer_mode == "choice"
                else "sqrt(freq_a*freq_b)*normalized_idf"
            ),
            "answer_mode": answer_mode,
            "gold_used": False,
        }
    return enriched


def score_current_rollouts(
    evidence: dict,
    current_completions: Iterable[str],
    *,
    answer_mode: str = "numeric",
) -> list[float]:
    """Look up immutable FCE rewards for current-policy answers."""
    scores = evidence.get("scores", {})
    return [
        float(scores.get(extract_answer(completion, answer_mode), 0.0))
        for completion in current_completions
    ]
