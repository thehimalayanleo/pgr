#!/usr/bin/env python3

from fce_reward import attach_frozen_cross_evidence, score_current_rollouts


def solution(answer: int, wording: str) -> str:
    return f"{wording}\nTherefore the result is \\boxed{{{answer}}}."


def make_item(index: int, answers_a: list[int], answers_b: list[int]) -> dict:
    return {
        "index": index,
        "prompt_hash": str(index),
        "prompt": f"problem {index}",
        "panel_a": [
            solution(answer, f"panel A route {index}-{position}")
            for position, answer in enumerate(answers_a)
        ],
        "panel_b": [
            solution(answer, f"panel B derivation {index}-{position}")
            for position, answer in enumerate(answers_b)
        ],
        "candidates": [],
    }


def test_cross_panel_support_is_rewarded() -> None:
    items = [
        make_item(0, [18, 18, 7, 8], [18, 18, 6, 5]),
        make_item(1, [9, 1, 2, 3], [9, 4, 5, 6]),
    ]
    bank = attach_frozen_cross_evidence(items, panel_size=4)
    evidence = bank[0]["frozen_evidence"]
    rewards = score_current_rollouts(
        evidence,
        [solution(18, "current route"), solution(999, "wrong route")],
    )
    assert rewards[0] > 0
    assert rewards[1] == 0


def test_more_cross_support_scores_higher() -> None:
    items = [
        make_item(0, [18, 18, 7, 9], [18, 18, 6, 5]),
        make_item(1, [21, 1, 2, 3], [21, 4, 5, 6]),
    ]
    bank = attach_frozen_cross_evidence(items, panel_size=4)
    first = bank[0]["frozen_evidence"]["scores"]["18"]
    second = bank[1]["frozen_evidence"]["scores"]["21"]
    assert first > second


def test_global_generic_answer_is_downweighted() -> None:
    items = [
        make_item(0, [0, 0, 18, 7], [0, 0, 18, 6]),
        make_item(1, [0, 0, 21, 8], [0, 0, 21, 5]),
        make_item(2, [0, 0, 24, 9], [0, 0, 24, 4]),
    ]
    bank = attach_frozen_cross_evidence(items, panel_size=4)
    assert "0" not in bank[0]["frozen_evidence"]["scores"]
    assert bank[0]["frozen_evidence"]["scores"]["18"] > 0


def test_current_collapse_cannot_change_frozen_scores() -> None:
    items = [
        make_item(0, [18, 18, 7, 9], [18, 18, 6, 5]),
        make_item(1, [21, 1, 2, 3], [21, 4, 5, 6]),
    ]
    bank = attach_frozen_cross_evidence(items, panel_size=4)
    evidence = bank[0]["frozen_evidence"]
    before = dict(evidence["scores"])
    assert score_current_rollouts(
        evidence,
        [solution(999, "collapsed clone")] * 8,
    ) == [0.0] * 8
    assert evidence["scores"] == before


def test_choice_evidence_uses_choice_parser() -> None:
    items = [
        {
            "panel_a": [
                "Route one. Final answer: B.",
                r"Route two. \boxed{B}.",
                "Wrong route. Final answer: A.",
                "Other route. Final answer: D.",
            ],
            "panel_b": [
                "Independent reasoning says option B.",
                r"Separate derivation gives \boxed{B}.",
                "Wrong route. Final answer: C.",
                "Other route. Final answer: D.",
            ],
        },
        {
            "panel_a": [
                "First says B. Final answer: B.",
                "Second distinct route. Final answer: B.",
                "Third independent route. Final answer: B.",
                "Fourth separate route. Final answer: B.",
            ],
            "panel_b": [
                "One says B. Final answer: B.",
                "Two independently says B. Final answer: B.",
                "Three separately says B. Final answer: B.",
                "Four also derives B. Final answer: B.",
            ],
        },
    ]
    enriched = attach_frozen_cross_evidence(
        items,
        panel_size=4,
        answer_mode="choice",
    )
    evidence = enriched[0]["frozen_evidence"]
    assert evidence["answer_mode"] == "choice"
    assert evidence["formula"] == "sqrt(freq_a*freq_b)"
    assert evidence["scores"]["B"] > 0
    rewards = score_current_rollouts(
        evidence,
        [r"Final answer: \boxed{B}.", r"Final answer: \boxed{A}."],
        answer_mode="choice",
    )
    assert rewards[0] > 0
    assert rewards[1] == 0
