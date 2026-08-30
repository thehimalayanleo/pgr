#!/usr/bin/env python3

from fcc_reward import build_frozen_target, extract_answer, score_current_rollouts


def solution(answer: int, wording: str) -> str:
    return f"{wording}\nTherefore the result is \\boxed{{{answer}}}."


def test_accepts_two_independent_panels() -> None:
    panel_a = [
        solution(18, "Add the first pair, then include the remainder."),
        solution(18, "Combining all quantities gives the total."),
        solution(21, "A mistaken multiplication gives this."),
        solution(7, "An unrelated route gives this."),
    ]
    panel_b = [
        solution(18, "Summing the components independently yields the answer."),
        solution(18, "A second derivation reaches the same total."),
        solution(17, "This route drops one unit."),
        solution(9, "This route uses the wrong operation."),
    ]
    target = build_frozen_target(panel_a, panel_b)
    assert target.accepted
    assert target.target == "18"
    rewards = score_current_rollouts(
        target,
        [solution(18, "current correct route"), solution(20, "current wrong route")],
    )
    assert rewards == [1.0, 0.0]


def test_disagreement_abstains() -> None:
    panel_a = [
        solution(18, "path a"),
        solution(18, "path b"),
        solution(7, "path c"),
        solution(9, "path d"),
    ]
    panel_b = [
        solution(20, "route one"),
        solution(20, "route two"),
        solution(8, "route three"),
        solution(10, "route four"),
    ]
    target = build_frozen_target(panel_a, panel_b)
    assert not target.accepted
    assert target.reason == "panels_disagree"
    assert score_current_rollouts(target, [solution(18, "candidate")]) == [0.0]


def test_exact_clone_support_is_rejected() -> None:
    clone = solution(18, "identical derivation")
    panel_a = [clone, clone, solution(7, "other"), solution(9, "other again")]
    panel_b = [
        solution(18, "independent route one"),
        solution(18, "independent route two"),
        solution(7, "other route"),
        solution(9, "last route"),
    ]
    target = build_frozen_target(panel_a, panel_b)
    assert not target.accepted
    assert target.reason == "panel_a_weak"


def test_current_collapse_cannot_move_target() -> None:
    panel_a = [
        solution(18, "anchor route one"),
        solution(18, "anchor route two"),
        solution(7, "anchor error one"),
        solution(9, "anchor error two"),
    ]
    panel_b = [
        solution(18, "second anchor route one"),
        solution(18, "second anchor route two"),
        solution(6, "second error one"),
        solution(5, "second error two"),
    ]
    target = build_frozen_target(panel_a, panel_b)
    wrong_clone = solution(999, "current policy collapsed here")
    assert score_current_rollouts(target, [wrong_clone] * 8) == [0.0] * 8
    assert target.target == "18"


def test_currency_comma_percent_and_unit_canonicalization() -> None:
    assert extract_answer(r"The result is \boxed{\$18,000}.") == "18000"
    assert extract_answer(r"The result is \boxed{18,000 dollars}.") == "18000"
    assert extract_answer(r"The result is \boxed{25\%}.") == "25"
    assert extract_answer(r"The result is \boxed{460}.") == "460"


def test_choice_extraction_requires_an_explicit_answer_cue() -> None:
    assert extract_answer(r"Final answer: \boxed{c}.", "choice") == "C"
    assert extract_answer(
        r"Initially \boxed{A}, but after checking, final answer: \boxed{D}.",
        "choice",
    ) == "D"
    assert extract_answer("After checking, the answer is option B.", "choice") == "B"
    assert extract_answer("A cat and a dog appear in the explanation.", "choice") is None
    assert extract_answer("Consider possibilities A through D.", "choice") is None


def test_choice_panels_build_and_score_frozen_targets() -> None:
    panel_a = [
        r"First independent argument. Final answer: \boxed{B}.",
        r"Second distinct derivation. The answer is B.",
        r"Alternative says A. Final answer: \boxed{A}.",
        r"Alternative says D. Final answer: \boxed{D}.",
    ]
    panel_b = [
        r"Elimination leaves option B as the answer.",
        r"A separate analysis gives \boxed{B}.",
        r"This route picks C. Final answer: \boxed{C}.",
        r"This route picks D. Final answer: \boxed{D}.",
    ]
    target = build_frozen_target(panel_a, panel_b, answer_mode="choice")
    assert target.accepted
    assert target.target == "B"
    assert score_current_rollouts(
        target,
        [r"Final answer: \boxed{B}.", r"Final answer: \boxed{A}."],
        answer_mode="choice",
    ) == [1.0, 0.0]
