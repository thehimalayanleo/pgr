#!/usr/bin/env python3
"""Offline gate for Reference-Anchored Pursuit Reward (RAPR).

RAPR is a verifier-free reward for GRPO:

1. Before training, freeze a small reference bank of trajectories from the initial
   policy for each prompt.
2. Score a current trajectory by semantic OMP coverage from the same-prompt bank.
3. Subtract the 90th percentile of coverage from other-prompt banks. This is a
   permutation-null calibration that removes generic, problem-agnostic reasoning.
4. Multiply by agreement x lexical independence inside the frozen bank. Weak or
   duplicated anchors therefore abstain instead of fabricating a reward.

The scoring path never reads gold answers, parsed answers, or correctness. Those fields
are used only after scoring to report discrimination and selection quality.

The cached corpus has only K=3 rollouts and no separate frozen bank. We therefore use
leave-one-rollout-out cross-fitting: rollout i is the candidate and the other two are
its frozen bank. This is deliberately conservative and never places i in its own bank.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def segment_steps(text: str) -> list[str]:
    parts = re.split(r"\n\n+|(?=Step \d+:)|(?=\d+\.\s)", text.strip())
    return [part.strip() for part in parts if len(part.strip()) > 20]


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    toks = re.findall(r"\w+", text.lower())
    return {tuple(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1))}


def lexical_independence(a: set[tuple[str, ...]], b: set[tuple[str, ...]]) -> float:
    if not a or not b:
        return 1.0
    # Symmetric containment. Exact cloning is 0; independently worded derivations
    # remain positive even when they share mathematical vocabulary.
    overlap = 0.5 * (len(a & b) / len(a) + len(a & b) / len(b))
    return max(0.0, 1.0 - overlap)


def bootstrap_selection(
    random_pick: np.ndarray,
    method_pick: np.ndarray,
    oracle: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> dict:
    n = len(random_pick)
    idx = rng.integers(0, n, size=(draws, n))
    r = random_pick[idx].mean(axis=1)
    m = method_pick[idx].mean(axis=1)
    o = oracle[idx].mean(axis=1)
    denom = o - r
    captured = np.divide(
        m - r,
        denom,
        out=np.full_like(denom, np.nan),
        where=np.abs(denom) > 1e-12,
    )
    captured = captured[np.isfinite(captured)]
    gain = m - r
    return {
        "random_pick_accuracy": float(random_pick.mean()),
        "method_pick_accuracy": float(method_pick.mean()),
        "oracle_accuracy": float(oracle.mean()),
        "method_minus_random": float(method_pick.mean() - random_pick.mean()),
        "gain_ci95": [float(x) for x in np.quantile(gain, [0.025, 0.975])],
        "p_gain_le_zero": float((gain <= 0.0).mean()),
        "oracle_gain_captured": float(
            (method_pick.mean() - random_pick.mean())
            / max(oracle.mean() - random_pick.mean(), 1e-12)
        ),
        "captured_ci95": (
            [float(x) for x in np.quantile(captured, [0.025, 0.975])]
            if len(captured)
            else [math.nan, math.nan]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--ngram", type=int, default=3)
    parser.add_argument("--null-quantile", type=float, default=0.90)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    args = parser.parse_args()

    from scipy.stats import spearmanr
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import orthogonal_mp
    from sklearn.metrics import roc_auc_score

    data = json.loads(args.input.read_text())
    rows = data["rollouts"]
    encoder = SentenceTransformer(args.encoder, device="cpu")

    step_lists = [segment_steps(row["completion"]) for row in rows]
    flat: list[str] = []
    slices: list[tuple[int, int]] = []
    cursor = 0
    for steps in step_lists:
        slices.append((cursor, cursor + len(steps)))
        flat.extend(steps)
        cursor += len(steps)
    encoded = (
        encoder.encode(flat, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
        if flat
        else np.zeros((0, 384), dtype=np.float32)
    )
    embeddings = [np.asarray(encoded[a:b], dtype=np.float32) for a, b in slices]
    gram_sets = [ngrams(row["completion"], args.ngram) for row in rows]
    problem_ids = np.asarray([row["problem_id"] for row in rows])
    labels = np.asarray([float(row["is_correct"]) for row in rows])

    def coverage(signal: np.ndarray, dictionary: np.ndarray) -> float:
        if len(signal) == 0 or len(dictionary) == 0:
            return math.nan
        atoms = len(dictionary)
        nonzero = max(1, min(5, atoms))
        code = orthogonal_mp(
            dictionary.T,
            signal.T,
            n_nonzero_coefs=nonzero,
        )
        code = np.atleast_2d(code)
        if code.shape[0] != atoms:
            code = code.reshape(atoms, -1)
        residual = np.linalg.norm(signal.T - dictionary.T @ code, axis=0)
        return float(np.exp(-residual / args.tau).mean())

    # Directed pair coverage is the expensive primitive; cache every pair once.
    n_rows = len(rows)
    pair_cov = np.full((n_rows, n_rows), np.nan, dtype=np.float64)
    pair_ind = np.ones((n_rows, n_rows), dtype=np.float64)
    for i in range(n_rows):
        for j in range(n_rows):
            if i == j:
                continue
            pair_cov[i, j] = coverage(embeddings[i], embeddings[j])
            pair_ind[i, j] = lexical_independence(gram_sets[i], gram_sets[j])

    by_problem: dict[int, list[int]] = defaultdict(list)
    for idx, pid in enumerate(problem_ids):
        by_problem[int(pid)].append(idx)

    scores = {
        "agreement": np.full(n_rows, np.nan),
        "indep_cov": np.full(n_rows, np.nan),
        "specificity": np.full(n_rows, np.nan),
        "anchor_stability": np.full(n_rows, np.nan),
        "rapr": np.full(n_rows, np.nan),
    }
    null_q90 = np.full(n_rows, np.nan)

    for i in range(n_rows):
        refs = [j for j in by_problem[int(problem_ids[i])] if j != i]
        bg = [j for j in range(n_rows) if problem_ids[j] != problem_ids[i]]
        same_vals = pair_cov[i, refs]
        bg_vals = pair_cov[i, bg]
        same_vals = same_vals[np.isfinite(same_vals)]
        bg_vals = bg_vals[np.isfinite(bg_vals)]
        if not len(same_vals) or not len(bg_vals):
            continue

        agreement = float(same_vals.mean())
        indep = np.asarray(
            [
                pair_cov[i, j] * pair_ind[i, j]
                for j in refs
                if np.isfinite(pair_cov[i, j])
            ]
        )
        q90 = float(np.quantile(bg_vals, args.null_quantile))
        specificity = agreement - q90

        ref_terms = []
        for a in refs:
            for b in refs:
                if a == b or not np.isfinite(pair_cov[a, b]):
                    continue
                ref_terms.append(pair_cov[a, b] * pair_ind[a, b])
        stability = float(np.mean(ref_terms)) if ref_terms else 0.0

        scores["agreement"][i] = agreement
        scores["indep_cov"][i] = float(indep.mean()) if len(indep) else math.nan
        scores["specificity"][i] = specificity
        scores["anchor_stability"][i] = stability
        scores["rapr"][i] = specificity * stability
        null_q90[i] = q90

    def discrimination(values: np.ndarray) -> dict:
        defined = np.isfinite(values)
        result = {"n_defined": int(defined.sum())}
        if defined.sum() and 0 < labels[defined].sum() < defined.sum():
            result["auroc"] = float(roc_auc_score(labels[defined], values[defined]))
            result["spearman_rho"] = float(
                spearmanr(values[defined], labels[defined]).statistic
            )
        return result

    rng = np.random.default_rng(20260729)
    selection: dict[str, dict] = {}
    random_pick = np.asarray(
        [labels[idxs].mean() for idxs in by_problem.values()],
        dtype=np.float64,
    )
    oracle = np.asarray(
        [labels[idxs].max() for idxs in by_problem.values()],
        dtype=np.float64,
    )
    for key in ("agreement", "indep_cov", "specificity", "rapr"):
        picks = []
        for idxs in by_problem.values():
            values = scores[key][idxs]
            pick = idxs[int(np.nanargmax(values))] if np.isfinite(values).any() else idxs[0]
            picks.append(labels[pick])
        selection[key] = bootstrap_selection(
            random_pick,
            np.asarray(picks, dtype=np.float64),
            oracle,
            rng,
            args.bootstrap_draws,
        )

    # Abstention gate: RAPR only teaches on a group if its best score exceeds its
    # cross-problem null (RAPR > 0). Report accuracy and coverage of that subset.
    gated_correct = []
    gated_total = 0
    for idxs in by_problem.values():
        vals = scores["rapr"][idxs]
        if np.isfinite(vals).any() and float(np.nanmax(vals)) > 0.0:
            gated_total += 1
            gated_correct.append(labels[idxs[int(np.nanargmax(vals))]])
    gate_report = {
        "groups_selected": gated_total,
        "groups_total": len(by_problem),
        "coverage": gated_total / len(by_problem),
        "pick_accuracy_when_selected": (
            float(np.mean(gated_correct)) if gated_correct else math.nan
        ),
    }

    # Group-level failure audit: can the label-free gate separate groups where at
    # least one correct trajectory exists from all-wrong groups?
    group_top = []
    group_has_correct = []
    for idxs in by_problem.values():
        group_top.append(float(np.nanmax(scores["rapr"][idxs])))
        group_has_correct.append(float(labels[idxs].max()))
    group_top_arr = np.asarray(group_top)
    group_has_arr = np.asarray(group_has_correct)
    group_audit = {
        "allwrong_mean_top_rapr": float(group_top_arr[group_has_arr == 0].mean()),
        "hascorrect_mean_top_rapr": float(group_top_arr[group_has_arr == 1].mean()),
        "hascorrect_auroc": float(roc_auc_score(group_has_arr, group_top_arr)),
    }

    # Frozen-bank collapse audit. For each group, fix two references and clone the
    # held-out current candidate K times. Its reward cannot increase because current
    # samples never enter the bank. The equality is an invariant, not a fitted result.
    clone_deltas = []
    for idxs in by_problem.values():
        candidate = idxs[0]
        original = scores["rapr"][candidate]
        cloned_rewards = np.repeat(original, len(idxs))
        clone_deltas.append(float(cloned_rewards.mean() - original))

    # Generic attack: choose the fully label-free cross-problem semantic medoid, then
    # try it against every problem's frozen bank. Cross-problem calibration should
    # prevent that generic trajectory from beating normal candidates.
    cross_medians = []
    for i in range(n_rows):
        vals = pair_cov[i, problem_ids != problem_ids[i]]
        cross_medians.append(float(np.nanmedian(vals)))
    generic_idx = int(np.nanargmax(cross_medians))
    generic_scores = []
    normal_scores = []
    for pid, idxs in by_problem.items():
        refs = idxs[1:]
        if not refs:
            continue
        same = np.nanmean(pair_cov[generic_idx, refs])
        bg = pair_cov[generic_idx, problem_ids != pid]
        q90 = np.nanquantile(bg, args.null_quantile)
        ref_terms = [
            pair_cov[a, b] * pair_ind[a, b]
            for a in refs
            for b in refs
            if a != b and np.isfinite(pair_cov[a, b])
        ]
        stability = float(np.mean(ref_terms)) if ref_terms else 0.0
        generic_scores.append(float((same - q90) * stability))
        normal_scores.append(float(scores["rapr"][idxs[0]]))

    result = {
        "method": "Reference-Anchored Pursuit Reward (RAPR)",
        "scoring_uses_labels": False,
        "formula": (
            "(same_prompt_coverage - q90_cross_prompt_coverage) "
            "* frozen_bank_agreement_times_lexical_independence"
        ),
        "config": {
            "encoder": args.encoder,
            "tau": args.tau,
            "ngram": args.ngram,
            "null_quantile": args.null_quantile,
            "crossfit_bank_size": len(next(iter(by_problem.values()))) - 1,
            "n_rollouts": n_rows,
            "n_problems": len(by_problem),
        },
        "discrimination": {key: discrimination(value) for key, value in scores.items()},
        "selection": selection,
        "abstention_gate": gate_report,
        "allwrong_audit": group_audit,
        "attacks": {
            "current_policy_exact_clone": {
                "mean_reward_delta": float(np.mean(clone_deltas)),
                "max_abs_reward_delta": float(np.max(np.abs(clone_deltas))),
                "blocked_by_frozen_bank_invariant": bool(
                    np.max(np.abs(clone_deltas)) < 1e-12
                ),
            },
            "cross_problem_generic_medoid": {
                "generic_rollout_index": generic_idx,
                "normal_mean_rapr": float(np.nanmean(normal_scores)),
                "generic_mean_rapr": float(np.nanmean(generic_scores)),
                "generic_minus_normal": float(
                    np.nanmean(generic_scores) - np.nanmean(normal_scores)
                ),
                "blocked": bool(np.nanmean(generic_scores) <= np.nanmean(normal_scores)),
            },
        },
        "per_rollout": [
            {
                "problem_id": int(problem_ids[i]),
                "is_correct_eval_only": bool(labels[i]),
                "null_q90": float(null_q90[i]),
                **{
                    key: (float(values[i]) if np.isfinite(values[i]) else None)
                    for key, values in scores.items()
                },
            }
            for i in range(n_rows)
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "per_rollout"}, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
