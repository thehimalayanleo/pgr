# OMP RL reward final audit

Date: 2026-08-21

## Verdict

The frozen-encoder OMP reward line is closed as a negative result for the tested
reasoning setting. The implementation can create dense rewards and valid policy
gradients, but the reward does not reliably identify better reasoning. No OMP RL
quality or verifier-free policy-learning claim is supported.

This decision is based on the frozen artifacts below. It does not depend on the
unfinished G1 constant-terminal attribution ablation.

## Evidence ladder

### 1. Direct OMP step rewards failed the quality gate

The fixed 150-rollout GSM8K panel compared five frozen-encoder formulations.

| Reward | Spearman rho with correctness | Best-of-K accuracy |
|---|---:|---:|
| Single-step OMP residual | `0.022` | `0.200` |
| Prefix-conditioned OMP | `-0.045` | `0.160` |
| Leave-one-out policy likelihood | `-0.108` | `0.160` |
| Pair concatenation OMP | `-0.003` | `0.120` |
| Pair-difference OMP | `0.071` | `0.160` |

Random candidate accuracy was `0.147`; the oracle best-of-K ceiling was `0.320`.
None of the tested signals approached the oracle ceiling or showed a robust
relationship with correctness.

Artifact: `offline_test_pairwise_n50.json`

### 2. The strongest apparent OMP result was label leakage

Success-dictionary OMP originally scored the same successful rollouts used to
construct its dictionary. That self-inclusion encoded the outcome label in the
reference set and produced an artificial AUROC of `1.0`.

Under a cross-fitted dictionary, success-dictionary OMP fell to AUROC `0.426` and
Spearman `-0.091`. The leave-one-out contrastive score reached AUROC `0.443`.
Both are retained as negative results, and their earlier positive claims are
retracted.

Artifact: `loo_audit_results.json`

### 3. Frozen-reference OMP did not rescue selection

RAPR prevented the current policy from rewriting its reward target. It achieved
AUROC `0.748` on 150 cached rollouts, but its load-bearing candidate-selection
result failed:

- random pick accuracy: `0.147`
- RAPR pick accuracy: `0.160`
- method minus random: `+0.013`
- paired 95% interval: `[-0.067, +0.093]`
- one-sided `p(gain <= 0)`: `0.383`
- oracle gain captured: `7.69%`

RAPR also selected in all 50 groups rather than using its intended abstention
mechanism. Online training was correctly not promoted.

Artifact: `experiments/reference_anchor/rapr_offline_results.json`

### 4. Training activity was not a quality result

Historical runs reported nonzero gradient norms, reward variance, and occasional
training-time success. Those measurements show that the reward reached the loss.
They do not establish that OMP credit improved held-out policy quality.

Several historical majority, random, and consensus arms were also invalidated by
a reward-source control-path bug. Their diagnostic terminal changed, but the loss
still consumed gold-derived step rewards. The bug was repaired, and deterministic
tests now prove that the selected reward source rebuilds the arrays consumed by
the loss.

Artifact: `experiments/reward_source_fix/AUDIT_2026-07-29.md`

## Separate result: FCE-GRPO

FCE-GRPO is the strongest verifier-free RL result in the repository, but it is
not an OMP reward. It freezes cross-panel answer evidence from the initial policy.
In the registered SmolLM2/GSM8K experiment:

- base accuracy: `29.6%`
- exact-reward-multiset permutation control: `33.6%`
- FCE-GRPO: `51.4%`
- FCE minus control: `+17.8` percentage points
- paired 95% interval: `[+13.2, +22.4]`

Full policy learning remains single-model, single-seed, and limited to
canonicalizable numeric answers.

Artifact: `experiments/frozen_cross_consensus/RESULTS_2026-07-29.md`

## Decision

Do not spend more GPU time on the current OMP reconstruction-reward family.
The optional G1 experiment can attribute an old leaked L2a training effect to
constant terminal credit, but it cannot make OMP a valid reward.

Reopen the OMP line only if a new, cross-fitted signal passes all of these gates
before online training:

1. no gold labels, correctness-conditioned references, or self-inclusion in the
   scoring path;
2. positive candidate-selection gain with a paired 95% lower bound above zero;
3. replication on two model families and two task domains;
4. an online comparison against base, gold GRPO, and an exact-reward-multiset
   permutation control;
5. untouched held-out evaluation with frozen hashes and promotion thresholds.

## Integrity record

SHA-256 values checked on 2026-08-21:

```text
248ad50f42b3335adf3cd18a7b13f6ca656ee6bfbaf28904e0153d6e5c42a964  offline_test_pairwise_n50.json
8bbeb04c032d0e3b0d468c7f7d2611f1d66cc1caca09b73284cf29d01f5e45a3  loo_audit_results.json
2d53bc3eb1f9a6c0cd2a9087d0c91a2996ab7ca3b60c8ecf77e2dc2c2360fdde  experiments/reference_anchor/rapr_offline_results.json
6d922d3aa714ea462ab86f5c78eef2e3925f52ac63782c68ba7b8f3c707677b8  experiments/reward_source_fix/AUDIT_2026-07-29.md
8ba5dfb94ea49f856fe199dfe0aa605cdcb7797a22347abf45e4589c7a01cd8f  experiments/frozen_cross_consensus/RESULTS_2026-07-29.md
380cb2a60a5b3ed226998765567fe15038b653211e7d27893e7cd4d64e87de5d  step_pgr_trainer.py
```

The remote trainer self-test passed reward-source rewiring, full-token constant
advantage coverage, answer canonicalization, causal credit, length normalization,
clipping, and top-k filtering. The focused FCE reward and matched-control tests
also passed. No GPU training was started because the 5090 was occupied by an
unrelated active experiment.
