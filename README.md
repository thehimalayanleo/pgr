# PGR research audit

This repository tested whether sparse pursuit over frozen step embeddings could
provide verifier-free rewards for reasoning-model RL.

## Final status

The original OMP reward hypothesis is **rejected in the tested setting**.
Frozen-encoder OMP reconstruction scores did not grade GSM8K reasoning quality,
and the apparent success-dictionary and contrastive results did not survive
leak-free evaluation. A frozen-reference successor, RAPR, had modest correctness
discrimination but failed its candidate-selection gate. We therefore do not claim
that OMP rewards improve policy learning.

The complete decision record is in
[`OMP_RL_REWARD_FINAL_AUDIT_2026-08-21.md`](OMP_RL_REWARD_FINAL_AUDIT_2026-08-21.md).

## What survived

| Component | Result | Status |
|---|---|---|
| Single-step OMP residual | Spearman rho `0.022`; best-of-K `0.200` | Negative |
| Prefix OMP, LOO likelihood, pair features | absolute Spearman rho at most `0.108` | Negative |
| Success-dictionary OMP | Cross-fitted AUROC `0.426` | Retracted due to label leakage |
| Contrastive step reward | Leave-one-out AUROC `0.443` | Retracted due to label leakage |
| RAPR frozen-reference OMP | AUROC `0.748`, but selection `0.160` versus random `0.147`, CI `[-0.067, 0.093]` | Failed promotion |
| Trainer reward plumbing | Reward-source and all-token advantage invariants pass | Engineering result only |

Nonzero gradients, reward variance, and completed optimizer steps show that a
reward reaches the loss. They do not show that the reward points toward better
answers.

## Separate positive result

Frozen Cross-Evidence GRPO, or FCE-GRPO, is a later verifier-free method in this
repository. It uses immutable cross-panel answer evidence, not OMP or frozen
embedding reconstruction. On the registered SmolLM2/GSM8K run, it improved
held-out accuracy from `29.6%` to `51.4%`; an exact-reward-multiset permutation
control reached `33.6%`. See
[`experiments/frozen_cross_consensus/RESULTS_2026-07-29.md`](experiments/frozen_cross_consensus/RESULTS_2026-07-29.md).

FCE-GRPO does not establish cross-model policy learning, non-numeric transfer,
step-local credit, or superiority to gold GRPO.

## Reproduce the audits

The principal saved artifacts are:

- `offline_test_pairwise_n50.json`
- `loo_audit_results.json`
- `experiments/reference_anchor/rapr_offline_results.json`
- `experiments/reward_source_fix/AUDIT_2026-07-29.md`
- `experiments/frozen_cross_consensus/RESULTS_2026-07-29.md`

Run the trainer's deterministic plumbing checks with:

```bash
python step_pgr_trainer.py
```

The historical workshop draft is retained as `PAPER_DRAFT.md`, but it is marked
superseded because its positive OMP and contrastive claims predate the leakage and
reward-source audits.
