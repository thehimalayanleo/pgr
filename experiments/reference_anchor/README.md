# Reference-Anchored Pursuit Reward (RAPR)

RAPR is a verifier-free GRPO reward that does not let the current policy define its own
target.

For each prompt, sample a small trajectory bank from the initial policy and freeze it for
the full run. A current rollout earns reward for semantic OMP coverage by that same-prompt
bank, minus a high-quantile cross-prompt coverage null. The residual is multiplied by
agreement and lexical independence inside the frozen bank. Weak, duplicated, or generic
anchors abstain.

This differs from on-policy majority and `indep_cov` in one load-bearing way: current
rollouts never enter the reference dictionary. Making the current group collapse therefore
cannot increase reward by changing the target.

`reference_anchor_offline.py` performs a label-free cross-fitted evaluation on the cached
GSM8K corpus and reports discrimination, best-of-K selection, all-wrong-group behavior,
exact-clone resistance, and a generic-text attack. Gold correctness is evaluation-only.
