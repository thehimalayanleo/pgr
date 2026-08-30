# Reward-source loss-path audit

The July verifier-free sweeps changed `per_rollout_terminals` after
`per_rollout_step_rewards` had already been computed from gold. With L2a and contrastive
disabled, the non-gold terminal affected diagnostics but not the loss.

The repaired trainer separates raw OMP scoring from terminal blending, selects the desired
gold/majority/random/consensus terminals, and then rebuilds the exact reward arrays consumed
by advantage normalization. Its regression suite contains a load-bearing invariant that
swapping terminal vectors swaps the loss-side step rewards.

`run_fixed_source_gate.sh` reruns the clean SmolLM2 group-normalized instrument with unique
checkpoint names. Majority is launched only if fixed-path random trails gold by more than
three percentage points.
