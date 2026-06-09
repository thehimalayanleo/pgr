# PGR — Pursuit-Graded Reward

Dense per-step RL reward for reasoning models, built from Orthogonal Matching Pursuit. Works without a verifier.

## The Problem

Binary GRPO gives zero gradient when every rollout fails. On Level 5 MATH problems with a sub-7B model, this happens on ~100% of groups. The model generates hundreds of reasoning tokens per rollout and learns nothing.

## The Idea

Replace the binary terminal reward with a per-step quality score from sparse pursuit.

1. Collect correct solution traces, encode each step as a vector.
2. Fit a dictionary `D` of prototypical reasoning moves via dictionary learning.
3. At training time, score each rollout step by its OMP reconstruction error against `D`.
4. Combine with the terminal reward when available, or use the per-step signal alone (oracle-free).

```
For each rollout step s_i:
  e_i     = OMP_reconstruction_error(encode(s_i), D)
  r_step  = exp(-e_i / tau)

total_reward = alpha * mean(r_step) + (1 - alpha) * terminal
```

Default: `alpha=0.5`, `tau=0.3`, `terminal in {0, 1}`. Oracle-free mode sets `terminal=0` and trains on step rewards alone.

## Results

Qwen2.5-3B-Instruct on MATH-Hard Level 5, 200 training steps, A100-80GB.

### grad_norm per step

| Step | Binary GRPO | PGR |
|------|------|------|
| 10   | 0.000                   | 8.25 |
| 20   | 6.125 (spike)           | 9.56 |
| 30   | 0.038                   | 6.97 |
| 50   | 0.043                   | 8.50 |
| 80   | 0.041                   | 9.81 |
| 100  | 0.062 (reward_std=0.00) | 8.94 (reward_std=0.04) |

### Summary

| Metric | Binary GRPO | PGR |
|---|---|---|
| Mean grad_norm | 0.00–0.09 | 7–12 |
| reward_std at step 100 | 0.00 | 0.03–0.06 |
| Groups with nonzero gradient | ~40% | 100% |
| Dictionary drift collected | n/a | 172 steps |
| Training-time rollout success | ~0% | ~8.5% |
| Requires verifier | yes | no |
| Minimum group size k | 8–16 | 4 |

## Dictionary Drift

The dictionary updates online from steps in correct rollouts:

```
initial D  =  fit(gold solution steps)

every N training steps:
  if buffer has >= 50 new steps from correct rollouts:
    D  =  refit(D, new_steps)   # warm-started, 100 iters
```

Over 200 steps on Level 5 MATH: 172 steps collected, dictionary refreshed once.

## How to Run

```bash
modal run modal_dictionary.py                # build dictionary (once)
modal run modal_smoke_test.py                # end-to-end pipeline check
modal run --detach modal_train.py --mode pgr # train
modal run modal_eval.py                      # eval
```

See [WIKI.md](WIKI.md) for full run order, cost, and troubleshooting.

## Files

| File | What it does |
|---|---|
| `pgr_reward.py` | `PGRReward` class, prime-rl compatible |
| `modal_dictionary.py` | Offline dictionary learning |
| `modal_smoke_test.py` | 5-check pipeline test |
| `modal_train.py` | PGR or binary GRPO training, with drift |
| `modal_eval.py` | Eval checkpoints on MATH-Hard test |

## Compute Request (Prime Intellect)

| Phase | Hardware | Node-days |
|---|---|---|
| Dictionary learning, multi-domain | 4x A100 80GB | 35 |
| PGR vs baselines (MATH, AIME) | 8x H100 SXM | 60 |
| Science domain runs (GPQA, SciBench) | 8x H100 SXM | 50 |
| Ablations (tau, drift, oracle-free) | 4x H100 | 40 |
| Total | ~185 H100-equivalent node-days | 185 |
