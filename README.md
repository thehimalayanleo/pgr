# PGR — Pursuit-Graded Reward

> Dense per-step RL reward for reasoning models using Orthogonal Matching Pursuit — no verifier required.

---

## The Problem

Binary GRPO gives zero gradient when every rollout fails. On hard problems this happens constantly — the model generates hundreds of reasoning tokens and learns nothing.

```
Binary GRPO on Level 5 MATH (Qwen2.5-3B, 200 steps)

grad_norm
  10 │
   8 │
   6 │                     ▲ spike
   4 │
   2 │
   0 │▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
     └─────────────────────────────────────────▶ step
      0                  100                 200

PGR on the same problems

grad_norm
  10 │████████████████████████████████████████
   8 │████████████████████████████████████████
   6 │
   4 │
   2 │
   0 │
     └─────────────────────────────────────────▶ step
      0                  100                 200
```

---

## The Idea

Replace the binary terminal reward with a dense, per-step quality score built from sparse pursuit.

```
┌─────────────────────────────────────────────────────────────┐
│                    PGR Pipeline                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Gold solutions    ──▶  Encoder (22M)  ──▶  Step embeddings │
│  (MATH, GPQA, ...)                                          │
│                              │                              │
│                              ▼                              │
│                     Dictionary Learning                     │
│                     D ∈ ℝ^{256×384}                        │
│                     (prototypical reasoning moves)          │
│                              │                              │
│         ┌────────────────────┘                              │
│         │          At training time                         │
│         ▼                                                   │
│  Rollout step  ──▶  OMP reconstruction  ──▶  recon_error   │
│                                                             │
│  step_reward = exp(−error / τ)          ──▶  [0, 1]        │
│                                                             │
│  total_reward = α × mean(step_rewards) + (1−α) × terminal  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Results

Empirical comparison on **Qwen2.5-3B-Instruct**, **MATH-Hard Level 5**, **200 training steps**.

| Metric | Binary GRPO | PGR |
|---|---|---|
| Mean `grad_norm` | 0.00–0.09 (≈ 0 most steps) | **7–12 (every step)** |
| `reward_std` at step 100 | 0.00 (dead zone) | **0.03–0.06** |
| Gradient-carrying steps | ~40% (correct answers) | **100%** |
| Requires verifier | Yes | **No** |
| Minimum group size k | 8–16 | **4** |

```
                  ~100× higher gradient norm
                  Binary ░░░░░░░░░░░░░░░░░░░░░░░░  ≈ 0.04
                  PGR    ████████████████████████  ≈ 8.5
```

---

## Dictionary Drift

The dictionary updates during training using steps from correct rollouts:

```
initial_dict  ←  gold solutions (offline, once)

every N steps:
  if buffer has ≥ 50 new steps from correct rollouts:
    dict ← refit(current_atoms, new_steps)   ← warm start, 100 iters
    D[:] = new_atoms                          ← in-place, reward fn picks up immediately
```

Over 200 steps on Level 5 MATH: **172 steps collected**, dictionary refreshed once.

---

## How to Run

```bash
# 1. Build reasoning dictionary (once)
modal run modal_dictionary.py

# 2. Smoke test — verify pipeline end to end (~15 min, ~$1.50)
modal run modal_smoke_test.py

# 3. Train
modal run --detach modal_train.py --mode pgr

# 4. Eval
modal run modal_eval.py
```

See [WIKI.md](WIKI.md) for full run order, cost reference, and troubleshooting.

---

## Reward Formula

```
For each rollout step s_i:
  e_i  = OMP_reconstruction_error(embed(s_i), D)
  r_i  = exp(−e_i / τ)              τ = temperature (default 0.3)

total = α · mean(r_i) + (1−α) · terminal
                                    α = 0.5, terminal ∈ {0, 1}
```

Oracle-free mode (no verifier): set `terminal = 0`, train purely on step rewards.

---

## Files

| File | What it does |
|---|---|
| `pgr_reward.py` | Core `PGRReward` class — prime-rl compatible |
| `modal_dictionary.py` | Offline dictionary learning from solution traces |
| `modal_smoke_test.py` | 5-check end-to-end pipeline test |
| `modal_train.py` | PGR or binary GRPO training, with dictionary drift |
| `modal_eval.py` | Eval checkpoints on MATH-Hard test set |

---

*Ajinkya Kiran Mulay · Research Scientist, Meta · [thepursuits.xyz](https://thepursuits.xyz)*
