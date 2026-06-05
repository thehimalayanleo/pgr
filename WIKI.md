# PGR — Project Wiki

Pursuit-Graded Reward (PGR) is a dense, per-step reward signal for reasoning RL built on
Orthogonal Matching Pursuit (OMP). It replaces the binary terminal reward used in GRPO/RLVR
with a step-level quality score derived from a learned reasoning dictionary — no verifier needed.

---

## Table of Contents

1. [Setup](#1-setup)
2. [File Overview](#2-file-overview)
3. [Run Order](#3-run-order)
4. [Detailed Commands](#4-detailed-commands)
5. [Cost Reference](#5-cost-reference)
6. [Reading Results](#6-reading-results)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Setup

### Conda environment
```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate pursuit_env   # Python 3.11, torch 2.11
```

> ⚠️ Do NOT use `m1_env` — it runs Python 3.14 where torch is broken.

### Modal authentication
```bash
modal setup   # one-time, opens browser
```

### WandB secret (needed for modal_train.py only)
```bash
modal secret create wandb-secret WANDB_API_KEY=<your_key>
```

### Install dependencies locally (for import checks)
```bash
pip install -r requirements.txt
```

---

## 2. File Overview

```
pgr/
├── pgr_reward.py          # Core PGRReward class — prime-rl compatible reward wrapper
├── modal_smoke_test.py    # End-to-end pipeline check (5 assertions, ~15 min, ~$1.50)
├── modal_dictionary.py    # Offline dictionary learning from MATH-Hard solutions
├── modal_train.py         # PGR or binary GRPO training run, one mode at a time
├── modal_eval.py          # Eval base / binary / PGR checkpoints in parallel
├── requirements.txt       # Pinned dependencies
└── WIKI.md                # This file
```

### Key design decisions
- **Dataset**: `lighteval/MATH-Hard` (Level 5 only, MIT license). `hendrycks/competition_math`
  is DMCA'd; `lighteval/MATH` was removed from the Hub.
- **Encoder**: `BAAI/bge-small-en-v1.5` (22M params, 384d embeddings, unit-normalized).
- **Dictionary**: `N_ATOMS=256` atoms for real runs, `64` in smoke test.
- **TRL version**: `0.14.0` — GRPO was introduced here. `0.9.x`–`0.13.x` do not have it.
  In `0.14.0`, the trainer takes `processing_class=` not `tokenizer=`.
- **GPU**: A100-80GB for training (~$2.50/hr). Enough VRAM for 3B, ~1.5× slower than H100.
- **Volume**: `pgr-artifacts` Modal volume persists the dictionary and checkpoints across runs.

---

## 3. Run Order

Always run in this order. Each step depends on the previous.

```
[1] modal_smoke_test.py     verify the full pipeline works         ~15 min  ~$1.50
        ↓  (dictionary is cached in volume after this)
[2] modal_train.py pgr      train PGR on MATH-Hard                ~2.5 hr  ~$6–7
        ↓  (only if rewards look healthy)
[3] modal_train.py binary   train binary GRPO baseline            ~2.5 hr  ~$6–7
        ↓
[4] modal_eval.py           eval all three checkpoints            ~15 min  ~$0.50
```

**Total: ~$14–16** for a full PGR vs binary GRPO comparison on 500 steps.

---

## 4. Detailed Commands

All commands assume you are in `~/github-repos/pgr` with `pursuit_env` active.

### Activate environment
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate pursuit_env
cd ~/github-repos/pgr
```

### [1] Smoke test — run this first every time
```bash
modal run modal_smoke_test.py
```

Runs 5 checks end-to-end:
- `[1]` Dictionary build + load
- `[2]` Encoder unit-norm
- `[3]` OMP discriminability (correct vs shuffled steps)
- `[4]` Reward output in [0,1], correct > incorrect
- `[5]` 20-step GRPO training loop completes

All 5 must show `✅` before proceeding. The dictionary is saved to the Modal volume
and reused by subsequent runs — no need to rebuild it.

### [2] Dictionary only (optional, if you want a bigger dict before training)
```bash
modal run modal_dictionary.py
```

Builds a full `N_ATOMS=256` dictionary from all 2,300 MATH-Hard train problems.
The smoke test builds a smaller `N_ATOMS=64` dict. For real training runs, rebuild
with 256 atoms first.

### [3] Training — PGR
```bash
modal run modal_train.py --mode pgr
```

Optional flags:
```bash
modal run modal_train.py --mode pgr --max-steps 500 --k 4
```

| Flag | Default | Description |
|---|---|---|
| `--mode` | `pgr` | `pgr` or `binary` |
| `--max-steps` | `500` | Number of gradient steps |
| `--k` | `4` | Rollouts per group (GRPO group size) |

Checkpoint saved to `/artifacts/checkpoints/pgr_final` in the Modal volume.

### [4] Training — binary GRPO baseline
```bash
modal run modal_train.py --mode binary
```

Run only after PGR training looks healthy (rewards increasing, grad norm nonzero).

### [5] Eval
```bash
modal run modal_eval.py
```

Evaluates three checkpoints in parallel on 100 MATH-Hard test problems:
- `Qwen/Qwen2.5-3B-Instruct` (base model)
- `/artifacts/checkpoints/binary_final`
- `/artifacts/checkpoints/pgr_final`

Prints an accuracy table. This is the number that goes in the Prime Intellect grant.

---

## 5. Cost Reference

| Job | GPU | Est. time | Cost |
|---|---|---|---|
| Smoke test | A10G | 15 min | ~$1.50 |
| Dictionary (full, 256 atoms) | A10G | 15 min | ~$0.30 |
| PGR training (500 steps) | A100-80GB | 2.5 hrs | ~$6.50 |
| Binary GRPO training (500 steps) | A100-80GB | 2.5 hrs | ~$6.50 |
| Eval (3 checkpoints, parallel) | A10G ×3 | 15 min | ~$0.50 |
| **Full pipeline** | | | **~$15** |

Modal rates (pay-per-second, no idle billing):
- A10G: ~$0.60/hr
- A100-80GB: ~$2.50/hr
- H100: ~$3.95/hr

---

## 6. Reading Results

### Smoke test — what to look for

| Metric | Target | Notes |
|---|---|---|
| `omp_gap` | > 0 | Positive = signal is real. Small gap (0.02) is expected with 64 atoms. |
| `reward_correct` | > `reward_incorrect` | Terminal anchor doing its job. |
| `loss` in Check 5 | May show 0.0 | Normal for GRPO — it logs clipped ratio loss, not cross-entropy. |
| `grad_norm` | > 0 | Nonzero = gradients flowing. |

### Training run — what to look for in WandB

| Metric | Healthy | Unhealthy |
|---|---|---|
| `reward` | Slowly increasing | Flat at 0 (binary on hard problems = dead zone) |
| `reward_std` | > 0 | Zero = all rollouts identical, no learning signal |
| `grad_norm` | Nonzero, stable | NaN = reward scale too large, lower lr |
| `kl` | Small (< 0.1) | Large = policy collapsing away from reference |
| `completion_length` | ~200–400 tokens | Saturating at max = model not finishing thoughts |

### The key comparison (grant figure)

After both training runs and eval:

```
base model:   acc = X.XX
binary GRPO:  acc = X.XX   ← may not improve on Level 5 (dead zone)
PGR:          acc = X.XX   ← should improve even on Level 5
```

Even a small accuracy gap (2–3%) on Level 5 problems is a strong result given only 500 steps.

---

## 7. Troubleshooting

### `GRPOConfig` import error
TRL version is wrong. Must be `>=0.14.0`. Check with:
```bash
python -c "import trl; print(trl.__version__)"
```
Fix: `pip install trl==0.14.0`

### `Dataset 'lighteval/MATH' doesn't exist`
The original dataset was removed. Use `lighteval/MATH-Hard` — already fixed in all scripts.
`hendrycks/competition_math` is also gone (DMCA takedown).

### `tokenizer` parameter error in GRPOTrainer
TRL 0.14.0 renamed this to `processing_class`. All scripts use the correct name.

### `torch` broken in `m1_env`
Python 3.14 + torch is broken on M1. Use `pursuit_env` (Python 3.11) instead.

### Loss shows 0.0 throughout training
Expected GRPO behavior — it logs clipped ratio loss which starts near zero.
Check `train_loss` in the final summary line and `grad_norm` in WandB instead.

### Dictionary not found during training
Run `modal_smoke_test.py` first — it builds and commits the dictionary to the
`pgr-artifacts` volume. Training scripts load from there.

### `volume.commit()` errors
If Modal complains about `commit()` inside the container, it's a Modal SDK version issue.
The volume auto-commits on clean function exit — remove the explicit `volume.commit()` call.
