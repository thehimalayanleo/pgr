# SUPERSEDED DRAFT: Frozen-Encoder Per-Step Rewards Don't Grade Math Reasoning, But Contrastive Group Structure Does

> **Status, 2026-08-21:** This draft is preserved only as research history. Its
> positive contrastive and success-dictionary OMP claims did not survive the
> label-leakage audit. Cross-fitted success-dictionary OMP reached AUROC `0.426`,
> and leave-one-out contrastive scoring reached AUROC `0.443`. Historical
> majority, random, and consensus training comparisons were separately
> invalidated by a reward-source control-path bug. Do not cite the positive
> claims below. The current verdict is recorded in
> `OMP_RL_REWARD_FINAL_AUDIT_2026-08-21.md`.

**Workshop submission draft - COLM 2026**

## Abstract (TL;DR)

We test whether *frozen-encoder per-step rewards* can substitute for trained Process Reward Models (PRMs) in dense-credit RL post-training. On 150 GSM8K rollouts from Qwen2.5-1.5B, five distinct per-step formulations - single-step OMP residual, prefix-conditioned OMP, leave-one-out policy likelihood, pair-wise concatenation, pair-wise difference - all yield Spearman ρ ≤ |0.11| with trajectory correctness. **A frozen sentence encoder cannot grade math reasoning step quality.** Two complementary findings rescue per-step credit in this regime: (i) GRPO's group normalization mathematically collapses any preserved per-step variance by **~3500×**, explaining null results across reward designs; (ii) reframing the K-rollout group as a **self-contained labeled batch** lets the encoder measure step *distance* (easy) rather than step *quality* (impossible), yielding two new reward signals - *contrastive direction* (ρ = +0.674) and *success-dictionary OMP* (ρ = +0.740) - both reaching the oracle best-of-K pick ceiling without using the correctness label as a per-step input.

## 1. Setup and Problem

We study per-step reward signals for RL post-training of reasoning LLMs. Two regimes exist in practice: trajectory-only (binary terminal reward + GRPO; mature, used by DeepSeek-R1, Qwen-Math) and per-step (Process Reward Model + PPO; powerful but requires a trained PRM). We ask: **can a frozen pretrained encoder substitute for the PRM?**

Setup: Qwen2.5-3B-Instruct fine-tuned with GRPO on MATH-Hard (Hendrycks et al., 2021). K=4 rollouts per prompt, max_completion_length=512. Encoder: BAAI/bge-small-en-v1.5 (384-d). All experiments are limited by a fixed compute budget appropriate for a workshop submission (≤100 H100 hours).

## 2. Per-Step Rewards from Frozen Encoders Are Noise

We test five per-step reward formulations on 150 GSM8K rollouts (Qwen2.5-1.5B, K=3, T=0.8):

| Method | Formulation | Spearman ρ with correctness | Best-of-K pick |
|---|---|---:|---:|
| Single-step OMP | `r_k = exp(-‖e_k − D · OMP(e_k, D)‖/τ)` | +0.022 | 0.200 |
| Prefix-conditioned OMP | marginal residual of cum. prefix `e_{1:k}` vs `e_{1:k-1}` | -0.045 | 0.160 |
| LOO likelihood | mean log P(step_k tokens \| prefix) under policy | -0.108 | 0.160 |
| Pair-wise concat | OMP residual on embed(step_{k-1} + step_k) | -0.003 | 0.120 |
| Pair-wise diff | OMP residual on embed(pair) − embed(step_{k-1}) | +0.071 | 0.160 |

All five lie within sampling noise of ρ = 0. Best-of-K picking with these rewards performs at or below random (15%) - none approaches the oracle ceiling of 32% (problems with at least one correct rollout in the group). The pretrained encoder captures lexical/topical patterns but no information about mathematical correctness at the step level.

This is consistent with DeepSeek-R1, Math-Shepherd, and OmegaPRM all using *trained* PRMs (small models fine-tuned on labeled correct/incorrect intermediate steps) rather than frozen-encoder similarity.

## 3. Mechanism: GRPO Group Normalization Collapses Per-Step Variance

Even when a per-step reward function has non-trivial structure, GRPO's group-relative advantage normalization can erase it. We measure within-rollout advantage variance (the variance of `A[t]` across token positions within a single rollout) under two advantage estimators applied to the same per-step rewards:

| Advantage estimator | within_rollout_adv_var |
|---|---:|
| GRPO group-normalized: `(r − group_mean) / group_std` | **0.05** |
| PPO-style EMA baseline: `(r − r_EMA) / σ_EMA` | **150.8** |

A **~3500× variance gap** at identical per-step rewards. GRPO group normalization pools rewards across K rollouts × N steps and re-centers them; the within-rollout component is destroyed. PPO-style scalar baselines (Schulman et al., 2017) preserve it.

This explains why prior negative results on "per-step rewards under GRPO" do not necessarily indict the per-step signal - the algorithm itself is destructive to it.

## 4. The Group Is a Self-Contained Labeled Batch

GRPO already samples K rollouts per prompt and labels each with binary correctness. We use this as a **per-group labeled training set for a step-level reward**:

> *Don't ask the encoder to grade a step. Ask it to measure distance between two attempts at the same step position.*

### 4.1 Contrastive direction (Level 1)

For each prompt with mixed success/failure in the group:
```
c_succ[k] = mean of embed(step_k) over successful rollouts
c_fail[k] = mean of embed(step_k) over failed rollouts
direction[k] = normalize(c_succ[k] − c_fail[k])
per-step score = (embed(step_k) − c_fail[k]) · direction[k]
```

The correctness labels generate the centroids; the encoder only contributes pairwise distances.

### 4.2 Success-dictionary OMP (Level 2a)

We pool successful step embeddings into a per-group dictionary `D_succ` and ask the OMP framework: "how well is this rollout's step k explained by winning patterns?"

```
D_succ = stack(embed(step_k_j) for all successful rollouts j and positions k)
reward[k] = exp(- ‖e_k − D_succ · OMP(e_k, D_succ, n_nonzero=5)‖ / τ)
```

This is position-agnostic (set-based) and captures the linear span of success patterns through OMP combinations.

### 4.3 Offline validation (n=50 GSM8K, K=3)

| Method | ρ | Pick acc |
|---|---:|---:|
| Random pick / base accuracy | - | 0.147 |
| **Oracle ceiling** | - | **0.320** |
| Single OMP (baseline) | +0.022 | 0.200 |
| Contrastive direction (L1) | +0.674 | 0.320 ✓ |
| **Success-dictionary OMP (L2a)** | **+0.740** | **0.320 ✓** |

Both L1 and L2a reach the oracle picking ceiling *without using the correctness label as a per-step reward input* (correctness labels only generate the contrast centroids / dictionary). L2a wins on ρ because its set-based formulation is robust to position misalignment and length variation across rollouts.

## 5. Online Training Results

We train Qwen2.5-3B-Instruct with each reward formulation on MATH-Hard. All eval numbers are n=100, max_tokens=1024, T=0.7.

| Config | Train steps | KL at end | Eval acc |
|---|---:|---:|---:|
| Trajectory GRPO baseline | 100 | 3e-4 | ~50% |
| Single OMP (PGR original) | 100 | 3e-4 | ~49% |
| GRPO + positional terminal | 50 | 3e-4 | 48% |
| PPO + uniform terminal | 50 | 3e-4 | 48% |
| PPO + positional terminal | 50 | 3e-4 | 50% |
| **Contrastive (L1)** | 50 | 3e-4 | **50%** |
| **L2a (success-dict OMP)** | 50 | - | *in progress* |

Online results land within ±3% of the trajectory baseline - within the n=100 CI of ±10%. At this compute budget (50–100 training steps, LR=1e-6, KL ≈ 3e-4), no per-step formulation produces statistically distinguishable eval lift. The model barely moves.

**This is consistent with the broader literature**: DeepSeek-R1, Math-Shepherd, and similar PRM-based RL pipelines use 10k+ training steps to demonstrate per-step credit advantage. Our compute budget cannot reach that regime.

## 6. Contributions and Limitations

**Contributions.** (1) A clean negative: five frozen-encoder per-step formulations are noise on math reasoning. (2) A mechanism: GRPO group-norm collapses per-step variance by ~3500×, independent of reward design. (3) Two new positive results: contrastive direction and success-dictionary OMP both achieve ρ ≈ 0.7 with correctness - the first non-trivial per-step signals from a frozen encoder that don't leak the trajectory label per step.

**Limitations.** (1) Online null results due to limited compute budget; we cannot rule out lift at 1000+ training steps. (2) GSM8K offline tests; MATH-Hard online tests. (3) `contrast_group_frac` declines from 0.30 → 0.05 over training as success rate rises and groups become all-correct or all-failed - practical use will need a fallback (e.g., L2a falls back to OMP, or hybridize with positional terminal on no-contrast groups).

**Practical recommendation.** Without a trained PRM and below ≈1000 training steps, trajectory GRPO is the right default. Above that, our L2a method is the cheapest known per-step signal that doesn't require labeled step data.

## 7. Code and Reproducibility

All code, configs, cached rollouts, and offline test results: `github.com/thehimalayanleo/thepursuits/pgr`. Modal artifact volumes are public.
