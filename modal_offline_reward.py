# modal_offline_reward.py
"""
Offline reward signal comparison: generate rollouts from base Qwen2.5-3B,
score them with multiple reward functions, measure alignment with correctness.

This doesn't train anything. It answers the question:
  "Which reward signal is most informative about rollout correctness?"

Reward signals compared:
  - Binary       (1 if final answer correct, else 0)
  - Confidence   (mean log-prob of generated tokens)
  - PGR step     (mean step-level OMP reward)
  - PGR step+T   (alpha=0.5 blend with binary terminal)
  - PGR oracle-free (alpha=1.0, step rewards only)

Metrics:
  - Spearman correlation with binary correctness
  - Group-level reward_std distribution
  - Fraction of groups with zero variance (dead-zone analog)

Runtime: ~30 min on A100  (we use A100 for fast 3B inference)
Cost:    ~$1.50
"""

import modal

app = modal.App("pgr-offline-reward")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "vllm==0.6.3",          # pinned for transformers 4.46 compat
        "transformers==4.46.2",
        "datasets",
        "sentence-transformers",
        "scikit-learn",
        "scipy",
        "numpy",
    )
)

volume = modal.Volume.from_name("pgr-artifacts")


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/artifacts": volume},
)
def offline_reward_analysis(
    n_problems: int = 50,
    k_rollouts: int = 4,
    temperature: float = 0.8,
    max_tokens: int = 512,
    model_id: str = "Qwen/Qwen2.5-3B-Instruct",
):
    import os, re, json
    import numpy as np
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import orthogonal_mp
    from scipy.stats import spearmanr
    from vllm import LLM, SamplingParams

    # ── Load dictionary + encoder ───────────────────────────────────────
    D = np.load("/artifacts/dictionary_atoms.npy", allow_pickle=False)
    print(f"Dictionary: {D.shape}")
    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def seg(text):
        parts = re.split(r'\n\n+|(?=Step \d+:)', text.strip())
        return [p.strip() for p in parts if len(p.strip()) > 20]

    def omp_per_step_rewards(steps, tau=0.3):
        if not steps:
            return np.array([0.0])
        emb = encoder.encode(steps, normalize_embeddings=True, batch_size=64)
        codes = orthogonal_mp(D.T, emb.T, n_nonzero_coefs=5)
        errs = np.linalg.norm(emb.T - D.T @ codes, axis=0)
        return np.exp(-errs / tau)

    def extract_answer(text):
        idx = text.find("\\boxed{")
        if idx == -1:
            return None
        i = idx + len("\\boxed{")
        depth = 1
        out = []
        while i < len(text) and depth > 0:
            c = text[i]
            if c == "{":
                depth += 1
                out.append(c)
            elif c == "}":
                depth -= 1
                if depth > 0:
                    out.append(c)
            else:
                out.append(c)
            i += 1
        return "".join(out).strip() if depth == 0 else None

    # ── Load problems ───────────────────────────────────────────────────
    ds = load_dataset("lighteval/MATH-Hard", split="test")
    problems = list(ds)[:n_problems]
    print(f"Loaded {len(problems)} problems")

    # ── Generate k rollouts per problem with logprobs ────────────────────
    print(f"\nLoading {model_id} into vLLM...")
    llm = LLM(model=model_id, max_model_len=1536, gpu_memory_utilization=0.85)
    params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=k_rollouts,
        logprobs=1,  # one logprob per generated token
    )

    print(f"Generating {n_problems} × {k_rollouts} = {n_problems * k_rollouts} rollouts...")
    prompts = [f"Solve step by step:\n{ex['problem']}\n\nSolution:" for ex in problems]
    outputs = llm.generate(prompts, params)
    print(f"Generation done. Outputs: {len(outputs)}")

    # ── Score every rollout with every reward function ──────────────────
    rows = []
    for ex, output in zip(problems, outputs):
        gold = extract_answer(ex["solution"])
        for rollout in output.outputs:
            text = rollout.text
            pred = extract_answer(text)
            is_correct = int(pred is not None and gold is not None and pred == gold)

            # Confidence: mean log-prob of generated tokens
            if rollout.logprobs:
                logprobs_per_tok = []
                for tok_logprobs in rollout.logprobs:
                    # vLLM logprobs is {tok_id: Logprob(...)} for top-k. Pick max
                    if tok_logprobs:
                        lp = max(lp.logprob for lp in tok_logprobs.values())
                        logprobs_per_tok.append(lp)
                conf = float(np.exp(np.mean(logprobs_per_tok))) if logprobs_per_tok else 0.0
            else:
                conf = 0.0

            # PGR per-step rewards
            steps = seg(text)
            step_rewards = omp_per_step_rewards(steps)
            mean_step_reward = float(step_rewards.mean())

            # Reward function values
            r_binary       = float(is_correct)
            r_confidence   = conf
            r_pgr_oraclefree = mean_step_reward
            r_pgr_step_term  = 0.5 * mean_step_reward + 0.5 * is_correct

            rows.append({
                "problem_id": ex.get("problem", "")[:80],
                "is_correct": is_correct,
                "r_binary": r_binary,
                "r_confidence": r_confidence,
                "r_pgr_oraclefree": r_pgr_oraclefree,
                "r_pgr_step_term": r_pgr_step_term,
                "n_steps": len(steps),
            })

    print(f"\nScored {len(rows)} rollouts")

    # ── Analysis: correlation with correctness ───────────────────────────
    print("\n" + "="*60)
    print("  SPEARMAN ρ WITH CORRECTNESS")
    print("="*60)
    correctness = [r["is_correct"] for r in rows]
    results = {"per_rollout": {}, "per_group_std": {}, "dead_zone_rate": {}}

    for reward_name in ["r_binary", "r_confidence", "r_pgr_oraclefree", "r_pgr_step_term"]:
        values = [r[reward_name] for r in rows]
        if len(set(values)) > 1 and len(set(correctness)) > 1:
            rho, p = spearmanr(values, correctness)
        else:
            rho, p = 0.0, 1.0
        print(f"  {reward_name:25s}  ρ = {rho:+.4f}  p = {p:.2e}")
        results["per_rollout"][reward_name] = {"spearman_rho": round(rho, 4), "p_value": round(p, 6)}

    # ── Analysis: group-level reward variance (the "dead zone" angle) ────
    print("\n" + "="*60)
    print("  GROUP-LEVEL REWARD VARIANCE (k-rollout groups)")
    print("="*60)
    # Reshape: each problem has k_rollouts grouped together
    n_groups = len(rows) // k_rollouts
    for reward_name in ["r_binary", "r_confidence", "r_pgr_oraclefree", "r_pgr_step_term"]:
        stds = []
        for g in range(n_groups):
            group_vals = [rows[g*k_rollouts + i][reward_name] for i in range(k_rollouts)]
            stds.append(float(np.std(group_vals)))
        stds = np.array(stds)
        zero_var_rate = float(np.mean(stds < 0.01))
        mean_std = float(stds.mean())
        median_std = float(np.median(stds))
        print(f"  {reward_name:25s}  mean σ = {mean_std:.4f}  median σ = {median_std:.4f}  zero-σ rate = {zero_var_rate*100:.1f}%")
        results["per_group_std"][reward_name] = {
            "mean": round(mean_std, 4),
            "median": round(median_std, 4),
        }
        results["dead_zone_rate"][reward_name] = round(zero_var_rate, 4)

    # ── Save raw data ───────────────────────────────────────────────────
    out_path = "/artifacts/offline_reward_analysis.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "n_problems": n_problems,
                "k_rollouts": k_rollouts,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "model": model_id,
            },
            "summary": results,
            "rollouts": rows,
        }, f, indent=2)
    volume.commit()
    print(f"\nResults saved to {out_path}")

    print("\n" + "="*60)
    print("  HEADLINE")
    print("="*60)
    binary_zero = results["dead_zone_rate"]["r_binary"]
    pgr_zero = results["dead_zone_rate"]["r_pgr_step_term"]
    conf_zero = results["dead_zone_rate"]["r_confidence"]
    print(f"  Binary reward dead-zone rate:     {binary_zero*100:.1f}%")
    print(f"  Confidence reward dead-zone rate: {conf_zero*100:.1f}%")
    print(f"  PGR reward dead-zone rate:        {pgr_zero*100:.1f}%")
    return results


@app.local_entrypoint()
def main(n_problems: int = 50, k_rollouts: int = 4):
    print(f"Launching offline reward analysis: {n_problems} problems × {k_rollouts} rollouts")
    results = offline_reward_analysis.remote(n_problems=n_problems, k_rollouts=k_rollouts)
    print(json.dumps(results, indent=2) if False else results)


import json
