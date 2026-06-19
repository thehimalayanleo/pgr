# modal_train.py
"""
PGR or binary GRPO training run — one mode at a time.

Usage:
  modal run modal_train.py --mode pgr     # ~$7, ~2.5 hrs on A100
  modal run modal_train.py --mode binary  # baseline, run after pgr looks good

GPU: A100-80GB (~$2.50/hr). Enough VRAM for 3B, ~1.5x slower than H100.
"""

import modal
import sys

app = modal.App("pgr-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.46.2",
        "trl==0.14.0",
        "datasets",
        "accelerate==0.34.2",
        "sentence-transformers",
        "scikit-learn",
        "numpy",
        "wandb",
        "peft",
    )
)

volume = modal.Volume.from_name("pgr-artifacts")


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=10800,         # 3 hrs to be safe on A100
    volumes={"/artifacts": volume},
)
def train(
    mode: str = "pgr",
    max_steps: int = 500,
    k: int = 4,
    alpha: float = 0.5,
    seed: int = 42,
    model_id: str = "Qwen/Qwen2.5-3B-Instruct",
):
    import os, re, torch
    import numpy as np
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import GRPOConfig, GRPOTrainer
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import orthogonal_mp

    assert mode in ("pgr", "binary"), f"mode must be 'pgr' or 'binary', got '{mode}'"

    # Seed everything for reproducibility across seeds
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n=== Training mode: {mode.upper()} | steps: {max_steps} | k: {k} | seed: {seed} ===\n")

    # ── Load dictionary (built by modal_smoke_test or modal_dictionary) ───
    dict_path = "/artifacts/dictionary_atoms.npy"
    assert os.path.exists(dict_path), (
        "Dictionary not found at /artifacts/dictionary_atoms.npy. "
        "Run modal_smoke_test.py or modal_dictionary.py first."
    )
    D = np.load(dict_path)
    print(f"Loaded dictionary: {D.shape}")

    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

    # ── Helpers ───────────────────────────────────────────────────────────
    def seg(text):
        parts = re.split(r'\n\n+|(?=Step \d+:)', text.strip())
        return [p.strip() for p in parts if len(p.strip()) > 20]

    def omp_rewards(steps, tau=0.3):
        if not steps:
            return np.array([0.0])
        emb   = encoder.encode(steps, normalize_embeddings=True, batch_size=64)
        codes = orthogonal_mp(D.T, emb.T, n_nonzero_coefs=5)
        errs  = np.linalg.norm(emb.T - D.T @ codes, axis=0)
        return np.exp(-errs / tau)

    def extract_answer(text):
        m = re.search(r'\\boxed\{(.+?)\}', text)
        return m.group(1).strip() if m else None

    # ── Dataset ───────────────────────────────────────────────────────────
    ds   = load_dataset("lighteval/MATH-Hard", split="train")
    hard = ds.map(
        lambda x: {
            "prompt": f"Solve step by step:\n{x['problem']}\n\nSolution:",
            "answer": x["solution"],
        },
        remove_columns=ds.column_names,
    )
    print(f"Dataset: {len(hard)} hard problems")

    # ── Reward functions ──────────────────────────────────────────────────
    def pgr_reward(completions, **kwargs):
        answer = kwargs.get("answer", [""] * len(completions))
        rewards = []
        for completion, ans in zip(completions, answer):
            steps   = seg(completion)
            sr      = omp_rewards(steps)
            mean_sr = float(sr.mean())
            pred    = extract_answer(completion)
            gold    = extract_answer(ans)
            term    = 1.0 if (pred and gold and pred == gold) else 0.0
            rewards.append(alpha * mean_sr + (1 - alpha) * term)
        return rewards

    def binary_reward(completions, **kwargs):
        answer = kwargs.get("answer", [""] * len(completions))
        rewards = []
        for completion, ans in zip(completions, answer):
            pred = extract_answer(completion)
            gold = extract_answer(ans)
            rewards.append(1.0 if (pred and gold and pred == gold) else 0.0)
        return rewards

    reward_fn = pgr_reward if mode == "pgr" else binary_reward

    # ── Model ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # ── Training ──────────────────────────────────────────────────────────
    out_dir = f"/artifacts/checkpoints/{mode}_seed{seed}_steps{max_steps}"
    config = GRPOConfig(
        output_dir=out_dir,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        num_generations=k,
        max_completion_length=512,
        learning_rate=1e-6,
        logging_steps=10,
        save_steps=25,                  # checkpoint every 25 steps for preemption resilience
        save_total_limit=2,             # keep only the 2 most recent — saves volume space
        report_to="none",
        run_name=f"{mode}_steps{max_steps}_k{k}_seed{seed}",
        gradient_accumulation_steps=4,
        warmup_steps=20,
        bf16=True,
        dataloader_num_workers=0,
        seed=seed,
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=hard,
        reward_funcs=[reward_fn],
        processing_class=tokenizer,
    )

    # Auto-resume from latest checkpoint if one exists in out_dir
    resume = os.path.isdir(out_dir) and any(
        d.startswith("checkpoint-") for d in os.listdir(out_dir)
    )
    if resume:
        print(f"Resuming from latest checkpoint in {out_dir}")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_model(f"{out_dir}_final")
    volume.commit()

    result = {
        "mode": mode,
        "steps": max_steps,
        "seed": seed,
        "checkpoint": f"{out_dir}_final",
    }
    print(f"\nDone: {result}")
    return result


@app.local_entrypoint()
def main(
    mode: str = "pgr",
    max_steps: int = 500,
    k: int = 4,
    seed: int = 42,
):
    """
    Args:
      --mode      pgr or binary (default: pgr)
      --max-steps gradient steps (default: 500)
      --k         rollouts per group (default: 4)
      --seed      random seed (default: 42)
    """
    print(f"Launching {mode.upper()} | steps={max_steps} k={k} seed={seed}")
    result = train.remote(mode=mode, max_steps=max_steps, k=k, seed=seed)
    print(result)
