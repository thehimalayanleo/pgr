# modal_train.py
"""
PGR vs binary GRPO mini training run.
Spawns both jobs in parallel on H100s.

Runtime: ~2 hours each (parallel)
Cost:    ~$18-20 total
"""

import modal

app = modal.App("pgr-train-smoke")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers", "trl>=0.9.0",
        "datasets", "accelerate", "sentence-transformers",
        "scikit-learn", "numpy", "wandb", "peft"
    )
)

volume = modal.Volume.from_name("pgr-artifacts")
secret = modal.Secret.from_name("wandb-secret")  # modal secret create wandb-secret WANDB_API_KEY=...


@app.function(
    image=image,
    gpu="H100",
    timeout=7200,
    volumes={"/artifacts": volume},
    secrets=[secret]
)
def train(
    mode: str = "pgr",        # "pgr" or "binary"
    max_steps: int = 500,
    k: int = 4,
    alpha: float = 0.5,
    model_id: str = "Qwen/Qwen2.5-3B-Instruct"
):
    import os, re, time, torch
    import numpy as np
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import GRPOConfig, GRPOTrainer
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import orthogonal_mp

    os.environ["WANDB_PROJECT"] = "pgr-smoke"

    D = np.load("/artifacts/dictionary_atoms.npy")
    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def segment_steps(text):
        steps = re.split(r'\n\n+|(?=Step \d+:)', text.strip())
        return [s.strip() for s in steps if len(s.strip()) > 20]

    def omp_step_rewards(texts, tau=0.3):
        if not texts:
            return np.array([0.0])
        emb   = encoder.encode(texts, normalize_embeddings=True, batch_size=64)
        codes = orthogonal_mp(D.T, emb.T, n_nonzero_coefs=5)
        errors = np.linalg.norm(emb.T - D.T @ codes, axis=0)
        return np.exp(-errors / tau)

    def extract_answer(text):
        m = re.search(r'\\boxed\{(.+?)\}', text)
        return m.group(1).strip() if m else None

    # Dataset: hard problems only
    ds = load_dataset("lighteval/MATH", split="train")
    hard = ds.filter(lambda x: x["level"] in ("Level 4", "Level 5"))
    hard = hard.map(lambda x: {
        "prompt": f"Solve step by step:\n{x['problem']}\n\nSolution:",
        "answer": x["solution"],
    })

    # Reward functions
    def pgr_reward(completions, prompts, answer, **kwargs):
        rewards = []
        for completion, ans in zip(completions, answer):
            steps  = segment_steps(completion)
            sr     = omp_step_rewards(steps)
            mean_sr = float(sr.mean()) if len(sr) else 0.0
            pred   = extract_answer(completion)
            gold   = extract_answer(ans)
            term   = 1.0 if (pred and gold and pred == gold) else 0.0
            rewards.append(alpha * mean_sr + (1 - alpha) * term)
        return rewards

    def binary_reward(completions, prompts, answer, **kwargs):
        rewards = []
        for completion, ans in zip(completions, answer):
            pred = extract_answer(completion)
            gold = extract_answer(ans)
            rewards.append(1.0 if (pred and gold and pred == gold) else 0.0)
        return rewards

    reward_fn = pgr_reward if mode == "pgr" else binary_reward

    # Model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    config = GRPOConfig(
        output_dir=f"/artifacts/checkpoints/{mode}",
        max_steps=max_steps,
        per_device_train_batch_size=1,
        num_generations=k,
        max_completion_length=400,
        learning_rate=1e-6,
        logging_steps=10,
        save_steps=250,
        report_to="wandb",
        run_name=f"{mode}_smoke_{max_steps}steps_k{k}",
        gradient_accumulation_steps=4,
        warmup_steps=20,
        bf16=True,
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=hard,
        reward_funcs=[reward_fn],
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(f"/artifacts/checkpoints/{mode}_final")
    volume.commit()

    return {
        "mode": mode,
        "steps": max_steps,
        "saved_to": f"/artifacts/checkpoints/{mode}_final"
    }


@app.local_entrypoint()
def main():
    # Run PGR and binary GRPO in parallel
    pgr_job    = train.spawn(mode="pgr",    max_steps=500)
    binary_job = train.spawn(mode="binary", max_steps=500)

    pgr_result    = pgr_job.get()
    binary_result = binary_job.get()

    print("PGR:   ", pgr_result)
    print("Binary:", binary_result)
