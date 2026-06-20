# modal_step_train.py
"""
Step-level PGR training on Modal.

Uses StepLevelGRPOTrainer (in step_pgr_trainer.py) which applies per-step
advantages at the token level rather than collapsing to a trajectory scalar.

Usage:
  modal run --detach modal_step_train.py --max-steps 100 --seed 42

Output checkpoint: /artifacts/checkpoints/step_pgr_seed{seed}_steps{N}_final
"""

import modal

app = modal.App("pgr-step-train")

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
        "peft",
    )
    .add_local_file("step_pgr_trainer.py", remote_path="/root/step_pgr_trainer.py")
)

volume = modal.Volume.from_name("pgr-artifacts")


@app.function(
    image=image,
    gpu="H100",
    timeout=14400,
    volumes={"/artifacts": volume},
)
def train(
    max_steps: int = 100,
    k: int = 4,
    alpha: float = 0.5,
    tau: float = 0.3,
    seed: int = 42,
    step_advantage_mode: str = "pooled",
    model_id: str = "Qwen/Qwen2.5-3B-Instruct",
):
    import os, sys, re, random
    import numpy as np
    import torch

    sys.path.insert(0, "/root")
    from step_pgr_trainer import StepLevelGRPOTrainer

    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import GRPOConfig
    from sentence_transformers import SentenceTransformer

    # Seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n=== STEP-LEVEL PGR | steps: {max_steps} | k: {k} | seed: {seed} "
          f"| alpha: {alpha} | mode: {step_advantage_mode} ===\n")

    # ── Load dictionary + encoder ────────────────────────────────────────
    dict_path = "/artifacts/dictionary_atoms.npy"
    assert os.path.exists(dict_path), "Dictionary not found — run modal_dictionary.py first"
    D = np.load(dict_path)
    print(f"Dictionary: {D.shape}")
    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

    # ── Dataset (must include 'answer' so we can compute terminal reward) ──
    ds   = load_dataset("lighteval/MATH-Hard", split="train")
    hard = ds.map(
        lambda x: {
            "prompt": f"Solve step by step:\n{x['problem']}\n\nSolution:",
            "answer": x["solution"],
        },
        remove_columns=ds.column_names,
    )
    print(f"Dataset: {len(hard)} problems")

    # ── Model + tokenizer ────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # ── Training ─────────────────────────────────────────────────────────
    out_dir = f"/artifacts/checkpoints/step_pgr_seed{seed}_steps{max_steps}"
    config = GRPOConfig(
        output_dir=out_dir,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        num_generations=k,
        max_completion_length=512,
        learning_rate=1e-6,
        logging_steps=10,
        save_steps=25,
        save_total_limit=4,                          # keep step-25, 50, 75, 100
        report_to="none",
        run_name=f"step_pgr_seed{seed}_steps{max_steps}_alpha{alpha:.1f}",
        gradient_accumulation_steps=4,
        warmup_steps=20,
        bf16=True,
        dataloader_num_workers=0,
        seed=seed,
    )

    trainer = StepLevelGRPOTrainer(
        model=model,
        args=config,
        train_dataset=hard,
        processing_class=tokenizer,
        encoder=encoder,
        dictionary=D,
        alpha=alpha,
        tau=tau,
        step_advantage_mode=step_advantage_mode,
    )

    # Auto-resume from latest checkpoint if one exists
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
        "mode": "step_pgr",
        "steps": max_steps,
        "seed": seed,
        "alpha": alpha,
        "step_advantage_mode": step_advantage_mode,
        "checkpoint": f"{out_dir}_final",
    }
    print(f"\nDone: {result}")
    return result


@app.local_entrypoint()
def main(
    max_steps: int = 100,
    k: int = 4,
    seed: int = 42,
    alpha: float = 0.5,
    step_advantage_mode: str = "pooled",
):
    print(f"Launching STEP-PGR | steps={max_steps} k={k} seed={seed} alpha={alpha} mode={step_advantage_mode}")
    result = train.remote(
        max_steps=max_steps, k=k, seed=seed, alpha=alpha,
        step_advantage_mode=step_advantage_mode,
    )
    print(result)
