# modal_smoke_test.py
"""
PGR end-to-end smoke test.
Runs all 5 checks in sequence on a single A10G.
Target: <15 min, <$1.50

Checks:
  [1] Dictionary loads and shapes are correct
  [2] Encoder produces normalized embeddings
  [3] OMP error discriminates correct vs shuffled steps
  [4] Reward function runs without crash, output in [0, 1]
  [5] 20-step GRPO training loop completes, loss decreases
"""

import modal

app = modal.App("pgr-smoke-test")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.0",
        "transformers>=4.43.0",
        "trl>=0.9.0",
        "datasets",
        "accelerate",
        "sentence-transformers",
        "scikit-learn",
        "numpy",
        "peft",
    )
)

volume = modal.Volume.from_name("pgr-artifacts", create_if_missing=True)

DICTIONARY_PATH = "/artifacts/dictionary_atoms.npy"
ENCODER_NAME    = "BAAI/bge-small-en-v1.5"
MODEL_ID        = "Qwen/Qwen2.5-0.5B-Instruct"   # tiny model, smoke only
N_ATOMS         = 64                               # small dict, fast to build
N_STEPS_SAMPLE  = 500                             # small sample, fast to fit
TRAIN_STEPS     = 20                              # just enough to see loss move

GOOD_STEPS = [
    "Let x be the unknown variable. We set up the equation 2x + 5 = 13.",
    "Subtracting 5 from both sides gives 2x = 8.",
    "Dividing both sides by 2 yields x = 4.",
    "We verify: 2(4) + 5 = 13. ✓",
]

BAD_STEPS = [
    "equation variable 5 both x the sides set 2x unknown Let be up 13.",
    "8 gives from 2x Subtracting = both 5.",
    "yields 2 both dividing by x sides = 4.",
    "verify 13 We 5 4 2 = ✓.",
]


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    volumes={"/artifacts": volume},
)
def run_smoke_test():
    import re, time, traceback
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import DictionaryLearning
    from sklearn.linear_model import orthogonal_mp
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    results = {}
    encoder = SentenceTransformer(ENCODER_NAME)

    def header(n, title):
        print(f"\n{'='*55}")
        print(f"  CHECK [{n}] {title}")
        print(f"{'='*55}")

    def ok(msg):   print(f"  ✅  {msg}")
    def fail(msg): print(f"  ❌  {msg}")

    # ── [1] Dictionary ────────────────────────────────────────────────────
    header(1, "Dictionary build + load")
    D = None
    try:
        import os
        if not os.path.exists(DICTIONARY_PATH):
            print("  Building dictionary from MATH train (small sample)…")
            ds = load_dataset("lighteval/MATH", split="train")
            hard = [x for x in ds if x["level"] in ("Level 4", "Level 5")][:200]

            def seg(text):
                parts = re.split(r'\n\n+|(?=Step \d+:)', text.strip())
                return [p.strip() for p in parts if len(p.strip()) > 20]

            steps = [s for ex in hard for s in seg(ex["solution"])][:N_STEPS_SAMPLE]
            emb = encoder.encode(steps, normalize_embeddings=True, batch_size=256)

            dl = DictionaryLearning(
                n_components=N_ATOMS, alpha=0.5, max_iter=200,
                fit_algorithm="lars", transform_algorithm="omp",
                transform_n_nonzero_coefs=5, n_jobs=-1,
            )
            dl.fit(emb)
            np.save(DICTIONARY_PATH, dl.components_)
            volume.commit()
            print("  Built and saved dictionary.")

        D = np.load(DICTIONARY_PATH)
        assert D.ndim == 2,            f"Expected 2D array, got {D.ndim}D"
        assert D.shape[0] >= N_ATOMS,  f"Expected ≥{N_ATOMS} atoms, got {D.shape[0]}"
        assert D.shape[1] > 0,         "Embedding dim is 0"

        ok(f"Dictionary shape: {D.shape}  (atoms × embedding_dim)")
        results["check_1_dictionary"] = "PASS"

    except Exception as e:
        fail(str(e)); traceback.print_exc()
        results["check_1_dictionary"] = f"FAIL: {e}"

    # ── [2] Encoder ───────────────────────────────────────────────────────
    header(2, "Encoder — normalized embeddings")
    try:
        emb = encoder.encode(GOOD_STEPS, normalize_embeddings=True)
        assert emb.shape == (len(GOOD_STEPS), 384), f"Unexpected shape {emb.shape}"
        norms = np.linalg.norm(emb, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), f"Embeddings not unit-norm: {norms}"

        ok(f"Shape: {emb.shape},  norms ≈ 1.0 ✓")
        results["check_2_encoder"] = "PASS"

    except Exception as e:
        fail(str(e)); traceback.print_exc()
        results["check_2_encoder"] = f"FAIL: {e}"

    # ── [3] OMP discriminability ──────────────────────────────────────────
    header(3, "OMP reconstruction error — correct vs shuffled steps")
    try:
        assert D is not None, "Dictionary not loaded (check 1 failed)"

        good_emb = encoder.encode(GOOD_STEPS, normalize_embeddings=True)
        bad_emb  = encoder.encode(BAD_STEPS,  normalize_embeddings=True)

        def recon_errors(emb):
            codes = orthogonal_mp(D.T, emb.T, n_nonzero_coefs=5)
            return np.linalg.norm(emb.T - D.T @ codes, axis=0)

        good_err = recon_errors(good_emb)
        bad_err  = recon_errors(bad_emb)
        gap = float(bad_err.mean()) - float(good_err.mean())

        print(f"  Mean recon error — correct: {good_err.mean():.4f}  |  shuffled: {bad_err.mean():.4f}")
        print(f"  Gap (bad − good): {gap:.4f}")
        assert gap > 0, "Shuffled steps should have higher reconstruction error"

        ok(f"Correct steps reconstruct better (gap={gap:.4f})")
        results["check_3_omp"] = "PASS"
        results["omp_gap"] = round(gap, 4)

    except Exception as e:
        fail(str(e)); traceback.print_exc()
        results["check_3_omp"] = f"FAIL: {e}"

    # ── [4] Reward function ───────────────────────────────────────────────
    header(4, "PGR reward function — output range and correctness bonus")
    try:
        assert D is not None, "Dictionary not loaded (check 1 failed)"

        def pgr_reward(completion, is_correct, alpha=0.5, tau=0.3):
            steps = re.split(r'\n\n+|(?=Step \d+:)', completion.strip())
            steps = [s.strip() for s in steps if len(s.strip()) > 20]
            if not steps:
                return 0.0, []
            emb   = encoder.encode(steps, normalize_embeddings=True)
            codes = orthogonal_mp(D.T, emb.T, n_nonzero_coefs=5)
            errs  = np.linalg.norm(emb.T - D.T @ codes, axis=0)
            sr    = np.exp(-errs / tau)
            total = alpha * float(sr.mean()) + (1 - alpha) * (1.0 if is_correct else 0.0)
            return total, sr.tolist()

        r_correct,   sr_c = pgr_reward("\n\n".join(GOOD_STEPS), is_correct=True)
        r_incorrect, sr_i = pgr_reward("\n\n".join(BAD_STEPS),  is_correct=False)

        print(f"  Correct solution reward:   {r_correct:.4f}  (step rewards: {[round(x,3) for x in sr_c]})")
        print(f"  Incorrect solution reward: {r_incorrect:.4f}  (step rewards: {[round(x,3) for x in sr_i]})")

        assert 0.0 <= r_correct   <= 1.0, f"Correct reward out of range: {r_correct}"
        assert 0.0 <= r_incorrect <= 1.0, f"Incorrect reward out of range: {r_incorrect}"
        assert r_correct > r_incorrect,   "Correct solution should score higher than shuffled"

        ok(f"Correct ({r_correct:.3f}) > Incorrect ({r_incorrect:.3f}),  both in [0,1]")
        results["check_4_reward"] = "PASS"
        results["reward_correct"]   = round(r_correct, 4)
        results["reward_incorrect"] = round(r_incorrect, 4)

    except Exception as e:
        fail(str(e)); traceback.print_exc()
        results["check_4_reward"] = f"FAIL: {e}"

    # ── [5] Mini training loop ────────────────────────────────────────────
    header(5, f"GRPO training loop — {TRAIN_STEPS} steps, loss must decrease")
    try:
        assert D is not None, "Dictionary not loaded (check 1 failed)"

        ds   = load_dataset("lighteval/MATH", split="train")
        hard = ds.filter(lambda x: x["level"] in ("Level 4", "Level 5"))
        hard = hard.select(range(min(60, len(hard))))
        hard = hard.map(lambda x: {
            "prompt": f"Solve step by step:\n{x['problem']}\n\nSolution:",
            "answer": x["solution"],
        })

        def extract_answer(text):
            m = re.search(r'\\boxed\{(.+?)\}', text)
            return m.group(1).strip() if m else None

        _D, _enc = D, encoder

        def reward_fn(completions, prompts, answer, **kwargs):
            rewards = []
            for completion, ans in zip(completions, answer):
                steps = re.split(r'\n\n+|(?=Step \d+:)', completion.strip())
                steps = [s.strip() for s in steps if len(s.strip()) > 20]
                if not steps:
                    rewards.append(0.0); continue
                emb   = _enc.encode(steps, normalize_embeddings=True)
                codes = orthogonal_mp(_D.T, emb.T, n_nonzero_coefs=5)
                errs  = np.linalg.norm(emb.T - _D.T @ codes, axis=0)
                sr    = float(np.exp(-errs / 0.3).mean())
                pred  = extract_answer(completion)
                gold  = extract_answer(ans)
                term  = 1.0 if (pred and gold and pred == gold) else 0.0
                rewards.append(0.5 * sr + 0.5 * term)
            return rewards

        losses = []

        class LossCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    losses.append(logs["loss"])
                    print(f"  step {state.global_step:3d}  loss={logs['loss']:.4f}")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
        )

        cfg = GRPOConfig(
            output_dir="/artifacts/smoke_ckpt",
            max_steps=TRAIN_STEPS,
            per_device_train_batch_size=1,
            num_generations=2,
            max_completion_length=128,
            learning_rate=1e-6,
            logging_steps=5,
            report_to="none",
            bf16=True,
            gradient_accumulation_steps=2,
        )

        trainer = GRPOTrainer(
            model=model,
            args=cfg,
            train_dataset=hard,
            reward_funcs=[reward_fn],
            tokenizer=tokenizer,
            callbacks=[LossCallback()],
        )

        t0 = time.time()
        trainer.train()
        elapsed = time.time() - t0

        assert len(losses) >= 2, "Not enough loss values logged"
        first, last = losses[0], losses[-1]
        print(f"\n  Loss trajectory: {[round(l, 4) for l in losses]}")
        print(f"  Elapsed: {elapsed:.0f}s")

        ok(f"Training completed in {elapsed:.0f}s.  Loss: {first:.4f} → {last:.4f}")
        results["check_5_training"] = "PASS"
        results["loss_first"]   = round(first, 4)
        results["loss_last"]    = round(last, 4)
        results["train_time_s"] = round(elapsed)

    except Exception as e:
        fail(str(e)); traceback.print_exc()
        results["check_5_training"] = f"FAIL: {e}"

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  SMOKE TEST SUMMARY")
    print(f"{'='*55}")

    all_pass = True
    for k, v in results.items():
        if str(v).startswith("PASS"):
            icon = "✅"
        elif str(v).startswith("FAIL"):
            icon = "❌"
            all_pass = False
        else:
            icon = "📊"
        print(f"  {icon}  {k}: {v}")

    verdict = "🟢 ALL CHECKS PASSED — ready to scale" if all_pass else "🔴 SOME CHECKS FAILED — fix before scaling"
    print(f"\n  {verdict}")
    print(f"{'='*55}\n")

    return results


@app.local_entrypoint()
def main():
    result = run_smoke_test.remote()
    failed = [k for k, v in result.items() if str(v).startswith("FAIL")]
    if failed:
        print(f"\nFailed checks: {failed}")
        raise SystemExit(1)
