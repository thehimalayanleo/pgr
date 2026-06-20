# modal_smoke_test.py
"""
PGR end-to-end smoke test.
Runs all 5 checks in sequence on a single A10G.
Target: <15 min, <$1.50

Checks:
  [1] Dictionary builds and loads with correct shape
  [2] Encoder produces unit-normalized embeddings
  [3] OMP error discriminates correct vs shuffled steps
  [4] Reward function output is in [0, 1], correct > incorrect
  [5] 20-step GRPO training loop completes without crash
"""

import modal

app = modal.App("pgr-smoke-test")

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
)

volume = modal.Volume.from_name("pgr-artifacts", create_if_missing=True)

DICTIONARY_PATH = "/artifacts/dictionary_atoms.npy"
ENCODER_NAME    = "BAAI/bge-small-en-v1.5"
MODEL_ID        = "Qwen/Qwen2.5-0.5B-Instruct"
N_ATOMS         = 64
N_STEPS_SAMPLE  = 500
TRAIN_STEPS     = 20

# Fixed examples for checks 3 & 4
GOOD_STEPS = [
    "Let x be the unknown variable. We set up the equation 2x + 5 = 13.",
    "Subtracting 5 from both sides gives 2x = 8.",
    "Dividing both sides by 2 yields x = 4.",
    "We verify: 2(4) + 5 = 13. The answer is correct.",
]

BAD_STEPS = [
    "equation variable 5 both x the sides set 2x unknown Let be up 13.",
    "8 gives from 2x Subtracting both sides equals 5.",
    "yields 2 both dividing by x sides result 4.",
    "verify 13 We 5 4 2 equals correct answer.",
]


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    volumes={"/artifacts": volume},
)
def run_smoke_test():
    import os, re, time, traceback
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

    def seg(text):
        parts = re.split(r'\n\n+|(?=Step \d+:)', text.strip())
        return [p.strip() for p in parts if len(p.strip()) > 20]

    def recon_errors(D, emb):
        """OMP reconstruction error. D: (n_atoms, d), emb: (n_steps, d)"""
        codes = orthogonal_mp(D.T, emb.T, n_nonzero_coefs=5)  # (n_atoms, n_steps)
        residual = emb.T - D.T @ codes                          # (d, n_steps)
        return np.linalg.norm(residual, axis=0)                 # (n_steps,)

    # ── [1] Dictionary ────────────────────────────────────────────────────
    header(1, "Dictionary build + load")
    D = None
    try:
        if not os.path.exists(DICTIONARY_PATH):
            print("  No cached dictionary found — building from MATH train…")
            ds   = load_dataset("lighteval/MATH-Hard", split="train")
            hard = list(ds)[:200]
            steps = [s for ex in hard for s in seg(ex["solution"])][:N_STEPS_SAMPLE]

            emb = encoder.encode(steps, normalize_embeddings=True, batch_size=256)

            dl = DictionaryLearning(
                n_components=N_ATOMS,
                alpha=0.5,
                max_iter=200,
                fit_algorithm="lars",
                transform_algorithm="omp",
                transform_n_nonzero_coefs=5,
                n_jobs=-1,
            )
            dl.fit(emb)
            os.makedirs("/artifacts", exist_ok=True)
            np.save(DICTIONARY_PATH, dl.components_)
            # Commit so the file persists in the volume after this function exits
            volume.commit()
            print("  Dictionary built and committed to volume.")

        D = np.load(DICTIONARY_PATH, allow_pickle=False)
        assert D.ndim == 2,           f"Expected 2D array, got {D.ndim}D"
        assert D.shape[0] == N_ATOMS, f"Expected {N_ATOMS} atoms, got {D.shape[0]}"
        assert D.shape[1] > 0,        "Embedding dim is 0"

        ok(f"Dictionary shape: {D.shape}  ({D.shape[0]} atoms × {D.shape[1]}d)")
        results["check_1_dictionary"] = "PASS"

    except Exception as e:
        fail(str(e)); traceback.print_exc()
        results["check_1_dictionary"] = f"FAIL: {e}"

    # ── [2] Encoder ───────────────────────────────────────────────────────
    header(2, "Encoder — unit-normalized embeddings")
    try:
        emb   = encoder.encode(GOOD_STEPS, normalize_embeddings=True)
        norms = np.linalg.norm(emb, axis=1)

        assert emb.ndim == 2,                       f"Expected 2D output, got {emb.ndim}D"
        assert emb.shape[0] == len(GOOD_STEPS),     f"Wrong number of rows: {emb.shape[0]}"
        assert np.allclose(norms, 1.0, atol=1e-4),  f"Embeddings not unit-norm: {norms}"

        ok(f"Shape: {emb.shape},  norms all ≈ 1.0")
        results["check_2_encoder"] = "PASS"

    except Exception as e:
        fail(str(e)); traceback.print_exc()
        results["check_2_encoder"] = f"FAIL: {e}"

    # ── [3] OMP discriminability ──────────────────────────────────────────
    header(3, "OMP error — correct steps reconstruct better than shuffled")
    try:
        assert D is not None, "Skipped — check 1 failed"

        good_emb = encoder.encode(GOOD_STEPS, normalize_embeddings=True)
        bad_emb  = encoder.encode(BAD_STEPS,  normalize_embeddings=True)

        good_err = recon_errors(D, good_emb)
        bad_err  = recon_errors(D, bad_emb)

        mean_good = float(good_err.mean())
        mean_bad  = float(bad_err.mean())
        gap       = mean_bad - mean_good

        print(f"  Correct step mean error:  {mean_good:.4f}")
        print(f"  Shuffled step mean error: {mean_bad:.4f}")
        print(f"  Gap (bad − good):         {gap:.4f}")

        assert gap > 0, (
            f"Shuffled steps should have HIGHER error than correct steps. "
            f"Got gap={gap:.4f}. Dictionary may be too small or encoder mismatch."
        )

        ok(f"Gap = {gap:.4f}  (correct reconstructs better ✓)")
        results["check_3_omp"]  = "PASS"
        results["omp_gap"]      = round(gap, 4)
        results["mean_err_correct"]  = round(mean_good, 4)
        results["mean_err_shuffled"] = round(mean_bad, 4)

    except Exception as e:
        fail(str(e)); traceback.print_exc()
        results["check_3_omp"] = f"FAIL: {e}"

    # ── [4] Reward function ───────────────────────────────────────────────
    header(4, "PGR reward — output in [0,1], correct > incorrect")
    try:
        assert D is not None, "Skipped — check 1 failed"

        def pgr_reward(completion, is_correct, alpha=0.5, tau=0.3):
            steps = seg(completion)
            if not steps:
                return 0.0, []
            emb   = encoder.encode(steps, normalize_embeddings=True)
            errs  = recon_errors(D, emb)
            sr    = np.exp(-errs / tau)
            term  = 1.0 if is_correct else 0.0
            total = alpha * float(sr.mean()) + (1 - alpha) * term
            return total, sr.tolist()

        r_good, sr_good = pgr_reward("\n\n".join(GOOD_STEPS), is_correct=True)
        r_bad,  sr_bad  = pgr_reward("\n\n".join(BAD_STEPS),  is_correct=False)

        print(f"  Correct   reward: {r_good:.4f}  step scores: {[round(x,3) for x in sr_good]}")
        print(f"  Incorrect reward: {r_bad:.4f}  step scores: {[round(x,3) for x in sr_bad]}")

        assert 0.0 <= r_good <= 1.0, f"Correct reward out of [0,1]: {r_good}"
        assert 0.0 <= r_bad  <= 1.0, f"Incorrect reward out of [0,1]: {r_bad}"
        assert r_good > r_bad,       f"Correct ({r_good:.4f}) should beat incorrect ({r_bad:.4f})"

        ok(f"Correct ({r_good:.3f}) > Incorrect ({r_bad:.3f}), both in [0,1]")
        results["check_4_reward"]    = "PASS"
        results["reward_correct"]    = round(r_good, 4)
        results["reward_incorrect"]  = round(r_bad,  4)

    except Exception as e:
        fail(str(e)); traceback.print_exc()
        results["check_4_reward"] = f"FAIL: {e}"

    # ── [5] Training loop ─────────────────────────────────────────────────
    header(5, f"GRPO training loop — {TRAIN_STEPS} steps, no crash")
    try:
        assert D is not None, "Skipped — check 1 failed"

        # Build dataset — keep only prompt + answer columns
        ds   = load_dataset("lighteval/MATH-Hard", split="train")
        hard = ds.select(range(min(60, len(ds))))
        hard = hard.map(
            lambda x: {
                "prompt": f"Solve step by step:\n{x['problem']}\n\nSolution:",
                "answer": x["solution"],
            },
            remove_columns=hard.column_names,   # drop all original columns
        )

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

        _D, _enc = D, encoder

        # TRL 0.9.x calls: reward_func(completions, prompts=..., **other_cols)
        # "answer" arrives via **kwargs; "prompts" is passed explicitly by TRL
        def reward_fn(completions, **kwargs):
            answer  = kwargs.get("answer", [""] * len(completions))
            rewards = []
            for completion, ans in zip(completions, answer):
                steps = seg(completion)
                if not steps:
                    rewards.append(0.0)
                    continue
                emb   = _enc.encode(steps, normalize_embeddings=True)
                errs  = recon_errors(_D, emb)
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
                    losses.append(float(logs["loss"]))
                    print(f"  step {state.global_step:3d}  loss={logs['loss']:.4f}")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        cfg = GRPOConfig(
            output_dir="/artifacts/smoke_ckpt",
            max_steps=TRAIN_STEPS,
            per_device_train_batch_size=1,
            num_generations=2,
            max_completion_length=128,
            learning_rate=1e-6,
            logging_steps=5,
            save_steps=999,          # don't save mid-run
            report_to="none",
            bf16=True,
            gradient_accumulation_steps=2,
            dataloader_num_workers=0,
        )

        trainer = GRPOTrainer(
            model=model,
            args=cfg,
            train_dataset=hard,
            reward_funcs=[reward_fn],
            processing_class=tokenizer,
            callbacks=[LossCallback()],
        )

        t0 = time.time()
        trainer.train()
        elapsed = round(time.time() - t0)

        print(f"\n  Loss trajectory: {[round(l, 4) for l in losses]}")
        print(f"  Elapsed: {elapsed}s")

        # Smoke test: just need training to complete and log at least one loss
        assert len(losses) >= 1, "No loss values logged — training may have crashed silently"

        first, last = losses[0], losses[-1]
        ok(f"Completed in {elapsed}s.  Loss: {first:.4f} → {last:.4f}")
        results["check_5_training"] = "PASS"
        results["loss_first"]       = round(first, 4)
        results["loss_last"]        = round(last,  4)
        results["train_time_s"]     = elapsed

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
        elif str(v).startswith("FAIL") or str(v).startswith("Skipped"):
            icon = "❌"
            all_pass = False
        else:
            icon = "📊"
        print(f"  {icon}  {k}: {v}")

    verdict = (
        "🟢 ALL CHECKS PASSED — ready to scale"
        if all_pass else
        "🔴 SOME CHECKS FAILED — fix before scaling"
    )
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
