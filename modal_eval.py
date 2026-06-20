# modal_eval.py
"""
Evaluate base model, binary GRPO checkpoint, and PGR checkpoint
against MATH Level 5 problems.

All three evals run in parallel.
Runtime: ~15 min
Cost:    ~$0.50
"""

import modal
from modal_config import image_inference, volume, VOLUME_MOUNT, DATASET_NAME

app = modal.App("pgr-eval")


@app.function(
    image=image_inference,
    gpu="A10G",
    timeout=1800,
    volumes=VOLUME_MOUNT,
)
def eval_checkpoint(checkpoint_path: str, n_problems: int = 100):
    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    from pgr_utils import extract_boxed_answer

    ds   = load_dataset(DATASET_NAME, split="test")
    hard = list(ds)[:n_problems]

    # Bumped: max_tokens=1024 (training completion_length ~500, need headroom for answer)
    # Use sampling temperature=0.7 to match training distribution (not greedy)
    llm    = LLM(model=checkpoint_path, max_model_len=2048)
    params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024, seed=42)

    import json, os
    correct = 0
    safe_label = checkpoint_path.replace("/", "_").replace(":", "_")
    result_path = f"/artifacts/eval_results/{safe_label}.json"
    os.makedirs("/artifacts/eval_results", exist_ok=True)

    for i, ex in enumerate(hard):
        prompt = f"Solve step by step:\n{ex['problem']}\n\nSolution:"
        out    = llm.generate([prompt], params)[0].outputs[0].text
        pred   = extract_boxed_answer(out)
        gold   = extract_boxed_answer(ex["solution"])
        if pred and gold and pred == gold:
            correct += 1

        # Persist after EVERY problem so we never lose progress
        with open(result_path, "w") as f:
            json.dump({
                "checkpoint": checkpoint_path,
                "correct": correct,
                "total_attempted": i + 1,
                "total_planned": n_problems,
                "running_acc": round(correct / (i + 1), 4),
            }, f, indent=2)
        if (i + 1) % 5 == 0:        # commit every 5 problems
            volume.commit()

    # Final save
    final = {
        "checkpoint": checkpoint_path,
        "accuracy": round(correct / n_problems, 4),
        "correct": correct,
        "total": n_problems,
    }
    with open(result_path, "w") as f:
        json.dump(final, f, indent=2)
    volume.commit()
    print(f"\n[FINAL] {checkpoint_path}: {correct}/{n_problems} = {final['accuracy']:.4f}")
    return final


@app.local_entrypoint()
def main():
    checkpoints = [
        "Qwen/Qwen2.5-3B-Instruct",                                        # base
        "/artifacts/checkpoints/binary_seed42_steps300_final",             # 300-step binary
        "/artifacts/checkpoints/pgr_seed42_steps300/checkpoint-275",       # 275-step PGR (92% of 300)
    ]

    results = list(eval_checkpoint.map(checkpoints))

    print("\n=== EVAL RESULTS (Qwen2.5-3B, MATH-Hard Level 5, 100 problems) ===")
    for r in results:
        label = r["checkpoint"].split("/")[-1] if "/" in r["checkpoint"] else r["checkpoint"]
        print(f"  {label:50s}  acc={r['accuracy']:.4f}  ({r['correct']}/{r['total']})")
