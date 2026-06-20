# modal_eval_all.py
"""
Eval ALL existing checkpoints in the volume.
Incremental save per-problem so preemption-safe.
"""

import modal
from modal_config import image_inference, volume, VOLUME_MOUNT, DATASET_NAME

app = modal.App("pgr-eval-all")


@app.function(
    image=image_inference,
    gpu="A10G",
    timeout=2400,                # 40 min budget per checkpoint
    volumes=VOLUME_MOUNT,
)
def eval_checkpoint(checkpoint_path: str, n_problems: int = 100):
    import json, os
    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    from pgr_utils import extract_boxed_answer

    ds   = load_dataset(DATASET_NAME, split="test")
    hard = list(ds)[:n_problems]

    safe_label = checkpoint_path.replace("/", "_").replace(":", "_")
    result_path = f"/artifacts/eval_results/{safe_label}.json"
    os.makedirs("/artifacts/eval_results", exist_ok=True)

    # Resume support — skip problems already done
    start = 0
    correct = 0
    if os.path.exists(result_path):
        try:
            prev = json.loads(open(result_path).read())
            start = prev.get("total_attempted", 0)
            correct = prev.get("correct", 0)
            if start >= n_problems:
                print(f"[ALREADY DONE] {checkpoint_path}")
                return prev
            print(f"[RESUMING] {checkpoint_path} from problem {start}")
        except Exception:
            pass

    llm    = LLM(model=checkpoint_path, max_model_len=2048)
    params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024, seed=42)

    for i in range(start, n_problems):
        ex = hard[i]
        prompt = f"Solve step by step:\n{ex['problem']}\n\nSolution:"
        out    = llm.generate([prompt], params)[0].outputs[0].text
        pred   = extract_boxed_answer(out)
        gold   = extract_boxed_answer(ex["solution"])
        if pred and gold and pred == gold:
            correct += 1

        with open(result_path, "w") as f:
            json.dump({
                "checkpoint": checkpoint_path,
                "correct": correct,
                "total_attempted": i + 1,
                "total_planned": n_problems,
                "running_acc": round(correct / (i + 1), 4),
            }, f, indent=2)
        if (i + 1) % 5 == 0:
            volume.commit()

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
    # All checkpoints worth evaluating
    checkpoints = [
        # base
        "Qwen/Qwen2.5-3B-Instruct",
        # 100-step runs, 3 seeds × 2 modes (full 6-cell ablation)
        "/artifacts/checkpoints/binary_seed42_steps100_final",
        "/artifacts/checkpoints/binary_seed43_steps100_final",
        "/artifacts/checkpoints/binary_seed44_steps100_final",
        "/artifacts/checkpoints/pgr_seed42_steps100_final",
        "/artifacts/checkpoints/pgr_seed43_steps100_final",
        "/artifacts/checkpoints/pgr_seed44_steps100_final",
        # 300-step runs (the high-signal eval)
        "/artifacts/checkpoints/binary_seed42_steps300_final",
        "/artifacts/checkpoints/pgr_seed42_steps300/checkpoint-275",
    ]

    print(f"Evaluating {len(checkpoints)} checkpoints × 100 problems each")
    results = list(eval_checkpoint.map(checkpoints))

    print("\n=== EVAL RESULTS (Qwen2.5-3B, MATH-Hard Level 5, n=100) ===\n")
    for r in results:
        label = r["checkpoint"].split("/")[-1] if "/" in r["checkpoint"] else r["checkpoint"]
        print(f"  {label:55s}  acc={r['accuracy']:.4f}  ({r['correct']}/{r['total']})")
