# modal_eval.py
"""
Evaluate base model, binary GRPO checkpoint, and PGR checkpoint
against MATH Level 5 problems.

All three evals run in parallel.
Runtime: ~15 min
Cost:    ~$0.50
"""

import modal

app = modal.App("pgr-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "vllm", "datasets", "numpy")
)

volume = modal.Volume.from_name("pgr-artifacts")


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    volumes={"/artifacts": volume}
)
def eval_checkpoint(checkpoint_path: str, n_problems: int = 100):
    import re
    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    def extract_answer(text):
        m = re.search(r'\\boxed\{(.+?)\}', text)
        return m.group(1).strip() if m else None

    ds   = load_dataset("lighteval/MATH-Hard", split="test")
    hard = list(ds)[:n_problems]

    llm    = LLM(model=checkpoint_path, max_model_len=1024)
    params = SamplingParams(temperature=0.0, max_tokens=400)

    correct = 0
    for ex in hard:
        prompt = f"Solve step by step:\n{ex['problem']}\n\nSolution:"
        out    = llm.generate([prompt], params)[0].outputs[0].text
        pred   = extract_answer(out)
        gold   = extract_answer(ex["solution"])
        if pred and gold and pred == gold:
            correct += 1

    return {
        "checkpoint": checkpoint_path,
        "accuracy": round(correct / n_problems, 4),
        "correct": correct,
        "total": n_problems,
    }


@app.local_entrypoint()
def main():
    checkpoints = [
        "Qwen/Qwen2.5-3B-Instruct",               # base model
        "/artifacts/checkpoints/binary_final",     # binary GRPO
        "/artifacts/checkpoints/pgr_final",        # PGR
    ]

    results = list(eval_checkpoint.map(checkpoints))

    print("\n=== EVAL RESULTS ===")
    for r in results:
        label = r["checkpoint"].split("/")[-1]
        print(f"  {label:25s}  acc={r['accuracy']:.3f}  ({r['correct']}/{r['total']})")
