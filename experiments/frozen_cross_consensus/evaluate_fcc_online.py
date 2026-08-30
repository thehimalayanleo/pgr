#!/usr/bin/env python3
"""Deterministic, gold-separated evaluation for an FCC-trained checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from fcc_reward import extract_answer
from fce_tasks import (
    gold_for_example,
    load_task_dataset,
    prompt_for_example,
    task_spec,
)


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--dataset", default="openai/gsm8k")
    parser.add_argument("--split", default=None)
    parser.add_argument("--offset", type=int, default=500)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = task_spec(args.dataset)
    selected_split = args.split or spec.eval_split
    dataset = list(load_task_dataset(args.dataset, selected_split))
    selected = dataset[args.offset : args.offset + args.n]
    if len(selected) != args.n:
        raise ValueError(
            f"requested test[{args.offset}:{args.offset + args.n}], "
            f"but only {len(selected)} examples are available"
        )
    problems = [
        {
            "prompt": prompt_for_example(example, args.dataset),
            "gold": gold_for_example(example, args.dataset),
        }
        for example in selected
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    records: list[dict] = []
    started = time.time()
    for start in range(0, len(problems), args.batch_size):
        batch = problems[start : start + args.batch_size]
        encoded = tokenizer(
            [item["prompt"] for item in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to("cuda")
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = output[:, encoded.input_ids.shape[1] :]
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for item, text in zip(batch, texts):
            prediction = extract_answer(text, spec.answer_mode)
            records.append(
                {
                    "prompt_hash": hashlib.sha256(
                        item["prompt"].encode("utf-8")
                    ).hexdigest(),
                    "gold": item["gold"],
                    "prediction": prediction,
                    "correct": bool(prediction == item["gold"]),
                    "completion": text,
                }
            )
        print(
            f"{len(records)}/{len(problems)} "
            f"correct={sum(record['correct'] for record in records)} "
            f"elapsed={time.time() - started:.0f}s",
            flush=True,
        )

    correct = sum(record["correct"] for record in records)
    result = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "split": selected_split,
        "answer_mode": spec.answer_mode,
        "offset": args.offset,
        "n": len(records),
        "decoding": "greedy",
        "correct": correct,
        "accuracy": correct / len(records),
        "records": records,
    }
    write_json_atomic(args.output, result)
    print(json.dumps({key: result[key] for key in result if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
