#!/usr/bin/env python3
"""Generate a frozen two-panel pseudo-target bank with the initial policy only.

The output intentionally contains no gold answers. A third independent candidate panel
is included for evaluation and power analysis, but never participates in target building.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from fcc_reward import build_frozen_target
from fce_reward import attach_frozen_cross_evidence
from fce_tasks import load_task_dataset, prompt_for_example, task_spec


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def panel_seed(base_seed: int, prompt_key: str, panel_name: str) -> int:
    material = f"{base_seed}:{prompt_key}:{panel_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63)


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="openai/gsm8k")
    parser.add_argument(
        "--split",
        default=None,
        help="dataset split (defaults to the task adapter's training split)",
    )
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--panel-size", type=int, default=4)
    parser.add_argument("--candidate-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument(
        "--max-new-items",
        type=int,
        default=0,
        help=(
            "stop after this many newly generated prompts while leaving an "
            "atomically resumable partial bank; 0 means no limit"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--omit-candidates",
        action="store_true",
        help="generate only the two frozen panels (for online training banks)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.save_every <= 0:
        parser.error("--save-every must be positive")
    if args.max_new_items < 0:
        parser.error("--max-new-items must be nonnegative")
    spec = task_spec(args.dataset)
    args.split = args.split or spec.train_split

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset = load_task_dataset(args.dataset, args.split)
    prompts = [
        prompt_for_example(example, args.dataset)
        for example in list(dataset)[: args.n]
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    existing: dict[str, dict] = {}
    if args.output.exists():
        prior = json.loads(args.output.read_text())
        expected = {
            "model": args.model,
            "dataset": args.dataset,
            "split": args.split,
            "panel_size": args.panel_size,
            "candidate_panel_size": (
                0 if args.omit_candidates else args.candidate_size
            ),
            "seed": args.seed,
            "candidate_panel_included": not args.omit_candidates,
            "answer_mode": spec.answer_mode,
        }
        mismatches = {
            key: (prior.get(key), value)
            for key, value in expected.items()
            if (
                prior.get(key, "numeric" if key == "answer_mode" else None)
                != value
            )
        }
        if mismatches:
            raise ValueError(
                f"existing bank configuration mismatch: {mismatches}"
            )
        existing = {item["prompt_hash"]: item for item in prior.get("items", [])}

    items: list[dict] = []
    new_items = 0
    for index, prompt in enumerate(prompts):
        key = prompt_hash(prompt)
        if key in existing:
            items.append(existing[key])
            continue

        panels: dict[str, list[str]] = {}
        seeds: dict[str, int] = {}
        panel_names = ("a", "b") if args.omit_candidates else ("a", "b", "candidate")
        for panel_name in panel_names:
            seed = panel_seed(args.seed, key, panel_name)
            seeds[panel_name] = seed
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            completions: list[str] = []
            sample_count = (
                args.candidate_size
                if panel_name == "candidate"
                else args.panel_size
            )
            for start in range(0, sample_count, args.batch_size):
                size = min(args.batch_size, sample_count - start)
                encoded = tokenizer(
                    [prompt] * size,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                ).to("cuda")
                with torch.no_grad():
                    output = model.generate(
                        **encoded,
                        max_new_tokens=args.max_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=0.95,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                completions.extend(
                    tokenizer.batch_decode(
                        output[:, encoded.input_ids.shape[1] :],
                        skip_special_tokens=True,
                    )
                )
            panels[panel_name] = completions

        panel_a = panels["a"]
        panel_b = panels["b"]
        candidates = panels.get("candidate", [])
        decision = build_frozen_target(
            panel_a,
            panel_b,
            answer_mode=spec.answer_mode,
        )
        items.append(
            {
                "index": index,
                "prompt_hash": key,
                "prompt": prompt,
                "panel_a": panel_a,
                "panel_b": panel_b,
                "candidates": candidates,
                "panel_seeds": seeds,
                "frozen_target": decision.to_dict(),
            }
        )
        new_items += 1
        if (index + 1) % args.save_every == 0:
            payload = {
                "method": "Frozen Cross-Evidence / Cross-Consensus bank",
                "model": args.model,
                "dataset": args.dataset,
                "dataset_config": spec.config,
                "split": args.split,
                "answer_mode": spec.answer_mode,
                "panel_size": args.panel_size,
                "candidate_panel_size": (
                    0 if args.omit_candidates else args.candidate_size
                ),
                "seed": args.seed,
                "candidate_panel_included": not args.omit_candidates,
                "frozen_evidence_attached": False,
                "gold_stored": False,
                "items": items,
                # Only the final, globally calibrated write may be complete.
                "partial": True,
            }
            write_json_atomic(args.output, payload)
        if (index + 1) % 25 == 0:
            accepted = sum(item["frozen_target"]["accepted"] for item in items)
            print(
                f"{index + 1}/{len(prompts)} accepted={accepted}/{len(items)}",
                flush=True,
            )
        if args.max_new_items and new_items >= args.max_new_items:
            break

    if len(items) < len(prompts):
        # The regular cadence may not align with max_new_items. Always leave a
        # complete atomic partial write before voluntarily returning the GPU.
        payload = {
            "method": "Frozen Cross-Evidence / Cross-Consensus bank",
            "model": args.model,
            "dataset": args.dataset,
            "dataset_config": spec.config,
            "split": args.split,
            "answer_mode": spec.answer_mode,
            "panel_size": args.panel_size,
            "candidate_panel_size": (
                0 if args.omit_candidates else args.candidate_size
            ),
            "seed": args.seed,
            "candidate_panel_included": not args.omit_candidates,
            "frozen_evidence_attached": False,
            "gold_stored": False,
            "items": items,
            "partial": True,
        }
        write_json_atomic(args.output, payload)
        print(
            f"paused {args.output} at {len(items)}/{len(prompts)} prompts "
            f"after {new_items} new item(s)",
            flush=True,
        )
        return

    # Recompute every derived field at finalization. This makes a resumed bank
    # adopt the current canonicalizer even when its raw panels were saved earlier.
    for item in items:
        item["frozen_target"] = build_frozen_target(
            item["panel_a"],
            item["panel_b"],
            answer_mode=spec.answer_mode,
        ).to_dict()
    items = attach_frozen_cross_evidence(
        items,
        panel_size=args.panel_size,
        answer_mode=spec.answer_mode,
    )
    payload = {
        "method": "Frozen Cross-Evidence / Cross-Consensus bank",
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": spec.config,
        "split": args.split,
        "answer_mode": spec.answer_mode,
        "panel_size": args.panel_size,
        "candidate_panel_size": (
            0 if args.omit_candidates else args.candidate_size
        ),
        "seed": args.seed,
        "candidate_panel_included": not args.omit_candidates,
        "frozen_evidence_attached": True,
        "gold_stored": False,
        "items": items,
        "partial": False,
    }
    write_json_atomic(args.output, payload)
    print(f"wrote {args.output} ({len(items)} prompts)")


if __name__ == "__main__":
    main()
