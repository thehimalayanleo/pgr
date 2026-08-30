#!/usr/bin/env python3
"""Synthetic regression test for the fail-closed FCE online auditor."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import audit_fce_online as audit
from build_fcc_bank import panel_seed, prompt_hash
from fcc_reward import build_frozen_target
from fce_reward import attach_frozen_cross_evidence


def completion(reasoning: str, answer: str) -> str:
    return f"{reasoning}\nTherefore the result is \\boxed{{{answer}}}."


def main() -> None:
    audit.MODEL = "synthetic-model"
    audit.TRAIN_N = 2
    audit.PANEL_SIZE = 2
    audit.BASE_SEED = 7
    audit.MIN_ACCEPTED = 1
    audit.HELDOUT_OFFSET = 0
    audit.HELDOUT_N = 2
    audit.BOOTSTRAP_DRAWS = 100

    train_rows = [
        {"question": "Alice has one apple."},
        {"question": "Bob has two pears."},
    ]
    test_rows = [
        {"question": "Carol has three plums.", "answer": "work #### 3"},
        {"question": "Dan has four figs.", "answer": "work #### 4"},
    ]
    panels = [
        (
            [
                completion("Count Alice's single fruit.", "1"),
                completion("There is exactly one object.", "1"),
            ],
            [
                completion("A direct tally gives one.", "1"),
                completion("No addition is needed; the count is one.", "1"),
            ],
        ),
        (
            [
                completion("Count Bob's pair of fruit.", "2"),
                completion("The collection contains two objects.", "2"),
            ],
            [
                completion("A direct tally gives two.", "2"),
                completion("One plus one makes two.", "2"),
            ],
        ),
    ]
    items = []
    for index, (row, (panel_a, panel_b)) in enumerate(zip(train_rows, panels)):
        prompt = audit.prompt_for(row["question"])
        key = prompt_hash(prompt)
        items.append(
            {
                "index": index,
                "prompt_hash": key,
                "prompt": prompt,
                "panel_a": panel_a,
                "panel_b": panel_b,
                "candidates": [],
                "panel_seeds": {
                    name: panel_seed(audit.BASE_SEED, key, name)
                    for name in ("a", "b")
                },
                "frozen_target": build_frozen_target(
                    panel_a,
                    panel_b,
                ).to_dict(),
            }
        )
    items = attach_frozen_cross_evidence(
        items,
        panel_size=audit.PANEL_SIZE,
    )
    assert all(item["frozen_evidence"]["scores"] for item in items)

    bank = {
        "method": "Frozen Cross-Evidence / Cross-Consensus bank",
        "model": audit.MODEL,
        "dataset": audit.DATASET,
        "split": "train",
        "panel_size": audit.PANEL_SIZE,
        "candidate_panel_size": 0,
        "seed": audit.BASE_SEED,
        "candidate_panel_included": False,
        "frozen_evidence_attached": True,
        "gold_stored": False,
        "items": items,
        "partial": False,
    }

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bank_path = root / "bank.json"
        bank_path.write_text(json.dumps(bank))
        summary = audit.audit_bank(bank_path, train_rows, test_rows)
        assert summary["items"] == 2
        assert summary["accepted_evidence_prompts"] == 2
        assert summary["frozen_evidence_recomputed_exactly"] is True

        expected = audit.expected_eval_rows(test_rows)
        records = [
            {
                "prompt_hash": expected[0]["prompt_hash"],
                "gold": "3",
                "prediction": "3",
                "correct": True,
                "completion": "three",
            },
            {
                "prompt_hash": expected[1]["prompt_hash"],
                "gold": "4",
                "prediction": "5",
                "correct": False,
                "completion": "five",
            },
        ]
        evaluation = {
            "checkpoint": audit.MODEL,
            "dataset": audit.DATASET,
            "split": "test",
            "offset": 0,
            "n": 2,
            "decoding": "greedy",
            "correct": 1,
            "accuracy": 0.5,
            "records": records,
        }
        eval_path = root / "eval.json"
        eval_path.write_text(json.dumps(evaluation))
        eval_summary, correctness = audit.audit_evaluation(
            eval_path,
            checkpoint=audit.MODEL,
            expected_rows=expected,
        )
        assert eval_summary["accuracy"] == 0.5
        assert correctness == [1.0, 0.0]

        model_dir = root / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        (model_dir / "tokenizer.json").write_text("{}")
        (model_dir / "tokenizer_config.json").write_text("{}")
        (model_dir / "model.safetensors").write_bytes(b"weights")
        model_summary = audit.audit_final_model(model_dir)
        assert model_summary["weight_files"]["model.safetensors"]["bytes"] == 7

        broken = json.loads(bank_path.read_text())
        broken["items"][0]["candidates"] = ["leak"]
        bank_path.write_text(json.dumps(broken))
        try:
            audit.audit_bank(bank_path, train_rows, test_rows)
        except ValueError as error:
            assert "candidate leakage" in str(error)
        else:
            raise AssertionError("auditor accepted a contaminated training bank")

    print("online-audit regression passed")


if __name__ == "__main__":
    main()
