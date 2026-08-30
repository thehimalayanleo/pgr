#!/usr/bin/env python3
"""Focused regression test for FCE's streamed exact-resume contract."""

from __future__ import annotations

import ast
import json
import os
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


class FakeStepLevelGRPOTrainer:
    def _get_output_dir(self, trial=None):
        return self.output_dir

    def _save_checkpoint(self, model, trial, metrics=None):
        checkpoint = Path(self.output_dir) / f"checkpoint-{self.state.global_step}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "model.safetensors").write_bytes(b"model")
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": self.state.global_step})
        )

    def _save_rng_state(self, output_dir):
        torch.save(
            {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "cpu": torch.random.get_rng_state(),
            },
            Path(output_dir) / "rng_state.pth",
        )

    def _load_optimizer_and_scheduler(self, checkpoint):
        raise AssertionError(
            "custom streamed-resume checkpoints must bypass the parent "
            "monolithic optimizer loader"
        )


def load_resume_trainer_class():
    source = Path(__file__).with_name("local_fcc_train.py").read_text()
    tree = ast.parse(source)
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    trainer_class = next(
        node
        for node in main.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ResumableStepLevelGRPOTrainer"
    )
    module = ast.Module(body=[trainer_class], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "StepLevelGRPOTrainer": FakeStepLevelGRPOTrainer,
        "PREFIX_CHECKPOINT_DIR": "checkpoint",
        "json": json,
        "os": os,
        "tempfile": tempfile,
        "torch": torch,
    }
    exec(compile(module, "local_fcc_train.py", "exec"), namespace)
    return namespace["ResumableStepLevelGRPOTrainer"]


def main() -> None:
    trainer_class = load_resume_trainer_class()
    with tempfile.TemporaryDirectory() as directory:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW(
            [parameter],
            lr=1.0,
            betas=(0.8, 0.95),
            weight_decay=0.1,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: max(0.0, 1.0 - step / 100.0),
        )
        for _ in range(17):
            parameter.grad = torch.tensor(0.25)
            optimizer.step()
            scheduler.step()
        expected_epoch = scheduler.last_epoch
        expected_lr = scheduler.get_last_lr()
        expected_optimizer_state = {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in optimizer.state[parameter].items()
        }

        trainer = object.__new__(trainer_class)
        trainer.output_dir = directory
        trainer.args = SimpleNamespace(
            save_only_model=True,
            should_save=True,
            world_size=1,
            process_index=0,
        )
        trainer.state = SimpleNamespace(global_step=50)
        trainer.lr_scheduler = scheduler
        trainer.optimizer = optimizer
        trainer._save_checkpoint(None, None)

        checkpoint = Path(directory) / "checkpoint-50"
        manifest = json.loads(
            (checkpoint / "lightweight_resume_state.json").read_text()
        )
        assert manifest["global_step"] == 50
        assert manifest["scheduler_state_saved"] is True
        assert manifest["rng_state_saved"] is True
        assert manifest["optimizer_state_saved"] is True
        assert manifest["optimizer_reset_on_resume"] is False
        assert manifest["format"] == "fce-streamed-exact-resume-v2"
        assert (checkpoint / "scheduler.pt").is_file()
        assert (checkpoint / "rng_state.pth").is_file()
        assert (
            checkpoint
            / manifest["optimizer_state_dir"]
            / "metadata.pt"
        ).is_file()

        fresh_parameter = torch.nn.Parameter(parameter.detach().clone())
        fresh_optimizer = torch.optim.AdamW(
            [fresh_parameter],
            lr=0.25,
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )
        trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            fresh_optimizer,
            lambda step: max(0.0, 1.0 - step / 100.0),
        )
        trainer.optimizer = fresh_optimizer
        assert trainer.lr_scheduler.last_epoch != expected_epoch
        assert fresh_optimizer.param_groups[0]["lr"] != expected_lr[0]
        # Transformers loads optimizer/scheduler before replacing self.state
        # with trainer_state.json, so the loader must not trust this stale value.
        trainer.state = SimpleNamespace(global_step=0)
        trainer._load_optimizer_and_scheduler(str(checkpoint))
        assert trainer.lr_scheduler.last_epoch == expected_epoch
        assert trainer.lr_scheduler.get_last_lr() == expected_lr
        assert fresh_optimizer.param_groups[0]["lr"] == expected_lr[0]
        assert fresh_optimizer.param_groups[0]["betas"] == (0.8, 0.95)
        assert fresh_optimizer.param_groups[0]["weight_decay"] == 0.1
        restored_state = fresh_optimizer.state[fresh_parameter]
        assert restored_state.keys() == expected_optimizer_state.keys()
        for key, expected in expected_optimizer_state.items():
            actual = restored_state[key]
            if torch.is_tensor(expected):
                assert torch.equal(actual, expected)
            else:
                assert actual == expected

        # The first update after resume must be identical to uninterrupted Adam.
        parameter.grad = torch.tensor(-0.4)
        fresh_parameter.grad = torch.tensor(-0.4)
        optimizer.step()
        scheduler.step()
        fresh_optimizer.step()
        trainer.lr_scheduler.step()
        assert torch.equal(fresh_parameter, parameter)
        assert trainer.lr_scheduler.state_dict() == scheduler.state_dict()
        for key, expected in optimizer.state[parameter].items():
            actual = fresh_optimizer.state[fresh_parameter][key]
            if torch.is_tensor(expected):
                assert torch.equal(actual, expected)
            else:
                assert actual == expected

        (checkpoint / "lightweight_resume_state.json").unlink()
        try:
            trainer._load_optimizer_and_scheduler(str(checkpoint))
        except ValueError as error:
            assert "lacks certified exact-resume state" in str(error)
        else:
            raise AssertionError("uncertified model-only checkpoint was accepted")

    print("resume-state regression passed")


if __name__ == "__main__":
    main()
