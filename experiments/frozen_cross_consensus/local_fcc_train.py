"""
LOCAL L2a training on RTX 5090 (32GB VRAM, sm_120).

Runs the same L2a step-PGR trainer as our Modal smoke, but locally.
Handles 5090 quirks:
  - Uses fp32 for OMP (sklearn), bf16 for the policy model
  - K=4 with Qwen-3B fits in 32GB comfortably
  - K=8 with Qwen-3B may be tight — halve max_completion_length if OOM

Usage (from repo root, with pursuit_env or equivalent):
    python local_5090_train.py --max-steps 200 --seed 42 --k 4 --use-l2a
    python local_5090_train.py --max-steps 200 --seed 42 --k 8 --use-l2a --max-completion 256

Runtime rough estimate (5090 vs H100):
    5090 is ~50-70% of H100 speed for LLM inference
    100 steps × ~60s/step ≈ 100 min
    200 steps ≈ 3.5 hours
    500 steps ≈ 9 hours (overnight)

Saves checkpoint to ./checkpoints/<name>_final for evaluation.
"""

import argparse, hashlib, json, os, sys, random, tempfile
import numpy as np
import torch
from fce_data import FROZEN_REWARD_SOURCES, prepare_training_example
from fce_tasks import load_task_dataset, task_spec
# peft 0.19 checks `isinstance(w, torch.distributed.tensor.DTensor)` without importing
# the submodule; on this torch build it isn't auto-imported and LoRA injection dies.
# MUST live at module level: a function-level `import torch.distributed.tensor` binds
# `torch` as a function-local and breaks every earlier torch reference in that function.
try:
    import torch.distributed.tensor  # noqa: F401
except Exception:
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=4, help="num_generations per group")
    p.add_argument("--use-l2a", action="store_true", help="enable L2a (else trajectory baseline)")
    p.add_argument("--use-hybrid", action="store_true", help="hybrid reward: terminal + length + confidence")
    p.add_argument("--reward-source", default="gold",
                   choices=[
                       "gold",
                       "majority",
                       "random",
                       "consensus",
                       "fcc",
                       "fce",
                       "fce_permuted",
                   ],
                   help="gold=RLVR (verifier); majority=verifier-free self-consistency; "
                        "random=spurious-reward control; consensus=independence-weighted "
                        "semantic agreement without answer parsing; fcc=frozen "
                        "two-panel cross-consensus; fce=frozen cross-panel "
                        "evidence scores; fce_permuted=matched control with the "
                        "same group rewards uniformly shuffled across trajectories")
    p.add_argument("--fcc-bank", default=None,
                   help="complete gold-free frozen bank JSON; required for FCC/FCE sources")
    p.add_argument("--random-reward-p", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.5,
                   help="blend: alpha*OMP_step + (1-alpha)*terminal. alpha=0 => pure "
                        "terminal; with --terminal-spread constant and "
                        "--step-advantage-mode group_mean this reduces EXACTLY to GRPO "
                        "group-norm (one scalar advantage per rollout).")
    p.add_argument("--step-advantage-mode", default="ema",
                   choices=["pooled", "group_mean", "ema"],
                   help="ema = PPO-style running baseline (preserves per-step credit but "
                        "turns ANY reward into self-distillation: random matched gold on "
                        "both Qwen and SmolLM2). group_mean = GRPO-style group-relative "
                        "baseline (robust to reward noise; random should FAIL).")
    p.add_argument("--terminal-spread", default="positional",
                   choices=["last_only", "uniform", "omp_weighted", "positional",
                            "constant", "signed_positional"],
                   help="how the terminal reward is distributed across steps. "
                        "'positional' only shapes SUCCESSFUL rollouts (terminal*ramp is "
                        "0 everywhere when terminal=0); 'signed_positional' fixes that.")
    p.add_argument("--use-lora", action="store_true",
                   help="LoRA (r=16) instead of full finetune. Required for 7B on the "
                        "32GB 5090: full-FT Adam states alone are ~56GB for 7B; frozen "
                        "base + LoRA lands ~20GB. Final save merges adapters into the "
                        "base model so local_5090_eval.py can load the checkpoint dir "
                        "with plain AutoModelForCausalLM.")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="gradient checkpointing (~30%% slower, large activation-memory "
                        "saving). Required for 7B on 32GB: without it the logprob "
                        "forward peaks at 30.7GB and OOMs at step 1.")
    p.add_argument("--resume-from", default=None,
                   help="path to a checkpoint-N dir to resume from")
    # Each 1.5B checkpoint is ~8.7GB (optimizer state dominates). Saving every 25
    # steps and keeping 3 means ~26GB of rolling disk churn, which on this 29GB box
    # is a real contributor to OOM-killer pressure when another job shares the host.
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--save-total-limit", type=int, default=1)
    p.add_argument("--max-completion", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--dataset", default="lighteval/MATH-Hard",
                   choices=[
                       "lighteval/MATH-Hard",
                       "openai/gsm8k",
                       "cais/mmlu",
                   ])
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--encoder", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--dict-path", default="./dictionary_atoms.npy")
    p.add_argument("--output-suffix", default="local5090")
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        print(f"CUDA: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
    else:
        print("WARN: CUDA not available — this will be very slow on CPU/MPS.")

    experiment_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, experiment_dir)
    from fcc_step_pgr_trainer import StepLevelGRPOTrainer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import GRPOConfig
    from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

    class ResumableStepLevelGRPOTrainer(StepLevelGRPOTrainer):
        """Save exact resume state without a monolithic Adam serialization spike."""

        _resume_manifest = "lightweight_resume_state.json"
        _resume_format = "fce-streamed-exact-resume-v2"

        @staticmethod
        def _tree_to_cpu(value):
            if torch.is_tensor(value):
                return value.detach().cpu()
            if isinstance(value, dict):
                return {
                    key: ResumableStepLevelGRPOTrainer._tree_to_cpu(item)
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [
                    ResumableStepLevelGRPOTrainer._tree_to_cpu(item)
                    for item in value
                ]
            if isinstance(value, tuple):
                return tuple(
                    ResumableStepLevelGRPOTrainer._tree_to_cpu(item)
                    for item in value
                )
            return value

        @staticmethod
        def _tree_to_device(value, device):
            if torch.is_tensor(value):
                return value.to(device=device)
            if isinstance(value, dict):
                return {
                    key: ResumableStepLevelGRPOTrainer._tree_to_device(item, device)
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [
                    ResumableStepLevelGRPOTrainer._tree_to_device(item, device)
                    for item in value
                ]
            if isinstance(value, tuple):
                return tuple(
                    ResumableStepLevelGRPOTrainer._tree_to_device(item, device)
                    for item in value
                )
            return value

        @classmethod
        def _optimizer_state_to_device(cls, state, parameter, group):
            """Match torch Optimizer's special placement policy for step tensors."""
            restored = {}
            for key, value in state.items():
                device = parameter.device
                if (
                    key == "step"
                    and not group.get("capturable", False)
                    and not group.get("fused", False)
                ):
                    device = torch.device("cpu")
                restored[key] = cls._tree_to_device(value, device)
            return restored

        def _save_optimizer_stream(self, output_dir):
            """Serialize one parameter state at a time to bound host-RAM usage."""
            if self.optimizer is None:
                raise ValueError("cannot checkpoint FCE before optimizer creation")

            temporary = tempfile.mkdtemp(
                prefix=".optimizer-stream-",
                dir=output_dir,
            )
            param_groups = []
            state_files = []
            parameter_index = 0
            for group in self.optimizer.param_groups:
                parameters = list(group["params"])
                param_groups.append(
                    {
                        "param_count": len(parameters),
                        "options": {
                            key: value
                            for key, value in group.items()
                            if key != "params"
                        },
                    }
                )
                for parameter in parameters:
                    filename = f"state-{parameter_index:06d}.pt"
                    destination = os.path.join(temporary, filename)
                    torch.save(
                        self._tree_to_cpu(
                            self.optimizer.state.get(parameter, {})
                        ),
                        destination + ".tmp",
                    )
                    os.replace(destination + ".tmp", destination)
                    state_files.append(filename)
                    parameter_index += 1

            metadata = {
                "format": self._resume_format,
                "param_groups": param_groups,
                "state_files": state_files,
                "parameter_count": parameter_index,
            }
            metadata_path = os.path.join(temporary, "metadata.pt")
            torch.save(metadata, metadata_path + ".tmp")
            os.replace(metadata_path + ".tmp", metadata_path)

            # The completed directory becomes visible atomically. If a process is
            # killed earlier, the dot-prefixed directory is ignored by the runner.
            completed_name = os.path.basename(temporary)[1:]
            completed = os.path.join(output_dir, completed_name)
            os.replace(temporary, completed)
            return completed_name, parameter_index

        def _load_optimizer_stream(self, checkpoint, manifest):
            directory_name = manifest.get("optimizer_state_dir")
            if not isinstance(directory_name, str) or "/" in directory_name:
                raise ValueError("invalid FCE streamed optimizer directory")
            directory = os.path.join(checkpoint, directory_name)
            metadata_path = os.path.join(directory, "metadata.pt")
            if not os.path.isfile(metadata_path):
                raise ValueError("FCE streamed optimizer metadata is missing")
            metadata = torch.load(
                metadata_path,
                map_location="cpu",
                weights_only=False,
            )
            if metadata.get("format") != self._resume_format:
                raise ValueError("invalid FCE streamed optimizer format")

            saved_groups = metadata.get("param_groups", [])
            if len(saved_groups) != len(self.optimizer.param_groups):
                raise ValueError("FCE optimizer parameter-group count mismatch")
            current_parameters = []
            for current_group, saved_group in zip(
                self.optimizer.param_groups,
                saved_groups,
            ):
                parameters = list(current_group["params"])
                if saved_group.get("param_count") != len(parameters):
                    raise ValueError("FCE optimizer parameter count mismatch")
                for key, value in saved_group.get("options", {}).items():
                    current_group[key] = value
                current_parameters.extend(
                    (parameter, current_group)
                    for parameter in parameters
                )

            state_files = metadata.get("state_files", [])
            if (
                metadata.get("parameter_count") != len(current_parameters)
                or len(state_files) != len(current_parameters)
                or manifest.get("optimizer_parameter_count")
                != len(current_parameters)
            ):
                raise ValueError("FCE optimizer state-file count mismatch")

            self.optimizer.state.clear()
            for (parameter, group), filename in zip(
                current_parameters,
                state_files,
            ):
                if (
                    not isinstance(filename, str)
                    or os.path.basename(filename) != filename
                ):
                    raise ValueError("invalid FCE optimizer state filename")
                state_path = os.path.join(directory, filename)
                if not os.path.isfile(state_path):
                    raise ValueError("FCE optimizer state file is missing")
                state = torch.load(
                    state_path,
                    map_location="cpu",
                    weights_only=False,
                )
                self.optimizer.state[parameter] = self._optimizer_state_to_device(
                    state,
                    parameter,
                    group,
                )

        def _save_checkpoint(self, model, trial, metrics=None):
            super()._save_checkpoint(model, trial, metrics)
            if not self.args.save_only_model or not self.args.should_save:
                return

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(
                run_dir,
                f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}",
            )

            scheduler_tmp = os.path.join(output_dir, "scheduler.pt.tmp")
            torch.save(self.lr_scheduler.state_dict(), scheduler_tmp)
            os.replace(scheduler_tmp, os.path.join(output_dir, "scheduler.pt"))

            # Trainer suppresses RNG saving when save_only_model=True. Restore it
            # explicitly so resumed generation continues the seed-42 sample stream.
            with tempfile.TemporaryDirectory(dir=output_dir) as temporary:
                self._save_rng_state(temporary)
                for name in os.listdir(temporary):
                    if name.startswith("rng_state") and name.endswith(".pth"):
                        os.replace(
                            os.path.join(temporary, name),
                            os.path.join(output_dir, name),
                        )

            optimizer_directory, optimizer_parameter_count = (
                self._save_optimizer_stream(output_dir)
            )
            manifest = {
                "format": self._resume_format,
                "global_step": self.state.global_step,
                "scheduler_state_saved": True,
                "rng_state_saved": True,
                "optimizer_state_saved": True,
                "optimizer_state_format": "streamed-per-parameter-v1",
                "optimizer_state_dir": optimizer_directory,
                "optimizer_parameter_count": optimizer_parameter_count,
                "optimizer_reset_on_resume": False,
            }
            manifest_path = os.path.join(output_dir, self._resume_manifest)
            manifest_tmp = manifest_path + ".tmp"
            with open(manifest_tmp, "w") as handle:
                json.dump(manifest, handle, indent=2)
                handle.write("\n")
            os.replace(manifest_tmp, manifest_path)

        def _load_optimizer_and_scheduler(self, checkpoint):
            if checkpoint is None or not self.args.save_only_model:
                return super()._load_optimizer_and_scheduler(checkpoint)

            manifest_path = os.path.join(checkpoint, self._resume_manifest)
            scheduler_path = os.path.join(checkpoint, "scheduler.pt")
            if not os.path.isfile(manifest_path) or not os.path.isfile(scheduler_path):
                raise ValueError(
                    "model-only checkpoint lacks certified exact-resume state"
                )
            with open(manifest_path) as handle:
                manifest = json.load(handle)
            trainer_state_path = os.path.join(checkpoint, "trainer_state.json")
            with open(trainer_state_path) as handle:
                checkpoint_state = json.load(handle)
            if (
                manifest.get("format") != self._resume_format
                or manifest.get("global_step")
                != checkpoint_state.get("global_step")
                or manifest.get("scheduler_state_saved") is not True
                or manifest.get("rng_state_saved") is not True
                or manifest.get("optimizer_state_saved") is not True
                or manifest.get("optimizer_reset_on_resume") is not False
            ):
                raise ValueError("invalid FCE exact-resume manifest")
            self._load_optimizer_stream(checkpoint, manifest)
            self.lr_scheduler.load_state_dict(
                torch.load(scheduler_path, map_location="cpu")
            )
            restored_lrs = self.lr_scheduler.get_last_lr()
            if len(restored_lrs) != len(self.optimizer.param_groups):
                raise ValueError("scheduler/optimizer parameter-group mismatch")
            for group, learning_rate in zip(
                self.optimizer.param_groups,
                restored_lrs,
            ):
                group["lr"] = learning_rate

    if args.use_hybrid:
        label = "hybrid"
    elif args.use_l2a:
        label = "l2a"
    else:
        label = "trajectory"
    # Tag the reward source so verifier-free arms don't overwrite the RLVR baseline.
    if args.reward_source != "gold":
        label = f"{label}_{args.reward_source}"
    print(f"\n=== LOCAL 5090 SMOKE [{label} | K={args.k} | steps={args.max_steps}] ===\n")

    # Exact trajectory GRPO needs no auxiliary encoder or OMP dictionary,
    # regardless of whether its terminal comes from gold, majority, or FCE.
    # Keep those models only for explicit step-reward ablations.
    pure_trajectory = (
        args.alpha == 0
        and args.terminal_spread == "constant"
        and args.reward_source != "consensus"
        and not args.use_l2a
        and not args.use_hybrid
    )
    if pure_trajectory:
        D = None
        encoder = None
        print(
            "Pure trajectory GRPO: no sentence encoder or OMP dictionary loaded"
        )
    else:
        from sentence_transformers import SentenceTransformer

        assert os.path.exists(args.dict_path), f"Missing dictionary at {args.dict_path}"
        D = np.load(args.dict_path)
        encoder = SentenceTransformer(
            args.encoder,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    # Dataset
    spec = task_spec(args.dataset)
    ds = load_task_dataset(args.dataset, spec.train_split)

    def prep(ex):
        return prepare_training_example(
            ex,
            dataset_name=args.dataset,
            reward_source=args.reward_source,
        )

    train = ds.map(prep, remove_columns=ds.column_names)
    if args.reward_source in FROZEN_REWARD_SOURCES:
        if not args.fcc_bank:
            p.error("--fcc-bank is required with frozen FCC/FCE reward sources")
        bank = json.load(open(args.fcc_bank))
        if bank.get("partial") or bank.get("gold_stored") is not False:
            raise ValueError("FCC bank must be complete and declare gold_stored=false")
        if bank.get("model") != args.model:
            raise ValueError(
                f"FCC bank model mismatch: {bank.get('model')} != {args.model}"
            )
        if (
            bank.get("dataset") != args.dataset
            or bank.get("split") != spec.train_split
        ):
            raise ValueError("FCC online training requires a matching train-split bank")
        if bank.get("answer_mode", "numeric") != spec.answer_mode:
            raise ValueError("FCC online training bank has the wrong answer mode")
        if bank.get("candidate_panel_included") is not False:
            raise ValueError("FCC online training bank must omit evaluation candidates")
        panel_size = int(bank.get("panel_size", 0))
        if panel_size <= 0:
            raise ValueError("FCC bank has an invalid panel size")
        seen_hashes = set()
        for item in bank.get("items", []):
            if item["prompt_hash"] in seen_hashes:
                raise ValueError("FCC bank contains duplicate prompt hashes")
            seen_hashes.add(item["prompt_hash"])
            if (
                len(item.get("panel_a", [])) != panel_size
                or len(item.get("panel_b", [])) != panel_size
            ):
                raise ValueError("FCC bank panel length does not match metadata")
        if args.reward_source == "fcc":
            accepted_hashes = {
                item["prompt_hash"]
                for item in bank.get("items", [])
                if item.get("frozen_target", {}).get("accepted")
            }
        else:
            if bank.get("frozen_evidence_attached") is not True:
                raise ValueError("FCE bank lacks finalized frozen evidence")
            accepted_hashes = {
                item["prompt_hash"]
                for item in bank.get("items", [])
                if item.get("frozen_evidence", {}).get("scores")
            }
        train = train.filter(
            lambda ex: hashlib.sha256(ex["prompt"].encode("utf-8")).hexdigest()
            in accepted_hashes
        )
        if len(train) == 0:
            raise ValueError("FCC bank has no accepted targets matching this dataset")
        # Enforce the no-verifier contract in the actual training examples. The
        # isolated trainer also refuses to parse answer fields in every
        # verifier-free mode.
        train = train.map(lambda ex: {"prompt": ex["prompt"], "answer": ""})
        print(
            f"{args.reward_source.upper()} bank coverage: "
            f"training on {len(train)} accepted prompts"
        )

    # Model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )

    out_dir = f"./checkpoints/{args.output_suffix}_{label}_seed{args.seed}_steps{args.max_steps}_k{args.k}"
    config = GRPOConfig(
        output_dir=out_dir, max_steps=args.max_steps,
        per_device_train_batch_size=1, num_generations=args.k,
        max_completion_length=args.max_completion, learning_rate=args.lr,
        logging_steps=5, save_steps=args.save_steps,
        # Keep the previous certified checkpoint while the next checkpoint is
        # being serialized. A killed write can therefore never erase the last
        # scheduler/RNG-complete resume point.
        save_total_limit=max(args.save_total_limit, 2),
        # Both OOM kills on the 29GB box landed just after a checkpoint save
        # (died at step 489 with a save at 475; died at 113 with a save at 100).
        # The optimizer state is what makes a checkpoint 8.7GB, and serializing it
        # spikes host RAM enough for the OOM killer to pick this process whenever
        # another job is resident. save_only_model avoids Transformers' monolithic
        # optimizer serialization; the trainer above streams Adam state one
        # parameter at a time and certifies optimizer, scheduler, and RNG together.
        save_only_model=True,
        report_to="none", run_name=f"{args.output_suffix}_{label}_k{args.k}",
        gradient_accumulation_steps=4, warmup_steps=5, bf16=True,
        gradient_checkpointing=args.grad_checkpoint,
        gradient_checkpointing_kwargs=({"use_reentrant": False}
                                       if args.grad_checkpoint else None),
        dataloader_num_workers=0, seed=args.seed,
    )
    if args.grad_checkpoint and args.use_lora:
        # Standard PEFT+GC recipe: without this, checkpointed segments see inputs with
        # requires_grad=False (base weights frozen) and backward through LoRA breaks.
        model.enable_input_require_grads()
    peft_config = None
    if args.use_lora:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
    trainer = ResumableStepLevelGRPOTrainer(
        model=model, args=config, train_dataset=train, processing_class=tokenizer,
        encoder=encoder, dictionary=D,
        alpha=args.alpha, tau=0.3, step_advantage_mode=args.step_advantage_mode,
        terminal_spread=args.terminal_spread, advantage_clip=5.0, length_normalize=False,
        use_l2a=args.use_l2a, l2a_weight=1.0,
        use_hybrid=args.use_hybrid,
        reward_source=args.reward_source,
        random_reward_p=args.random_reward_p,
        fcc_bank_path=args.fcc_bank,
        answer_mode=spec.answer_mode,
        peft_config=peft_config,
    )
    if args.resume_from:
        # torch >= 2.6 flipped torch.load to weights_only=True, which rejects the
        # numpy globals inside a HF checkpoint's rng_state.pth and makes resume die
        # instantly. Safe to relax here: we are loading a checkpoint we just wrote.
        _orig_load = torch.load

        def _load_full(*a, **kw):
            kw["weights_only"] = False
            return _orig_load(*a, **kw)

        torch.load = _load_full
        print(f"RESUMING from {args.resume_from}")
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()
    if args.use_lora:
        # Merge adapters into the base weights so the final dir is a plain HF model
        # that local_5090_eval.py can load with AutoModelForCausalLM. Saving the raw
        # PeftModel would write only adapter_{config,model} and the eval would fail.
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(f"{out_dir}_final")
        tokenizer.save_pretrained(f"{out_dir}_final")
    else:
        trainer.save_model(f"{out_dir}_final")
    print(f"\n✓ Saved checkpoint to {out_dir}_final")


if __name__ == "__main__":
    main()
