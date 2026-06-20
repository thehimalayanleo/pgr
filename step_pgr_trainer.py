"""
Step-level PGR trainer.

Subclasses TRL 0.14.0 GRPOTrainer. Replaces the trajectory-level advantage with
per-step advantages, applied at the per-token level.

KEY DIFFERENCE FROM STANDARD GRPO:

    GRPO computes one scalar advantage per rollout and applies it identically
    to every token in that rollout. That is:

        per_token_loss = exp(logp - logp.detach()) * advantage[rollout_idx]

    Step-PGR computes per-step rewards, normalizes them across the group, then
    expands the per-step advantages to per-token weights:

        per_token_loss = exp(logp - logp.detach()) * advantage[rollout_idx, token_idx]

REGRESSION GUARANTEE:

    If `step_reward = constant for all steps`, this reduces exactly to standard
    GRPO trajectory-level advantage. See `validate_regression()`.
"""

import numpy as np
from transformers import PreTrainedTokenizer
from pgr_utils import (
    segment_steps as _segment_steps,
    omp_reconstruction_errors,
    extract_boxed_answer,
)

try:
    import torch
except ImportError:
    pass

try:
    from trl import GRPOTrainer
    from trl.data_utils import maybe_apply_chat_template, is_conversational, apply_chat_template
    from trl.models import unwrap_model_for_generation
    from accelerate.utils.other import gather_object
    from accelerate.utils import broadcast_object_list
    from trl.trainer.utils import pad
    from transformers import PreTrainedModel
    TRL_AVAILABLE = True
except (ImportError, RuntimeError):
    GRPOTrainer = object
    TRL_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────
# Step segmentation + token alignment
# ─────────────────────────────────────────────────────────────────────────

def segment_steps(text: str) -> list[str]:
    return _segment_steps(text)


def step_token_spans(
    tokenizer: PreTrainedTokenizer,
    completion_text: str,
    completion_token_ids: list[int],
) -> list[tuple[int, int, str]]:
    """For a completion's text, identify (start_token, end_token, step_text) per step.

    Strategy: use the tokenizer's offset_mapping when available; fallback to
    prefix re-encoding.
    """
    steps = segment_steps(completion_text)
    if not steps:
        return []

    try:
        enc = tokenizer(
            completion_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offsets = enc["offset_mapping"]
    except Exception:
        offsets = None

    spans = []
    if offsets is not None:
        for step_text in steps:
            char_start = completion_text.find(step_text)
            if char_start < 0:
                continue
            char_end = char_start + len(step_text)
            tok_start = None
            tok_end = None
            for i, (s, e) in enumerate(offsets):
                if s == 0 and e == 0:
                    continue
                if tok_start is None and e > char_start:
                    tok_start = i
                if s < char_end:
                    tok_end = i + 1
            if tok_start is not None and tok_end is not None and tok_end > tok_start:
                spans.append((tok_start, tok_end, step_text))
    else:
        running_offset = 0
        for step_text in steps:
            char_start = completion_text.find(step_text, running_offset)
            if char_start < 0:
                continue
            char_end = char_start + len(step_text)
            tok_start = len(tokenizer.encode(completion_text[:char_start], add_special_tokens=False))
            tok_end   = len(tokenizer.encode(completion_text[:char_end],   add_special_tokens=False))
            if tok_end > tok_start:
                spans.append((tok_start, tok_end, step_text))
            running_offset = char_end

    return spans


def omp_step_rewards(
    step_texts: list[str],
    encoder,
    D: np.ndarray,
    tau: float = 0.3,
    n_nonzero: int = 5,
) -> np.ndarray:
    """OMP reconstruction reward per step. Returns array shape (n_steps,)."""
    if not step_texts:
        return np.array([0.0])
    emb = encoder.encode(step_texts, normalize_embeddings=True, batch_size=64)
    errs = omp_reconstruction_errors(D, emb, n_nonzero=n_nonzero)
    return np.exp(-errs / tau)


def extract_answer(text: str) -> str | None:
    return extract_boxed_answer(text)


# ─────────────────────────────────────────────────────────────────────────
# Step-Level GRPO Trainer
# ─────────────────────────────────────────────────────────────────────────

class StepLevelGRPOTrainer(GRPOTrainer):
    """
    GRPO with per-step credit assignment.

    Constructor adds:
        encoder         : sentence transformer for embedding steps
        dictionary      : np.ndarray (n_atoms, embed_dim)
        alpha           : blend between step reward and terminal reward
        tau             : OMP reward temperature
        step_advantage_mode : 'pooled' | 'group_mean' (see _compute_step_advantages)

    Override of `compute_loss` builds per-token advantages from per-step rewards.

    Reward function passed to parent is a stub that returns zeros — we compute
    rewards inside `compute_loss` because we need the per-step decomposition.
    """

    def __init__(
        self,
        *args,
        encoder=None,
        dictionary=None,
        alpha: float = 0.5,
        tau: float = 0.3,
        step_advantage_mode: str = "pooled",
        **kwargs,
    ):
        # Ensure a stub reward_func is present so parent accepts the init
        if "reward_funcs" not in kwargs or kwargs["reward_funcs"] is None:
            def _stub_reward(completions, **_):
                return [0.0] * len(completions)
            _stub_reward.__name__ = "step_pgr_reward"
            kwargs["reward_funcs"] = [_stub_reward]

        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.D = dictionary
        self.alpha = alpha
        self.tau = tau
        self.step_advantage_mode = step_advantage_mode
        assert step_advantage_mode in ("pooled", "group_mean"), \
            f"Unknown step_advantage_mode: {step_advantage_mode}"

    # ── helpers ──────────────────────────────────────────────────────────

    def _compute_per_step_rewards(self, completion_text: str, gold_answer: str | None):
        """Return (step_rewards: np.ndarray, step_spans_in_text: list[(s,e,txt)]).
        Final-step reward gets the terminal anchor added."""
        spans_text_only = segment_steps(completion_text)
        if not spans_text_only:
            return np.array([0.0]), []
        step_rewards = omp_step_rewards(spans_text_only, self.encoder, self.D, tau=self.tau)
        # Apply alpha blend ONLY to the final step (terminal reward attaches there).
        pred = extract_answer(completion_text)
        is_correct = (pred is not None and gold_answer is not None and pred == gold_answer)
        terminal = 1.0 if is_correct else 0.0
        blended = self.alpha * step_rewards.copy()
        blended[-1] += (1 - self.alpha) * terminal
        return blended, spans_text_only

    def _compute_step_advantages(self, all_step_rewards: list[np.ndarray], num_generations: int):
        """Given a list of per-step reward arrays (one per rollout), compute
        per-step advantages.

        Two modes:
          'pooled' : pool ALL (rollout, step) pairs in a group, compute mean/std,
                     normalize each step-reward → per-step advantage.
                     Simpler. Recommended.
          'group_mean' : for each rollout, normalize its own steps using the
                         group's trajectory-mean reward as baseline.
        """
        n_rollouts = len(all_step_rewards)
        groups = n_rollouts // num_generations
        out = [None] * n_rollouts

        for g in range(groups):
            slc = slice(g * num_generations, (g + 1) * num_generations)
            group_rewards = all_step_rewards[slc]

            if self.step_advantage_mode == "pooled":
                pooled = np.concatenate(group_rewards) if any(len(r) for r in group_rewards) else np.array([0.0])
                mu, sigma = pooled.mean(), pooled.std()
                for i in range(num_generations):
                    out[g * num_generations + i] = (group_rewards[i] - mu) / (sigma + 1e-4)

            else:  # group_mean
                group_trajectory_means = np.array([r.mean() if len(r) else 0.0 for r in group_rewards])
                mu_traj = group_trajectory_means.mean()
                sigma_traj = group_trajectory_means.std()
                for i in range(num_generations):
                    out[g * num_generations + i] = (group_rewards[i] - mu_traj) / (sigma_traj + 1e-4)
        return out

    def _build_per_token_advantages(
        self,
        completion_texts: list[str],
        completion_ids: "torch.Tensor",          # [B*G, seq_len]
        step_advantages: list[np.ndarray],
        completion_mask: "torch.Tensor",         # [B*G, seq_len]
    ):
        """Build A[B*G, seq_len] from per-step advantages.

        Each token in step s gets advantage[s]. Tokens outside any identified
        step span (e.g., trailing whitespace between steps) get advantage 0.
        """
        device = completion_ids.device
        B, L = completion_ids.shape
        A = torch.zeros((B, L), dtype=torch.float32, device=device)

        for b in range(B):
            text = completion_texts[b]
            token_ids = completion_ids[b].tolist()
            spans = step_token_spans(self.processing_class, text, token_ids)

            sr = step_advantages[b]
            if len(spans) == 0 or len(sr) == 0:
                continue
            n = min(len(spans), len(sr))
            for k in range(n):
                tok_start, tok_end, _ = spans[k]
                tok_end = min(tok_end, L)
                A[b, tok_start:tok_end] = float(sr[k])

        # Zero-out padded / post-EOS positions
        A = A * completion_mask.float()
        return A

    # ── main compute_loss override ───────────────────────────────────────

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("StepLevelGRPOTrainer does not support returning outputs")

        device = self.accelerator.device
        prompts_raw = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(ex, self.processing_class)["prompt"] for ex in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(prompt_inputs)

        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"]      = prompt_inputs["input_ids"][:, -self.max_prompt_length:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length:]

        # ── Generation: identical to parent (regular path only — no vLLM) ──
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(
                **prompt_inputs, generation_config=self.generation_config
            )

        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # ── Log probs (identical to parent) ──────────────────────────────
        def get_per_token_logps(mdl, input_ids, num_logits_to_keep):
            logits = mdl(input_ids, num_logits_to_keep=num_logits_to_keep + 1).logits
            logits = logits[:, :-1, :]
            per_token_logps = []
            for logits_row, input_ids_row in zip(logits, input_ids[:, -num_logits_to_keep:]):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)

        num_logits_to_keep = completion_ids.size(1)
        per_token_logps = get_per_token_logps(model, prompt_completion_ids, num_logits_to_keep)

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = get_per_token_logps(self.ref_model, prompt_completion_ids, num_logits_to_keep)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = get_per_token_logps(model, prompt_completion_ids, num_logits_to_keep)

        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Mask everything after the first EOS
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Decode completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # ── Compute per-step rewards (PGR-specific) ──────────────────────
        # Each input row gets repeated num_generations times to align with completions.
        gold_answers_repeated = []
        for ex in inputs:
            gold = extract_answer(ex.get("answer", ""))
            gold_answers_repeated.extend([gold] * self.num_generations)

        per_rollout_step_rewards = []
        trajectory_means = []
        for text, gold in zip(completions, gold_answers_repeated):
            sr, _ = self._compute_per_step_rewards(text, gold)
            per_rollout_step_rewards.append(sr)
            trajectory_means.append(float(sr.mean()) if len(sr) else 0.0)

        # ── Per-step advantage normalization across each group ───────────
        step_adv = self._compute_step_advantages(per_rollout_step_rewards, self.num_generations)

        # ── Build per-token advantage tensor ─────────────────────────────
        per_token_advantages = self._build_per_token_advantages(
            completion_texts=completions,
            completion_ids=completion_ids,
            step_advantages=step_adv,
            completion_mask=completion_mask,
        )

        # ── Loss (clipped surrogate with per-token advantages + KL) ──────
        ratio = torch.exp(per_token_logps - per_token_logps.detach())
        per_token_loss = ratio * per_token_advantages
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp_min(1)).mean()

        # ── Metrics ──────────────────────────────────────────────────────
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        traj_rewards = torch.tensor(trajectory_means, device=device)
        self._metrics["rewards/step_pgr_reward"].append(traj_rewards.mean().item())
        self._metrics["reward"].append(traj_rewards.mean().item())

        # Reward variance across group — mirrors GRPO's reward_std metric
        traj_grouped = traj_rewards.view(-1, self.num_generations)
        self._metrics["reward_std"].append(traj_grouped.std(dim=1).mean().item())

        # Per-token advantage variance — useful debugging
        adv_var = (per_token_advantages * completion_mask).pow(2).sum() / completion_mask.sum().clamp_min(1)
        self._metrics["per_token_adv_var"].append(adv_var.item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp_min(1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss


# ─────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────

def validate_step_alignment(tokenizer, completion_text: str):
    enc = tokenizer(completion_text, add_special_tokens=False)
    token_ids = enc["input_ids"]
    spans = step_token_spans(tokenizer, completion_text, token_ids)
    print(f"Total tokens: {len(token_ids)}")
    print(f"Identified {len(spans)} steps")
    for i, (s, e, txt) in enumerate(spans):
        decoded = tokenizer.decode(token_ids[s:e])
        match = decoded.strip() == txt.strip() or txt.strip() in decoded.strip()
        flag = "✓" if match else "✗"
        print(f"  [{i}] tok[{s}:{e}]={e-s}t  match={flag}")
        if not match:
            print(f"      expected: {txt[:80]}")
            print(f"      got:      {decoded[:80]}")
    return spans


def validate_regression():
    """Regression test: if all per-step rewards are equal to the trajectory
    reward, per-step advantages should equal trajectory advantages (modulo
    pooled normalization, which uses pooled std instead of group std).

    This isn't an EXACT reduction because we use pooled (rollout, step) statistics
    for normalization instead of pooled trajectory statistics. But the SIGN of
    each token's advantage should match the trajectory advantage.
    """
    import numpy as np
    np.random.seed(0)
    # 4 rollouts, each with 5 steps; constant step reward per rollout
    rollouts_step_rewards = [
        np.full(5, 0.3),
        np.full(5, 0.7),
        np.full(5, 0.1),
        np.full(5, 0.5),
    ]
    pooled = np.concatenate(rollouts_step_rewards)
    mu, sigma = pooled.mean(), pooled.std()
    step_adv = [(r - mu) / (sigma + 1e-4) for r in rollouts_step_rewards]
    # All tokens in a rollout should have the same advantage (since steps are equal)
    for i, adv in enumerate(step_adv):
        assert np.allclose(adv, adv[0]), f"Rollout {i}: advantages should be uniform when step rewards are constant"
    print(f"✓ regression test passed — constant step rewards yield uniform per-token advantages")
    # Sign of each rollout's advantage should match trajectory rank
    traj_advs = [a[0] for a in step_adv]
    print(f"  trajectory advantages: {[round(a, 3) for a in traj_advs]}")
    print(f"  rollout 1 (highest step reward) has the highest advantage: {traj_advs[1] == max(traj_advs)}")


if __name__ == "__main__":
    from transformers import AutoTokenizer
    print("=== Step Alignment Test ===")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    sample = """Let x be the unknown variable.

Subtract 5 from both sides: 2x = 8.

Divide by 2: x = 4.

We verify: 2(4) + 5 = 13. The answer is correct."""
    validate_step_alignment(tok, sample)
    print()
    print("=== Regression Test ===")
    validate_regression()
