# modal_step_pgr.py
"""
STEP-LEVEL PGR — per-step credit assignment.

Difference from modal_train.py:
  - Standard GRPO collapses per-step rewards to a scalar before computing advantage,
    so every token in a rollout gets the SAME gradient weight.
  - Step-level PGR computes per-step rewards r_t, then per-step advantages A_t,
    then expands each step's advantage to all tokens in that step.

Token-level loss per rollout:
  loss_t = -A[step(t)] * log_prob(token_t) + beta * KL_t

This makes high-PGR-score steps push UP, low-PGR-score steps push DOWN, all within
the same rollout. Recovers the per-step information PGR computes.
"""

import modal
from modal_config import image_training, VOLUME_MOUNT

app = modal.App("pgr-step-train-sketch")


# ─────────────────────────────────────────────────────────────────────────
# DESIGN SKETCH — not yet runnable. Implementation notes below.
# ─────────────────────────────────────────────────────────────────────────
"""
ARCHITECTURE:

class StepLevelGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, step_reward_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_reward_fn = step_reward_fn  # returns list-of-list-of-floats

    def compute_loss(self, model, inputs, return_outputs=False):
        # inputs has: prompt_ids, completion_ids, attention_masks, ...
        # We need:
        #   1. Decode completion_ids → text for each rollout
        #   2. Run step_reward_fn(text) → list of (step_text, step_reward)
        #   3. Tokenize each step_text to find its token span in completion_ids
        #   4. Build token-level advantage tensor [batch, seq_len]
        #   5. Compute standard GRPO clipped surrogate loss with per-token advantages
        #   6. Add KL term as usual
        pass


KEY IMPLEMENTATION CHALLENGES:

(1) Step → token alignment
    The reward function operates on TEXT (split on \n\n or "Step N:").
    The training loop operates on TOKENS (Qwen2.5 tokenizer output).
    We need: for each step, the contiguous token range [start_tok, end_tok).

    APPROACH:
      a. tokenizer.encode(text_prefix_through_step_i, add_special_tokens=False)
         → cumulative token count up to end of step i
      b. step_token_ranges[i] = (cumtok[i-1], cumtok[i])

    EDGE CASES:
      - First step: start_tok = 0
      - Trailing whitespace / formatting between steps
      - Step boundary may fall mid-BPE-token (rare for Qwen but possible)

(2) Per-step group advantage
    Standard GRPO: advantage_i = (R_i - mean(R)) / std(R) over the k rollouts.

    Step-level: for each STEP POSITION, normalize across the k rollouts.
    But rollouts have DIFFERENT NUMBERS of steps. Options:
      a. Per-(rollout, step) advantage normalized only by that rollout's mean/std
         → loses cross-rollout comparison
      b. Per-(rollout, step) advantage = (r_step - mean_all_steps_across_group) / std
         → simpler, treats each step as an independent observation
      c. Position-aligned: only normalize step 0 across all rollouts, step 1 across
         rollouts that have a step 1, etc.
         → handles variable-length cleanly but per-position group size shrinks

    RECOMMEND (b): per-step advantage uses pooled mean/std across all steps in the group.
                   Simpler. Closest to standard GRPO. Empirically tested in PRM literature.

(3) Token-level advantage tensor
    After (1) and (2):
      A[b, t] = step_advantage[b, step_of_token_t]   for t in completion span
      A[b, t] = 0                                     for prompt tokens

    Apply to per-token log_prob in clipped surrogate, identical to standard PPO.

(4) KL term unchanged
    KL is per-token regardless. β · KL(π_θ(token_t) || π_ref(token_t))
    Sums over the same token range.

(5) Validation
    Set step_reward = constant (same for all steps) → should reduce to trajectory GRPO.
    Set step_reward = binary terminal only on last step, 0 elsewhere → reduces to
    something like sparse-credit RL. Useful regression test.


PRACTICAL TODO (in order):
  [ ] Subclass GRPOTrainer from TRL 0.14.0
  [ ] Override _prepare_inputs to attach step_token_ranges and step_advantages tensors
  [ ] Override compute_loss to use per-token advantages
  [ ] Test step alignment on 1 batch (assertions on shape, monotonicity, totals)
  [ ] Regression test: constant step_reward should reproduce trajectory PGR exactly
  [ ] Smoke run: 20 steps, 3B model, check no NaN/inf, gradient flowing
  [ ] Full run: 300 steps, 3B, seed 42, compare grad_norm trajectory vs trajectory PGR
  [ ] Eval: PGR_step vs PGR_traj vs binary on MATH-Hard test

TIME ESTIMATE (focused, no Modal preemption):
  Day 1: Subclassing + step alignment (3-4 hrs)
  Day 2: Loss computation + regression test (3-4 hrs)
  Day 3: Smoke + full run (Modal compute, ~5 hrs wall)
  Day 4: Eval + writeup (compute, ~30 min eval; writing afterward)


RISKS:
  - TRL 0.14.0 internals may change behavior on subclass — read source carefully
  - Step-token alignment could leak gradients if mis-aligned
  - Per-step advantage variance could be HUGE on short steps (need clipping)
  - Modal H100 preemption still a risk regardless of implementation


FALLBACK if step-level doesn't ship by deadline:
  Submit trajectory-PGR with explicit limitation:
    "Our implementation aggregates per-step rewards to a trajectory scalar before
     advantage estimation, discarding the per-step credit information. Step-level
     advantage estimation is the natural next experiment, which we sketch in
     Appendix X."
"""


@app.function(
    image=image_training,
    gpu="H100",
    timeout=14400,
    volumes=VOLUME_MOUNT,
)
def train_step_pgr(
    max_steps: int = 300,
    k: int = 4,
    alpha: float = 0.5,
    tau: float = 0.3,
    seed: int = 42,
    model_id: str = "Qwen/Qwen2.5-3B-Instruct",
):
    """STUB — implementation forthcoming.

    Will follow the architecture sketched in the docstring above.
    """
    raise NotImplementedError(
        "Step-level PGR training loop is in design. "
        "See module docstring for architecture and implementation plan."
    )


@app.local_entrypoint()
def main():
    print("step-level PGR — design sketch only. See modal_step_pgr.py docstring.")
    print("Implementation forthcoming.")
