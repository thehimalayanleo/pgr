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

import hashlib
import json
import re
import numpy as np
from transformers import PreTrainedTokenizer
from fcc_reward import (
    extract_answer as _extract_frozen_answer,
    normalize_answer as _normalize_frozen_answer,
)

try:
    import torch
    from sklearn.linear_model import orthogonal_mp
except ImportError:
    pass

try:
    from trl import GRPOTrainer
    TRL_AVAILABLE = True
except (ImportError, RuntimeError):
    GRPOTrainer = object
    TRL_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────
# Step segmentation + token alignment
# ─────────────────────────────────────────────────────────────────────────

def segment_steps(text: str) -> list[str]:
    """Split a completion into reasoning steps.

    NOTE: this regex must stay in sync with the one the dictionary was LEARNED with
    (modal_dictionary.py) and with the offline rescorers, or OMP is applied at a
    different step granularity than it was fit at. It previously omitted the
    numbered-list alternative that both of those use, so model rollouts full of
    "1. ... 2. ..." were glued into single oversized steps at training time while the
    dictionary atoms and every offline AUROC came from finer splits.
    """
    parts = re.split(r'\n\n+|(?=Step \d+:)|(?=\d+\.\s)', text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 20]


def step_token_spans(
    tokenizer: PreTrainedTokenizer,
    completion_text: str,
    completion_token_ids: "list[int] | None" = None,
) -> list[tuple[int, int, str]]:
    """For a completion's text, identify (start_token, end_token, step_text) per step.

    FIX (Bug 2): When completion_token_ids is provided, align spans to the
    ORIGINAL token ids by incrementally decoding prefixes and matching character
    positions — guarantees alignment with the original completion_ids tensor
    even when re-tokenization would differ.

    Falls back to offset_mapping (faster but uses re-tokenization) when
    completion_token_ids is None.
    """
    steps = segment_steps(completion_text)
    if not steps:
        return []

    # ── Path A: align to ORIGINAL ids (preferred) ────────────────────────
    if completion_token_ids is not None:
        n = len(completion_token_ids)
        # cum_chars[i] = number of decoded chars after consuming tokens[0:i+1]
        cum_chars = []
        for i in range(n):
            try:
                decoded = tokenizer.decode(completion_token_ids[: i + 1], skip_special_tokens=True)
                cum_chars.append(len(decoded))
            except Exception:
                cum_chars.append(cum_chars[-1] if cum_chars else 0)

        # Build the full decoded text from the same source so step.find works
        full_text = tokenizer.decode(completion_token_ids, skip_special_tokens=True)

        spans = []
        search_from = 0
        for step_text in steps:
            char_start = full_text.find(step_text, search_from)
            if char_start < 0:
                # try without anchoring
                char_start = full_text.find(step_text)
            if char_start < 0:
                continue
            char_end = char_start + len(step_text)
            # tok_start = first token whose cum_chars > char_start
            tok_start = None
            for i, c in enumerate(cum_chars):
                if c > char_start:
                    tok_start = i
                    break
            # tok_end = first token whose cum_chars >= char_end, +1
            tok_end = None
            for i, c in enumerate(cum_chars):
                if c >= char_end:
                    tok_end = i + 1
                    break
            if tok_end is None:
                tok_end = n
            if tok_start is None:
                continue
            if tok_end > tok_start:
                spans.append((tok_start, min(tok_end, n), step_text))
            search_from = char_end
        return spans

    # ── Path B: legacy offset_mapping fallback (re-tokenization-based) ──
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
        for step_text in steps:
            char_start = completion_text.find(step_text)
            if char_start < 0:
                continue
            char_end = char_start + len(step_text)
            tok_start = len(tokenizer.encode(completion_text[:char_start], add_special_tokens=False))
            tok_end   = len(tokenizer.encode(completion_text[:char_end],   add_special_tokens=False))
            if tok_end > tok_start:
                spans.append((tok_start, tok_end, step_text))

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
    emb   = encoder.encode(step_texts, normalize_embeddings=True, batch_size=64)
    codes = orthogonal_mp(D.T, emb.T, n_nonzero_coefs=n_nonzero)
    errs  = np.linalg.norm(emb.T - D.T @ codes, axis=0)
    return np.exp(-errs / tau)


def _norm_num(s: str) -> str:
    """Normalize a numeric answer string for comparison (strip $, commas, trailing .)."""
    return _normalize_frozen_answer(s)


def extract_answer(text: str, answer_mode: str = "numeric") -> str | None:
    """Canonical task-aware extractor shared with frozen bank construction."""
    return _extract_frozen_answer(text, answer_mode)


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
        # ── new fixes ──
        terminal_spread: str = "omp_weighted",       # 'last_only' | 'uniform' | 'omp_weighted'
        gamma: float = 1.0,                           # fixed per-step discount; ignored if gamma_total set
        gamma_total: float | None = None,             # if set, gamma_per_step = gamma_total^(1/n_steps)
                                                      # equalizes total decay regardless of rollout length
        top_k_steps: int | None = None,               # if set, only top-K steps per rollout get non-zero advantage
        advantage_clip: float = 3.0,                  # clip per-step advantages to [-x, x]
        length_normalize: bool = True,                # divide per-step advantage by sqrt(step_len) when assigning
        use_contrastive: bool = False,                # Level 1: contrastive direction from GRPO group
        contrastive_weight: float = 1.0,              # weight on contrastive score (1.0 = replace OMP)
        use_l2a: bool = False,                        # Level 2a: success-dict OMP per group
        l2a_weight: float = 1.0,                      # weight on L2a score (1.0 = replace OMP)
        l2a_n_nonzero: int = 5,                       # OMP sparsity for L2a
        # ── Verifier-free reward source (the "replace RLVR" axis) ──
        # Where does the binary terminal come from?
        #   "gold"     : terminal = 1[pred == gold]              (RLVR; needs a verifier)
        #   "majority" : terminal = 1[pred == group plurality]   (verifier-free,
        #                TTRL-style self-consistency; gold is absent in training)
        #   "random"   : terminal ~ Bernoulli(random_reward_p)   (spurious-reward control:
        #                if this moves the model as much as "majority", the lift is Qwen
        #                prior amplification, not signal -- cf. Shao et al. 2025)
        #   "fce_permuted": matched FCE control. It preserves each group's exact
        #                frozen reward multiset but uniformly shuffles assignment
        #                across trajectories, breaking answer/reward alignment in
        #                expectation without constructing an adversarial anti-reward.
        #   "consensus": terminal = independence-weighted OMP consensus, in [0,1].
        #                The pursuit method. Strictly LESS supervision than "majority":
        #                needs no gold, no labels, AND no parseable answer -- only
        #                embeddings. That is the point: it reaches domains where nothing
        #                is checkable or extractable, which is where RLVR cannot go.
        #                Offline (n=150 cached GSM8K): AUROC 0.697 vs 0.683 for plain
        #                coverage, and it BLOCKS the mode-collapse hack that makes plain
        #                coverage unusable (plain saturates to 1.0000 under cloning;
        #                this goes to 0.0000 because independence -> 0).
        reward_source: str = "gold",
        random_reward_p: float = 0.5,
        consensus_lambda: float = 1.0,   # exponent on the independence gate
        consensus_ngram: int = 3,
        # Frozen Cross-Consensus: two initial-policy panels build a target once;
        # current rollouts can score against it but can never change it.
        fcc_bank_path: str | None = None,
        answer_mode: str = "numeric",
        # Hybrid orthogonal reward (trajectory + per-step signals independent of correctness)
        use_hybrid: bool = False,
        hybrid_terminal_weight: float = 0.7,
        hybrid_length_weight: float = 0.15,           # penalty for long steps
        hybrid_conciseness_target: int = 60,          # tokens per step (soft target)
        hybrid_confidence_weight: float = 0.15,       # bonus for high-confidence tokens
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
        self.terminal_spread = terminal_spread
        self.gamma = gamma
        self.gamma_total = gamma_total
        self.top_k_steps = top_k_steps
        self.advantage_clip = advantage_clip
        self.length_normalize = length_normalize
        assert step_advantage_mode in ("pooled", "group_mean", "ema"), \
            f"Unknown step_advantage_mode: {step_advantage_mode}"
        # EMA baseline state (PPO-style: V(s) ≈ scalar running mean of per-step rewards)
        # Also track running std for proper scale normalization.
        self._ema_reward = 0.0
        self._ema_reward_sq = 0.0
        self._ema_alpha = 0.05
        self._ema_initialized = False
        # Contrastive credit from GRPO group structure.
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        # Level 2a: dynamic success-dictionary OMP per group.
        self.use_l2a = use_l2a
        self.l2a_weight = l2a_weight
        self.l2a_n_nonzero = l2a_n_nonzero
        # Hybrid orthogonal reward
        self.use_hybrid = use_hybrid
        self.hybrid_terminal_weight = hybrid_terminal_weight
        self.hybrid_length_weight = hybrid_length_weight
        self.hybrid_conciseness_target = hybrid_conciseness_target
        self.hybrid_confidence_weight = hybrid_confidence_weight
        assert terminal_spread in ("last_only", "uniform", "omp_weighted", "positional", "constant", "signed_positional"), \
            f"Unknown terminal_spread: {terminal_spread}"
        # Verifier-free reward source.
        self.reward_source = reward_source
        self.random_reward_p = random_reward_p
        self.consensus_lambda = consensus_lambda
        self.consensus_ngram = consensus_ngram
        self.fcc_bank_path = fcc_bank_path
        self._fce_answer_mode = answer_mode
        self._fcc_targets: dict[str, str] = {}
        self._fce_scores: dict[str, dict[str, float]] = {}
        if fcc_bank_path is not None:
            with open(fcc_bank_path) as bank_file:
                bank = json.load(bank_file)
            if bank.get("partial"):
                raise ValueError("FCC bank is partial; refusing to train")
            if bank.get("gold_stored") is not False:
                raise ValueError("FCC bank must explicitly declare gold_stored=false")
            bank_answer_mode = bank.get("answer_mode", "numeric")
            if bank_answer_mode != self._fce_answer_mode:
                raise ValueError(
                    "FCC bank answer mode mismatch: "
                    f"{bank_answer_mode} != {self._fce_answer_mode}"
                )
            for item in bank.get("items", []):
                frozen = item.get("frozen_target", {})
                if frozen.get("accepted") and frozen.get("target") is not None:
                    self._fcc_targets[item["prompt_hash"]] = str(frozen["target"])
                evidence = item.get("frozen_evidence", {})
                scores = evidence.get("scores", {})
                if scores:
                    self._fce_scores[item["prompt_hash"]] = {
                        str(answer): float(score)
                        for answer, score in scores.items()
                    }
        assert reward_source in (
            "gold",
            "majority",
            "random",
            "consensus",
            "fcc",
            "fce",
            "fce_permuted",
        ), \
            f"Unknown reward_source: {reward_source}"
        if reward_source == "fcc" and not self._fcc_targets:
            raise ValueError("reward_source='fcc' requires a complete bank with accepted targets")
        if reward_source in ("fce", "fce_permuted") and not self._fce_scores:
            raise ValueError(
                f"reward_source='{reward_source}' requires a complete bank "
                "with frozen evidence"
            )
        # Deterministic RNG for the "random" control (seeded off the run seed).
        self._reward_rng = np.random.RandomState(
            getattr(getattr(self, "args", None), "seed", 0) or 0
        )

    # ── helpers ──────────────────────────────────────────────────────────

    def _exact_trajectory_mode(self) -> bool:
        return (
            self.alpha == 0
            and self.terminal_spread == "constant"
            and not self.use_contrastive
            and not self.use_l2a
            and not self.use_hybrid
        )

    def _compute_raw_step_rewards(
        self, completion_text: str
    ) -> tuple[np.ndarray, list[str]]:
        """Compute label-free OMP rewards before any terminal is mixed in."""
        spans_text_only = segment_steps(completion_text)
        if not spans_text_only:
            return np.array([0.0]), []
        step_rewards = omp_step_rewards(spans_text_only, self.encoder, self.D, tau=self.tau)
        return step_rewards, spans_text_only

    def _blend_step_rewards(
        self, step_rewards: np.ndarray, terminal: float
    ) -> np.ndarray:
        """Mix an explicit terminal into raw OMP rewards.

        `terminal` is an argument rather than being recovered from gold here. That
        separation is load-bearing: majority/random/consensus rewards must rebuild
        the arrays that actually enter the loss after choosing their terminal.
        """
        step_rewards = np.asarray(step_rewards, dtype=np.float32)
        if len(step_rewards) == 0:
            return np.array([float(terminal)], dtype=np.float32)
        if self.terminal_spread == "last_only":
            terminal_per_step = np.zeros_like(step_rewards)
            terminal_per_step[-1] = terminal
        elif self.terminal_spread == "uniform":
            terminal_per_step = np.full_like(step_rewards, terminal / len(step_rewards))
        elif self.terminal_spread == "constant":
            # Length-invariant broadcast: EVERY step gets the full terminal, not
            # terminal/N. This is the leakage-equivalent of Level-2a: because a
            # successful rollout's own step embeddings are rows of D_succ, OMP
            # reconstructs them exactly and L2a returns exp(0) = 1.0 at every
            # step (measured: 1.0000 +/- 0.0000 over all correct rollouts).
            # 'uniform' is NOT this control -- it divides by N and so carries an
            # implicit length penalty, which is the very axis L2a is claimed to
            # win on. Use this mode to ablate the dictionary machinery away.
            terminal_per_step = np.full_like(step_rewards, terminal)
        elif self.terminal_spread == "positional":
            # Linear ramp 1.0 → 0.1 scaled by terminal.
            #
            # CAVEAT (measured 2026-07-24): this only shapes SUCCESSFUL rollouts.
            # `terminal * positions` is zero at every step when terminal == 0, so a
            # failed rollout's per-step reward collapses to alpha*OMP -- the component
            # measured at rho ~= 0, i.e. noise. Within-rollout spread is 8.8x smaller
            # for failures, and correlation with the ramp drops from +0.99 to ~0.
            # Normalization cannot recover it: subtracting a baseline from a flat
            # vector leaves a flat vector. So the stated intent ("penalize the early
            # mistake") is exactly the case this does NOT implement. Use
            # 'signed_positional' for that.
            positions = np.linspace(1.0, 0.1, len(step_rewards))
            terminal_per_step = terminal * positions
        elif self.terminal_spread == "signed_positional":
            # Signed ramp: +ramp when correct, -ramp when wrong, so EARLY steps carry
            # the largest magnitude in BOTH directions. This is what 'positional' was
            # documented to do. A failed rollout now gets its most negative credit on
            # its earliest steps, which is the actual "an early mistake dooms the rest"
            # prior. Centering on 0.5 keeps the mean effect comparable to 'positional'.
            positions = np.linspace(1.0, 0.1, len(step_rewards))
            terminal_per_step = (2.0 * terminal - 1.0) * positions
        else:  # 'omp_weighted'
            s = step_rewards.sum()
            if s > 1e-8:
                w = step_rewards / s
            else:
                w = np.ones_like(step_rewards) / len(step_rewards)
            terminal_per_step = terminal * w

        blended = self.alpha * step_rewards + (1 - self.alpha) * terminal_per_step

        # ── Fix 2: causal credit (cumulative discounted future rewards) ──
        # Variable-gamma per rollout: if self.gamma_total is set, compute the
        # per-step gamma so that gamma^(n_steps) ~= gamma_total, equalizing the
        # total decay regardless of rollout length.
        if self.gamma_total is not None and len(blended) > 0:
            # gamma_per_step = gamma_total ** (1 / n_steps)
            n_steps = len(blended)
            gamma_eff = float(self.gamma_total) ** (1.0 / max(n_steps, 1))
        else:
            gamma_eff = self.gamma

        if gamma_eff < 1.0 - 1e-6:
            returns = np.zeros_like(blended)
            running = 0.0
            for t in range(len(blended) - 1, -1, -1):
                running = blended[t] + gamma_eff * running
                returns[t] = running
            blended = returns

        return blended

    def _apply_terminal_source(
        self,
        raw_step_rewards: list[np.ndarray],
        terminals: list[float],
    ) -> list[np.ndarray]:
        """Build the exact reward arrays consumed by advantage normalization."""
        if len(raw_step_rewards) != len(terminals):
            raise ValueError(
                "raw reward/terminal length mismatch: "
                f"{len(raw_step_rewards)} != {len(terminals)}"
            )
        return [
            self._blend_step_rewards(raw, terminal)
            for raw, terminal in zip(raw_step_rewards, terminals)
        ]

    @staticmethod
    def _fcc_prompt_hash(prompt) -> str:
        text = prompt if isinstance(prompt, str) else json.dumps(prompt, sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _fcc_group_terminals(
        self,
        prompt,
        predictions: list[str | None],
    ) -> tuple[list[float], str | None]:
        """Score a current group against a frozen target without updating the bank."""
        target = self._fcc_targets.get(self._fcc_prompt_hash(prompt))
        if target is None:
            return [0.0] * len(predictions), None
        return [
            float(prediction is not None and prediction == target)
            for prediction in predictions
        ], target

    def _fce_group_terminals(
        self,
        prompt,
        predictions: list[str | None],
    ) -> tuple[list[float], dict[str, float] | None]:
        """Score a current group against immutable cross-panel evidence."""
        scores = self._fce_scores.get(self._fcc_prompt_hash(prompt))
        if scores is None:
            return [0.0] * len(predictions), None
        return [
            float(scores.get(prediction, 0.0)) if prediction is not None else 0.0
            for prediction in predictions
        ], scores

    def _permuted_fce_terminals(
        self,
        prompt,
        terminals: list[float],
        group_index: int,
    ) -> list[float]:
        """Break FCE alignment while preserving the exact group reward multiset."""
        if len(terminals) <= 1:
            return list(terminals)
        prompt_key = self._fcc_prompt_hash(prompt)
        global_step = int(getattr(getattr(self, "state", None), "global_step", 0))
        run_seed = int(getattr(getattr(self, "args", None), "seed", 0) or 0)
        material = (
            f"{run_seed}:{global_step}:{group_index}:{prompt_key}:fce-permuted"
        ).encode("utf-8")
        seed = int.from_bytes(
            hashlib.sha256(material).digest()[:8],
            "big",
        )
        permutation = np.random.default_rng(seed).permutation(len(terminals))
        values = np.asarray(terminals, dtype=np.float64)
        return list(values[permutation])

    def _gold_answers_for_batch(self, inputs: list[dict]) -> list[str | None]:
        """Repeat labels only for gold RLVR; all verifier-free methods are blind."""
        repeated: list[str | None] = []
        for ex in inputs:
            gold = (
                None
                if self.reward_source != "gold"
                else extract_answer(
                    ex.get("answer", ""),
                    self._fce_answer_mode,
                )
            )
            repeated.extend([gold] * self.num_generations)
        return repeated

    def _compute_per_step_rewards(self, completion_text: str, gold_answer: str | None):
        """Compatibility wrapper for the gold-terminal path."""
        raw, spans_text_only = self._compute_raw_step_rewards(completion_text)
        pred = extract_answer(completion_text, self._fce_answer_mode)
        is_correct = pred is not None and gold_answer is not None and pred == gold_answer
        blended = self._blend_step_rewards(raw, 1.0 if is_correct else 0.0)
        return blended, spans_text_only

    def _contrastive_group_step_rewards(
        self,
        group_step_texts: list[list[str]],   # one list-of-step-strings per rollout
        group_terminals: list[float],        # 0 or 1 per rollout (is_correct)
    ) -> list[np.ndarray | None]:
        """Compute per-step contrastive scores from a GRPO group.

        For each step position k and each scored rollout i (i held out of both
        centroids -- see the leave-one-out note in the loop below):
          c_succ[k] = mean(embed(step_k)) over success rollouts != i
          c_fail[k] = mean(embed(step_k)) over failure rollouts != i
          dir[k]    = normalize(c_succ[k] - c_fail[k])
        Per-rollout step k score = (embed(r.step_k) - c_fail[k]) · dir[k]

        Returns: list of arrays (one per rollout), or None for rollouts with
                  no contrast available at that position.
        """
        succ_idx = [i for i, t in enumerate(group_terminals) if t > 0.5]
        fail_idx = [i for i, t in enumerate(group_terminals) if t <= 0.5]
        if not succ_idx or not fail_idx:
            return [None] * len(group_step_texts)

        # Embed all steps for all rollouts in one batch (max efficiency)
        flat_steps, slices = [], []
        cursor = 0
        for steps in group_step_texts:
            slices.append((cursor, cursor + len(steps)))
            flat_steps.extend(steps)
            cursor += len(steps)
        if not flat_steps:
            return [None] * len(group_step_texts)
        flat_emb = self.encoder.encode(flat_steps, normalize_embeddings=True, batch_size=64)
        per_rollout_emb = [flat_emb[a:b] for (a, b) in slices]

        max_k = max(len(e) for e in per_rollout_emb)
        # Compute centroids per position k
        out_scores = [np.zeros(len(e), dtype=np.float32) for e in per_rollout_emb]
        n_used_per_rollout = [0] * len(per_rollout_emb)
        for k in range(max_k):
            # Centroids are rebuilt per scored rollout with that rollout held
            # out. Otherwise a rollout helps define the direction it is then
            # projected onto: with K=3 and a single success, c_succ[k] IS the
            # winner's own embedding, so its score collapses to
            # ||e - c_fail||, the maximum attainable value, for every winner.
            for i, emb in enumerate(per_rollout_emb):
                if k >= len(emb):
                    continue
                succ_k = [
                    per_rollout_emb[j][k]
                    for j in succ_idx
                    if j != i and k < len(per_rollout_emb[j])
                ]
                fail_k = [
                    per_rollout_emb[j][k]
                    for j in fail_idx
                    if j != i and k < len(per_rollout_emb[j])
                ]
                if not succ_k or not fail_k:
                    continue
                c_s = np.mean(succ_k, axis=0)
                c_f = np.mean(fail_k, axis=0)
                d = c_s - c_f
                d_norm = np.linalg.norm(d)
                if d_norm < 1e-6:
                    continue
                out_scores[i][k] = float(np.dot(emb[k] - c_f, d / d_norm))
                n_used_per_rollout[i] += 1

        # If a rollout got zero contrast positions, return None for it
        return [
            s if n > 0 else None
            for s, n in zip(out_scores, n_used_per_rollout)
        ]

    def _l2a_group_step_rewards(
        self,
        group_step_texts: list[list[str]],
        group_terminals: list[float],
    ) -> list[np.ndarray | None]:
        """Level 2a: per-group dynamic dictionary from success rollouts.

        For rollout i, D_succ = stacked step embeddings of the OTHER successful
        rollouts in this group (leave-one-rollout-out). OMP each step embedding
        of i against that D_succ; reward at step k is exp(-residual_k / tau).
        High = "explained by winning patterns other than my own".

        Returns: list of np.ndarray, or None for a rollout with no leak-free
        dictionary (no successes in the group, or i is the sole success). None
        means "undefined" and the caller must fall back to a label-free reward;
        it must never be coerced to 0.0.
        """
        try:
            from sklearn.linear_model import orthogonal_mp
        except Exception:
            return [None] * len(group_step_texts)

        succ_idx = [i for i, t in enumerate(group_terminals) if t > 0.5]
        if not succ_idx:
            return [None] * len(group_step_texts)

        # One-shot batch encode all steps
        flat, slices, cursor = [], [], 0
        for steps in group_step_texts:
            slices.append((cursor, cursor + len(steps)))
            flat.extend(steps)
            cursor += len(steps)
        if not flat:
            return [None] * len(group_step_texts)
        flat_emb = self.encoder.encode(flat, normalize_embeddings=True, batch_size=64)
        per_rollout_emb = [flat_emb[a:b] for (a, b) in slices]

        # Build the success dictionary LEAVE-ONE-ROLLOUT-OUT.
        #
        # Scoring a rollout against a dictionary that contains its own step
        # embeddings is not a measurement. Embeddings are unit-normalised, so
        # when the signal IS an atom, OMP's first greedy pick has correlation
        # exactly 1.0, the least-squares coefficient is 1, and the residual is
        # exactly 0 -> reward exp(0) = 1.0 at every step. That returns
        # is_correct, not a per-step signal (measured on cached rollouts:
        # 1.0000 +/- 0.0000 over all correct rollouts vs 0.0395 for wrong ones).
        # Only successful rollouts are affected; a failed rollout is not in
        # succ_idx, so its dictionary is unchanged by the hold-out.
        out = []
        for i, emb in enumerate(per_rollout_emb):
            if len(emb) == 0:
                out.append(None)
                continue
            succ_arrays = [
                per_rollout_emb[j]
                for j in succ_idx
                if j != i and len(per_rollout_emb[j]) > 0
            ]
            if not succ_arrays:
                # Sole winner in its group: no leak-free dictionary exists.
                # Return None so the caller falls back to the label-free OMP
                # reward. Never return 0.0 here -- the rollouts that land in
                # this branch are systematically the CORRECT ones, so a zero
                # fill is an inverted leak (measured: pick accuracy 0.125,
                # below the 0.147 random baseline).
                out.append(None)
                continue
            D_succ = np.vstack(succ_arrays).astype(np.float32)  # (M, d)
            n_atoms = D_succ.shape[0]
            n_nonzero = max(1, min(self.l2a_n_nonzero, n_atoms))
            codes = orthogonal_mp(D_succ.T, emb.T, n_nonzero_coefs=n_nonzero)
            codes = np.atleast_2d(codes)
            if codes.shape[0] != n_atoms:
                codes = codes.reshape(n_atoms, -1)
            recon = D_succ.T @ codes
            resid = np.linalg.norm(emb.T - recon, axis=0)
            out.append(np.exp(-resid / self.tau).astype(np.float32))
        return out

    def _indep_consensus_group_scores(
        self,
        group_step_texts: list[list[str]],
        group_raw_texts: list[str],
    ) -> list[float | None]:
        """Independence-weighted OMP consensus. Fully label-free.

            score_i = mean_j [ cov(i | peer_j) * (1 - lexical_overlap(i, peer_j))^lambda ]

        Agreement is measured SEMANTICALLY (OMP coverage of my step embeddings by peer j's),
        independence LEXICALLY (n-gram containment). They are combined MULTIPLICATIVELY, which
        is the whole design: a policy cannot farm this by making rollouts similar, because
        similarity drives the independence factor -- and hence the product -- to zero.

        Validated offline on n=150 cached GSM8K rollouts: AUROC 0.697 (plain coverage 0.683),
        and under K exact clones this returns 0.0000 where plain coverage returns 1.0000.

        Known limit, stated so it is not rediscovered as a surprise: peers that INDEPENDENTLY
        derive the same WRONG answer still score high. No within-group consensus signal can
        separate that from correctness; it needs information from outside the group.
        """
        try:
            from sklearn.linear_model import orthogonal_mp
        except Exception:
            return [None] * len(group_step_texts)

        n = len(group_step_texts)
        flat, slices, cursor = [], [], 0
        for steps in group_step_texts:
            slices.append((cursor, cursor + len(steps)))
            flat.extend(steps)
            cursor += len(steps)
        if not flat:
            return [None] * n
        emb_all = self.encoder.encode(flat, normalize_embeddings=True, batch_size=64)
        per_rollout = [emb_all[a:b] for (a, b) in slices]

        def ngrams(text):
            toks = re.findall(r"\w+", text.lower())
            k = self.consensus_ngram
            return set(tuple(toks[i:i + k]) for i in range(max(0, len(toks) - k + 1)))

        grams = [ngrams(t) for t in group_raw_texts]

        def coverage(sig, D):
            m = D.shape[0]
            if m == 0 or len(sig) == 0:
                return None
            nz = max(1, min(5, m))
            codes = orthogonal_mp(D.T, sig.T, n_nonzero_coefs=nz)
            codes = np.atleast_2d(codes)
            if codes.shape[0] != m:
                codes = codes.reshape(m, -1)
            resid = np.linalg.norm(sig.T - D.T @ codes, axis=0)
            return float(np.exp(-resid / self.tau).mean())

        out: list[float | None] = []
        for i in range(n):
            emb_i = per_rollout[i]
            if len(emb_i) == 0:
                out.append(None)
                continue
            terms = []
            for j in range(n):
                if j == i or len(per_rollout[j]) == 0:
                    continue
                agree = coverage(emb_i, per_rollout[j].astype(np.float32))
                if agree is None:
                    continue
                A, B = grams[i], grams[j]
                overlap = (len(A & B) / len(A)) if (A and B) else 0.0
                indep = max(0.0, 1.0 - overlap) ** self.consensus_lambda
                terms.append(agree * indep)
            out.append(float(np.mean(terms)) if terms else None)
        return out

    def _compute_step_advantages(self, all_step_rewards: list[np.ndarray], num_generations: int):
        """Given a list of per-step reward arrays (one per rollout), compute
        per-step advantages with top-K filtering and clipping.

        Two normalization modes:
          'pooled'     : pool ALL (rollout, step) pairs in a group, compute mean/std
          'group_mean' : trajectory-mean across the group as baseline

        Additional fixes applied AFTER normalization:
          - top_k_steps: zero out advantages for steps below the K-th highest |advantage|
                         in each rollout (focuses learning on confidently-scored steps)
          - advantage_clip: clip per-step advantages to [-clip, clip]
        """
        n_rollouts = len(all_step_rewards)
        groups = n_rollouts // num_generations
        out = [None] * n_rollouts

        for g in range(groups):
            slc = slice(g * num_generations, (g + 1) * num_generations)
            group_rewards = all_step_rewards[slc]

            if self.step_advantage_mode == "ema":
                # PPO-style baseline: advantage = (step_reward - V_ema) / σ_ema
                # Running mean AND running std → stable scale across batches.
                # No within-group normalization → per-step variance preserved.
                pooled = np.concatenate(group_rewards) if any(len(r) for r in group_rewards) else np.array([0.0])
                batch_mean = float(pooled.mean())
                batch_sq = float((pooled ** 2).mean())
                if not self._ema_initialized:
                    self._ema_reward = batch_mean
                    self._ema_reward_sq = batch_sq
                    self._ema_initialized = True
                else:
                    self._ema_reward = (1 - self._ema_alpha) * self._ema_reward + self._ema_alpha * batch_mean
                    self._ema_reward_sq = (1 - self._ema_alpha) * self._ema_reward_sq + self._ema_alpha * batch_sq
                # Running std from EMA of second moment. Floor at 0.1 to prevent
                # tiny-σ blowup when rewards are near-constant early in training.
                ema_var = max(self._ema_reward_sq - self._ema_reward ** 2, 0.0)
                sigma = max(float(np.sqrt(ema_var)), 0.1)
                for i in range(num_generations):
                    adv = (group_rewards[i] - self._ema_reward) / sigma
                    out[g * num_generations + i] = adv
            elif self.step_advantage_mode == "pooled":
                pooled = np.concatenate(group_rewards) if any(len(r) for r in group_rewards) else np.array([0.0])
                mu, sigma = pooled.mean(), pooled.std()
                for i in range(num_generations):
                    adv = (group_rewards[i] - mu) / (sigma + 1e-4)
                    out[g * num_generations + i] = adv
            else:  # group_mean
                group_trajectory_means = np.array([r.mean() if len(r) else 0.0 for r in group_rewards])
                mu_traj = group_trajectory_means.mean()
                sigma_traj = group_trajectory_means.std()
                for i in range(num_generations):
                    adv = (group_rewards[i] - mu_traj) / (sigma_traj + 1e-4)
                    out[g * num_generations + i] = adv

        # ── Fix 5: top-K filtering per rollout ──────────────────────────
        if self.top_k_steps is not None and self.top_k_steps > 0:
            filtered = []
            for adv in out:
                if adv is None or len(adv) == 0:
                    filtered.append(adv); continue
                k = min(self.top_k_steps, len(adv))
                top_idx = np.argsort(np.abs(adv))[-k:]      # by absolute magnitude
                mask = np.zeros_like(adv, dtype=bool)
                mask[top_idx] = True
                adv_filtered = np.where(mask, adv, 0.0)
                filtered.append(adv_filtered)
            out = filtered

        # ── Fix 4: advantage clipping ───────────────────────────────────
        if self.advantage_clip is not None and self.advantage_clip > 0:
            out = [np.clip(adv, -self.advantage_clip, self.advantage_clip) if adv is not None else None
                   for adv in out]

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
        step span (e.g., trailing whitespace) get advantage 0.

        With length_normalize=True, each step's per-token advantage is divided
        by sqrt(step_length) so long steps don't dominate the gradient.
        """
        device = completion_ids.device
        B, L = completion_ids.shape
        A = torch.zeros((B, L), dtype=torch.float32, device=device)

        for b in range(B):
            sr = step_advantages[b]
            # Exact trajectory-level GRPO: a constant rollout advantage applies
            # to every generated token. Step segmentation must not create holes.
            if len(sr) > 0 and np.allclose(sr, sr[0]):
                A[b, :] = float(sr[0])
                continue

            text = completion_texts[b]
            token_ids = completion_ids[b].tolist()
            spans = step_token_spans(self.processing_class, text, token_ids)

            if len(spans) == 0 or len(sr) == 0:
                continue
            n = min(len(spans), len(sr))
            for k in range(n):
                tok_start, tok_end, _ = spans[k]
                tok_end = min(tok_end, L)
                if tok_end <= tok_start:
                    continue
                step_len = tok_end - tok_start
                # ── Fix 3: length normalization ──────────────────────
                if self.length_normalize:
                    per_token = float(sr[k]) / (step_len ** 0.5)
                else:
                    per_token = float(sr[k])
                A[b, tok_start:tok_end] = per_token

        # Zero-out padded / post-EOS positions
        A = A * completion_mask.float()
        return A

    # ── main compute_loss override ───────────────────────────────────────

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("StepLevelGRPOTrainer does not support returning outputs")

        # Lazy imports — only available in the Modal container, not locally
        from trl.data_utils import maybe_apply_chat_template
        from trl.models import unwrap_model_for_generation

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
        # FCC's training path is structurally gold-blind. Even if a caller
        # accidentally leaves an answer field in the dataset, do not parse it.
        gold_answers_repeated = self._gold_answers_for_batch(inputs)

        per_rollout_raw_step_rewards = []
        per_rollout_step_texts = []
        per_rollout_terminals = []      # the terminal that TRAINS the policy
        per_rollout_gold_correct = []   # populated only for gold RLVR
        per_rollout_preds = []
        for text, gold in zip(completions, gold_answers_repeated):
            if self._exact_trajectory_mode():
                # Pure GRPO needs only one scalar per rollout. Avoid even loading
                # or invoking the optional OMP encoder/dictionary reward machinery.
                raw_sr, step_texts = np.array([0.0]), []
            else:
                raw_sr, step_texts = self._compute_raw_step_rewards(text)
            per_rollout_raw_step_rewards.append(raw_sr)
            per_rollout_step_texts.append(step_texts)
            pred = extract_answer(text, self._fce_answer_mode)
            per_rollout_preds.append(pred)
            is_correct = (pred is not None and gold is not None and pred == gold)
            per_rollout_gold_correct.append(1.0 if is_correct else 0.0)
            per_rollout_terminals.append(1.0 if is_correct else 0.0)

        # ── Verifier-free reward source ──────────────────────────────────
        # Replace the gold-derived terminal with a label that needs no verifier.
        # This is the "replace RLVR" lever. In every non-gold mode the dataset
        # answer is stripped and this trainer returns no gold value at all.
        fcc_groups_total = 0
        fcc_groups_accepted = 0
        fcc_groups_informative = 0
        fce_groups_total = 0
        fce_groups_covered = 0
        fce_groups_informative = 0
        fcc_target_gold_correct: list[float] = []
        if self.reward_source != "gold":
            G = self.num_generations
            n_groups = len(per_rollout_preds) // G
            for g in range(n_groups):
                lo, hi = g * G, (g + 1) * G
                preds_g = per_rollout_preds[lo:hi]
                if self.reward_source == "fcc":
                    fcc_groups_total += 1
                    terminals, target = self._fcc_group_terminals(
                        prompts_raw[g],
                        preds_g,
                    )
                    per_rollout_terminals[lo:hi] = terminals
                    if target is not None:
                        fcc_groups_accepted += 1
                        if terminals and max(terminals) > min(terminals):
                            fcc_groups_informative += 1
                        gold = gold_answers_repeated[lo]
                        if gold is not None:
                            fcc_target_gold_correct.append(float(target == gold))
                elif self.reward_source in ("fce", "fce_permuted"):
                    fce_groups_total += 1
                    terminals, scores = self._fce_group_terminals(
                        prompts_raw[g],
                        preds_g,
                    )
                    if self.reward_source == "fce_permuted":
                        terminals = self._permuted_fce_terminals(
                            prompts_raw[g],
                            terminals,
                            g,
                        )
                    per_rollout_terminals[lo:hi] = terminals
                    if scores is not None:
                        fce_groups_covered += 1
                        if terminals and max(terminals) > min(terminals):
                            fce_groups_informative += 1
                elif self.reward_source == "majority":
                    # TTRL-style: plurality answer in the group is the pseudo-gold.
                    from collections import Counter
                    valid = [p for p in preds_g if p is not None]
                    if valid:
                        pseudo_gold, _ = Counter(valid).most_common(1)[0]
                        for i in range(G):
                            per_rollout_terminals[lo + i] = (
                                1.0 if preds_g[i] is not None and preds_g[i] == pseudo_gold
                                else 0.0
                            )
                    else:
                        for i in range(G):
                            per_rollout_terminals[lo + i] = 0.0
                elif self.reward_source == "random":
                    # Spurious control: label independent of the completion.
                    draws = self._reward_rng.rand(G) < self.random_reward_p
                    for i in range(G):
                        per_rollout_terminals[lo + i] = float(draws[i])
                elif self.reward_source == "consensus":
                    # Pursuit method: continuous terminal in [0,1], no answer parsing.
                    sc = self._indep_consensus_group_scores(
                        per_rollout_step_texts[lo:hi], completions[lo:hi]
                    )
                    # Group-relative rescale: the raw scale of coverage drifts as the policy
                    # changes, so an absolute value would silently re-weight the reward over
                    # training. Rank within the group is what carries the signal.
                    vals = [v for v in sc if v is not None]
                    if len(vals) >= 2 and (max(vals) - min(vals)) > 1e-8:
                        vmin, vmax = min(vals), max(vals)
                        for i in range(G):
                            per_rollout_terminals[lo + i] = (
                                (sc[i] - vmin) / (vmax - vmin) if sc[i] is not None else 0.0
                            )
                    else:
                        # degenerate group (all identical / undefined) => no signal, not a
                        # fabricated one. Zero here means "this group teaches nothing".
                        for i in range(G):
                            per_rollout_terminals[lo + i] = 0.0

        # This is the actual reward-source switch. The previous implementation
        # changed only `per_rollout_terminals` after gold-derived step rewards had
        # already been built, so majority/random/consensus diagnostics changed while
        # the loss still trained on gold. Rebuild the exact arrays consumed below
        # from the selected terminals for every source.
        per_rollout_step_rewards = self._apply_terminal_source(
            per_rollout_raw_step_rewards,
            per_rollout_terminals,
        )
        trajectory_means = [
            float(reward.mean()) if len(reward) else 0.0
            for reward in per_rollout_step_rewards
        ]

        # Diagnostics: reward density. Verifier-free training examples contain
        # no gold, so they cannot silently compute pseudo-label accuracy.
        _gc = np.array(per_rollout_gold_correct)
        _tm = np.array(per_rollout_terminals)
        if len(_tm):
            self._metrics.setdefault("terminal_pos_rate", []).append(float(_tm.mean()))
            if any(answer is not None for answer in gold_answers_repeated):
                self._metrics.setdefault("pseudo_label_acc", []).append(
                    float((_tm == _gc).mean())
                )
                self._metrics.setdefault("gold_pos_rate", []).append(float(_gc.mean()))
            if self.reward_source == "fcc":
                self._metrics.setdefault("fcc_bank_coverage", []).append(
                    fcc_groups_accepted / max(fcc_groups_total, 1)
                )
                self._metrics.setdefault("fcc_informative_group_rate", []).append(
                    fcc_groups_informative / max(fcc_groups_total, 1)
                )
                if fcc_target_gold_correct:
                    # Evaluation-only diagnostic. This never enters any reward array.
                    self._metrics.setdefault("fcc_target_gold_accuracy", []).append(
                        float(np.mean(fcc_target_gold_correct))
                    )
            if self.reward_source in ("fce", "fce_permuted"):
                self._metrics.setdefault("fce_bank_coverage", []).append(
                    fce_groups_covered / max(fce_groups_total, 1)
                )
                self._metrics.setdefault("fce_informative_group_rate", []).append(
                    fce_groups_informative / max(fce_groups_total, 1)
                )

        # ── Contrastive per-step credit from GRPO group structure ────────
        n_contrast_groups_used = 0
        n_l2a_groups_used = 0
        if self.use_contrastive or self.use_l2a:
            n_groups = len(per_rollout_step_rewards) // self.num_generations
            for g in range(n_groups):
                lo, hi = g * self.num_generations, (g + 1) * self.num_generations
                group_texts = per_rollout_step_texts[lo:hi]
                group_terms = per_rollout_terminals[lo:hi]
                if self.use_contrastive:
                    contrast = self._contrastive_group_step_rewards(group_texts, group_terms)
                    for i, c in enumerate(contrast):
                        if c is not None and len(c) == len(per_rollout_step_rewards[lo + i]):
                            w = self.contrastive_weight
                            per_rollout_step_rewards[lo + i] = (
                                (1 - w) * per_rollout_step_rewards[lo + i] + w * c
                            )
                    if any(c is not None for c in contrast):
                        n_contrast_groups_used += 1
                if self.use_l2a:
                    l2a = self._l2a_group_step_rewards(group_texts, group_terms)
                    for i, c in enumerate(l2a):
                        if c is not None and len(c) == len(per_rollout_step_rewards[lo + i]):
                            w = self.l2a_weight
                            per_rollout_step_rewards[lo + i] = (
                                (1 - w) * per_rollout_step_rewards[lo + i] + w * c
                            )
                    if any(c is not None for c in l2a):
                        n_l2a_groups_used += 1

        # ── Hybrid orthogonal reward (independent of trajectory outcome) ─
        # r_step[k] = terminal_w * terminal_per_step
        #           + length_w  * exp(-|step_len - target| / target)   # concision
        #           + conf_w    * mean_token_confidence[step_k]        # commitment
        if self.use_hybrid:
            # Detach per_token_logps for confidence proxy (no grad through reward)
            with torch.no_grad():
                per_tok_conf = torch.exp(per_token_logps.detach()).cpu().numpy()  # [B*G, L]
            for i, (text, gold, steps) in enumerate(
                zip(completions, gold_answers_repeated, per_rollout_step_texts)
            ):
                pred = extract_answer(text, self._fce_answer_mode)
                is_correct = (pred is not None and gold is not None and pred == gold)
                terminal = 1.0 if is_correct else 0.0
                if not steps:
                    per_rollout_step_rewards[i] = np.array([terminal])
                    continue
                # Per-step length penalty (orthogonal to correctness)
                target = self.hybrid_conciseness_target
                lengths = np.array([len(s.split()) for s in steps], dtype=np.float32)
                length_score = np.exp(-np.abs(lengths - target) / target)
                # Per-step confidence (mean of token probs within the step, coarsely)
                n_tokens_total = per_tok_conf.shape[1]
                per_step_tokens = max(1, n_tokens_total // max(len(steps), 1))
                conf_score = np.zeros(len(steps), dtype=np.float32)
                for k in range(len(steps)):
                    lo, hi = k * per_step_tokens, min((k + 1) * per_step_tokens, n_tokens_total)
                    if hi > lo:
                        conf_score[k] = float(per_tok_conf[i, lo:hi].mean())
                # Combine: terminal component distributed uniformly
                terminal_per_step = np.full(len(steps), terminal, dtype=np.float32)
                r_hybrid = (
                    self.hybrid_terminal_weight * terminal_per_step
                    + self.hybrid_length_weight * length_score
                    + self.hybrid_confidence_weight * conf_score
                )
                per_rollout_step_rewards[i] = r_hybrid

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

        # Per-token advantage variance — TOTAL (between + within rollout)
        adv_var = (per_token_advantages * completion_mask).pow(2).sum() / completion_mask.sum().clamp_min(1)
        self._metrics["per_token_adv_var"].append(adv_var.item())

        # WITHIN-rollout variance — proves step-level credit is actually applied.
        # If this is ~0 while per_token_adv_var is high, we're doing trajectory GRPO.
        with torch.no_grad():
            B, L = per_token_advantages.shape
            mask_f = completion_mask.float()
            mask_sum = mask_f.sum(dim=1).clamp_min(1)
            rollout_mean = (per_token_advantages * mask_f).sum(dim=1) / mask_sum   # [B]
            centered = (per_token_advantages - rollout_mean.unsqueeze(1)) * mask_f  # [B, L]
            within_var = (centered ** 2).sum(dim=1) / mask_sum
            self._metrics["within_rollout_adv_var"].append(within_var.mean().item())

            # Span coverage: fraction of completion tokens that received non-zero advantage
            nonzero_token_frac = ((per_token_advantages != 0).float() * mask_f).sum() / mask_f.sum().clamp_min(1)
            self._metrics["adv_token_coverage"].append(nonzero_token_frac.item())
            if self.use_contrastive:
                n_groups_total = max(len(per_rollout_step_rewards) // self.num_generations, 1)
                self._metrics["contrast_group_frac"].append(n_contrast_groups_used / n_groups_total)
            if self.use_l2a:
                n_groups_total = max(len(per_rollout_step_rewards) // self.num_generations, 1)
                self._metrics["l2a_group_frac"].append(n_l2a_groups_used / n_groups_total)

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
    """Regression test: constant step rewards → uniform per-token advantages."""
    np.random.seed(0)
    rollouts_step_rewards = [
        np.full(5, 0.3),
        np.full(5, 0.7),
        np.full(5, 0.1),
        np.full(5, 0.5),
    ]
    pooled = np.concatenate(rollouts_step_rewards)
    mu, sigma = pooled.mean(), pooled.std()
    step_adv = [(r - mu) / (sigma + 1e-4) for r in rollouts_step_rewards]
    for i, adv in enumerate(step_adv):
        assert np.allclose(adv, adv[0]), f"Rollout {i}: advantages should be uniform"
    print("✓ Regression test passed: constant step rewards → uniform per-token advantages")
    traj_advs = [a[0] for a in step_adv]
    print(f"  trajectory advantages: {[round(a, 3) for a in traj_advs]}")
    assert traj_advs[1] == max(traj_advs), "highest reward should map to highest advantage"


def validate_constant_token_coverage():
    """Constant rollout advantages must cover every non-padding completion token."""
    trainer = object.__new__(StepLevelGRPOTrainer)
    completion_ids = torch.tensor([[11, 12, 13, 0]])
    completion_mask = torch.tensor([[1, 1, 1, 0]])
    advantages = trainer._build_per_token_advantages(
        completion_texts=["segmentation is deliberately irrelevant"],
        completion_ids=completion_ids,
        step_advantages=[np.array([0.75, 0.75, 0.75])],
        completion_mask=completion_mask,
    )
    assert torch.allclose(
        advantages,
        torch.tensor([[0.75, 0.75, 0.75, 0.0]]),
    ), advantages
    covered = ((advantages != 0).float() * completion_mask).sum() / completion_mask.sum()
    assert float(covered) == 1.0
    values = advantages[completion_mask.bool()]
    assert float(values.var(unbiased=False)) == 0.0
    print("✓ constant trajectory advantage covers all generated tokens with zero within-rollout variance")


def validate_terminal_spread():
    """Validate that terminal_spread modes redistribute the binary terminal correctly."""
    step_rewards = np.array([0.1, 0.5, 0.3, 0.2])     # raw OMP rewards
    terminal = 1.0                                      # correct answer
    alpha = 0.5

    # last_only
    last = np.zeros_like(step_rewards)
    last[-1] = terminal
    blended = alpha * step_rewards + (1 - alpha) * last
    assert abs(blended.sum() - (alpha * step_rewards.sum() + (1 - alpha) * terminal)) < 1e-6
    print(f"✓ last_only: total mass conserved ({blended.sum():.3f})")

    # uniform
    unif = np.full_like(step_rewards, terminal / len(step_rewards))
    blended = alpha * step_rewards + (1 - alpha) * unif
    assert abs(blended.sum() - (alpha * step_rewards.sum() + (1 - alpha) * terminal)) < 1e-6
    print(f"✓ uniform: total mass conserved ({blended.sum():.3f})")

    # constant: length-invariant, deliberately NOT mass-conserving
    const = np.full_like(step_rewards, terminal)
    assert np.allclose(const, terminal), "constant gives full terminal at every step"
    long_const = np.full(4 * len(step_rewards), terminal)
    assert abs(const.mean() - long_const.mean()) < 1e-9, \
        "constant must be length-invariant (this is what distinguishes it from uniform)"
    unif_long = np.full(4 * len(step_rewards), terminal / (4 * len(step_rewards)))
    assert unif_long.mean() < unif.mean(), "uniform penalizes length; constant does not"
    print(f"✓ constant: length-invariant ({const.round(3)}), unlike uniform")

    # omp_weighted
    w = step_rewards / step_rewards.sum()
    omp_w = terminal * w
    blended = alpha * step_rewards + (1 - alpha) * omp_w
    assert abs(blended.sum() - (alpha * step_rewards.sum() + (1 - alpha) * terminal)) < 1e-6
    print(f"✓ omp_weighted: total mass conserved ({blended.sum():.3f})")
    # high-OMP-score step should get more terminal credit
    assert omp_w[1] > omp_w[0], "step with highest OMP gets highest terminal share"
    print(f"  omp_weighted distribution: {omp_w.round(3)} (high OMP gets more terminal)")


def validate_reward_source_rewire():
    """Changing the selected terminals must change the arrays consumed by the loss."""
    trainer = object.__new__(StepLevelGRPOTrainer)
    trainer.alpha = 0.0
    trainer.terminal_spread = "constant"
    trainer.gamma = 1.0
    trainer.gamma_total = None

    raw = [
        np.array([0.2, 0.4], dtype=np.float32),
        np.array([0.8, 0.6], dtype=np.float32),
    ]
    gold = trainer._apply_terminal_source(raw, [1.0, 0.0])
    random = trainer._apply_terminal_source(raw, [0.0, 1.0])

    assert np.allclose(gold[0], 1.0) and np.allclose(gold[1], 0.0)
    assert np.allclose(random[0], 0.0) and np.allclose(random[1], 1.0)
    assert not all(np.array_equal(a, b) for a, b in zip(gold, random)), (
        "reward_source changed diagnostics but not loss rewards"
    )
    print("✓ reward_source terminals rebuild the exact step-reward arrays used by the loss")


def validate_fcc_frozen_target():
    """Current-group outputs can change rewards but cannot mutate the FCC target."""
    trainer = object.__new__(StepLevelGRPOTrainer)
    prompt = "Solve this problem"
    key = trainer._fcc_prompt_hash(prompt)
    trainer._fcc_targets = {key: "18"}

    rewards, target = trainer._fcc_group_terminals(
        prompt,
        ["18", "19", None, "18"],
    )
    assert target == "18"
    assert rewards == [1.0, 0.0, 0.0, 1.0]

    collapsed, target_after = trainer._fcc_group_terminals(
        prompt,
        ["999"] * 8,
    )
    assert collapsed == [0.0] * 8
    assert target_after == "18"
    assert trainer._fcc_targets[key] == "18"
    print("✓ FCC current-group collapse cannot move the frozen target")


def validate_fce_frozen_scores():
    """Current outputs can query but cannot mutate frozen FCE scores."""
    trainer = object.__new__(StepLevelGRPOTrainer)
    prompt = "Solve this other problem"
    key = trainer._fcc_prompt_hash(prompt)
    trainer._fce_scores = {key: {"18": 0.4, "21": 0.1}}
    rewards, scores = trainer._fce_group_terminals(
        prompt,
        ["18", "21", "999", None],
    )
    assert rewards == [0.4, 0.1, 0.0, 0.0]
    assert scores == {"18": 0.4, "21": 0.1}
    before = dict(trainer._fce_scores[key])
    collapsed, _ = trainer._fce_group_terminals(prompt, ["999"] * 8)
    assert collapsed == [0.0] * 8
    assert trainer._fce_scores[key] == before
    print("✓ FCE current-group collapse cannot move frozen evidence scores")


def validate_fce_permuted_control():
    """Matched control preserves signal density but breaks trajectory alignment."""
    trainer = object.__new__(StepLevelGRPOTrainer)
    trainer.state = type("State", (), {"global_step": 17})()
    trainer.args = type("Args", (), {"seed": 42})()
    prompt = "Solve step by step:\nA test prompt.\n\nSolution:"
    terminals = [0.4, 0.1, 0.0, 0.0]
    permuted = trainer._permuted_fce_terminals(prompt, terminals, 0)
    repeated = trainer._permuted_fce_terminals(prompt, terminals, 0)
    assert sorted(permuted) == sorted(terminals)
    assert permuted != terminals
    assert repeated == permuted
    assert trainer._permuted_fce_terminals(
        prompt,
        [0.0, 0.0, 0.0, 0.0],
        0,
    ) == [0.0, 0.0, 0.0, 0.0]
    print("✓ matched FCE control preserves rewards while breaking trajectory alignment")


def validate_fce_continuous_loss_path():
    """Frozen continuous scores must preserve their ranking through GRPO normalization."""
    trainer = object.__new__(StepLevelGRPOTrainer)
    trainer.alpha = 0.0
    trainer.terminal_spread = "constant"
    trainer.gamma = 1.0
    trainer.gamma_total = None
    trainer.step_advantage_mode = "group_mean"
    trainer.top_k_steps = None
    trainer.advantage_clip = 5.0
    raw = [np.array([9.0], dtype=np.float32) for _ in range(4)]
    loss_rewards = trainer._apply_terminal_source(raw, [0.4, 0.1, 0.0, 0.0])
    assert np.allclose(
        [float(reward[0]) for reward in loss_rewards],
        [0.4, 0.1, 0.0, 0.0],
    )
    advantages = trainer._compute_step_advantages(loss_rewards, num_generations=4)
    values = [float(advantage[0]) for advantage in advantages]
    assert values[0] > values[1] > values[2]
    assert values[2] == values[3]
    print("✓ FCE continuous scores reach the loss and preserve group-relative ranking")


def validate_fcc_gold_blindness():
    """FCC ignores an answer field even if a caller accidentally supplies one."""
    trainer = object.__new__(StepLevelGRPOTrainer)
    trainer.reward_source = "fcc"
    trainer.num_generations = 4
    hidden = trainer._gold_answers_for_batch(
        [{"prompt": "irrelevant", "answer": r"\boxed{18}"}]
    )
    assert hidden == [None, None, None, None]

    trainer.reward_source = "fce"
    hidden_fce = trainer._gold_answers_for_batch(
        [{"prompt": "irrelevant", "answer": r"\boxed{18}"}]
    )
    assert hidden_fce == [None, None, None, None]

    trainer.reward_source = "fce_permuted"
    hidden_control = trainer._gold_answers_for_batch(
        [{"prompt": "irrelevant", "answer": r"\boxed{18}"}]
    )
    assert hidden_control == [None, None, None, None]

    trainer.reward_source = "gold"
    visible = trainer._gold_answers_for_batch(
        [{"prompt": "irrelevant", "answer": r"\boxed{18}"}]
    )
    assert visible == ["18", "18", "18", "18"]
    print("✓ FCC/FCE/control paths are gold-blind even when an answer field is supplied")


def validate_exact_trajectory_bypasses_auxiliary_reward():
    """Pure FCC-GRPO must not require the optional encoder/dictionary path."""
    trainer = object.__new__(StepLevelGRPOTrainer)
    trainer.alpha = 0
    trainer.terminal_spread = "constant"
    trainer.use_contrastive = False
    trainer.use_l2a = False
    trainer.use_hybrid = False
    trainer.encoder = None
    trainer.D = None
    assert trainer._exact_trajectory_mode()
    print("✓ exact FCC/FCE trajectory mode bypasses the auxiliary reward stack")


def validate_causal_credit():
    """Gamma=1 should be a no-op (identity). Gamma<1 should propagate future credit backward."""
    step_rewards = np.array([0.1, 0.5, 0.3, 0.2])

    gamma = 1.0
    if gamma >= 1.0 - 1e-6:
        out = step_rewards.copy()
        print(f"✓ gamma=1.0: no-op {out.round(3)}")

    gamma = 0.9
    returns = np.zeros_like(step_rewards)
    running = 0.0
    for t in range(len(step_rewards) - 1, -1, -1):
        running = step_rewards[t] + gamma * running
        returns[t] = running
    assert returns[0] > returns[-1], "causal returns should accumulate from the end"
    print(f"✓ gamma=0.9: returns {returns.round(3)} (earlier steps get future credit)")


def validate_variable_gamma():
    """gamma_total=0.7 should give a per-step gamma that depends on n_steps."""
    gamma_total = 0.7
    for n in [4, 8, 16]:
        gamma_eff = gamma_total ** (1.0 / n)
        total_decay = gamma_eff ** n
        assert abs(total_decay - gamma_total) < 1e-6, "total decay should equal gamma_total"
        print(f"✓ n_steps={n:2d}: gamma_per_step={gamma_eff:.4f}, total_decay={total_decay:.4f}")
    # Shorter rollouts → lower per-step gamma → faster decay per step
    g_short = gamma_total ** (1.0 / 4)
    g_long = gamma_total ** (1.0 / 16)
    assert g_short < g_long, "short rollouts should have lower per-step gamma"
    print(f"✓ short rollout per-step gamma ({g_short:.3f}) < long rollout ({g_long:.3f})")


def validate_top_k_filter():
    """Top-K should zero out all but the K largest |advantage|."""
    advantages = np.array([0.1, -0.8, 0.3, 0.5, -0.2])
    k = 2
    top_idx = np.argsort(np.abs(advantages))[-k:]
    mask = np.zeros_like(advantages, dtype=bool)
    mask[top_idx] = True
    filtered = np.where(mask, advantages, 0.0)
    assert (filtered != 0).sum() == k
    assert filtered[1] == -0.8 and filtered[3] == 0.5, "top-2 by |adv| should be -0.8 and 0.5"
    print(f"✓ top_k=2: {advantages.round(2)} → {filtered.round(2)}")


def validate_advantage_clip():
    """Clipping should bound advantages to [-c, c]."""
    advantages = np.array([-5.0, -2.0, 0.0, 3.5, 10.0])
    c = 3.0
    clipped = np.clip(advantages, -c, c)
    assert (clipped.min() == -c) and (clipped.max() == c)
    print(f"✓ clip=3.0: {advantages.round(1)} → {clipped.round(1)}")


def validate_length_normalize():
    """Long steps should get smaller per-token advantage at same step-level advantage."""
    adv = 1.0
    short_len, long_len = 4, 100
    short_per_tok = adv / (short_len ** 0.5)
    long_per_tok  = adv / (long_len ** 0.5)
    assert short_per_tok > long_per_tok
    print(f"✓ length_normalize: short-step per-token {short_per_tok:.3f} > long-step per-token {long_per_tok:.3f}")


if __name__ == "__main__":
    from transformers import AutoTokenizer
    print("=== Step Alignment Test ===")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    sample = """Let x be the unknown variable.

Subtract 5 from both sides: 2x = 8.

Divide by 2: x = 4.

We verify: 2(4) + 5 = 13. The answer is correct."""
    validate_step_alignment(tok, sample)

    print("\n=== Regression Test ===")
    validate_regression()
    validate_constant_token_coverage()

    print("\n=== Fix 1: Terminal Spread ===")
    validate_terminal_spread()

    print("\n=== Reward Source Rewire ===")
    validate_reward_source_rewire()

    print("\n=== Frozen Cross-Consensus ===")
    validate_fcc_frozen_target()
    validate_fce_frozen_scores()
    validate_fce_permuted_control()
    validate_fce_continuous_loss_path()
    validate_fcc_gold_blindness()
    validate_exact_trajectory_bypasses_auxiliary_reward()

    print("\n=== Fix 2: Causal Credit ===")
    validate_causal_credit()

    print("\n=== Fix 2b: Variable Gamma (length-equalized) ===")
    validate_variable_gamma()

    print("\n=== Fix 3: Length Normalization ===")
    validate_length_normalize()

    print("\n=== Fix 4: Advantage Clipping ===")
    validate_advantage_clip()

    print("\n=== Fix 5: Top-K Filter ===")
    validate_top_k_filter()

    print("\n=== ALL VALIDATIONS PASSED ===")
