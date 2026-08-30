# FCE-GRPO claim ledger — 2026-07-29

## Proposed method

Frozen Cross-Evidence GRPO (FCE-GRPO) samples two domain-separated 12-rollout panels from
the initial policy. It retains every answer observed in both panels with at least two
non-clone supporting paths. Each retained answer receives a frozen score equal to
geometric cross-panel support times normalized inverse frequency across prompts.

Current-policy rollouts receive the frozen score associated with their extracted answer.
They cannot update the score table. Unseen answers receive zero. Strict single-target FCC
is retained as a baseline; FCE is primary because it is denser without becoming on-policy.

## No-verifier boundary

The FCE training bank stores prompts, initial-policy completions, frozen scores, and
generation metadata. It declares `gold_stored=false`.

Exact FCE training:

- strips answer fields from training examples;
- refuses to parse answer fields even if one is accidentally supplied;
- does not load or invoke a reward model, verifier model, sentence encoder, OMP
  dictionary, web service, or gold-answer checker;
- uses answer extraction only to compare current text with a frozen initial-policy
  text target.

Gold is loaded only by separate evaluation scripts after bank construction or training.

## Certified implementation invariants

- Reward-source terminals rebuild the exact arrays consumed by the loss.
- Constant trajectory advantages cover every generated token.
- Constant trajectory advantages have zero within-rollout token variance, up to
  floating-point roundoff.
- Current-policy clone collapse cannot move frozen targets or evidence scores.
- FCE ignores an answer field in training mode.
- Exact FCE trajectory mode bypasses the auxiliary step-reward stack.
- Interrupted bank generation is resumable with deterministic, prompt-derived,
  panel-separated seeds.
- Model-only Transformers checkpoints are supplemented with atomically certified,
  per-parameter streamed optimizer state. Adam moments, linear-scheduler position,
  and the seed-42 RNG stream therefore resume exactly without the measured host-RAM
  spike from monolithic optimizer serialization.

## Evidence gates

1. SmolLM2 selection: 500 GSM8K test prompts, two 12-sample target panels and one
   independent 4-sample candidate panel.
2. Qwen selection: same design, run only if SmolLM2 passes.
3. Each family must capture at least 70% of oracle candidate-selection gain with
   paired-bootstrap `p < 0.05`, and at least 30% of candidate groups must contain
   both target-matching and non-matching rollouts so GRPO has nonzero relative signal.
4. Only then build a gold-free 1,000-prompt training bank. At least 200 prompts must
   contain nonempty FCE evidence before online training.
5. Run 1,000 exact GRPO steps with K=4, seed 42, and LR 2e-5.
6. Compare the trained policy with the untrained base using deterministic greedy
   decoding on untouched GSM8K test items 500:1000.
7. A viability claim requires at least +3 percentage points and paired-bootstrap
   `p < 0.05`.
8. Run a matched trajectory-permutation control regardless of the primary outcome,
   with identical reward multisets, density, training hyperparameters, and evaluation
   prompts. An identified FCE-learning claim additionally requires FCE to beat this
   control with paired-bootstrap `p < 0.05`.

## Current evidence

- Four pure FCC and four pure FCE behavior tests pass.
- The isolated trainer regression suite passes all reward-source, all-token,
  immutability, gold-blindness, and auxiliary-bypass checks.
- The focused resume-state regression verifies exact Adam/scheduler restoration,
  an identical first post-resume update, RNG-state persistence, restoration before
  `TrainerState` assignment, and rejection of uncertified model-only checkpoints.
- The planned online bank (`train[:1000]`) and held-out evaluation slice
  (`test[500:1000]`) have zero exact-question or normalized-question overlap.
- The execution environment is pinned at TRL 0.14.0, Transformers 4.46.2, and
  PyTorch 2.11.0+cu128; the inherited GRPO KL coefficient is `beta=0.04`.
- The pre-registered SmolLM2 selection gate completed on 500 GSM8K test prompts.
  FCE improved independent four-candidate selection from 26.05% random accuracy
  to 48.40%, a gain of 22.35 percentage points (paired-bootstrap 95% CI
  [19.60, 25.10], `p(gain <= 0) = 0` over 50,000 draws).
- FCE captured 77.20% of the available oracle candidate-selection gain and
  produced informative mixed-reward groups on 56.60% of prompts, passing all
  three pre-registered SmolLM2 gates.
- Strict single-target FCC improved selection by 15.00 percentage points but
  captured only 51.81% of oracle gain, failing the pre-registered 70% capture
  gate. This supports the graded multi-answer FCE construction over FCC for the
  next stage.
- The pre-registered Qwen replication completed on 500 GSM8K test prompts.
  FCE improved independent four-candidate selection from 40.30% to 66.55%,
  a gain of 26.25 percentage points (paired-bootstrap 95% CI
  [23.45, 29.05], `p(gain <= 0) = 0` over 50,000 draws).
- Qwen FCE captured 81.78% of oracle gain and produced informative groups on
  72.00% of prompts, passing all three pre-registered cross-family gates.
- Qwen strict FCC captured only 59.35% of oracle gain and again failed the
  pre-registered 70% capture gate.
- Re-running both Qwen evaluators from the sealed bank produced byte-identical
  result artifacts.
- The gold-free online bank contains 1,000 prompts, with frozen evidence on 895.
  The audit recomputed every evidence score exactly, confirmed that no gold or
  candidate panel was stored, and found zero normalized or exact overlap with
  held-out GSM8K test items 500:1000.
- A clean 1,000-step SmolLM2 FCE-GRPO run improved deterministic greedy held-out
  accuracy from 148/500 (29.6%) to 257/500 (51.4%), a gain of 21.8 percentage
  points (paired-bootstrap 95% CI [17.4, 26.2], `p(gain <= 0) = 0`).
- The unconditional matched control ran cleanly from step 0 with the same base
  model, bank, seed, schedule, K=4 groups, and exact reward multiset in every
  group. It uniformly permuted reward assignment across trajectories and scored
  168/500 (33.6%).
- FCE exceeded that matched control by 17.8 percentage points
  (paired-bootstrap 95% CI [13.2, 22.4], `p(gain <= 0) = 0`), passing the
  pre-registered attribution gate. FCE was uniquely correct on 123 prompts and
  the control was uniquely correct on 34.
- The control's checkpoint-1000 history contains all 200 expected five-step log
  records, 100% frozen-bank coverage, 64.05% mean nonzero-token advantage
  coverage, and maximum within-rollout advantage variance `2.49e-15`. Its final
  weight hash exactly equals its checkpoint-1000 weight hash.
- The behavior audit is descriptive rather than a new gate. It found 100%
  parseable FCE completions versus 78.6% for the matched control and 68.2% for
  the base. On the 393 prompts where both FCE and control were parseable, FCE
  still led 53.18% to 42.75%, so the accuracy result is not only recovery from
  unparseable base outputs.
- The final execution audit passed all integrity checks, including paired prompt
  identity, immutable source and protocol hashes, the complete uniform
  permutation domain, clean live invocation, training artifacts, and explicit
  exclusion of every interrupted or failed-replay attempt.

## Not yet established

- Cross-model replication of actual policy learning; Qwen currently replicates
  candidate selection, not the full 1,000-step training result.
- Generalization beyond canonicalizable numeric GSM8K-style answers.
- Step-local reasoning credit; the demonstrated dense signal broadcasts one
  trajectory advantage to all generated tokens.
- Superiority to gold GRPO or to other verifier-free baselines.

Within the registered SmolLM2/GSM8K setting, FCE-GRPO is now a validated
verifier-free training mechanism: it improves the held-out policy and beats a
reward-density- and reward-multiset-matched control. Claims of broad RLVR
replacement still require the replications listed above.
