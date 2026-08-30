# Frozen Cross-Evidence GRPO

FCE-GRPO uses the initial policy as its own frozen reward source:

1. Sample two independent 12-rollout reference panels for each prompt.
2. Retain every answer independently observed in both panels with at least two
   independently worded supporting paths.
3. Score retained open answers by geometric cross-panel support times normalized
   inverse frequency across prompts. For a fixed A-D choice vocabulary, use
   geometric cross-panel support without IDF.
4. Freeze the complete prompt-to-answer score table for the full GRPO run.
5. Reward current rollouts by table lookup; unseen answers receive zero.

## Exact training signal

For prompt `q`, answer `a`, panel size `P`, and a bank of `N` prompts, the
frozen reward is

```text
support(q,a) = sqrt((count_A(q,a) / P) * (count_B(q,a) / P))
idf(a)       = log((N + 1) / (document_frequency(a) + 1)) / log(N + 1)
reward(q,a)  = support(q,a) * idf(a)
```

For fixed-choice tasks, `idf(a) = 1`. Choice letters are intentionally reused
across every prompt, so letter rarity is not evidence and would introduce a
position bias. This prospective task adaptation is sealed in
`CHOICE_MODE_ADDENDUM_2026-07-30.json`.

The entry is retained only when the answer occurs in both panels and its
combined support contains at least two non-clone reasoning paths. During
training, each current completion is canonicalized to an answer and receives
that immutable lookup value or zero. For each K=4 group, the four trajectory
rewards are centered and scaled by their group mean and standard deviation,
then each trajectory's scalar advantage is broadcast to every generated token
in that trajectory before the clipped GRPO loss and KL penalty are applied.

This makes the learning signal dense over generated tokens, but it is
trajectory-level credit, not evidence that individual reasoning steps were
localized correctly. The implementation still assumes answers can be reliably
canonicalized. Numeric, boxed-math-surface, and fixed-choice adapters now
exist, but the completed policy-learning evidence remains GSM8K-only and does
not cover arbitrary open-ended text.

The current policy never enters target construction, so current-group collapse cannot move
the reward. Cross-panel support rejects one-panel accidents, the clone gate prevents
duplicated wording from masquerading as independent evidence, and inverse frequency
suppresses generic open answers shared across many unrelated prompts.

Strict single-target Frozen Cross-Consensus (FCC) remains an evaluated baseline. FCE is the
primary online method because it preserves all cross-confirmed evidence and produces denser
group-relative signal.

Each panel has a prompt-derived, domain-separated seed, so interrupted generation resumes
with the same panel samples and the two reference panels cannot accidentally share a
replayed RNG stream.

No gold answers are stored in the bank and no separate reward/verifier model is used.
The online entrypoint strips answer fields from every verifier-free training
example, and the isolated trainer refuses to parse an answer field in those
modes even if one is accidentally supplied.
In exact FCE-GRPO mode (`alpha=0`, constant terminal spread), it also bypasses OMP entirely
and does not load a sentence encoder or dictionary.
`evaluate_fcc_bank.py` loads gold only after bank construction to measure target precision,
selection gain, wrong-target reinforcement, and paired-bootstrap power.

Shared-GPU training uses model-only Transformers checkpoints to avoid the measured host-RAM
spike from monolithic Adam serialization. The trainer additionally streams optimizer state
one parameter at a time, then atomically certifies the optimizer, linear learning-rate
scheduler, and RNG stream together. The previous certified checkpoint remains available
until the next exact-resume checkpoint is complete.

Implementation:

- `fcc_reward.py`: pure two-panel target construction and immutable current-rollout reward.
- `fce_reward.py`: pure multi-answer frozen evidence construction and lookup reward.
- `build_fcc_bank.py`: resumable gold-free bank generation from the initial policy.
- `evaluate_fcc_bank.py`: evaluation-only target precision and selection-power gate.
- `evaluate_fce_bank.py`: primary FCE selection, signal-density, and power gate.
- `fcc_step_pgr_trainer.py`: isolated GRPO integration; it filters current terminals
  through the frozen bank and then uses the repaired loss-side reward-source path.
- `local_fcc_train.py`: runnable 5090 entrypoint.
- `run_fcc_selection_gate.sh`: SmolLM gate first, then Qwen only if SmolLM passes.
- `run_fcc_online_gate.sh`: only after both selection gates pass, builds a gold-free
  train bank, runs exact FCE-GRPO, and evaluates on untouched GSM8K test items 500:1000.
- `evaluate_fcc_online.py` / `analyze_fcc_online.py`: deterministic greedy evaluation
  and paired-bootstrap comparison against the untrained base model.
- `audit_fce_online.py`: fail-closed recomputation of bank evidence, dataset ordering,
  train/held-out separation, evaluation correctness, paired statistics, model artifacts,
  and source hashes.
- `analyze_fce_control.py`: paired attribution test against trajectory-permuted FCE,
  which preserves every group's exact reward multiset while breaking answer/reward
  alignment.
- `analyze_fce_wrong_consensus.py`: gold-separated development/locked-replication
  stress test for systematic wrong consensus.
- `fce_tasks.py`: frozen-preprocessing and evaluation adapters for GSM8K,
  MATH-Hard, and MMLU.
- `run_fce_replication_queue.sh`: fail-closed multi-seed, Qwen, domain-selection,
  gold-GRPO, and majority-baseline queue.

The online protocol gated training until selection captured at least 70% of oracle gain,
had paired-bootstrap `p < 0.05`, and yielded at least 30% informative mixed-reward candidate
groups on both model families. The final viability claim additionally required at least
+3 percentage points over the base model with paired-bootstrap `p < 0.05` on the held-out
500-example slice. All of these gates passed.

Because earlier GSM8K runs showed that random rewards can sometimes move GRPO policies,
the identified FCE learning claim additionally required FCE to beat a preregistered matched
control with paired-bootstrap `p < 0.05`. The control starts from the same base model and
uses the same bank, seed, schedule, group size, and reward distribution, but uniformly
shuffles each group's frozen rewards across trajectories. It runs regardless of the
primary outcome so the attribution test is not conditionally selected on held-out results.
This gate also passed.

The panel size is power-derived rather than tuned on FCE outcomes. At a 15% per-rollout
correct-answer rate, the probability that both panels contain at least two correct paths is
about 1.2% for panel size 4 versus 31.0% for panel size 12 (before accounting for wrong-answer
collisions). The gold-separated selection gate measures the actual precision and coverage.

## Validated result

On the registered SmolLM2/GSM8K experiment, deterministic greedy accuracy on
untouched test items 500:1000 was:

- base model: 148/500 (29.6%);
- matched reward-permutation control: 168/500 (33.6%);
- FCE-GRPO: 257/500 (51.4%).

FCE improved over the base by 21.8 percentage points (paired 95% CI
[17.4, 26.2]) and over the matched control by 17.8 points (paired 95% CI
[13.2, 22.4]); neither 50,000-draw bootstrap produced a non-positive gain.
The final execution audit passed.

## Locked wrong-consensus safeguard replication

A safeguard requiring an answer to occur at least twice in each frozen panel was
selected after a SmolLM2 development analysis and locked before its Qwen result
was inspected. On the untouched Qwen bank it reduced wrong-given-reinforcement
from 21.29% to 12.61%, a 40.78% relative reduction, while retaining 85.33% of
the baseline selection gain and producing informative rewards on 56.60% of
candidate groups. It passed all three locked replication gates.

The same rule narrowly missed the development gain-retention gate on SmolLM2
(77.63% retained versus the required 80%), so this is a cross-model robustness
replication, not evidence that the safeguard is cost-free on every policy.
Unanimous, diversely worded wrong consensus remains information-theoretically
indistinguishable from correct consensus without an additional signal.

This establishes a viable verifier-free training mechanism in the registered
setting. Full policy-learning replication on a second model family, broader
answer domains, step-local credit, and comparisons with gold GRPO and other
verifier-free baselines are preregistered and queued, but remain open until
their completed artifacts pass the stated gates.
