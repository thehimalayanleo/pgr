#!/usr/bin/env bash
set -euo pipefail

# Long, fail-closed 5090 queue for the preregistered GSM8K replications.
# Launch only after an explicit cross-task GPU handoff. This script first
# requires a full minute of zero CUDA processes, then keeps a reservation alive
# across every arm so an unrelated admission wrapper cannot race between jobs.

ROOT=/home/ajinkya/pgr
PY=/home/ajinkya/miniconda3/envs/pgr/bin/python
EXP="$ROOT/experiments/frozen_cross_consensus"
CHECKPOINTS="$ROOT/checkpoints/fce_replication_20260730"
LOGS="$ROOT/logs/fce_replication_20260730"
SMOL=HuggingFaceTB/SmolLM2-1.7B-Instruct
QWEN=Qwen/Qwen2.5-1.5B-Instruct
SMOL_BANK="$ROOT/checkpoints/fcc_smollm_train_n1000_p12.json"
QWEN_BANK="$ROOT/checkpoints/fcc_qwen_train_n1000_p12.json"
ACTIVE_PID=""
RESERVATION_PID=""
TRAIN_FINAL=""
LOCK_FILE=/tmp/pgr_fce_replication_20260730.lock
HANDOFF_TOKEN=/tmp/PGR_FCE_HANDOFF_20260730

mkdir -p "$CHECKPOINTS" "$LOGS"
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec 9>"$LOCK_FILE"
flock -n 9 || {
  echo "$(date -Is) another FCE replication supervisor owns $LOCK_FILE"
  exit 1
}

cleanup() {
  if [[ -n "$ACTIVE_PID" ]]; then
    kill -TERM "$ACTIVE_PID" 2>/dev/null || true
    wait "$ACTIVE_PID" 2>/dev/null || true
  fi
  if [[ -n "$RESERVATION_PID" ]]; then
    kill -TERM "$RESERVATION_PID" 2>/dev/null || true
    wait "$RESERVATION_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 143' INT TERM

cuda_pids() {
  nvidia-smi --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'
}

wait_for_exclusive_idle() {
  local stable=0
  while (( stable < 12 )); do
    local free_vram available_ram
    free_vram="$(
      nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
    )"
    available_ram="$(free -g | awk '/^Mem:/{print $7}')"
    if [[ -z "$(cuda_pids)" && "$free_vram" -ge 31000 \
      && "$available_ram" -ge 18 ]]; then
      stable=$((stable + 1))
      echo "$(date -Is) exclusive idle grace $stable/12"
    else
      stable=0
    fi
    sleep 5
  done
}

start_reservation() {
  local mib="$1"
  local deadline active
  PGR_FCE_REPLICATION_RESERVATION_BYTES="$((mib * 1024 * 1024))" \
    "$PY" -u -c \
    'import os,time,torch; n=int(os.environ["PGR_FCE_REPLICATION_RESERVATION_BYTES"]); x=torch.empty(n,dtype=torch.uint8,device="cuda"); print({"event":"fce_replication_reservation_ready","pid":os.getpid(),"bytes":n},flush=True); time.sleep(172800)' \
    pgr_fce_replication_reservation >> "$LOGS/reservation.log" 2>&1 &
  RESERVATION_PID=$!
  deadline=$((SECONDS + 90))
  while (( SECONDS < deadline )); do
    kill -0 "$RESERVATION_PID" 2>/dev/null || return 1
    active="$(cuda_pids)"
    if printf '%s\n' "$active" | grep -q -x "$RESERVATION_PID"; then
      echo "$(date -Is) reservation ready ${mib} MiB pid=$RESERVATION_PID"
      return 0
    fi
    sleep 1
  done
  return 1
}

foreign_cuda_present() {
  local pid
  while IFS= read -r pid; do
    pid="$(echo "$pid" | tr -d ' ')"
    [[ -z "$pid" ]] && continue
    [[ "$pid" == "$RESERVATION_PID" ]] && continue
    [[ -n "$ACTIVE_PID" && "$pid" == "$ACTIVE_PID" ]] && continue
    return 0
  done < <(cuda_pids)
  return 1
}

reservation_quiescent() {
  local stable=0
  while (( stable < 12 )); do
    if ! kill -0 "$RESERVATION_PID" 2>/dev/null \
      || foreign_cuda_present; then
      return 1
    fi
    stable=$((stable + 1))
    sleep 5
  done
  echo "$(date -Is) reservation quiescent 60 seconds"
}

archive_interrupted_dir() {
  local path="$1"
  if [[ -d "$path" ]]; then
    local archived="${path}_INTERRUPTED_$(date +%Y%m%dT%H%M%S)"
    mv "$path" "$archived"
    printf '%s\n' \
      "Interrupted run; heldout evaluation forbidden." \
      > "${archived}/CONFIRMATORY_STATUS.txt"
  fi
}

run_guarded() {
  local log="$1"
  local output_dir="$2"
  shift 2
  if foreign_cuda_present; then
    echo "$(date -Is) refusing launch: foreign CUDA process present" | tee -a "$log"
    return 75
  fi
  "$@" >> "$log" 2>&1 &
  ACTIVE_PID=$!
  local contaminated=0
  while kill -0 "$ACTIVE_PID" 2>/dev/null; do
    if ! kill -0 "$RESERVATION_PID" 2>/dev/null \
      || foreign_cuda_present; then
      contaminated=1
      echo "$(date -Is) contamination detected; terminating only PGR PID $ACTIVE_PID" \
        | tee -a "$log"
      kill -TERM "$ACTIVE_PID" 2>/dev/null || true
      sleep 10
      kill -KILL "$ACTIVE_PID" 2>/dev/null || true
      break
    fi
    sleep 5
  done
  local rc=0
  wait "$ACTIVE_PID" 2>/dev/null || rc=$?
  ACTIVE_PID=""
  if (( contaminated )); then
    archive_interrupted_dir "$output_dir"
    return 75
  fi
  if (( rc != 0 )); then
    archive_interrupted_dir "$output_dir"
    return "$rc"
  fi
}

model_complete() {
  local path="$1"
  [[ -s "$path/config.json" && -s "$path/tokenizer_config.json" ]] || return 1
  compgen -G "$path/*.safetensors" >/dev/null \
    || compgen -G "$path/pytorch_model*.bin" >/dev/null
}

eval_complete() {
  local path="$1"
  local checkpoint="$2"
  "$PY" - "$path" "$checkpoint" <<'PY'
import json,sys
from pathlib import Path
path=Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
try:
    result=json.loads(path.read_text())
except Exception:
    raise SystemExit(1)
ok=(
    result.get("checkpoint")==sys.argv[2]
    and result.get("dataset")=="openai/gsm8k"
    and result.get("split")=="test"
    and result.get("offset")==500
    and result.get("n")==500
    and result.get("decoding")=="greedy"
    and len(result.get("records",[]))==500
    and len({r.get("prompt_hash") for r in result["records"]})==500
)
raise SystemExit(0 if ok else 1)
PY
}

bank_complete() {
  local path="$1"
  local model="$2"
  "$PY" - "$path" "$model" <<'PY'
import json,sys
from pathlib import Path
path=Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
try:
    bank=json.loads(path.read_text())
except Exception:
    raise SystemExit(1)
ok=(
    bank.get("model")==sys.argv[2]
    and bank.get("dataset")=="openai/gsm8k"
    and bank.get("split")=="train"
    and bank.get("gold_stored") is False
    and bank.get("candidate_panel_included") is False
    and bank.get("frozen_evidence_attached") is True
    and not bank.get("partial",True)
    and bank.get("panel_size")==12
    and len(bank.get("items",[]))==1000
    and sum(bool(x.get("frozen_evidence",{}).get("scores")) for x in bank["items"])>=200
)
raise SystemExit(0 if ok else 1)
PY
}

selection_bank_complete() {
  local path="$1"
  local dataset="$2"
  local split="$3"
  "$PY" - "$path" "$QWEN" "$dataset" "$split" <<'PY'
import json,sys
from pathlib import Path
path=Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
try:
    bank=json.loads(path.read_text())
except Exception:
    raise SystemExit(1)
ok=(
    bank.get("model")==sys.argv[2]
    and bank.get("dataset")==sys.argv[3]
    and bank.get("split")==sys.argv[4]
    and bank.get("gold_stored") is False
    and bank.get("candidate_panel_included") is True
    and bank.get("frozen_evidence_attached") is True
    and not bank.get("partial",True)
    and bank.get("panel_size")==12
    and bank.get("candidate_panel_size")==4
    and len(bank.get("items",[]))==500
)
raise SystemExit(0 if ok else 1)
PY
}

build_and_evaluate_selection() {
  local dataset="$1"
  local split="$2"
  local tag="$3"
  local bank="$CHECKPOINTS/${tag}_selection_bank_n500_p12.json"
  local result="$CHECKPOINTS/${tag}_selection_result.json"
  if ! selection_bank_complete "$bank" "$dataset" "$split"; then
    run_guarded "$LOGS/${tag}_selection_bank.log" "" \
      env PYTHONPATH="$EXP" "$PY" -u "$EXP/build_fcc_bank.py" \
      --model "$QWEN" --dataset "$dataset" --split "$split" \
      --n 500 --panel-size 12 --candidate-size 4 \
      --max-tokens 384 --batch-size 12 --save-every 5 \
      --output "$bank"
    selection_bank_complete "$bank" "$dataset" "$split"
  fi
  if [[ ! -s "$result" ]]; then
    env PYTHONPATH="$EXP" "$PY" "$EXP/evaluate_fce_bank.py" \
      --bank "$bank" --output "$result" \
      > "$LOGS/${tag}_selection_analysis.log" 2>&1
  fi
}

run_eval() {
  local checkpoint="$1"
  local output="$2"
  local tag="$3"
  if eval_complete "$output" "$checkpoint"; then
    echo "$(date -Is) skip complete eval $tag"
    return
  fi
  run_guarded "$LOGS/${tag}_eval.log" "" \
    "$PY" -u "$EXP/evaluate_fcc_online.py" "$checkpoint" \
    --dataset openai/gsm8k --split test \
    --offset 500 --n 500 --max-tokens 384 --batch-size 8 \
    --output "$output"
  eval_complete "$output" "$checkpoint"
}

run_training() {
  local model="$1"
  local source="$2"
  local seed="$3"
  local suffix="$4"
  local bank="$5"
  local label=trajectory
  [[ "$source" == gold ]] || label="trajectory_${source}"
  local out_dir="$ROOT/checkpoints/${suffix}_${label}_seed${seed}_steps1000_k4"
  local final="${out_dir}_final"
  if model_complete "$final"; then
    echo "$(date -Is) skip complete training $suffix $source seed=$seed"
    TRAIN_FINAL="$final"
    return
  fi
  if [[ -e "$out_dir" || -e "$final" ]]; then
    echo "$(date -Is) refusing preexisting incomplete output $out_dir"
    return 1
  fi
  local command=(
    "$PY" -u "$EXP/local_fcc_train.py"
    --max-steps 1000 --seed "$seed" --k 4
    --dataset openai/gsm8k --model "$model"
    --max-completion 384 --lr 2e-5
    --output-suffix "$suffix" --save-steps 50 --save-total-limit 1
    --alpha 0 --step-advantage-mode group_mean
    --terminal-spread constant --reward-source "$source"
  )
  if [[ "$source" == fce || "$source" == fce_permuted ]]; then
    command+=(--fcc-bank "$bank")
  fi
  run_guarded "$LOGS/${suffix}_${source}_seed${seed}.log" "$out_dir" \
    "${command[@]}"
  model_complete "$final"
  TRAIN_FINAL="$final"
}

echo "$(date -Is) FCE replication supervisor waiting for explicit handoff token"
while [[ ! -s "$HANDOFF_TOKEN" ]]; do
  sleep 5
done
echo "$(date -Is) handoff token observed; waiting for exclusive idle"
wait_for_exclusive_idle

# Generate the harder-domain selection banks and the missing Qwen train bank
# under a 13.5 GiB admission reservation. These banks are resumable and contain
# no gold; their evaluators load labels only after score construction.
start_reservation 13500
reservation_quiescent
build_and_evaluate_selection lighteval/MATH-Hard test math_hard_qwen
build_and_evaluate_selection cais/mmlu test mmlu_qwen
if ! bank_complete "$QWEN_BANK" "$QWEN"; then
  run_guarded "$LOGS/qwen_train_bank.log" "" \
    env PYTHONPATH="$EXP" "$PY" -u "$EXP/build_fcc_bank.py" \
    --model "$QWEN" --dataset openai/gsm8k --split train \
    --n 1000 --panel-size 12 --max-tokens 384 --batch-size 12 \
    --save-every 5 --omit-candidates --output "$QWEN_BANK"
  bank_complete "$QWEN_BANK" "$QWEN"
fi

kill -TERM "$RESERVATION_PID"
wait "$RESERVATION_PID" 2>/dev/null || true
RESERVATION_PID=""
wait_for_exclusive_idle
start_reservation 6144
reservation_quiescent

# Base evaluations are shared across matched arms.
SMOL_BASE_EVAL="$CHECKPOINTS/smollm_base_heldout500.json"
QWEN_BASE_EVAL="$CHECKPOINTS/qwen_base_heldout500.json"
run_eval "$SMOL" "$SMOL_BASE_EVAL" smollm_base
run_eval "$QWEN" "$QWEN_BASE_EVAL" qwen_base

# Four new independent SmolLM training seeds, each with its matched permutation.
for seed in 43 44 45 46; do
  run_training "$SMOL" fce "$seed" "repl_smollm_s${seed}" "$SMOL_BANK"
  fce_final="$TRAIN_FINAL"
  run_training "$SMOL" fce_permuted "$seed" \
    "repl_smollm_s${seed}" "$SMOL_BANK"
  control_final="$TRAIN_FINAL"
  fce_eval="$CHECKPOINTS/smollm_seed${seed}_fce_heldout500.json"
  control_eval="$CHECKPOINTS/smollm_seed${seed}_permuted_heldout500.json"
  run_eval "$fce_final" "$fce_eval" "smollm_seed${seed}_fce"
  run_eval "$control_final" "$control_eval" "smollm_seed${seed}_permuted"
  "$PY" "$EXP/analyze_fce_control.py" \
    --fce "$fce_eval" --control "$control_eval" \
    --output "$CHECKPOINTS/smollm_seed${seed}_paired.json"
done

# Full-policy cross-model replication.
run_training "$QWEN" fce 42 repl_qwen_s42 "$QWEN_BANK"
qwen_fce_final="$TRAIN_FINAL"
run_training "$QWEN" fce_permuted 42 repl_qwen_s42 "$QWEN_BANK"
qwen_control_final="$TRAIN_FINAL"
QWEN_FCE_EVAL="$CHECKPOINTS/qwen_seed42_fce_heldout500.json"
QWEN_CONTROL_EVAL="$CHECKPOINTS/qwen_seed42_permuted_heldout500.json"
run_eval "$qwen_fce_final" "$QWEN_FCE_EVAL" qwen_seed42_fce
run_eval "$qwen_control_final" "$QWEN_CONTROL_EVAL" qwen_seed42_permuted
"$PY" "$EXP/analyze_fcc_online.py" \
  --base "$QWEN_BASE_EVAL" --trained "$QWEN_FCE_EVAL" \
  --output "$CHECKPOINTS/qwen_fce_vs_base.json"
"$PY" "$EXP/analyze_fce_control.py" \
  --fce "$QWEN_FCE_EVAL" --control "$QWEN_CONTROL_EVAL" \
  --output "$CHECKPOINTS/qwen_fce_vs_permuted.json"

# Clean matched gold and on-policy majority comparators. Majority is structurally
# gold-blind in the synced preprocessing/trainer code.
run_training "$SMOL" gold 42 repl_smollm_gold_s42 ""
gold_final="$TRAIN_FINAL"
run_training "$SMOL" majority 42 repl_smollm_majority_s42 ""
majority_final="$TRAIN_FINAL"
run_eval "$gold_final" "$CHECKPOINTS/smollm_gold_seed42_heldout500.json" \
  smollm_gold_seed42
run_eval "$majority_final" "$CHECKPOINTS/smollm_majority_seed42_heldout500.json" \
  smollm_majority_seed42
"$PY" "$EXP/analyze_paired_policy_evals.py" \
  --left "$ROOT/checkpoints/fce_online_heldout500_greedy.json" \
  --left-name fce_seed42 \
  --right "$CHECKPOINTS/smollm_majority_seed42_heldout500.json" \
  --right-name on_policy_majority_seed42 \
  --output "$CHECKPOINTS/smollm_fce_vs_majority_seed42.json"
"$PY" "$EXP/analyze_paired_policy_evals.py" \
  --left "$ROOT/checkpoints/fce_online_heldout500_greedy.json" \
  --left-name fce_seed42 \
  --right "$CHECKPOINTS/smollm_gold_seed42_heldout500.json" \
  --right-name gold_grpo_seed42 \
  --output "$CHECKPOINTS/smollm_fce_vs_gold_seed42.json"

"$PY" "$EXP/analyze_fce_multiseed.py" \
  --seed 43 --fce "$CHECKPOINTS/smollm_seed43_fce_heldout500.json" \
    --control "$CHECKPOINTS/smollm_seed43_permuted_heldout500.json" \
  --seed 44 --fce "$CHECKPOINTS/smollm_seed44_fce_heldout500.json" \
    --control "$CHECKPOINTS/smollm_seed44_permuted_heldout500.json" \
  --seed 45 --fce "$CHECKPOINTS/smollm_seed45_fce_heldout500.json" \
    --control "$CHECKPOINTS/smollm_seed45_permuted_heldout500.json" \
  --seed 46 --fce "$CHECKPOINTS/smollm_seed46_fce_heldout500.json" \
    --control "$CHECKPOINTS/smollm_seed46_permuted_heldout500.json" \
  --output "$CHECKPOINTS/smollm_multiseed_hierarchical.json"

echo "$(date -Is) FCE CORE REPLICATION QUEUE COMPLETE"
