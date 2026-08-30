#!/usr/bin/env bash
set -euo pipefail

# Conditional continuation of the core replication queue. It waits for the core
# supervisor to finish, reads each sealed domain-selection result, and performs
# policy training only for domains that passed every preregistered selection gate.

ROOT=/home/ajinkya/pgr
PY=/home/ajinkya/miniconda3/envs/pgr/bin/python
EXP="$ROOT/experiments/frozen_cross_consensus"
CORE="$ROOT/checkpoints/fce_replication_20260730"
OUT="$ROOT/checkpoints/fce_domain_policy_20260730"
LOGS="$ROOT/logs/fce_domain_policy_20260730"
QWEN=Qwen/Qwen2.5-1.5B-Instruct
ACTIVE_PID=""
RESERVATION_PID=""
TRAIN_FINAL=""

mkdir -p "$OUT" "$LOGS"
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec 9>/tmp/pgr_fce_domain_policy_20260730.lock
flock -n 9 || exit 1

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

wait_core_complete() {
  while pgrep -f \
    '^[b]ash experiments/frozen_cross_consensus/run_fce_replication_queue.sh$' \
    >/dev/null; do
    sleep 30
  done
  grep -q 'FCE CORE REPLICATION QUEUE COMPLETE' \
    "$ROOT/logs/fce_replication_20260730_supervisor.log"
}

selection_passed() {
  local result="$1"
  "$PY" - "$result" <<'PY'
import json,sys
from pathlib import Path
path=Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
try:
    result=json.loads(path.read_text())
except Exception:
    raise SystemExit(1)
gate=result.get("viability_gate",{})
ok=(
    gate.get("captures_at_least_70pct_oracle_gain") is True
    and gate.get("paired_bootstrap_p_lt_0_05") is True
    and gate.get("informative_group_rate_at_least_0_30") is True
)
raise SystemExit(0 if ok else 1)
PY
}

wait_idle() {
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
    else
      stable=0
    fi
    sleep 5
  done
}

start_reservation() {
  local mib="$1"
  PGR_FCE_DOMAIN_RESERVATION_BYTES="$((mib * 1024 * 1024))" \
    "$PY" -u -c \
    'import os,time,torch; n=int(os.environ["PGR_FCE_DOMAIN_RESERVATION_BYTES"]); x=torch.empty(n,dtype=torch.uint8,device="cuda"); print({"event":"fce_domain_reservation_ready","pid":os.getpid(),"bytes":n},flush=True); time.sleep(172800)' \
    pgr_fce_domain_reservation >> "$LOGS/reservation.log" 2>&1 &
  RESERVATION_PID=$!
  local deadline=$((SECONDS + 90))
  while (( SECONDS < deadline )); do
    kill -0 "$RESERVATION_PID" 2>/dev/null || return 1
    if cuda_pids | grep -q -x "$RESERVATION_PID"; then
      return 0
    fi
    sleep 1
  done
  return 1
}

foreign_cuda() {
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

quiescent() {
  local stable=0
  while (( stable < 12 )); do
    kill -0 "$RESERVATION_PID" 2>/dev/null || return 1
    foreign_cuda && return 1
    stable=$((stable + 1))
    sleep 5
  done
}

archive_interrupted() {
  local path="$1"
  if [[ -d "$path" ]]; then
    local archived="${path}_INTERRUPTED_$(date +%Y%m%dT%H%M%S)"
    mv "$path" "$archived"
    printf '%s\n' "Interrupted; heldout evaluation forbidden." \
      > "$archived/CONFIRMATORY_STATUS.txt"
  fi
}

run_guarded() {
  local log="$1"
  local output_dir="$2"
  shift 2
  foreign_cuda && return 75
  "$@" >> "$log" 2>&1 &
  ACTIVE_PID=$!
  local contaminated=0
  while kill -0 "$ACTIVE_PID" 2>/dev/null; do
    if ! kill -0 "$RESERVATION_PID" 2>/dev/null || foreign_cuda; then
      contaminated=1
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
  if (( contaminated || rc != 0 )); then
    archive_interrupted "$output_dir"
    (( contaminated )) && return 75
    return "$rc"
  fi
}

bank_complete() {
  local path="$1"
  local dataset="$2"
  local split="$3"
  "$PY" - "$path" "$dataset" "$split" <<'PY'
import json,sys
from pathlib import Path
path=Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
try:
    bank=json.loads(path.read_text())
except Exception:
    raise SystemExit(1)
accepted=sum(bool(x.get("frozen_evidence",{}).get("scores")) for x in bank.get("items",[]))
ok=(
    bank.get("model")=="Qwen/Qwen2.5-1.5B-Instruct"
    and bank.get("dataset")==sys.argv[2]
    and bank.get("split")==sys.argv[3]
    and bank.get("gold_stored") is False
    and bank.get("candidate_panel_included") is False
    and bank.get("frozen_evidence_attached") is True
    and not bank.get("partial",True)
    and bank.get("panel_size")==12
    and len(bank.get("items",[]))==1000
    and accepted>=200
)
print({"accepted":accepted,"required":200},flush=True)
raise SystemExit(0 if ok else 1)
PY
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
  local dataset="$3"
  "$PY" - "$path" "$checkpoint" "$dataset" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file():
    raise SystemExit(1)
try:
    x=json.loads(p.read_text())
except Exception:
    raise SystemExit(1)
ok=(
    x.get("checkpoint")==sys.argv[2]
    and x.get("dataset")==sys.argv[3]
    and x.get("offset")==500
    and x.get("n")==500
    and x.get("decoding")=="greedy"
    and len(x.get("records",[]))==500
    and len({r.get("prompt_hash") for r in x["records"]})==500
)
raise SystemExit(0 if ok else 1)
PY
}

run_eval() {
  local checkpoint="$1"
  local dataset="$2"
  local split="$3"
  local output="$4"
  local tag="$5"
  eval_complete "$output" "$checkpoint" "$dataset" && return
  run_guarded "$LOGS/${tag}_eval.log" "" \
    "$PY" -u "$EXP/evaluate_fcc_online.py" "$checkpoint" \
    --dataset "$dataset" --split "$split" \
    --offset 500 --n 500 --max-tokens 384 --batch-size 8 \
    --output "$output"
  eval_complete "$output" "$checkpoint" "$dataset"
}

run_training() {
  local dataset="$1"
  local source="$2"
  local suffix="$3"
  local bank="$4"
  local label="trajectory_${source}"
  local out_dir="$ROOT/checkpoints/${suffix}_${label}_seed42_steps1000_k4"
  local final="${out_dir}_final"
  if model_complete "$final"; then
    TRAIN_FINAL="$final"
    return
  fi
  [[ ! -e "$out_dir" && ! -e "$final" ]] || return 1
  run_guarded "$LOGS/${suffix}_${source}.log" "$out_dir" \
    "$PY" -u "$EXP/local_fcc_train.py" \
    --max-steps 1000 --seed 42 --k 4 --dataset "$dataset" \
    --model "$QWEN" --max-completion 384 --lr 2e-5 \
    --output-suffix "$suffix" --save-steps 50 --save-total-limit 1 \
    --alpha 0 --step-advantage-mode group_mean --terminal-spread constant \
    --reward-source "$source" --fcc-bank "$bank"
  model_complete "$final"
  TRAIN_FINAL="$final"
}

write_status() {
  local math_pass="$1"
  local mmlu_pass="$2"
  "$PY" - "$OUT/domain_queue_status.json" "$math_pass" "$mmlu_pass" <<'PY'
import json,sys
from pathlib import Path
payload={
    "math_hard_selection_passed":sys.argv[2]=="true",
    "mmlu_selection_passed":sys.argv[3]=="true",
    "training_policy":"run only after the corresponding preregistered selection gate",
    "queue_complete":True,
}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2)+"\n")
PY
}

echo "$(date -Is) domain continuation waiting for core queue" \
  > "$LOGS/supervisor.log"
wait_core_complete

MATH_RESULT="$CORE/math_hard_qwen_selection_result.json"
MMLU_RESULT="$CORE/mmlu_qwen_selection_result.json"
math_pass=false
mmlu_pass=false
selection_passed "$MATH_RESULT" && math_pass=true
selection_passed "$MMLU_RESULT" && mmlu_pass=true
math_hard_pass="$math_pass"
if [[ "$math_pass" == false && "$mmlu_pass" == false ]]; then
  write_status false false
  echo "$(date -Is) both domain selection gates failed; training correctly sealed" \
    >> "$LOGS/supervisor.log"
  exit 0
fi

wait_idle
start_reservation 13500
quiescent

declare -A DATASET SPLIT TRAIN_SPLIT
DATASET[math_hard]=lighteval/MATH-Hard
SPLIT[math_hard]=test
TRAIN_SPLIT[math_hard]=train
DATASET[mmlu]=cais/mmlu
SPLIT[mmlu]=test
TRAIN_SPLIT[mmlu]=auxiliary_train

for tag in math_hard mmlu; do
  pass_var="${tag}_pass"
  [[ "${!pass_var}" == true ]] || continue
  bank="$OUT/${tag}_qwen_train_n1000_p12.json"
  if ! bank_complete "$bank" "${DATASET[$tag]}" "${TRAIN_SPLIT[$tag]}"; then
    run_guarded "$LOGS/${tag}_train_bank.log" "" \
      env PYTHONPATH="$EXP" "$PY" -u "$EXP/build_fcc_bank.py" \
      --model "$QWEN" --dataset "${DATASET[$tag]}" \
      --split "${TRAIN_SPLIT[$tag]}" --n 1000 --panel-size 12 \
      --max-tokens 384 --batch-size 12 --save-every 5 \
      --omit-candidates --output "$bank"
    bank_complete "$bank" "${DATASET[$tag]}" "${TRAIN_SPLIT[$tag]}"
  fi
done

kill -TERM "$RESERVATION_PID"
wait "$RESERVATION_PID" 2>/dev/null || true
RESERVATION_PID=""
wait_idle
start_reservation 6144
quiescent

for tag in math_hard mmlu; do
  pass_var="${tag}_pass"
  [[ "${!pass_var}" == true ]] || continue
  dataset="${DATASET[$tag]}"
  split="${SPLIT[$tag]}"
  bank="$OUT/${tag}_qwen_train_n1000_p12.json"
  base_eval="$OUT/${tag}_qwen_base_heldout500.json"
  fce_eval="$OUT/${tag}_qwen_fce_heldout500.json"
  control_eval="$OUT/${tag}_qwen_permuted_heldout500.json"
  run_eval "$QWEN" "$dataset" "$split" "$base_eval" "${tag}_base"
  run_training "$dataset" fce "domain_${tag}_qwen" "$bank"
  fce_final="$TRAIN_FINAL"
  run_training "$dataset" fce_permuted "domain_${tag}_qwen" "$bank"
  control_final="$TRAIN_FINAL"
  run_eval "$fce_final" "$dataset" "$split" "$fce_eval" "${tag}_fce"
  run_eval "$control_final" "$dataset" "$split" "$control_eval" \
    "${tag}_permuted"
  "$PY" "$EXP/analyze_fcc_online.py" \
    --base "$base_eval" --trained "$fce_eval" \
    --output "$OUT/${tag}_fce_vs_base.json"
  "$PY" "$EXP/analyze_fce_control.py" \
    --fce "$fce_eval" --control "$control_eval" \
    --output "$OUT/${tag}_fce_vs_permuted.json"
done

write_status "$math_pass" "$mmlu_pass"
echo "$(date -Is) FCE DOMAIN POLICY QUEUE COMPLETE" >> "$LOGS/supervisor.log"
