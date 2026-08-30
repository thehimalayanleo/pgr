#!/usr/bin/env bash
set -uo pipefail

# Power-qualified, gold-separated evaluation of FCE (primary) and FCC (baseline)
# before any frozen-evidence training.

ROOT=/home/ajinkya/pgr
PY=/home/ajinkya/miniconda3/envs/pgr/bin/python
EXP="$ROOT/experiments/frozen_cross_consensus"
ACTIVE_PID=""

cd "$ROOT"
mkdir -p logs checkpoints
exec 8>"$ROOT/fixed_source_gate.lock"
echo "$(date -Is) waiting for frozen-evidence/source gate lock"
flock 8
echo "$(date -Is) FCE selection gate owns lock"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

foreign_big_busy() {
  if pgrep -f "[b]enchmark.runner" >/dev/null; then
    return 0
  fi
  local pid mem
  while IFS=, read -r pid mem; do
    pid="$(echo "$pid" | tr -d " ")"
    mem="$(echo "$mem" | tr -d " ")"
    [[ -z "$pid" ]] && continue
    if [[ -n "$ACTIVE_PID" && "$pid" == "$ACTIVE_PID" ]]; then
      continue
    fi
    if [[ "${mem:-0}" -ge 4000 ]]; then
      return 0
    fi
  done < <(
    nvidia-smi --query-compute-apps=pid,used_memory \
      --format=csv,noheader,nounits 2>/dev/null
  )
  return 1
}

wait_clear() {
  while true; do
    if foreign_big_busy; then
      :
    else
      free_ram="$(free -g | awk '/^Mem:/{print $7}')"
      free_vram="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)"
      if [[ "$free_ram" -ge 12 && "$free_vram" -ge 12000 ]]; then
        echo "$(date -Is) FCE capacity clear: RAM=${free_ram}G VRAM=${free_vram}MiB"
        return 0
      fi
    fi
    sleep 30
  done
}

bank_complete() {
  local path="$1"
  "$PY" - "$path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
data = json.loads(path.read_text())
raise SystemExit(
    0 if not data.get("partial", True)
    and data.get("frozen_evidence_attached") is True
    and len(data.get("items", [])) == 500
    else 1
)
PY
}

generate_bank() {
  local tag="$1"
  local model="$2"
  local bank="checkpoints/fcc_${tag}_test_n500_p12.json"
  local log="logs/fcc_${tag}_bank.log"
  local tries=0
  # Voluntary yields are normal on the shared 5090 and must not exhaust the
  # retry budget before an otherwise healthy atomic bank reaches 500 items.
  while ! bank_complete "$bank" && (( tries < 1000 )); do
    tries=$((tries + 1))
    wait_clear
    echo "$(date -Is) [$tag] generation attempt $tries" >> "$log"
    PYTHONPATH="$EXP" "$PY" -u "$EXP/build_fcc_bank.py" \
      --model "$model" \
      --split test \
      --n 500 \
      --panel-size 12 \
      --candidate-size 4 \
      --max-tokens 384 \
      --batch-size 8 \
      --output "$ROOT/$bank" \
      >> "$log" 2>&1 &
    ACTIVE_PID=$!
    while kill -0 "$ACTIVE_PID" 2>/dev/null; do
      if foreign_big_busy; then
        echo "$(date -Is) [$tag] yielding generation to foreign GPU work" >> "$log"
        kill -TERM "$ACTIVE_PID" 2>/dev/null || true
        sleep 20
        kill -KILL "$ACTIVE_PID" 2>/dev/null || true
        break
      fi
      sleep 30
    done
    wait "$ACTIVE_PID" 2>/dev/null
    rc=$?
    ACTIVE_PID=""
    echo "$(date -Is) [$tag] generation exit=$rc" >> "$log"
  done
  bank_complete "$bank"
}

evaluate_bank() {
  local tag="$1"
  local bank="$ROOT/checkpoints/fcc_${tag}_test_n500_p12.json"
  local result="$ROOT/checkpoints/fcc_${tag}_selection_result.json"
  PYTHONPATH="$EXP" "$PY" -u "$EXP/evaluate_fcc_bank.py" \
    --bank "$bank" \
    --output "$result" \
    > "$ROOT/logs/fcc_${tag}_selection_eval.log" 2>&1
}

evaluate_fce_bank() {
  local tag="$1"
  local bank="$ROOT/checkpoints/fcc_${tag}_test_n500_p12.json"
  local result="$ROOT/checkpoints/fce_${tag}_selection_result.json"
  PYTHONPATH="$EXP" "$PY" -u "$EXP/evaluate_fce_bank.py" \
    --bank "$bank" \
    --output "$result" \
    > "$ROOT/logs/fce_${tag}_selection_eval.log" 2>&1
}

passes_gate() {
  local result="$1"
  "$PY" - "$result" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1]))
gate = result["viability_gate"]
raise SystemExit(
    0 if gate["captures_at_least_70pct_oracle_gain"]
    and gate["paired_bootstrap_p_lt_0_05"]
    and gate["informative_group_rate_at_least_0_30"]
    else 1
)
PY
}

echo "$(date -Is) FCE/FCC SELECTION GATE START"
generate_bank smollm HuggingFaceTB/SmolLM2-1.7B-Instruct || exit 1
evaluate_bank smollm || exit 1
evaluate_fce_bank smollm || exit 1

if passes_gate "$ROOT/checkpoints/fce_smollm_selection_result.json"; then
  echo "$(date -Is) SmolLM FCE passed; testing Qwen family"
  generate_bank qwen Qwen/Qwen2.5-1.5B-Instruct || exit 1
  evaluate_bank qwen || exit 1
  evaluate_fce_bank qwen || exit 1
else
  echo "$(date -Is) SmolLM FCE failed pre-registered gate; Qwen and training gated off"
  exit 0
fi

if passes_gate "$ROOT/checkpoints/fce_qwen_selection_result.json"; then
  echo "$(date -Is) FCE PASSED BOTH MODEL FAMILIES; starting online GRPO arm"
  bash "$EXP/run_fcc_online_gate.sh" || exit 1
else
  echo "$(date -Is) Qwen FCE failed; online GRPO arm remains gated off"
fi
echo "$(date -Is) FCE/FCC SELECTION GATE DONE"
