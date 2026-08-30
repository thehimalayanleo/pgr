#!/usr/bin/env bash
set -uo pipefail

# Clean re-run of the GRPO reward-quality instrument after fixing the loss path.
#
# Previous non-gold arms changed `per_rollout_terminals` only after gold-derived
# step rewards had already been built. Their diagnostics differed, but their loss
# still consumed gold rewards. This queue uses the repaired trainer and unique
# checkpoint names; no prior random/majority checkpoint is reused.

ROOT=/home/ajinkya/pgr
PY=/home/ajinkya/miniconda3/envs/pgr/bin/python
MODEL=HuggingFaceTB/SmolLM2-1.7B-Instruct
EXPECTED_TRAINER_SHA=380cb2a60a5b3ed226998765567fe15038b653211e7d27893e7cd4d64e87de5d

cd "$ROOT"
mkdir -p logs checkpoints
exec 9>"$ROOT/fixed_source_gate.lock"
echo "$(date -Is) waiting for frozen-evidence/source gate lock"
flock 9
echo "$(date -Is) source gate owns lock"

actual_sha="$(sha256sum step_pgr_trainer.py | awk '{print $1}')"
if [[ "$actual_sha" != "$EXPECTED_TRAINER_SHA" ]]; then
  echo "trainer hash mismatch: expected $EXPECTED_TRAINER_SHA got $actual_sha"
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
ACTIVE_TRAIN_PID=""

foreign_big_busy() {
  if pgrep -f "[b]enchmark.runner" >/dev/null; then
    return 0
  fi
  local pid mem
  while IFS=, read -r pid mem; do
    pid="$(echo "$pid" | tr -d " ")"
    mem="$(echo "$mem" | tr -d " ")"
    [[ -z "$pid" ]] && continue
    if [[ -n "$ACTIVE_TRAIN_PID" && "$pid" == "$ACTIVE_TRAIN_PID" ]]; then
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
  local stable=0
  while (( stable < 2 )); do
    if foreign_big_busy; then
      stable=0
    else
      free_ram="$(free -g | awk '/^Mem:/{print $7}')"
      free_vram="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)"
      if [[ "$free_ram" -ge 18 && "$free_vram" -ge 24000 ]]; then
        stable=$((stable + 1))
        echo "$(date -Is) capacity stable $stable/2: RAM=${free_ram}G VRAM=${free_vram}MiB"
      else
        stable=0
      fi
    fi
    sleep 60
  done
}

latest_valid_checkpoint() {
  local out_dir="$1"
  local candidate
  while IFS= read -r candidate; do
    [[ -s "$candidate/trainer_state.json" ]] || continue
    if compgen -G "$candidate/*.safetensors" >/dev/null \
      || compgen -G "$candidate/pytorch_model*.bin" >/dev/null; then
      echo "$candidate"
      return 0
    fi
  done < <(
    find "$out_dir" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
      | sort -Vr
  )
  return 1
}

run_arm() {
  local tag="$1"
  local source="$2"
  local prefix="fixsrc_gn_${tag}"
  local label="trajectory"
  if [[ "$source" != "gold" ]]; then
    label="${label}_${source}"
  fi
  local final=""
  local tries=0

  while [[ -z "$(find checkpoints -maxdepth 1 -type d \
      -name "${prefix}_${label}_seed42_steps1000_k4_final" -print -quit)" ]] \
      && (( tries < 20 )); do
    tries=$((tries + 1))
    wait_clear

    local out_dir="checkpoints/${prefix}_${label}_seed42_steps1000_k4"
    local resume_args=()
    if [[ -d "$out_dir" ]]; then
      checkpoint_path="$(latest_valid_checkpoint "$out_dir" || true)"
      if [[ -n "$checkpoint_path" ]]; then
        resume_args=(--resume-from "$checkpoint_path")
      fi
    fi

    echo "$(date -Is) [$tag] attempt $tries source=$source resume=${resume_args[*]-}" \
      >> "logs/${prefix}_train.log"
    "$PY" -u local_5090_train.py \
      --max-steps 1000 \
      --seed 42 \
      --k 4 \
      --dataset openai/gsm8k \
      --model "$MODEL" \
      --max-completion 384 \
      --lr 2e-5 \
      --output-suffix "$prefix" \
      --save-steps 50 \
      --save-total-limit 1 \
      --alpha 0 \
      --step-advantage-mode group_mean \
      --terminal-spread constant \
      --reward-source "$source" \
      "${resume_args[@]}" \
      >> "logs/${prefix}_train.log" 2>&1 &
    train_pid=$!
    ACTIVE_TRAIN_PID="$train_pid"

    while kill -0 "$train_pid" 2>/dev/null; do
      if foreign_big_busy; then
        echo "$(date -Is) [$tag] yielding to foreign GPU work" \
          >> "logs/${prefix}_train.log"
        kill -TERM "$train_pid" 2>/dev/null || true
        sleep 20
        kill -KILL "$train_pid" 2>/dev/null || true
        break
      fi
      sleep 30
    done
    wait "$train_pid" 2>/dev/null
    rc=$?
    ACTIVE_TRAIN_PID=""
    echo "$(date -Is) [$tag] attempt $tries exit=$rc" \
      >> "logs/${prefix}_train.log"
    sleep 30
  done

  final="$(find checkpoints -maxdepth 1 -type d \
    -name "${prefix}_${label}_seed42_steps1000_k4_final" -print -quit)"
  if [[ -z "$final" ]]; then
    echo "$(date -Is) [$tag] failed after $tries attempts"
    return 1
  fi

  result="checkpoints/${prefix}_n500.json"
  if [[ ! -f "$result" ]]; then
    wait_clear
    "$PY" -u local_5090_eval.py "$final" \
      --n 500 \
      --dataset gsm8k \
      --out "$result" \
      > "logs/${prefix}_eval.log" 2>&1
  fi
}

echo "$(date -Is) FIXED REWARD-SOURCE GATE START"
run_arm gold gold || exit 1
run_arm random random || exit 1

gold_acc="$("$PY" -c 'import json; print(json.load(open("checkpoints/fixsrc_gn_gold_n500.json"))["accuracy"])')"
random_acc="$("$PY" -c 'import json; print(json.load(open("checkpoints/fixsrc_gn_random_n500.json"))["accuracy"])')"
echo "$(date -Is) instrument check: gold=$gold_acc random=$random_acc"

if "$PY" -c "import sys; sys.exit(0 if float('$random_acc') < float('$gold_acc') - 0.03 else 1)"; then
  echo "$(date -Is) instrument alive; running verifier-free majority"
  run_arm majority majority || exit 1
else
  echo "$(date -Is) instrument still cannot distinguish gold from random; majority gated off"
fi

"$PY" experiments/reward_source_fix/analyze_fixed_source_gate.py \
  > logs/fixed_source_gate_analysis.log 2>&1 || true
echo "$(date -Is) FIXED REWARD-SOURCE GATE DONE"
