#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/ajinkya/pgr
PY=/home/ajinkya/miniconda3/envs/pgr/bin/python
EXP="$ROOT/experiments/frozen_cross_consensus"
OUTPUT="$ROOT/checkpoints/fce_replication_20260730/final_program_audit.json"
LOG="$ROOT/logs/fce_replication_20260730/final_program_audit.log"

while pgrep -f \
  '^[b]ash experiments/frozen_cross_consensus/run_fce_domain_policy_queue.sh$' \
  >/dev/null; do
  sleep 30
done

"$PY" "$EXP/audit_fce_replication_program.py" \
  --root "$ROOT" --output "$OUTPUT" > "$LOG" 2>&1
rc=$?
printf '%s final_audit_exit_code=%s\n' "$(date -Is)" "$rc" >> "$LOG"
exit "$rc"
