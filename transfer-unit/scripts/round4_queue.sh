#!/usr/bin/env bash
# Round-4 lane queue (poison replication on a 2nd target task, B=codegen).
# Design + preregistered decision rule: reports/PREREG_ROUND4.md
# Baselines r3_none / r3_pyf60 are already in runs/ — do NOT re-run them.
# usage: round4_queue.sh <gpu> <cond> <seed_start> <seed_end>
GPU=$1; COND=$2; S0=$3; S1=$4
P=/mnt/bn/chobits-wx/jiayicheng/project/agent_catalyst
S=$P/transfer-unit/scripts/frozen_pipe_v2.sh
D=$P/transfer-unit/data
CG=$D/donor_codegen.jsonl; PF=$D/donor_pyfix.jsonl
EV="codegen:48"
# C/B = 2.0 for both conditions -> save_every 6, matching r3_pyf60 so the
# B-token checkpoint spacing stays equal across all four cells.
fail=0
for s in $(seq "$S0" "$S1"); do
  case "$COND" in
    cgfail60) bash $S r4_cgfail60_s$s $CG $CG fail 60000 "$s" "$GPU" "$EV" 30000 6 ;;
    pyfail60) bash $S r4_pyfail60_s$s $CG $PF fail 60000 "$s" "$GPU" "$EV" 30000 6 ;;
    *) echo "unknown cond: $COND (want cgfail60|pyfail60)" >&2; exit 2 ;;
  esac
  rc=$?
  # A run that trains but dies in eval leaves an empty curve.jsonl; that is a
  # failure, not a success, and must not be reported as one.
  curve=$P/runs/r4_${COND}_s$s/curve.jsonl
  if [ $rc -ne 0 ]; then
    echo "SEED_FAIL $COND s$s rc=$rc" >&2; fail=1
  elif [ ! -s "$curve" ]; then
    echo "SEED_FAIL $COND s$s: empty curve.jsonl (eval produced nothing)" >&2; fail=1
  fi
done
if [ $fail -ne 0 ]; then
  echo "QUEUE_FAILED $COND gpu$GPU seeds $S0-$S1" >&2; exit 1
fi
echo "QUEUE_DONE $COND gpu$GPU seeds $S0-$S1"
