#!/usr/bin/env bash
# Round-4 fan-out: 8 seeds of ONE condition, one seed per GPU, in parallel.
# usage: COND=cgfail60 bash r4_fanout.sh      (8x H100 worker)
COND=${COND:?set COND=cgfail60|pyfail60}
P=/mnt/bn/chobits-wx/jiayicheng/project/agent_catalyst
LOG=$P/runs/r4_fanout_$COND.log
exec > >(tee -a "$LOG") 2>&1
echo "===== r4 fanout [$COND] $(date -Is) host=$(hostname) ====="
nvidia-smi -L 2>&1 | head -8
pids=""
for g in 0 1 2 3 4 5 6 7; do
  s=$((g+1))
  bash $P/transfer-unit/scripts/round4_queue.sh "$g" "$COND" "$s" "$s" \
       > $P/runs/r4_${COND}_s${s}.out 2>&1 &
  pids="$pids $!:$s"
done
fail=0
for ps in $pids; do
  pid=${ps%%:*}; s=${ps##*:}
  if wait "$pid"; then echo "SEED_OK $COND s$s"; else echo "SEED_FAIL $COND s$s" >&2; fail=1; fi
done
echo "--- curve line counts ---"
for s in 1 2 3 4 5 6 7 8; do
  c=$P/runs/r4_${COND}_s${s}/curve.jsonl
  echo "  s$s: $( [ -s "$c" ] && wc -l < "$c" || echo 0 ) points"
done
echo "===== fanout [$COND] done fail=$fail $(date -Is) ====="
exit $fail
