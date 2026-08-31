#!/usr/bin/env bash
# Round-3 lane queue: each GPU runs one condition across a seed range, serially.
# usage: round3_queue.sh <gpu> <cond> <seed_start> <seed_end>
GPU=$1; COND=$2; S0=$3; S1=$4
P=/mnt/bn/chobits-wx/jiayicheng/project/agent_catalyst
S=$P/transfer-unit/scripts/frozen_pipe_v2.sh
D=$P/transfer-unit/data
CG=$D/donor_codegen.jsonl; MH=$D/donor_mathhard.jsonl; PF=$D/donor_pyfix.jsonl
EV="codegen:48"
for s in $(seq "$S0" "$S1"); do
  case "$COND" in
    none)  bash $S r3_none_s$s  $CG -   none   0     "$s" "$GPU" "$EV" 30000 2 ;;
    b90k)  bash $S r3_b90k_s$s  $CG -   none   0     "$s" "$GPU" "$EV" 90000 2 ;;
    mh15)  bash $S r3_mh15_s$s  $CG $MH clean  15000 "$s" "$GPU" "$EV" 30000 3 ;;
    pyf60) bash $S r3_pyf60_s$s $CG $PF errrec 60000 "$s" "$GPU" "$EV" 30000 6 ;;
  esac
done
echo "QUEUE_DONE $COND gpu$GPU seeds $S0-$S1"
