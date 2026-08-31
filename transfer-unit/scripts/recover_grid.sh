#!/usr/bin/env bash
# Recovery manifest for the sftc2 curve grid after the fleet outage.
# usage: recover_grid.sh <ssh_port> <ssh_host>   (run per revived worker; edit LANES)
# Completed (curve.jsonl w/ final): none_s1 m60_s1 py230_s1 m230_s1
# Pending (relaunch): none_s2 none_s3 py60_s1 py60_s2 py60_s3 m60_s2 m60_s3
#   py230_s2 py230_s3 m230_s2 m230_s3 neu230_s1 neu230_s2 neu230_s3 bfail_s3 errrec230_s1
# Partial ckpts exist for some (e.g. py230_s2 up to step54) — eval_ckpts can
# salvage those without retraining; retraining anyway is simpler+complete.
P=/mnt/bn/chobits-wx/jiayicheng/project/agent_catalyst
S=$P/transfer-unit/scripts/sft_curve_pipeline.sh
D=$P/transfer-unit/data
PF=$D/donor_pyfix.jsonl; MA=$D/donor_math.jsonl; NE=$D/donor_neutral.jsonl; SQ=$D/donor_sql.jsonl
PORT=$1; HOST=$2
run() { # name cdata csel ct seed gpu se
  ssh -o StrictHostKeyChecking=no -p $PORT tiger@$HOST \
    "cd $P && setsid nohup bash $S $1 $2 $3 30000 $4 $5 $6 $7 > runs/$1.out 2>&1 < /dev/null &"
}
# 8-lane assignment (edit for 4-lane workers)
run sftc2_none_s2    -    none   0      2 0 2
run sftc2_none_s3    -    none   0      3 1 2
run sftc2_py60_s1    $PF  random 60000  1 2 2
run sftc2_py230_s2   $PF  random 230000 2 3 6
run sftc2_m230_s2    $MA  random 230000 2 4 6
run sftc2_m60_s2     $MA  random 60000  2 5 2
run sftc2_neu230_s1  $NE  random 230000 1 6 6
run sftc2_errrec230_s1 $PF errrec 230000 1 7 6
echo "wave A launched; wave B (py60_s2/s3, m60_s3, py230_s3, m230_s3, neu230_s2/s3, bfail_s3) queue after"
