#!/bin/bash
# Stages 2-5 for one window, after stage1_masks.sh. Usage: stages.sh SEG "NAME A" "NAME B"
K=$HOME/bjj/x-mmagold/kp; P=$K/.venv/bin/python; s=$1; d=$K/out/$s
cd $K
run(){ local t0=$(date +%s); /usr/bin/time -f "%M KB" -o $d/$1.mem "${@:2}" > $d/$1.log 2>&1 || { echo "FAIL $s $1: $(tail -3 $d/$1.log)"; exit 1; }; echo "$s $1 $(( $(date +%s)-t0 ))s $(cat $d/$1.mem) $(tail -1 $d/$1.log)"; }
run identity $P identity.py $d "$2" "$3"
run keypoints $P keypoints.py $d models/rtmw-x_fp16.plan
run limbs $P limbs.py $d $HOME/.cache/tapnextpp/tapnextpp_512.ckpt
run candidates $P candidates.py $d
