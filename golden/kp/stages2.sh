#!/bin/bash
# Round 2 stages for segments on gx10: detector (RTMO variants only; the SAM 3 and DART detectors have their own
# runners) -> pose on every candidate -> identity2. Kit names come from segs.json. Usage: stages2.sh DET SEG...
K=$HOME/bjj/x-mmagold/kp; P=$K/.venv/bin/python; cd $K; det=$1; shift
for s in "$@"; do
  mkdir -p out/$s
  case $det in
    rtmoz) test -f out/$s/masks_$det.mot.txt || $P detect_rtmo.py segs/$s.mp4 models/rtmo-l_fp16.plan out/$s/masks_$det --zoom > out/$s/$det.log 2>&1 ;;
    rtmo)  test -f out/$s/masks_$det.mot.txt || $P detect_rtmo.py segs/$s.mp4 models/rtmo-l_fp16.plan out/$s/masks_$det > out/$s/$det.log 2>&1 ;;
  esac
  test -f out/$s/masks_$det.mot.txt || { echo "skip $s: no masks_$det.mot.txt"; continue; }
  test -f out/$s/pose_$det.npz || $P poseall.py out/$s $det models/rtmw-x_b8_fp16.plan > out/$s/pose_$det.log 2>&1
  A=$($P -c "import json;print(json.load(open('segs.json'))['$s']['names'][0])"); B=$($P -c "import json;print(json.load(open('segs.json'))['$s']['names'][1])")
  $P identity2.py out/$s $det "$A" "$B" > out/$s/identity_$det.log 2>&1 && $P idsum.py . $det $s || echo "FAIL $s $(tail -2 out/$s/identity_$det.log | cut -c1-200)"
done
