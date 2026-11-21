#!/bin/bash
# Stage 1: DARTF (SAM 3 INT8 TensorRT) "person" masks + tracks on each 25 fps window. Usage: stage1_masks.sh [tracker] seg...
W=$HOME/bjj/x-sam3d-dartf; D=$W/repos/DART/dartf; K=$HOME/bjj/x-mmagold/kp
export PATH=$W/.venv/bin:/usr/local/cuda/bin:$PATH LQ_PLUGINS=$D/plugins/lq_plugins.so
m=$1; shift; cd $D
for s in "$@"; do
  o=$K/out/$s/masks_$m; rm -rf $o; mkdir -p $K/out/$s
  X=""; [ $m != light ] && X="--vision-trunk assets/vision_int8_trunk.plan --trk-dir assets"
  /usr/bin/time -v python demo/run_video_x.py $K/segs/$s.mp4 $o.mp4 "person" --assets assets --groundmask assets/groundmask_c4_q32_phase_fp16.plan \
    --tracker $m $X --pipeline --no-render --duration 0 --dump-masks $o --stats $o.json --mot-out $o.mot.txt > $o.log 2>&1 \
    && echo "OK $s $m $(grep 'Maximum resident' $o.log)" || echo "FAIL $s $m $(tail -3 $o.log | tr '\n' ' ' | cut -c1-300)"
done
echo STAGE1_DONE
