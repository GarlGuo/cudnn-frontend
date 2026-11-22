#!/bin/bash
# Tile sweep of the SM100 / SM107 BF16 grouped GEMMs: one item per GPU, every arm interleaved in one process, both protocols.
#
#   run_sweep.sh [-g GPUS] [-e EP] [-o OUT] [-a ARMS] [-p PROTOCOLS] [-t ITEM_TIMEOUT_S] ITEM...
#     GPUS       comma list of GPU indices, one worker each (default 0,1,2,3)
#     EP         EP size for items given without one (default 8); an item may also be written EP:model.op, e.g. 4:deepseek_v3.fc1_fwd
#     ITEM       model.op (python quack_cmp_bench.py --list-items), e.g. deepseek_v3.fc1_fwd qwen38_2p4t.fc1_wgrad
#     ARMS       --arms of quack_cmp_bench.py (default "sweep": cudnn_dyn_256x{128,192,224,256}_c21, cudnn_dyn_256x256_c22,
#                cudnn (static 256x256 c2x1 default), quack_tile (quack 256x256 c2x1 dynamic, untuned), torch)
#     PROTOCOLS  default qcmp,bench (80 ms gated, then back to back; GC is disabled inside both timed loops)
#   Workers pull items from a shared queue in the given order. Output (OUT, default ./sweep_out_<UTC time>):
#     perf_ep<EP>_<item>.json / .log   per item (timing_all.<protocol>.timing.<arm>: us_median, tflops,
#                                      time_over_cudnn_dyn_256x256_c21; checks.<arm>: max rel diff vs torch; skipped)
#     rc.txt, env.txt, summary.md / summary.json, summary_tiles.md / summary_tiles.json (time over dyn 256x256 c2x1 per protocol)
#   Env: PYTHON (default python3; must import torch, cutlass and cudnn from this tree), BENCH_EXTRA (more bench args, default
#     "--runs 3 --warmup 5 --reps 50"), PEAK_TFLOPS (summary %-of-peak column, default 2500), QUACK_CACHE_DIR (default
#     ~/.cache/quack_cmp_sweep; QUACK_CACHE_AUTOTUNING=1 is set so any tuned quack arm tunes once), BENCH_LAUNCHER (default
#     $PYTHON; a wrapper command that takes the bench path first).
set -u
D=$(cd "$(dirname "$0")" && pwd); BENCH=$D/quack_cmp_bench.py
GPUS=0,1,2,3; EP=8; OUT=""; ARMS=sweep; PROTO=qcmp,bench; TMO=1800
while getopts "g:e:o:a:p:t:" opt; do
  case $opt in g) GPUS=$OPTARG ;; e) EP=$OPTARG ;; o) OUT=$OPTARG ;; a) ARMS=$OPTARG ;; p) PROTO=$OPTARG ;; t) TMO=$OPTARG ;;
               *) sed -n '2,20p' "$0"; exit 2 ;; esac
done
shift $((OPTIND - 1)); [ $# -gt 0 ] || { sed -n '2,20p' "$0"; exit 2; }
PYTHON=${PYTHON:-python3}; LAUNCH=${BENCH_LAUNCHER:-$PYTHON}; EXTRA=${BENCH_EXTRA:---runs 3 --warmup 5 --reps 50}
OUT=${OUT:-$PWD/sweep_out_$(date -u +%Y%m%dT%H%M%SZ)}; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
export QUACK_CACHE_AUTOTUNING=1 QUACK_CACHE_DIR=${QUACK_CACHE_DIR:-$HOME/.cache/quack_cmp_sweep} PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$QUACK_CACHE_DIR"
: > "$OUT/queue.txt"; : > "$OUT/claimed.txt"; : > "$OUT/rc.txt"
for it in "$@"; do case $it in *:*) echo "${it%%:*} ${it#*:}" ;; *) echo "$EP $it" ;; esac; done >> "$OUT/queue.txt"
{
  echo "host $(hostname) date $(date -u +%FT%TZ) gpus $GPUS arms $ARMS protocols $PROTO extra $EXTRA"
  (cd "$D" && git rev-parse HEAD 2>/dev/null) | sed 's/^/sha /'
  nvidia-smi --query-gpu=index,name,driver_version,memory.used,clocks.max.sm --format=csv 2>&1
  CUDA_VISIBLE_DEVICES=${GPUS%%,*} $LAUNCH "$BENCH" --list-items > /dev/null 2>&1 || echo "WARNING: bench --list-items failed"
  CUDA_VISIBLE_DEVICES=${GPUS%%,*} $PYTHON -c 'import torch, cutlass; print("torch", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_capability()); print("cutlass-dsl", cutlass.__version__)' 2>&1
} > "$OUT/env.txt"; cat "$OUT/env.txt"
claim() {  # first unclaimed queue line, or nothing
  ( flock 8
    while read -r ep it; do
      [ -n "$it" ] || continue; grep -qx "$ep $it" "$OUT/claimed.txt" && continue
      echo "$ep $it" >> "$OUT/claimed.txt"; echo "$ep $it"; break
    done < "$OUT/queue.txt" ) 8> "$OUT/queue.lock"; }
for g in ${GPUS//,/ }; do
  (while :; do
     read -r ep it <<< "$(claim)"; [ -n "${it:-}" ] || break
     tag=ep${ep}_$it; t0=$SECONDS; rc=0
     echo "$(date -u +%T) gpu$g start $tag"
     CUDA_VISIBLE_DEVICES=$g timeout --foreground "$TMO" $LAUNCH "$BENCH" --item "$it" --ep "$ep" --allow-non-sm90 --arms "$ARMS" \
       --protocol "$PROTO" $EXTRA --out "$OUT/perf_$tag.json" > "$OUT/perf_$tag.log" 2>&1 < /dev/null || rc=$?
     echo "$tag gpu$g rc=$rc secs=$((SECONDS - t0))" | tee -a "$OUT/rc.txt"
   done) &
done
wait
$LAUNCH "$BENCH" --summarize "$OUT" --peak-tflops "${PEAK_TFLOPS:-2500}" > "$OUT/summarize.log" 2>&1 || echo "summarize failed (see $OUT/summarize.log)"
cat "$OUT/summary_tiles.md" 2>/dev/null
for f in "$OUT"/perf_*.log; do grep -q "Traceback" "$f" && { echo "== $(basename "$f")"; grep -A12 -m1 "Traceback" "$f" | head -30; }; done
echo "OUT $OUT"; cat "$OUT/rc.txt"
! grep -qv 'rc=0 ' "$OUT/rc.txt"
