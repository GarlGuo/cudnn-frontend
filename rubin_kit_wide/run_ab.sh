#!/bin/bash
# run_ab.sh: lock-step BF16 wgrad A/B (stock vs wide vs stock A/A, all arms interleaved in every pass) on every GPU
# of this node, for a subset of (claim, round) units that fits one job's wall-clock cap. Any number of such jobs can
# write into the same or different OUT dirs; agg.py merges every file it is given (passes pooled per shape and claim).
# Env: FE_PYTHON  cuDNN-FE python/ root of the branch (required)
#      PY         python with torch + nvidia-cutlass-dsl that targets this GPU (default: python3)
#      OUT        output dir (default ./ab_out); files OUT/<claim>_round<k>.<JOB_TAG>.jsonl.r<rank> (+ .log)
#      JOB_TAG    unique per job (default <host>_<epoch>_<pid>), so jobs never overwrite each other
#      CLAIMS     clc | counter | clc,counter (default clc)
#      ROUNDS     round indices to run, e.g. 0 or 1,2 (default: all); round k = shapes k*NPROC .. k*NPROC+NPROC-1,
#                 the last round padded with shapes from the start
#      CAP        wall-clock seconds for the whole job (default 1620 = 27 min, for a 30 min job limit)
#      BUDGET     max seconds per unit (default 900). Each unit gets min(BUDGET, remaining CAP / units left - 45);
#                 the budget counts from process start, so it includes the unit's compiles and byte checks
#      MIN_UNIT   skip the remaining units when a unit would get less than this (default 240)
#      NPROC      ranks = GPUs (default: nvidia-smi count)   REPS  samples per arm per direction per pass (default 10)
#      SHAPES     model:ep:tokens list (default: the 10 GB200 shapes)   GATE_MS  idle gate (default 20)
#      EXTRA      e.g. stock100,wide100 (the same kernels forced to sm_100's 232448 B smem)
#      ARGS       extra wgrad_ls.py args (--no-cases, --no-api-check)
set -u
K=$(cd "$(dirname "$0")" && pwd)
: "${FE_PYTHON:?set FE_PYTHON to <cudnn-frontend checkout>/python}"
PY=${PY:-python3}; OUT=${OUT:-$PWD/ab_out}; CLAIMS=${CLAIMS:-${CLAIM:-clc}}; BUDGET=${BUDGET:-900}; REPS=${REPS:-10}
CAP=${CAP:-1620}; MIN_UNIT=${MIN_UNIT:-240}; JOB_TAG=${JOB_TAG:-$(hostname -s)_$(date +%s)_$$}
NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}
SHAPES=${SHAPES:-deepseek_v3:4:8192,qwen35_122b:4:8192,qwen35_397b:4:8192,glm53_flash:4:8192,qwen38_2p4t:4:8192,deepseek_v3:8:8192,qwen35_122b:8:8192,qwen35_397b:8:8192,glm53_flash:8:8192,qwen38_2p4t:8:8192}
mkdir -p "$OUT"
FE_SHA=$(cat "$FE_PYTHON/../SHA" 2>/dev/null || git -C "$FE_PYTHON" rev-parse --short=9 HEAD 2>/dev/null || echo unknown)
export FE_PYTHON FE_SHA CUDA_MODULE_LOADING=EAGER PYTHONUNBUFFERED=1 GATE_MS=${GATE_MS:-20}
IFS=, read -r -a SV <<< "$SHAPES"
n=${#SV[@]}; rounds=$(( (n + NPROC - 1) / NPROC ))
if [ -n "${ROUNDS:-}" ]; then IFS=, read -r -a RSEL <<< "$ROUNDS"; else RSEL=($(seq 0 $((rounds - 1)))); fi
IFS=, read -r -a CV <<< "$CLAIMS"
UNITS=(); for c in "${CV[@]}"; do for r in "${RSEL[@]}"; do UNITS+=("$c:$r"); done; done
echo "RUN_AB job $JOB_TAG claims $CLAIMS rounds ${RSEL[*]} of $rounds units ${#UNITS[@]} nproc $NPROC cap ${CAP}s budget<=$BUDGET reps $REPS gate ${GATE_MS}ms sha $FE_SHA"
T0=$SECONDS
for ((u = 0; u < ${#UNITS[@]}; u++)); do
  claim=${UNITS[$u]%%:*}; r=${UNITS[$u]##*:}
  left=$(( ${#UNITS[@]} - u )); remain=$(( CAP - (SECONDS - T0) ))
  b=$(( remain / left - 45 )); [ $b -gt $BUDGET ] && b=$BUDGET
  if [ $b -lt $MIN_UNIT ]; then
    for ((v = u; v < ${#UNITS[@]}; v++)); do echo "UNIT_SKIPPED ${UNITS[$v]} (cap)"; done; break
  fi
  rs=(); for ((i = 0; i < NPROC; i++)); do rs+=("${SV[$(( (r * NPROC + i) % n ))]}"); done
  shapes=$(IFS=,; echo "${rs[*]}")
  port=$($PY -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1])")
  base="$OUT/${claim}_round$r.$JOB_TAG"
  timeout --kill-after=20 $(( remain - 15 )) $PY -m torch.distributed.run --nnodes 1 --nproc-per-node $NPROC \
    --master-addr 127.0.0.1 --master-port $port "$K/wgrad_ls.py" --shapes "$shapes" --out "$base.jsonl" \
    --budget $b --reps $REPS --claim $claim ${EXTRA:+--extra $EXTRA} ${ARGS:-} > "$base.log" 2>&1
  echo "UNIT $claim round $r rc $? budget ${b}s at $((SECONDS - T0))s shapes $shapes"
done
$PY "$K/agg.py" "$OUT" | tee "$OUT/table.$JOB_TAG.md"
