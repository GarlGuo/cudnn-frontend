#!/bin/bash
# run_rule_ab.sh: lock-step BF16 wgrad A/B (stock vs wide, all arms interleaved in every pass), every GPU of this node
# running the same GEMM list, one torchrun per claim. Any number of jobs can write into the same OUT dir (per-job tags);
# agg_rule.py merges them.
# Env: FE_PYTHON  cuDNN-FE python/ root of the branch (required)
#      PY         python with torch + a nvidia-cutlass-dsl that targets this GPU (default python3)
#      OUT        output dir (default ./rule_out); files OUT/<set>.<claim>.<JOB_TAG>.jsonl.r<rank> (+ .log)
#      JOB_TAG    unique per job (default <host>_<epoch>_<pid>)
#      SET        GEMM set name below (default f1), or GEMMS=m:n:K:E,... directly
#      CLAIMS     clc,counter (default; use counter,clc for the mirrored job)
#      REVERSE    1 = run the GEMM list in reverse order (the mirrored job)
#      BUDGET     seconds per claim unit, compiles and byte checks included (default 600)
#      REPS       samples per arm per direction per pass (default 2)
#      EXTRA      extra arms (default stockT,wide100): stockT = stock at the target's smem, wide100 = wide at 232448 B
#      NPROC      ranks = GPUs (default: nvidia-smi count)   GATE_MS  idle gate (default 20)
set -u
K=$(cd "$(dirname "$0")" && pwd)
: "${FE_PYTHON:?set FE_PYTHON to <cudnn-frontend checkout>/python}"
PY=${PY:-python3}; OUT=${OUT:-$PWD/rule_out}; CLAIMS=${CLAIMS:-clc,counter}; BUDGET=${BUDGET:-600}; REPS=${REPS:-2}
JOB_TAG=${JOB_TAG:-$(hostname -s)_$(date +%s)_$$}; NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}; SET=${SET:-f1}
EXTRA=${EXTRA-stockT,wide100}
declare -A SETS=(
  # the 20 MoE GEMMs (5 models, EP4 / EP8 at 8192 tokens per rank; dW1 = 2I x H, dW2 = H x I; K rows per expert)
  [models_ep4]="4096:7168:1024:64,7168:2048:1024:64,2048:3072:1024:64,3072:1024:1024:64,2048:4096:768:128,4096:1024:768:128,4096:4096:1024:72,4096:2048:1024:72,4096:8192:768:128,8192:2048:768:128"
  [models_ep8]="4096:7168:2048:32,7168:2048:2048:32,2048:3072:2048:32,3072:1024:2048:32,2048:4096:1280:64,4096:1024:1280:64,4096:4096:2048:36,4096:2048:2048:36,4096:8192:1280:64,8192:2048:1280:64"
  # boundary families: the same per-expert GEMM with more or fewer experts, i.e. wide tiles per resident cluster
  # (wide tiles / 106 on VR200) of about 8, 12, 14, 17, 23, 26 and 30
  [f1]="3072:1024:2048:35,3072:1024:2048:53,3072:1024:2048:62,3072:1024:2048:75,3072:1024:2048:102,3072:1024:2048:115,3072:1024:2048:133"
  [f2]="2048:3072:2048:18,2048:3072:2048:27,2048:3072:2048:31,2048:3072:2048:38,2048:3072:2048:51,2048:3072:2048:57,2048:3072:2048:66"
  [f3]="4096:2048:1024:13,4096:2048:1024:20,4096:2048:1024:23,4096:2048:1024:28,4096:2048:1024:38,4096:2048:1024:43,4096:2048:1024:50"
  # other n and K: 6.8, 13.5, 27 (n 7168, K 2048), 9.7, 19.3 (8192 x 2048, K 1280), 14.5 (K 4096), 58 (K 1024)
  [f4]="4096:7168:2048:8,4096:7168:2048:16,4096:7168:2048:32,8192:2048:1280:16,8192:2048:1280:32,3072:1024:4096:64,3072:1024:1024:256"
)
GEMMS=${GEMMS:-${SETS[$SET]:?unknown SET $SET}}
mkdir -p "$OUT"
FE_SHA=$(git -C "$FE_PYTHON" rev-parse --short=9 HEAD 2>/dev/null || echo unknown)
export FE_PYTHON FE_SHA CUDA_MODULE_LOADING=EAGER PYTHONUNBUFFERED=1 GATE_MS=${GATE_MS:-20}
echo "RUN_RULE_AB job $JOB_TAG set $SET claims $CLAIMS reverse ${REVERSE:-0} nproc $NPROC budget ${BUDGET}s reps $REPS extra ${EXTRA:-none} sha $FE_SHA host $(hostname)"
nvidia-smi --query-gpu=index,name,clocks.sm,clocks.max.sm,power.draw,power.limit --format=csv,noheader
T0=$SECONDS
IFS=, read -r -a CV <<< "$CLAIMS"
for c in "${CV[@]}"; do
  port=$($PY -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1])")
  base="$OUT/$SET.$c.$JOB_TAG"
  timeout --kill-after=20 $(( BUDGET + 600 )) $PY -m torch.distributed.run --nnodes 1 --nproc-per-node $NPROC \
    --master-addr 127.0.0.1 --master-port $port "$K/rule_ls.py" --gemms "$GEMMS" --out "$base.jsonl" \
    --budget $BUDGET --reps $REPS --claim $c ${EXTRA:+--extra $EXTRA} $([ "${REVERSE:-0}" = 1 ] && echo --reverse) > "$base.log" 2>&1
  echo "UNIT $SET $c rc $? at $((SECONDS - T0))s"
done
$PY "$K/agg_rule.py" "$OUT" | tee "$OUT/table.$JOB_TAG.md" | tail -40
