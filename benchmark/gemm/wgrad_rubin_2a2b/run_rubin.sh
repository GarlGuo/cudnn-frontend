#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# run_rubin.sh: lock-step MXFP8 wgrad A/B on one node, all GPUs.
# Env: FE_PYTHON (required: <tree>/python), OUT (default ./lsw_out), BUDGET (s per round, default 600), REPS (10),
#      MODES ("0 1": CLC, atomic counter), PASSES ("a b": rank positions forward, then reversed), PY (python), TORCHRUN.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$PWD/lsw_out}; mkdir -p "$OUT"
PY=${PY:-python}; TORCHRUN=${TORCHRUN:-torchrun}
NG=$(nvidia-smi -L | wc -l)
SHAPES=(deepseek_v3:4:8192 qwen35_122b:4:8192 qwen35_397b:4:8192 glm53_flash:4:8192 qwen38_2p4t:4:8192
        deepseek_v3:8:8192 qwen35_122b:8:8192 qwen35_397b:8:8192 glm53_flash:8:8192 qwen38_2p4t:8:8192)
NS=${#SHAPES[@]}; NR=$(( (NS + NG - 1) / NG ))
echo "GPUs $NG rounds $NR tree $FE_PYTHON $(git -C "$FE_PYTHON/.." rev-parse --short=9 HEAD 2>/dev/null)"
nvidia-smi --query-gpu=index,name,clocks.max.sm,power.limit --format=csv,noheader
for mode in ${MODES:-0 1}; do
  for pass in ${PASSES:-a b}; do
    for ((r = 0; r < NR; r++)); do
      list=(); for ((i = 0; i < NG; i++)); do list+=("${SHAPES[$(( (r * NG + i) % NS ))]}"); done
      [ "$pass" = b ] && list=($(printf '%s\n' "${list[@]}" | tac))
      shapes=$(IFS=,; echo "${list[*]}")
      port=$($PY -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1])")
      tag=c${mode}_${pass}_r$r
      WAB_COUNTER=$mode FE_PYTHON=$FE_PYTHON CUDA_MODULE_LOADING=EAGER PYTHONUNBUFFERED=1 timeout --foreground --kill-after=30 $(( ${BUDGET:-600} + 900 )) \
        $TORCHRUN --nnodes 1 --nproc-per-node $NG --master-addr 127.0.0.1 --master-port $port \
        "$HERE/lsw_rubin.py" --shapes "$shapes" --out "$OUT/$tag.jsonl" --budget ${BUDGET:-600} --reps ${REPS:-10} > "$OUT/$tag.log" 2>&1
      echo "ROUND $tag $shapes rc $?"
    done
  done
done
$PY "$HERE/agg_rubin.py" "$OUT"/*.jsonl.r* | tee "$OUT/tables.md"
