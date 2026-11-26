#!/bin/bash
# run_tests.sh off|on [extra pytest args]: BF16/MXFP8 wgrad L0 + L1 test files of wip/wgrad-wide-bf16-sm107, with the
# wide BF16 kernel at its library default (off) or forced on through the test-only plugin wide_on.py
# (GroupedGemmWgradBf16API._use_wide_tiles = True; test_wide_tiles_are_off_by_default is deselected in that mode).
# Env: FE   the cuDNN-FE checkout of the branch (required); its python/cudnn must import (the compiled module
#           _compiled_module*.so copied in, or the branch installed), e.g. PYTHONPATH=$FE/python
#      PY   python (default python3)   MARK (default "L0 or L1")   OUT (default ./tests_out)
set -u
K=$(cd "$(dirname "$0")" && pwd)
MODE=${1:?off|on}; shift
: "${FE:?set FE to the cuDNN-FE checkout}"
PY=${PY:-python3}; MARK=${MARK:-L0 or L1}; OUT=${OUT:-$PWD/tests_out}; mkdir -p "$OUT"
FILES=(gemm/cutedsl/test_grouped_gemm_wgrad.py gemm/cutedsl/test_grouped_gemm_wgrad_atomic_counter.py
       gemm/cutedsl/test_grouped_gemm_wgrad_pdl.py gemm/cutedsl/test_grouped_gemm_wgrad_empty.py
       gemm/cutedsl/test_wgrad_tensormap_acquire.py gemm/cutedsl/test_grouped_gemm_wgrad_wide.py
       gemm/cutedsl/test_grouped_gemm_bf16_smem_capacity.py)
cd "$FE/test/python"
export PYTHONPATH="$K:$FE/python:$FE/test/python${PYTHONPATH:+:$PYTHONPATH}"
if [ "$MODE" = on ]; then
  export PYTEST_PLUGINS=wide_on WIDE_LOG_DIR="$OUT/wide_log_on"; rm -rf "$WIDE_LOG_DIR"
fi
$PY -m pytest -p no:cacheprovider -q -rfE -m "$MARK" "${FILES[@]}" "$@" 2>&1 | tee "$OUT/pytest_$MODE.log" | tail -40
if [ "$MODE" = on ]; then
  echo "WIDE_PICKED tests: $(cat "$WIDE_LOG_DIR"/*.tsv 2>/dev/null | awk -F'\t' '$2 == "True"' | cut -f1 | sort -u | wc -l)" \
       "of $(cat "$WIDE_LOG_DIR"/*.tsv 2>/dev/null | cut -f1 | sort -u | wc -l) that built a BF16 wgrad"
  cat "$WIDE_LOG_DIR"/*.tsv 2>/dev/null | awk -F'\t' '$2 == "True"' | cut -f1 | sort -u | sed 's/\[.*//' | uniq -c | sed 's/^/WIDE_PICKED_BY /'
fi
