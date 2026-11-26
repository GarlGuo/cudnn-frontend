#!/bin/bash
# setup_base_venv.sh: build BASE_VENV (default ~/venvs/cfe_quack) from PyPI and download.pytorch.org on a fresh H100 host.
# Same contents as the r-host cfe_quack: torch 2.14.0+cu130 (download.pytorch.org, sha256 pinned), nvidia-cutlass-dsl 4.8.0
# (+ libs-base/libs-cu12), quack-kernels 0.6.5 (sha256 pinned), apache-tvm-ffi 0.1.16, einops, torch-c-dlpack-ext, the
# pinned build tools (cmake 3.31.6, ninja 1.11.1.1, setuptools 80.9.0, wheel 0.45.1, nanobind 3.0.1) and the test deps
# (pytest 9.1.1, parameterized 0.9.0, numpy 2.5.2, cuda-python 13.3.1 as on the r hosts). No cudnn-frontend: run_s11.sh builds that into a copy.
#   Idempotent and flock-guarded: a finished venv carries .cfe_quack_ok and passes check(); a partial one from an
#   interrupted run of this script (.cfe_quack_building) is rebuilt; a foreign dir that fails check() is never touched.
#   Every pip call is "$V/bin/python -m pip". Env: BASE_VENV, PYTHON312 (interpreter override), PIP_EXTRA (extra pip args).
set -euo pipefail
V=${BASE_VENV:-$HOME/venvs/cfe_quack}; mkdir -p "$(dirname "$V")"
LOG=${SETUP_LOG:-$(dirname "$V")/.cfe_quack_setup.log}
exec 9> "$(dirname "$V")/.cfe_quack.lock"; flock 9
TORCH_WHL=torch-2.14.0+cu130-cp312-cp312-manylinux_2_28_x86_64.whl
QUACK_WHL=quack_kernels-0.6.5-py3-none-any.whl
check() {  # check <venv>: the pinned stack imports with the expected versions
  CUDA_VISIBLE_DEVICES= "$1/bin/python" -I -c "
import importlib.metadata as md, sys
import torch, cutlass, quack, tvm_ffi, einops, numpy, pytest
want = {'torch': '2.14.0+cu130', 'nvidia-cutlass-dsl': '4.8.0', 'quack-kernels': '0.6.5', 'apache-tvm-ffi': '0.1.16',
        'cmake': '3.31.6', 'ninja': '1.11.1.1', 'setuptools': '80.9.0', 'wheel': '0.45.1', 'nanobind': '3.0.1',
        'nvidia-cudnn-cu13': '9.24.0.43'}
bad = {k: md.version(k) for k in want if md.version(k) != want[k]}
assert not bad, f'version mismatch: {bad}'
assert sys.version_info[:2] == (3, 12), sys.version
assert sys.prefix != sys.base_prefix, 'not a venv'
print('BASE_VENV OK', sys.prefix, 'python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda,
      'cutlass', md.version('nvidia-cutlass-dsl'), 'quack', md.version('quack-kernels'), 'tvm_ffi', md.version('apache-tvm-ffi'))"
}
if [ -d "$V" ]; then
  if [ -f "$V/.cfe_quack_ok" ] && check "$V"; then exit 0; fi
  if [ -f "$V/.cfe_quack_building" ] && [ ! -f "$V/.cfe_quack_ok" ]; then echo "removing the partial build at $V"; rm -rf "$V"
  elif check "$V" > /dev/null 2>&1; then echo "adopting existing $V (passes the check)"; touch "$V/.cfe_quack_ok"; check "$V"; exit 0
  else echo "$V exists, was not built by this script and fails the check; move it away or set BASE_VENV"; exit 4; fi
fi
# A python 3.12 interpreter with venv support (the torch wheel is cp312).
find_py() {
  local c cands=("${PYTHON312:-}" python3.12 /usr/bin/python3.12 /usr/local/bin/python3.12 /gbr/root/usr/bin/python3.12
    /gbr/root/usr/local/bin/python3.12 python3)
  for f in /gbr/root/app/venvs/ours/pyvenv.cfg; do [ -f "$f" ] && cands+=("$(sed -n 's/^home *= *//p' "$f")/python3.12"); done
  command -v uv > /dev/null 2>&1 && cands+=("$(uv python find 3.12 2> /dev/null || true)")
  for c in "${cands[@]}"; do
    [ -n "$c" ] || continue
    command -v "$c" > /dev/null 2>&1 || [ -x "$c" ] || continue
    "$c" -I -c 'import sys, venv; assert sys.version_info[:2] == (3, 12)' > /dev/null 2>&1 && { command -v "$c" || echo "$c"; return 0; }
  done
  if command -v uv > /dev/null 2>&1 && uv python install 3.12 >> "$LOG" 2>&1; then uv python find 3.12; return 0; fi
  return 1
}
PY312=$(find_py) || { echo "no python 3.12 with venv found (tried PYTHON312, python3.12, /usr/bin, /gbr/root, the ours venv's home, uv)"; exit 5; }
echo "building $V with $PY312 ($("$PY312" -V 2>&1)); log $LOG"
mkdir -p "$V"; touch "$V/.cfe_quack_building"
if ! "$PY312" -m venv "$V" >> "$LOG" 2>&1; then  # Debian python without ensurepip: venv without pip, then get-pip
  "$PY312" -m venv --clear --without-pip "$V" >> "$LOG" 2>&1; touch "$V/.cfe_quack_building"
  curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$V/get-pip.py" && "$V/bin/python" -I "$V/get-pip.py" -q >> "$LOG" 2>&1 \
    || { echo "pip bootstrap failed (see $LOG)"; exit 5; }
fi
VP="$V/bin/python"; PIP=("$VP" -m pip --disable-pip-version-check ${PIP_EXTRA:-})
"${PIP[@]}" install -q --upgrade "pip>=24" >> "$LOG" 2>&1
W="$V/.wheels"; mkdir -p "$W"
"${PIP[@]}" download -q --no-deps torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130 -d "$W" >> "$LOG" 2>&1
"${PIP[@]}" download -q --no-deps quack-kernels==0.6.5 -d "$W" >> "$LOG" 2>&1
printf '%s  %s\n' 9c5693d3ca91743202034094ad3b3df20b5eb1373d5e0bc8a5d976be12c35302 "$TORCH_WHL" \
  df1ddd31366c82eb9692fe087c671209189dcbde53c909c81a865d4ee66b158d "$QUACK_WHL" > "$W/SHA256SUMS"
(cd "$W" && sha256sum -c SHA256SUMS) >> "$LOG" 2>&1 || { echo "WHEEL SHA MISMATCH (see $LOG)"; exit 4; }
# torch and its pinned CUDA 13.0 deps (cuda-toolkit 13.0.3, cuDNN 9.24.0.43, NCCL, triton) from the PyTorch cu130 index
"${PIP[@]}" install -q "$W/$TORCH_WHL" --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple \
  >> "$LOG" 2>&1 || { echo "torch install failed (see $LOG)"; exit 5; }
echo "torch==2.14.0+cu130" > "$W/constraints.txt"
"${PIP[@]}" install -q -c "$W/constraints.txt" "$W/$QUACK_WHL" nvidia-cutlass-dsl==4.8.0 apache-tvm-ffi==0.1.16 einops==0.8.2 \
  torch-c-dlpack-ext==0.1.5 >> "$LOG" 2>&1 || { echo "quack / cutlass-dsl install failed (see $LOG)"; exit 5; }
"${PIP[@]}" install -q -c "$W/constraints.txt" cmake==3.31.6 ninja==1.11.1.1 setuptools==80.9.0 wheel==0.45.1 nanobind==3.0.1 \
  pytest==9.1.1 parameterized==0.9.0 numpy==2.5.2 cuda-python==13.3.1 >> "$LOG" 2>&1 || { echo "build/test tools install failed (see $LOG)"; exit 5; }
check "$V" || { echo "the new venv fails the check (see $LOG); it stays marked partial"; exit 4; }
"${PIP[@]}" freeze > "$V/freeze.txt" 2> /dev/null || true
rm -rf "$W" "$V/get-pip.py"; touch "$V/.cfe_quack_ok"; rm -f "$V/.cfe_quack_building"
echo "built $V"
