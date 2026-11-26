#!/bin/bash
# setup.sh: per-host prep for the s59sm90perf package (kit branch wip/sm90-h100-kit-b, h100_kit_b/s59sm90perf/): a short
# perf replicate of the SM90 dGLU streaming signal cells, 644492a92 vs 228f92bab.  s57sm90plainperf3's setup.sh with git
# delivery instead of S3 and the trees moved one head forward:
#   base = 644492a92 = #2908 head (venv $HOME/venvs/cfe_pp_644492a9, shared with s59sm90regress / s59sm90hangrepro)
#   old  = 228f92bab = the previous #2908 head: $W/site_old (the venv's .so + 228f92bab's python/cudnn; no C++ change)
#   ctl  = base again in its own process (A/A control)
# Idempotent; one run at a time per host (flock on $W/.setup.lock); safe to rerun at any time, also while jobs run.
#   Success: $W/ready (holding the MD5SUMS md5).  rc 0 iff ready.
#   Env: KIT_REPO (the kit clone; default: this script's git checkout, else $W/KIT_REPO, else
#   $HOME/cudnnfe_sm90/kit-b/cudnn-frontend), BASE_VENV ($HOME/venvs/cfe_quack), CUDA_PATH (/usr/local/cuda-13.2).
#   Opens no ports, uploads nothing.
set -euo pipefail
RUN=s59sm90perf
SHA=644492a9
FULLSHA=644492a923c79bcae3b9214391b79a760a326742; GITTREE=4e5602db1098f6f2ac5b8fabc8fe45b39dc67465
# old = 228f92bab (the previous #2908 head, natively validated in s57/s58): its python/cudnn over this venv's .so
# (228f92bab..644492a92 changes no C++ file, only python/ and docs/)
OLD_SHA=228f92bab65ae26f9bac192a16e0a512fcbc2029; OLD_PYTREE=b839f6d31f8ceefeff42614410a7bbd6e5a82bc0
BASE_VENV=${BASE_VENV:-$HOME/venvs/cfe_quack}
HERE=$(cd "$(dirname "$0")" && pwd)
W=$HOME/cudnnfe_sm90/$RUN; mkdir -p "$W"
FILES="worker90.py plans.json setup_base_venv.sh setup.sh job.sh rss_watchdog.py jobs.jsonl caps.json"
exec 8> "$W/.setup.lock"; flock 8
LOG=$W/setup_$(date -u +%Y%m%dT%H%M%SZ)_$$.log; ST=$W/setup_state; mkdir -p "$ST"
exec > >(tee -a "$LOG") 2>&1
echo "setup $RUN on $(hostname) $(date -u +%FT%TZ)"
cd "$W"
# the kit clone (git delivery: the source trees come from its commits; no tarball, no S3)
if [ -z "${KIT_REPO:-}" ]; then
  KIT_REPO=$(git -C "$HERE" rev-parse --show-toplevel 2> /dev/null || true)
  [ -n "$KIT_REPO" ] || KIT_REPO=$(cat "$W/KIT_REPO" 2> /dev/null || true)
  [ -n "$KIT_REPO" ] || KIT_REPO=$HOME/cudnnfe_sm90/kit-b/cudnn-frontend
fi
src_from_git() {  # src_from_git <full sha>[:<path>] <git tree id> <dest dir> [prefix]: git archive (tree id checked)
  local got
  local spec=$1^{tree}; case $1 in *:*) spec=$1 ;; esac
  got=$(git -C "$KIT_REPO" rev-parse --verify -q "$spec" 2> /dev/null) || { echo "SETUP_FAIL $1 not in KIT_REPO=$KIT_REPO (clone the kit branch, see h100_kit_b/README.md)"; return 1; }
  [ "$got" = "$2" ] || { echo "SETUP_FAIL $1: tree $got, want $2"; return 1; }
  rm -rf "$3.tmp.$$"; mkdir "$3.tmp.$$"
  git -C "$KIT_REPO" archive ${4:+--prefix="$4"} "$1" | tar -x -C "$3.tmp.$$" || { echo "SETUP_FAIL git archive $1"; return 1; }
  mv "$3.tmp.$$" "$3"
}
[ -f MD5SUMS ] || { [ -f "$HERE/MD5SUMS" ] || { echo "SETUP_FAIL no MD5SUMS next to setup.sh"; exit 3; }; cp "$HERE/MD5SUMS" MD5SUMS.tmp.$$; mv MD5SUMS.tmp.$$ MD5SUMS; }
for f in $FILES; do
  [ -f "$f" ] && continue
  [ -f "$HERE/$f" ] && [ "$HERE" != "$W" ] || { echo "SETUP_FAIL $f missing (run setup.sh from the kit clone first)"; exit 3; }
  cp "$HERE/$f" "$f.tmp.$$"; mv "$f.tmp.$$" "$f"
done
md5sum -c MD5SUMS > "$ST/md5_check.txt" || { cat "$ST/md5_check.txt"; echo "SETUP_FAIL md5 check"; exit 3; }
if [ -e "$KIT_REPO/.git" ]; then echo "$KIT_REPO" > KIT_REPO; fi
echo "KIT_REPO=$KIT_REPO ($(git -C "$KIT_REPO" rev-parse --short HEAD 2> /dev/null || echo '?'))"
PKG_MD5=$(md5sum < MD5SUMS | cut -c1-32)
if [ "$(cat ready 2> /dev/null)" = "$PKG_MD5" ] && [ -d "$HOME/venvs/cfe_pp_$SHA" ] && [ -d src ] && [ -d site_old ]; then echo "ready"; exit 0; fi
rm -f ready
check_env() { CUDA_VISIBLE_DEVICES= "$1/bin/python" -I - "$2/python/cudnn/gemm/cutedsl/grouped" << 'EOF'
import hashlib, importlib, importlib.metadata as md, os, sys, cudnn, cutlass, torch, tvm_ffi, quack, quack.gemm_interface
ref_root = sys.argv[1]
assert cudnn.__file__.startswith(sys.prefix), ("cudnn outside the venv", cudnn.__file__, sys.prefix)
def tree(root):
    return {os.path.relpath(os.path.join(d, f), root): hashlib.md5(open(os.path.join(d, f), "rb").read()).hexdigest()
            for d, _, fs in os.walk(root) for f in fs if f.endswith(".py")}
inst, ref = tree(os.path.join(os.path.dirname(cudnn.__file__), "gemm", "cutedsl", "grouped")), tree(ref_root)
assert inst == ref, ("installed cudnn/gemm/cutedsl/grouped is not this archive's", sorted(set(inst) ^ set(ref)),
                     sorted(k for k in set(inst) & set(ref) if inst[k] != ref[k]))
for m in ("unfused.moe_grouped_gemm_sm90", "glu.moe_grouped_gemm_glu_sm90", "dglu.moe_grouped_gemm_dglu_sm90", "moe_streaming_input_block_tile_scheduler"):
    importlib.import_module("cudnn.gemm.cutedsl.grouped." + m)
for n in ("GroupedGemmSm90", "GroupedGemmGluSm90", "GroupedGemmDgluSm90"):
    getattr(cudnn, n)
assert md.version("quack-kernels") == "0.6.5", md.version("quack-kernels")
print("cudnn", cudnn.__file__, "backend", cudnn.backend_version(), "cutlass", md.version("nvidia-cutlass-dsl"), "torch", torch.__version__,
      "tvm_ffi", tvm_ffi.__version__, "quack", quack.__version__, "grouped .py files", len(inst))
EOF
}
[ -d src ] || src_from_git "$FULLSHA" "$GITTREE" "$W/src" || exit 3
mkdir -p "$HOME/venvs"
V=$HOME/venvs/cfe_pp_$SHA
(
  flock 9
  if [ -d "$V" ] && check_env "$V" "$W/src" > "$ST/env_check.txt" 2>&1; then echo "reusing $V"; exit 0; fi
  [ -d "$V" ] && { echo "$V exists but fails the import check (see env_check.txt); move it away and rerun"; exit 4; }
  BASE_VENV=$BASE_VENV bash "$W/setup_base_venv.sh" > "$ST/base_venv.txt" 2>&1 || { echo "BASE_VENV setup failed (see base_venv.txt)"; exit 4; }
  [ -x "$BASE_VENV/bin/python" ] || { echo "no $BASE_VENV"; exit 4; }
  echo "building $V from $BASE_VENV"
  T=$HOME/venvs/.cfe_pp.tmp.$(hostname).$$.s46; rm -rf "$T"; cp -a "$BASE_VENV" "$T"; rm -f "$T/.cfe_quack_ok" "$T/.cfe_quack_building" "$T/.qcmp_ok"
  TP=$(realpath "$("$T/bin/python" -I -c 'import sys; print(sys.prefix)')")
  [ "$TP" = "$(realpath "$T")" ] || { echo "sys.prefix '$TP' is not $T"; exit 4; }
  "$T/bin/python" -m pip --version > /dev/null 2>&1 || "$T/bin/python" -m ensurepip > /dev/null 2>&1
  "$T/bin/python" -m pip uninstall -y nvidia-cudnn-frontend > "$ST/cudnn_build.txt" 2>&1 || true
  "$T/bin/python" -m pip install --force-reinstall --no-deps cmake==3.31.6 ninja==1.11.1.1 >> "$ST/cudnn_build.txt" 2>&1
  "$T/bin/python" -m pip install setuptools==80.9.0 wheel==0.45.1 cmake==3.31.6 ninja==1.11.1.1 nanobind==3.0.1 >> "$ST/cudnn_build.txt" 2>&1
  CP=${CUDA_PATH:-/usr/local/cuda-13.2}; [ -d "$CP" ] || CP=/gbr/root/usr/local/cuda-13.2
  [ -d "$CP" ] || { echo "no CUDA 13.2 (set CUDA_PATH)"; exit 5; }
  CUDNN_PATH=$("$T/bin/python" -I -c "import importlib.util; print(importlib.util.find_spec('nvidia.cudnn').submodule_search_locations[0])") \
    CUDA_PATH=$CP VIRTUAL_ENV=$T PATH=$T/bin:$PATH \
    "$T/bin/python" -m pip install --no-deps --no-build-isolation "$W/src" >> "$ST/cudnn_build.txt" 2>&1 || { tail -20 "$ST/cudnn_build.txt"; echo "cudnn-fe build failed; left at $T"; exit 5; }
  check_env "$T" "$W/src" > "$ST/env_check.txt" 2>&1 || { cat "$ST/env_check.txt"; echo "new venv fails the import check; left at $T"; exit 4; }
  [ -e "$V" ] || mv "$T" "$V"
  rm -rf "$T"
) 9> "$HOME/venvs/.cfe_pp_$SHA.lock" > "$ST/venv_build.txt" 2>&1 || { echo "SETUP_FAIL venv: $(tail -3 "$ST/venv_build.txt" | tr '\n' ' ')"; exit 4; }
check_env "$V" "$W/src" > "$ST/env.txt" 2>&1 || { echo "SETUP_FAIL venv fails the check after the build"; exit 4; }
"$V/bin/python" -m pip freeze > "$ST/pip_freeze.txt" 2> /dev/null || true
SPK=$("$V/bin/python" -I -c "import os, cudnn; print(os.path.dirname(cudnn.__file__))")
# site_old: the venv's installed cudnn package (.so kept) with 228f92bab's python/cudnn over its .py files (s54c's variant
# site, built from git instead of variant_old.tar.gz)
rm -rf site_old.tmp site_old.py.tmp; mkdir -p site_old.tmp; cp -a "$SPK" site_old.tmp/cudnn
find site_old.tmp/cudnn -type f ! -name '*.so' -delete
src_from_git "$OLD_SHA:python/cudnn" "$OLD_PYTREE" "$W/site_old.py.tmp" cudnn/ || exit 3
cp -a site_old.py.tmp/cudnn/. site_old.tmp/cudnn/; rm -rf site_old.py.tmp
echo "$OLD_SHA" > site_old.tmp/SHA
rm -rf site_old; mv site_old.tmp site_old
echo "site_old: $(cat site_old/SHA)"
echo "$PKG_MD5" > ready
echo "setup done: ready $(date -u +%FT%TZ)"
