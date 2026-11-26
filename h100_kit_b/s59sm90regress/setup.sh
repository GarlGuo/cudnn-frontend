#!/bin/bash
# setup.sh: per-host prep for the s59sm90regress package (kit branch wip/sm90-h100-kit-b, h100_kit_b/s59sm90regress/):
# H100 regression of the #2908 head 644492a92 (sm90-streaming-signal).  s57sm90regress's setup.sh with one tree and git
# delivery instead of S3 (the source tree is `git archive` of the commit from the kit clone; no tarball, no S3):
#   sig = 644492a92  #2908 sm90-streaming-signal  (SM90 plain / GLU / dGLU with streaming_signal_capacity)
# Idempotent; safe to rerun; one setup at a time per host (flock on $W/.setup_sig.lock; the package files under
# $W/.setup.lock).
#   Usage: bash setup.sh [sig]
#   Steps:
#     0. W=$HOME/cudnnfe_sm90/s59sm90regress: the package files (copied from this script's directory), md5-checked
#        against MD5SUMS; the kit clone's path is kept in $W/KIT_REPO;
#     1. src_sig = git archive 644492a92 (tree id checked) and the venv $HOME/venvs/cfe_pp_644492a9 (reused if it passes
#        the import check, else a copy of BASE_VENV (setup_base_venv.sh) with cuDNN-FE built from src_sig; an existing venv
#        that fails the check is never touched; shared with s59sm90hangrepro / s59sm90perf under one lock);
#     2. collect on this host (one GPU: the first of CUDA_VISIBLE_DEVICES, else 0) every test_grouped_gemm*sm90*.py file
#        plus test_grouped_gemm_bf16_overlap.py, twice:
#          -m L0          -> $W/collected_sig.txt       (strict / san jobs)
#          -m "L0 or L1"  -> $W/collected_full_sig.txt  (fshard jobs)
#        each = the SM90 files' ids + the other files' ids containing "sm90" (sorted).  setup fails only when a
#        collection errors or records no id.
#   A tree that passes all of it gets $W/ready_sig (holding the MD5SUMS md5); a ready tree is skipped on a rerun.
#   rc 0 iff ready. Log: $W/setup_<UTC>_<pid>.log; state: $W/setup_state/.
#   Env: KIT_REPO (the kit clone; default: this script's git checkout, else $W/KIT_REPO, else
#   $HOME/cudnnfe_sm90/kit-b/cudnn-frontend), BASE_VENV ($HOME/venvs/cfe_quack), CUDA_PATH (/usr/local/cuda-13.2).
#   Opens no ports, uploads nothing.
set -euo pipefail
RUN=s59sm90regress
BASE_VENV=${BASE_VENV:-$HOME/venvs/cfe_quack}
HERE=$(cd "$(dirname "$0")" && pwd)
W=$HOME/cudnnfe_sm90/$RUN; mkdir -p "$W"
TREES=${*:-"sig"}
for tr in $TREES; do case $tr in sig) ;; *) echo "usage: bash setup.sh [sig]"; exit 2 ;; esac; done
declare -A SHA=([sig]=644492a9)
declare -A FULLSHA=([sig]=644492a923c79bcae3b9214391b79a760a326742)
declare -A GITTREE=([sig]=4e5602db1098f6f2ac5b8fabc8fe45b39dc67465)
# the import check: modules the tree has / has not, and its SM90 public names (exported; the signature against the Sm100
# twin is printed, not asserted)
GG=unfused.moe_grouped_gemm_sm90; WG=wgrad.moe_grouped_gemm_wgrad_sm90; GL=glu.moe_grouped_gemm_glu_sm90; DG=dglu.moe_grouped_gemm_dglu_sm90
SS=moe_streaming_input_block_tile_scheduler
declare -A HAS=([sig]="$GG $GL $DG $SS")
declare -A HASNOT=([sig]="$WG")
declare -A NAMES=([sig]="GroupedGemm:grouped_gemm_wrapper GroupedGemmGlu:grouped_gemm_glu_wrapper GroupedGemmDglu:grouped_gemm_dglu_wrapper")
FILES="setup_base_venv.sh setup.sh job.sh rss_watchdog.py jobs.jsonl caps.json"
LOG=$W/setup_$(date -u +%Y%m%dT%H%M%SZ)_$$.log; ST=$W/setup_state; mkdir -p "$ST"
exec > >(tee -a "$LOG") 2>&1
echo "setup $RUN on $(hostname) trees: $TREES $(date -u +%FT%TZ)"
cd "$W"
# the kit clone (git delivery: the source trees come from its commits)
if [ -z "${KIT_REPO:-}" ]; then
  KIT_REPO=$(git -C "$HERE" rev-parse --show-toplevel 2> /dev/null || true)
  [ -n "$KIT_REPO" ] || KIT_REPO=$(cat "$W/KIT_REPO" 2> /dev/null || true)
  [ -n "$KIT_REPO" ] || KIT_REPO=$HOME/cudnnfe_sm90/kit-b/cudnn-frontend
fi
src_from_git() {  # src_from_git <full sha> <git tree id> <dest dir>: git archive of the commit (tree id checked)
  local got
  got=$(git -C "$KIT_REPO" rev-parse --verify -q "$1^{tree}" 2> /dev/null) || { echo "SETUP_FAIL commit $1 not in KIT_REPO=$KIT_REPO (clone the kit branch, see h100_kit_b/README.md)"; return 1; }
  [ "$got" = "$2" ] || { echo "SETUP_FAIL commit $1: tree $got, want $2"; return 1; }
  rm -rf "$3.tmp.$$"; mkdir "$3.tmp.$$"
  git -C "$KIT_REPO" archive "$1" | tar -x -C "$3.tmp.$$" || { echo "SETUP_FAIL git archive $1"; return 1; }
  mv "$3.tmp.$$" "$3"
}
# 0. the package files, md5-checked (a file already in W is never replaced: a mismatch is an error, not a re-fetch)
(
  flock 8
  [ -f MD5SUMS ] || { [ -f "$HERE/MD5SUMS" ] || { echo "SETUP_FAIL no MD5SUMS next to setup.sh"; exit 3; }; cp "$HERE/MD5SUMS" MD5SUMS.tmp.$$; mv MD5SUMS.tmp.$$ MD5SUMS; }
  for f in $FILES; do
    [ -f "$f" ] && continue
    [ -f "$HERE/$f" ] && [ "$HERE" != "$W" ] || { echo "SETUP_FAIL $f missing (run setup.sh from the kit clone first)"; exit 3; }
    cp "$HERE/$f" "$f.tmp.$$"; mv "$f.tmp.$$" "$f"
  done
  md5sum -c MD5SUMS > "$ST/md5_check.txt.$$" || { cat "$ST/md5_check.txt.$$"; echo "SETUP_FAIL md5 check"; exit 3; }
  mv "$ST/md5_check.txt.$$" "$ST/md5_check.txt"
  if [ -e "$KIT_REPO/.git" ]; then echo "$KIT_REPO" > KIT_REPO; fi
) 8> "$W/.setup.lock" || exit 3
echo "KIT_REPO=$KIT_REPO ($(git -C "$KIT_REPO" rev-parse --short HEAD 2> /dev/null || echo '?'))"
PKG_MD5=$(md5sum < MD5SUMS | cut -c1-32)
check_env() { CUDA_VISIBLE_DEVICES= "$1/bin/python" -I - "$2/python/cudnn/gemm/cutedsl/grouped" "$3" "$4" "$5" << 'EOF'
import hashlib, importlib, importlib.util, inspect, os, sys, cudnn, cutlass, torch, tvm_ffi, quack
ref_root, has, hasnot, names = sys.argv[1], sys.argv[2].split(), sys.argv[3].split(), [n.split(":") for n in sys.argv[4].split()]
P = "cudnn.gemm.cutedsl.grouped."
assert cudnn.__file__.startswith(sys.prefix), ("cudnn outside the venv", cudnn.__file__, sys.prefix)
def tree(root):
    return {os.path.relpath(os.path.join(d, f), root): hashlib.md5(open(os.path.join(d, f), "rb").read()).hexdigest()
            for d, _, fs in os.walk(root) for f in fs if f.endswith(".py")}
inst, ref = tree(os.path.join(os.path.dirname(cudnn.__file__), "gemm", "cutedsl", "grouped")), tree(ref_root)
assert inst == ref, ("installed cudnn/gemm/cutedsl/grouped is not this archive's",
                     sorted(set(inst) ^ set(ref)), sorted(k for k in set(inst) & set(ref) if inst[k] != ref[k]))
frost_inst = tree(os.path.join(os.path.dirname(cudnn.__file__), "frost"))
frost_ref = tree(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(ref_root))), "frost"))  # <src>/python/cudnn/frost
assert frost_inst == frost_ref, "installed cudnn/frost is not this archive's"
mods = {m: importlib.import_module(P + m) for m in has}
for m in hasnot:
    assert importlib.util.find_spec(P + m) is None, f"{m} must not exist in this tree"
import cudnn.gemm.cutedsl.grouped as grouped
sigs = []
for cls, fn in names:  # the SM90 names: exported (the Sm100 twin's signature: reported)
    for a, b in ((cls + "Sm100", cls + "Sm90"), (fn + "_sm100", fn + "_sm90")):
        y = getattr(cudnn, b)
        assert b in grouped.__all__ and y is getattr(grouped, b), f"{b} not exported"
        x = getattr(cudnn, a, None)
        sigs.append(f"{b}:{'same' if x is not None and inspect.signature(x) == inspect.signature(y) else 'differs'}")
print("cudnn", cudnn.__file__, "backend", cudnn.backend_version(), "cutlass", cutlass.__version__, "torch", torch.__version__,
      "tvm_ffi", tvm_ffi.__version__, "quack", quack.__version__, "grouped .py files", len(inst), "frost .py files", len(frost_inst),
      "has", has, "has not", hasnot, "signature vs Sm100", sigs)
EOF
}
build_tree() {  # build_tree <tree>: src_<tree> + cfe_pp_<sha>; $ST/venv_<tree>.ok or .failed
  local tr=$1 sha=${SHA[$1]} V T TP CP
  V=$HOME/venvs/cfe_pp_$sha
  rm -f "$ST/venv_$tr.ok" "$ST/venv_$tr.failed"
  [ -d "src_$tr" ] || src_from_git "${FULLSHA[$tr]}" "${GITTREE[$tr]}" "$W/src_$tr" || { touch "$ST/venv_$tr.failed"; return 0; }
  mkdir -p "$HOME/venvs"
  (
    flock 9
    if [ -d "$V" ] && check_env "$V" "$W/src_$tr" "${HAS[$tr]}" "${HASNOT[$tr]}" "${NAMES[$tr]}" > "$ST/env_check_$tr.txt" 2>&1; then echo "reusing $V"; exit 0; fi
    [ -d "$V" ] && { echo "$V exists but fails the import check (see env_check_$tr.txt); move it away and rerun"; exit 4; }
    if ! BASE_VENV=$BASE_VENV bash "$W/setup_base_venv.sh" > "$ST/base_venv_$tr.txt" 2>&1; then
      echo "BASE_VENV setup failed (see base_venv_$tr.txt and $(dirname "$BASE_VENV")/.cfe_quack_setup.log)"; exit 4; fi
    [ -x "$BASE_VENV/bin/python" ] || { echo "no $BASE_VENV"; exit 4; }
    echo "building $V from $BASE_VENV"
    T=$HOME/venvs/.cfe_pp.tmp.$(hostname).$$.$tr; rm -rf "$T"; cp -a "$BASE_VENV" "$T"; rm -f "$T/.cfe_quack_ok" "$T/.cfe_quack_building" "$T/.qcmp_ok"
    TP=$(realpath "$("$T/bin/python" -I -c 'import sys; print(sys.prefix)')")
    [ "$TP" = "$(realpath "$T")" ] || { echo "sys.prefix '$TP' is not $T"; exit 4; }
    "$T/bin/python" -m pip --version > /dev/null 2>&1 || "$T/bin/python" -m ensurepip > /dev/null 2>&1
    "$T/bin/python" -m pip uninstall -y nvidia-cudnn-frontend > "$ST/cudnn_build_$tr.txt" 2>&1 || true
    "$T/bin/python" -m pip install --force-reinstall --no-deps cmake==3.31.6 ninja==1.11.1.1 >> "$ST/cudnn_build_$tr.txt" 2>&1
    "$T/bin/python" -m pip install setuptools==80.9.0 wheel==0.45.1 cmake==3.31.6 ninja==1.11.1.1 nanobind==3.0.1 >> "$ST/cudnn_build_$tr.txt" 2>&1
    CP=${CUDA_PATH:-/usr/local/cuda-13.2}; [ -d "$CP" ] || CP=/gbr/root/usr/local/cuda-13.2
    [ -d "$CP" ] || { echo "no CUDA 13.2 at /usr/local/cuda-13.2 or /gbr/root/usr/local/cuda-13.2 (set CUDA_PATH)"; exit 5; }
    CUDNN_PATH=$("$T/bin/python" -I -c "import importlib.util; print(importlib.util.find_spec('nvidia.cudnn').submodule_search_locations[0])") \
      CUDA_PATH=$CP VIRTUAL_ENV=$T PATH=$T/bin:$PATH \
      "$T/bin/python" -m pip install --no-deps --no-build-isolation "$W/src_$tr" >> "$ST/cudnn_build_$tr.txt" 2>&1 || { tail -20 "$ST/cudnn_build_$tr.txt"; echo "cudnn-fe build failed; left at $T"; exit 5; }
    check_env "$T" "$W/src_$tr" "${HAS[$tr]}" "${HASNOT[$tr]}" "${NAMES[$tr]}" > "$ST/env_check_$tr.txt" 2>&1 || { cat "$ST/env_check_$tr.txt"; echo "new venv fails the import check; left at $T"; exit 4; }
    [ -e "$V" ] || mv "$T" "$V"
    rm -rf "$T"
  ) 9> "$HOME/venvs/.cfe_pp_$sha.lock" > "$ST/venv_build_$tr.txt" 2>&1 || { echo "venv $tr ($sha) FAILED: $(tail -3 "$ST/venv_build_$tr.txt" | tr '\n' ' ')"; touch "$ST/venv_$tr.failed"; return 0; }
  check_env "$V" "$W/src_$tr" "${HAS[$tr]}" "${HASNOT[$tr]}" "${NAMES[$tr]}" > "$ST/env_$tr.txt" 2>&1 || { echo "venv $tr ($sha) fails the check after the build"; touch "$ST/venv_$tr.failed"; return 0; }
  "$V/bin/python" -m pip freeze > "$ST/pip_freeze_$tr.txt" 2> /dev/null || true
  echo "venv $tr ($sha) OK: $V"; touch "$ST/venv_$tr.ok"
}
G0=${CUDA_VISIBLE_DEVICES:-0}; G0=${G0%%,*}
collect() {  # collect <tree> <marker expr> <out>: on-host collection; record the ids; rc 0 iff no error and > 0 ids
  local tr=$1 mark=$2 out=$3 tag=$4 crc=0 n nall nerr files
  files=$(cd "$W/src_$tr/test/python" && ls gemm/cutedsl/test_grouped_gemm*sm90*.py gemm/cutedsl/test_grouped_gemm_bf16_overlap.py 2> /dev/null | tr '\n' ' ' || true)
  [ -f "$W/src_$tr/test/python/gemm/cutedsl/test_grouped_gemm_wgrad.py" ] && files="$files gemm/cutedsl/test_grouped_gemm_wgrad.py"
  (cd "$W/src_$tr/test/python" && CUDA_VISIBLE_DEVICES=$G0 PYTHONNOUSERSITE=1 "$HOME/venvs/cfe_pp_${SHA[$tr]}/bin/python" -m pytest -p no:cacheprovider \
     -m "$mark" -q --collect-only $files) > "$ST/collect_${tag}_$tr.log" 2> "$ST/collect_stderr_${tag}_$tr.log" || crc=$?
  grep '::' "$ST/collect_${tag}_$tr.log" | sort > "$ST/collect_all_${tag}_$tr.txt" || true
  awk -F'::' '$1 ~ /sm90[^\/]*\.py$/' "$ST/collect_all_${tag}_$tr.txt" > "$ST/$out.tmp" || true
  awk -F'::' '$1 !~ /sm90[^\/]*\.py$/' "$ST/collect_all_${tag}_$tr.txt" | grep -i 'sm90' >> "$ST/$out.tmp" || true
  sort -o "$ST/$out.tmp" "$ST/$out.tmp"
  n=$(wc -l < "$ST/$out.tmp"); nall=$(wc -l < "$ST/collect_all_${tag}_$tr.txt")
  nerr=$(cat "$ST/collect_${tag}_$tr.log" "$ST/collect_stderr_${tag}_$tr.log" | { grep -cE '^(ERROR|INTERNALERROR)|errors? during collection' || true; })
  echo "collect $tr -m '$mark': rc $crc, $nall ids in the files, $n recorded (SM90 files + the others' sm90 ids), error lines $nerr"
  awk -F'::' '{print $1}' "$ST/$out.tmp" | sort | uniq -c | sed "s/^/  collected_file $tag $tr /"
  if [ "$crc" = 0 ] && [ "$nerr" = 0 ] && [ "$n" -gt 0 ]; then mv "$ST/$out.tmp" "$W/$out"; return 0; fi
  { tail -n 20 "$ST/collect_${tag}_$tr.log" "$ST/collect_stderr_${tag}_$tr.log" | cut -c1-300; } || true
  echo "SETUP_FAIL collect $tr ($tag): rc $crc, error lines $nerr, $n ids"
  return 1
}
do_tree() {  # do_tree <tree>: under $W/.setup_<tree>.lock: ready check, build, both collections, ready_<tree>
  local tr=$1
  (
    flock 7
    if [ "$(cat "ready_$tr" 2> /dev/null)" = "$PKG_MD5" ] && [ -d "$HOME/venvs/cfe_pp_${SHA[$tr]}" ] && [ -d "src_$tr" ] \
       && [ -s "collected_$tr.txt" ] && [ -s "collected_full_$tr.txt" ]; then echo "tree $tr ready"; exit 0; fi
    rm -f "ready_$tr"
    build_tree "$tr"
    [ -f "$ST/venv_$tr.ok" ] || { echo "SETUP_FAIL venv $tr (see $ST/venv_build_$tr.txt)"; exit 1; }
    collect "$tr" L0 "collected_$tr.txt" l0 || exit 1
    collect "$tr" "L0 or L1" "collected_full_$tr.txt" full || exit 1
    echo "$PKG_MD5" > "ready_$tr"
    echo "tree $tr ready: venv OK, $(wc -l < "collected_$tr.txt") L0 ids, $(wc -l < "collected_full_$tr.txt") L0+L1 ids"
  ) 7> "$W/.setup_$tr.lock"
}
PIDS=(); TRS=()
for tr in $TREES; do do_tree "$tr" & PIDS+=($!); TRS+=("$tr"); done
fail=0
for i in "${!PIDS[@]}"; do wait "${PIDS[$i]}" || { echo "SETUP_FAIL tree ${TRS[$i]}"; fail=1; }; done  # explicit pids: a bare wait also waits for the tee
echo "setup done rc=$fail $(date -u +%FT%TZ)"
exit $fail
