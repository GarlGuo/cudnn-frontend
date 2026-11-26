#!/bin/bash
# job.sh: one job of the s59sm90perf package (kit branch wip/sm90-h100-kit-b, h100_kit_b/s59sm90perf/), run by the h100
# coordinator's dispatcher.  s57sm90plainperf3's job.sh (same-GPU interleaved A/B, stall watchdog, no periodic dump) with
# base = 644492a92 (venv cfe_pp_644492a9), old = 228f92bab ($W/site_old), ctl = base again (A/A), and no S3:
#   bash job.sh <job> <timeout_s> <rss_gb>
#     plans.json[<job>]: one dGLU shape, cells ready (needed: the base role's ready time calibrates the delayed producer)
#     and delay1.0; every role runs worker90.py as its own process on the first visible GPU (delayed cells: base's
#     producer cycles, shared), taking turns (one process launches at a time; pass order reversed every other pass: ABBA),
#     until the deadline (timeout - 240 s).  CUDA_MODULE_LOADING=EAGER.
#   Writes JOB_OUT/status.txt (PASS | FAIL | TIMEOUT | RSS_KILL | SETUP_FAIL | FOREIGN | KILLED); PASS: every role exited 0,
#   no COMPILE_FAIL / RUN_FAIL / SAMPLE_FAIL / STREAMED_DIFFERS line, the real rows of D equal across roles, GPU isolated.
#   When not PASS, JOB_OUT/fail_head.txt.  No upload: JOB_OUT is copied to $W/results/<JOB_ID>_a<ATTEMPT>/ (unless already
#   under $W) for report.py (which computes base/old and ctl/base per cell from the role logs).  Exits 0 iff PASS.
set -uo pipefail
RUN=s59sm90perf
SHA=644492a9
W=$HOME/cudnnfe_sm90/$RUN
JOB=${1:-}; TMO=${2:-}; RSS_GB=${3:-}; MODEL=$JOB; EP=-; DIR=-
case "$JOB" in *[!A-Za-z0-9_.-]* | "") echo "bad job '$JOB'"; exit 2 ;; esac
case "$TMO$RSS_GB" in *[!0-9]* | "") echo "timeout_s and rss_gb must be integers"; exit 2 ;; esac
JOB_ID=${JOB_ID:-local_${JOB}_$$}; ATTEMPT=${ATTEMPT:-1}
JOB_OUT=${JOB_OUT:-$W/job_out/${JOB_ID}_a$ATTEMPT}; mkdir -p "$JOB_OUT" || exit 2
LOG=$JOB_OUT/job.log; PGD=$JOB_OUT/pg; mkdir -p "$PGD"
START=$(date +%s); ST=""; RC=-; SUMM=""; PEAK_KB=0; EXTRA_STATUS=()
fail_head() {  # the first failure lines of a non-PASS job (report.py prints them)
  {
    echo "== status: $ST: $SUMM"
    for f in "$JOB_OUT"/*.log; do
      [ -s "$f" ] || continue
      echo "== $(basename "$f"): first failure lines"
      grep -m 20 -E 'STALL|BAD |COMPILE_FAIL|RUN_FAIL|SAMPLE_FAIL|STREAMED_DIFFERS|Traceback|Error|error:|SETUP_FAIL|Fatal|Segmentation|no barrier|ext_stall|SUMMARY|timeout' "$f" | cut -c1-300
    done
  } > "$JOB_OUT/fail_head.txt" 2> /dev/null || true
}
finish() {  # once: status.txt, fail_head.txt, the copy to $W/results, the exit code (0 iff PASS)
  local dst
  trap - EXIT
  touch "$PGD/STOP"; for f in "$PGD"/[0-9]*; do [ -e "$f" ] || continue; case $f in *.*) continue ;; esac; kill -KILL -- "-${f##*/}" 2> /dev/null || true; done
  [ -n "$ST" ] || { ST=SETUP_FAIL; SUMM=${SUMM:-job.sh stopped early (see job.log)}; }
  {
    echo "$ST"
    echo "job_id=$JOB_ID"; echo "attempt=$ATTEMPT"; echo "run=$RUN"; echo "sha=$SHA"; echo "model=$MODEL"; echo "ep=$EP"; echo "tokens=$DIR"
    echo "rc=$RC"; echo "secs=$(( $(date +%s) - START ))"
    echo "peak_rss_gb=$(awk -v k="$PEAK_KB" 'BEGIN{printf "%.1f", k / 1048576}')"; echo "rss_cap_gb=$RSS_GB"; echo "timeout_s=$TMO"
    echo "summary=$SUMM"; echo "host=$(hostname)"; echo "gpus=${CUDA_VISIBLE_DEVICES:-}"; echo "result_key=${RESULT_KEY:-}"; echo "job_out=$JOB_OUT"
    echo "end=$(date -u +%FT%TZ)"
    for l in "${EXTRA_STATUS[@]+"${EXTRA_STATUS[@]}"}"; do echo "$l"; done
  } > "$JOB_OUT/status.txt"
  rm -rf "${PGD:?}"
  echo "$(date -u +%FT%TZ) status $ST rc=$RC $SUMM" >> "$LOG"
  [ "$ST" = PASS ] || fail_head
  case "$JOB_OUT" in "$W"/*) ;; *)
    dst=$W/results/${JOB_ID}_a$ATTEMPT; mkdir -p "$W/results"; rm -rf "$dst.tmp"
    cp -a "$JOB_OUT" "$dst.tmp" 2> /dev/null && { rm -rf "$dst"; mv "$dst.tmp" "$dst"; } || echo "copy to $dst failed" >> "$LOG" ;;
  esac
  echo "$ST $JOB_ID ${SUMM}"
  [ "$ST" = PASS ] && exit 0
  exit 1
}
trap finish EXIT
trap 'ST=${ST:-KILLED}; SUMM="${SUMM:-killed by a signal}"; exit 1' TERM INT HUP
echo "$(date -u +%FT%TZ) $RUN job $JOB_ID attempt $ATTEMPT on $(hostname): $MODEL ep$EP $DIR timeout=${TMO}s rss_cap=${RSS_GB}G gpus=${CUDA_VISIBLE_DEVICES:-}" | tee "$LOG"
if [ "$(cat "$W/ready" 2> /dev/null)" != "$(md5sum < "$W/MD5SUMS" 2> /dev/null | cut -c1-32)" ]; then
  echo "tree not ready: running setup.sh" >> "$LOG"
  SETUP_SH=$W/setup.sh; [ -f "$SETUP_SH" ] || SETUP_SH=$(cd "$(dirname "$0")" && pwd)/setup.sh
  bash "$SETUP_SH" >> "$JOB_OUT/setup.log" 2>&1 || { ST=SETUP_FAIL; SUMM="setup.sh failed: $(grep SETUP_FAIL "$JOB_OUT/setup.log" | tail -1)"; exit 1; }
fi
V=$HOME/venvs/cfe_pp_$SHA
export PYTHONNOUSERSITE=1
export QUACK_CACHE_AUTOTUNING=1 QUACK_CACHE_DIR=${QUACK_CACHE_DIR:-$HOME/cudnnfe_sm90/quack_cache_q065_$(hostname -s)}
mkdir -p "$QUACK_CACHE_DIR"
unset CUTE_DSL_ARCH CUTE_DSL_KEEP CUTE_DSL_DUMP_DIR CUTE_DSL_COMPILER_OPT QUACK_ARCH PYTHONPATH VIRTUAL_ENV CUDNN_DETERMINISM_GOLDENS CUDNN_DETERMINISM_RECORD QUACK_FORCE_CACHE_UPDATE PYTORCH_CUDA_ALLOC_CONF
export CUDA_MODULE_LOADING=EAGER
pgid_of() { local s; s=$(cat "/proc/$1/stat" 2> /dev/null) || return 1; s=${s##*) }; set -- $s; echo "$3"; }
usr1_workers() {  # usr1_workers <pgid>: SIGUSR1 (faulthandler dump) to the group's python worker processes only (/proc; no pkill)
  local d c
  for d in /proc/[0-9]*; do
    [ "$(pgid_of "${d#/proc/}" 2> /dev/null)" = "$1" ] || continue
    c=$(tr '\0' ' ' < "$d/cmdline" 2> /dev/null) || continue
    case $c in *python*worker90.py*) case ${c%% *} in *python*) kill -USR1 "${d#/proc/}" 2> /dev/null || true ;; esac ;; esac
  done
}
run_pg() {  # run_pg <timeout s> <log> <dir> cmd...: own session / process group, RSS watchdog, TERM/KILL the whole group
  local tmo=$1 log=$2 dir=$3 pid pg t=0 rc=0 timed_out=0; shift 3
  setsid bash -c 'cd "$1" && shift && exec "$@"' run_pg "$dir" "$@" >> "$log" 2>&1 < /dev/null &
  pid=$!
  pg=$(pgid_of "$pid" || true); [ -n "$pg" ] || pg=$pid
  [ "$pg" = "$(pgid_of $$ || true)" ] && pg=$pid
  rm -f "$PGD/$pg".*; echo "$log" > "$PGD/$pg"
  while kill -0 "$pid" 2> /dev/null; do
    [ -f "$PGD/$pg.rsskill" ] && break
    if [ "$t" -ge "$tmo" ]; then
      timed_out=1; echo "run_pg: timeout ${tmo}s, killing process group $pg" >> "$log"
      usr1_workers "$pg"; sleep 5
      kill -TERM -- "-$pg" 2> /dev/null || true; sleep 30; kill -KILL -- "-$pg" 2> /dev/null || true; break
    fi
    ext_stall_check
    sleep 5; t=$((t + 5))
  done
  wait "$pid" 2> /dev/null || rc=$?
  kill -KILL -- "-$pg" 2> /dev/null || true
  PEAK_KB=$(cat "$PGD/$pg.peak" 2> /dev/null || echo 0)
  if [ -f "$PGD/$pg.rsskill" ]; then echo "run_pg: RSS watchdog killed process group $pg: $(cat "$PGD/$pg.rsskill")" >> "$log"; rc=125
  elif [ $timed_out = 1 ]; then rc=124; fi
  rm -f "$PGD/$pg" "$PGD/$pg".*
  return $rc
}
# The external fallback: a worker whose loop started (armed.<x> holds its pid) and whose watchdog thread stopped writing
# wd.<x> for 150 s: the watchdog thread cannot run (the main thread holds the GIL in a blocked C call).  Dump through
# faulthandler's SIGUSR1 handler (the main thread is blocked, so its frames are not moving), wait, SIGKILL; ext_stall.<x>.
EXT_DIR=""
ext_stall_check() {
  local a x pid wd age
  [ -n "$EXT_DIR" ] || return 0
  for a in "$EXT_DIR"/armed.*; do
    [ -e "$a" ] || continue
    x=${a##*/armed.}; pid=$(cat "$a" 2> /dev/null); wd=$EXT_DIR/wd.$x
    [ -n "$pid" ] && kill -0 "$pid" 2> /dev/null || continue
    [ -e "$EXT_DIR/ext_stall.$x" ] && continue
    [ -e "$wd" ] || wd=$a
    age=$(( $(date +%s) - $(stat -c %Y "$wd") ))
    if [ "$age" -gt 150 ]; then
      echo "$(date -u +%FT%TZ) ext_stall: $x pid $pid watchdog file ${age}s old: SIGUSR1 (faulthandler dump), then SIGKILL" | tee -a "$LOG" >> "$EXT_DIR/ext_stall.$x"
      nvidia-smi -q -d PERFORMANCE,UTILIZATION > "$JOB_OUT/nvsmi_stall_$x.txt" 2>&1 || true
      kill -USR1 "$pid" 2> /dev/null || true; sleep 10; kill -KILL "$pid" 2> /dev/null || true
    fi
  done
}
python3 -I "$W/rss_watchdog.py" "$PGD" "$RSS_GB" 0 2 > /dev/null 2>&1 &
G0=${CUDA_VISIBLE_DEVICES:-0}; G0=${G0%%,*}
busy_gpu() {
  local mem apps
  mem=$(nvidia-smi -i "$G0" --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | tr -d ' ' || echo 0)
  apps=$(nvidia-smi -i "$G0" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2> /dev/null | tr '\n' ';' || true)
  { [ -n "$apps" ] || [ "${mem:-0}" -gt "${IDLE_MEM_MIB:-2048}" ]; } && echo "gpu$G0(${mem}MiB: ${apps:-no process})"
  return 0
}
b=$(busy_gpu); [ -n "$b" ] && { ST=FOREIGN; SUMM="before the run: $b"; exit 1; }
EXTRA_STATUS+=("isolated_before=$(date -u +%FT%TZ)")
nvidia-smi --query-gpu=index,name,driver_version,memory.used,clocks.max.sm,clocks.max.mem,clocks.sm,clocks.mem,power.limit,temperature.gpu,clocks_throttle_reasons.active --format=csv > "$JOB_OUT/gpus.txt" 2>&1 || true
nvidia-smi -i "$G0" -q -d CLOCK,POWER,PERFORMANCE > "$JOB_OUT/gpu_clocks_before.txt" 2>&1 || true
"$V/bin/python" -m pip freeze > "$JOB_OUT/pip_freeze.txt" 2> /dev/null || true
RLOG=$JOB_OUT/run.log
"$V/bin/python" -I -c "import json, sys; json.dump(json.load(open(sys.argv[1]))[sys.argv[2]], open(sys.argv[3], 'w'))" "$W/plans.json" "$JOB" "$JOB_OUT/plan.json" \
  || { ST=FAIL; SUMM="no plan for job $JOB"; exit 1; }
SY=$JOB_OUT/sync; rm -rf "$SY"; mkdir -p "$SY"; echo 0 > "$SY/turn"
EXT_DIR=$SY
{
  echo '#!/bin/bash'
  echo "# every role of the plan as its own process on GPU $G0; exit codes in $SY/exit.<role>"
  echo "for role in \$(\"$V/bin/python\" -I -c \"import json; print(' '.join(json.load(open('$JOB_OUT/plan.json'))['roles']))\"); do"
  echo "  pp=''; [ \"\$role\" != base ] && pp=\"$W/site_\$role\""
  echo "  ( PYTHONPATH=\$pp CUDA_VISIBLE_DEVICES=$G0 END_AT=\$(( \$(date +%s) + $TMO - 240 )) \"$V/bin/python\" -u \"$W/worker90.py\" \"\$role\" \"$JOB_OUT/plan.json\" \"$SY\" > \"$JOB_OUT/\$role.log\" 2>&1; echo \$? > \"$SY/exit.\$role\" ) &"
  echo "done"
  echo "wait"
} > "$JOB_OUT/roles.sh"
rc=0
run_pg "$TMO" "$RLOG" "$W" bash "$JOB_OUT/roles.sh" || rc=$?
RC=$rc
os=""
for f in "$SY"/exit.*; do
  [ -e "$f" ] || continue; r=${f##*.}; e=$(cat "$f")
  if [ -e "$SY/ext_stall.$r" ]; then o=STALL_EXT; else case $e in 0) o=OK ;; 86) o=STALL ;; 139) o=SEGV ;; *) o=RC$e ;; esac; fi
  echo "role $r exit $e $o" >> "$RLOG"; os="$os $r=$o"; [ "$e" = 0 ] || [ $rc -ne 0 ] || rc=1
done
EXTRA_STATUS+=("outcome=$os")
nvidia-smi -i "$G0" -q -d CLOCK,POWER,PERFORMANCE > "$JOB_OUT/gpu_clocks_after.txt" 2>&1 || true
b=$(busy_gpu); [ -n "$b" ] && EXTRA_STATUS+=("foreign_after=$b")
fl=$(grep -hE "^(COMPILE_FAIL|RUN_FAIL|SAMPLE_FAIL|STREAMED_DIFFERS|STALL) " "$JOB_OUT"/*.log 2> /dev/null | head -3 | cut -c1-200 | tr '\n' ' ')
nck=$("$V/bin/python" -I -c "
import json, sys, glob, collections
by = collections.defaultdict(set)
n = 0
for f in glob.glob(sys.argv[1] + '/*.log'):
    for ln in open(f, errors='replace'):
        if ln.startswith('CHECKSUM '):
            d = json.loads(ln.split(' ', 1)[1]); c = d['cid'].split('|'); by[c[0] + '|' + '|'.join(c[2:])].add(json.dumps(d['rows'])); n += 1
print(sum(1 for v in by.values() if len(v) > 1) if n else 98)" "$JOB_OUT" 2> /dev/null || echo 99)
[ "$nck" = 0 ] || fl="$fl D bytes differ across roles in $nck groups (98: no checksum)"
if [ $rc -eq 125 ]; then ST=RSS_KILL
elif [ $rc -eq 124 ] || [ $rc -eq 137 ]; then ST=TIMEOUT
elif [ $rc -ne 0 ] || [ -n "$fl" ]; then ST=FAIL; SUMM="rc=$rc ${fl:-$(tail -1 "$RLOG" 2> /dev/null)}"
elif [ -n "$b" ]; then ST=FOREIGN; SUMM="after the run: $b"
else ST=PASS; SUMM="every role ran, D of the real rows byte-identical across roles and modes, isolated"; fi
sz=$(stat -c %s "$RLOG" 2> /dev/null || echo 0)
if [ "$sz" -gt 8000000 ]; then { head -c 4000000 "$RLOG"; echo; echo "[... truncated $sz bytes ...]"; tail -c 4000000 "$RLOG"; } > "$RLOG.t" && mv "$RLOG.t" "$RLOG"; fi
exit 0
