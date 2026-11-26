#!/bin/bash
# job.sh: one job of the s59sm90hangrepro package (kit branch wip/sm90-h100-kit-b, h100_kit_b/s59sm90hangrepro/), run by
# the h100 coordinator's dispatcher (jobs.jsonl "cmd").  s58sm90hangrepro's job.sh (the 58-case hang repro of the SM90
# dGLU streaming signal cell dglu / deepseek_v3 / EP4 / T4096) with the trees moved one head forward and no S3:
#   base = 644492a92 (venv cfe_pp_644492a9), old = 228f92bab ($W/site_old).  Reference checksum unchanged (s57/s58:
#   [-6823832267924107, -2178886971186175923]); producer cycles fixed at s57's shared calibration (hangrepro.py).
#   bash job.sh solo    <tree> <mode> <secs> <timeout_s> <rss_gb>   hangrepro.py <tree> <mode>: one process, back-to-back
#                                                                     launches for <secs>; mode ready | delay1.0 | delay0.5
#   bash job.sh fh      <tree> <mode> <secs> <period> <timeout_s> <rss_gb>
#                                                                     solo + faulthandler.dump_traceback_later(<period>,
#                                                                     repeat=True): does the periodic dump alone SIGSEGV?
#                                                                     (PASS either way; outcome= says SEGV or OK)
#   bash job.sh replica <secs> <timeout_s> <rss_gb>                  s57's job for this cell (3 roles base/old/ctl taking
#                                                                     turns, replica_plan.json) with worker90r.py (no
#                                                                     periodic dump, stall watchdog)
#   bash job.sh san     <tool> <tree> <timeout_s> <rss_gb>            compute-sanitizer <tool> (racecheck | synccheck) on
#                                                                     hangrepro.py <tree> san (ready + prestaged launches)
# Stall handling (never a periodic dump): the worker's watchdog thread exits rc 86 after printing every thread's stack
# (STALL line) when the loop makes no progress for 60 s; fallback here: when the watchdog thread itself stops (its
# wd.* file > 150 s old after the loop started), SIGUSR1 (faulthandler.register: a C-level dump of a blocked process),
# then SIGKILL, recorded as STALL_EXT.  CUDA_LAUNCH_BLOCKING unset.  CUDA_MODULE_LOADING=EAGER.
# Writes JOB_OUT/status.txt: PASS | FAIL | HAZARD | DEVICE_ERROR | TIMEOUT | RSS_KILL | SETUP_FAIL | FOREIGN | KILLED, and
#   outcome=OK | STALL | STALL_EXT | SEGV | BAD | ... per job; when not PASS, JOB_OUT/fail_head.txt.  No upload: JOB_OUT
#   is copied to $W/results/<JOB_ID>_a<ATTEMPT>/ (unless already under $W) for report.py.  Exits 0 iff PASS.
set -uo pipefail
RUN=s59sm90hangrepro
SHA=644492a9
W=$HOME/cudnnfe_sm90/$RUN
KIND=${1:-}; shift || true
case "$KIND" in
  solo) TREE=${1:-}; MODE=${2:-}; SECS=${3:-}; TMO=${4:-}; RSS_GB=${5:-}; PERIOD=0 ;;
  fh) TREE=${1:-}; MODE=${2:-}; SECS=${3:-}; PERIOD=${4:-}; TMO=${5:-}; RSS_GB=${6:-} ;;
  replica) TREE=all; MODE=replica; SECS=${1:-}; TMO=${2:-}; RSS_GB=${3:-}; PERIOD=0 ;;
  san) TOOL=${1:-}; TREE=${2:-}; MODE=san; SECS=0; TMO=${3:-}; RSS_GB=${4:-}; PERIOD=0
    case $TOOL in racecheck | synccheck) ;; *) echo "bad tool '$TOOL'"; exit 2 ;; esac ;;
  *) echo "bad kind '$KIND'"; exit 2 ;;
esac
case "$TREE" in base | old | all) ;; *) echo "bad tree '$TREE'"; exit 2 ;; esac
case "$MODE" in ready | delay1.0 | delay0.5 | replica | san) ;; *) echo "bad mode '$MODE'"; exit 2 ;; esac
case "$SECS$TMO$RSS_GB" in *[!0-9]* | "") echo "secs, timeout_s and rss_gb must be integers"; exit 2 ;; esac
case "$PERIOD" in *[!0-9.]* | "") echo "bad period '$PERIOD'"; exit 2 ;; esac
MODEL="$KIND.$TREE.$MODE${TOOL:+.$TOOL}"; EP=4; DIR=4096
JOB_ID=${JOB_ID:-local_${KIND}_${TREE}_${MODE}_$$}; ATTEMPT=${ATTEMPT:-1}
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
echo "$(date -u +%FT%TZ) $RUN job $JOB_ID attempt $ATTEMPT on $(hostname): $MODEL secs=$SECS period=$PERIOD timeout=${TMO}s rss_cap=${RSS_GB}G gpus=${CUDA_VISIBLE_DEVICES:-}" | tee "$LOG"
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
unset CUDA_LAUNCH_BLOCKING PYTHONFAULTHANDLER
export CUDA_MODULE_LOADING=EAGER
pgid_of() { local s; s=$(cat "/proc/$1/stat" 2> /dev/null) || return 1; s=${s##*) }; set -- $s; echo "$3"; }
usr1_workers() {  # usr1_workers <pgid>: SIGUSR1 (faulthandler dump) to the group's python worker processes only (/proc; no pkill)
  local d c
  for d in /proc/[0-9]*; do
    [ "$(pgid_of "${d#/proc/}" 2> /dev/null)" = "$1" ] || continue
    c=$(tr '\0' ' ' < "$d/cmdline" 2> /dev/null) || continue
    case $c in *python*hangrepro.py* | *python*worker90r.py*) case ${c%% *} in *python*) kill -USR1 "${d#/proc/}" 2> /dev/null || true ;; esac ;; esac
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
pp_of() { [ "$1" = base ] && echo "" || echo "$W/site_$1"; }
rc_name() {  # rc_name <exit code> [ext marker]: the outcome of one worker process
  if [ -n "${2:-}" ] && [ -e "$2" ]; then echo STALL_EXT; return; fi
  case $1 in 0) echo OK ;; 86) echo STALL ;; 139 | -11) echo SEGV ;; 134) echo ABORT ;; 1) echo BAD_OR_ERROR ;; *) echo "RC$1" ;; esac; }
tree_ok() {  # tree_ok <log> <tree>: the TREE line names the expected cudnn
  local f; f=$(grep -m1 '^TREE ' "$1" | sed 's/.*"cudnn": "\([^"]*\)".*/\1/')
  case $2 in base | ctl) case $f in "$V"/*) return 0 ;; esac ;; *) case $f in "$W/site_$2"/*) return 0 ;; esac ;; esac
  echo "tree $2 imported cudnn from '$f'"; return 1; }
case $KIND in
  solo | fh)
    WL=$JOB_OUT/worker.log; EXT_DIR=$JOB_OUT
    fhenv=(); [ "$KIND" = fh ] && fhenv=(FH_STRESS="$PERIOD")
    rc=0
    run_pg "$TMO" "$RLOG" "$W" env PYTHONPATH="$(pp_of "$TREE")" CUDA_VISIBLE_DEVICES="$G0" STALL_S=60 ${fhenv[@]+"${fhenv[@]}"} \
      bash -c '"$1" -u "$2" "$3" "$4" "$5" "$6" > "$7" 2>&1' x "$V/bin/python" "$W/hangrepro.py" "$TREE" "$MODE" "$SECS" "$JOB_OUT" "$WL" || rc=$?
    RC=$rc
    o=$(rc_name "$rc" "$JOB_OUT/ext_stall.$TREE.$MODE")
    [ $rc -eq 124 ] && o=TIMEOUT; [ $rc -eq 125 ] && o=RSS_KILL
    bad=$(grep -c '^BAD ' "$WL" 2> /dev/null); bad=${bad:-0}
    ck=$(grep -m1 '^CHECKSUM ' "$WL" | grep -c '"rows": \[-6823832267924107, -2178886971186175923\]')
    done_l=$(grep '^DONE ' "$WL" | tail -1 | cut -c1-200)
    launches=$(grep '^WIN ' "$WL" | tail -1 | sed 's/.*"launches": \([0-9]*\).*/\1/'); launches=${launches:-0}
    stall=$(grep -m1 '^STALL ' "$WL" | cut -c1-300)
    dumps=$(grep -c '^Timeout (' "$WL" 2> /dev/null); dumps=${dumps:-0}
    tm=$(tree_ok "$WL" "$TREE") || o="$o TREE_MISMATCH($tm)"
    [ "$ck" = 1 ] || o="$o REF_CHECKSUM_DIFFERS"
    [ "$bad" = 0 ] || o="$o BAD=$bad"
    EXTRA_STATUS+=("outcome=$o" "launches=$launches" "bad=$bad" "fh_dumps=$dumps" "worker_rc=$rc")
    [ -n "$stall" ] && EXTRA_STATUS+=("stall=$stall")
    if [ "$KIND" = fh ]; then
      case $o in SEGV | OK) ST=PASS ;; TIMEOUT) ST=TIMEOUT ;; RSS_KILL) ST=RSS_KILL ;; *) ST=FAIL ;; esac
      SUMM="fh period ${PERIOD}s: $o after $dumps periodic dumps, $launches launches ${stall:+; $stall}"
    else
      case $o in OK) ST=PASS ;; TIMEOUT) ST=TIMEOUT ;; RSS_KILL) ST=RSS_KILL ;; *) ST=FAIL ;; esac
      SUMM="$TREE $MODE: $o, $launches launches, $bad bad checksums${stall:+; $stall}${done_l:+; $done_l}"
    fi
    ;;
  replica)
    "$V/bin/python" -I -c "import json, sys; json.dump(json.load(open(sys.argv[1])), open(sys.argv[2], 'w'))" "$W/replica_plan.json" "$JOB_OUT/plan.json" \
      || { ST=FAIL; SUMM="no replica plan"; exit 1; }
    SY=$JOB_OUT/sync; rm -rf "$SY"; mkdir -p "$SY"; echo 0 > "$SY/turn"; EXT_DIR=$SY
    {
      echo '#!/bin/bash'
      echo "# every role of the plan as its own process on GPU $G0; exit codes in $SY/exit.<role>"
      echo "for role in \$(\"$V/bin/python\" -I -c \"import json; print(' '.join(json.load(open('$JOB_OUT/plan.json'))['roles']))\"); do"
      echo "  pp=''; [ \"\$role\" = old ] && pp=\"$W/site_old\""
      echo "  ( PYTHONPATH=\$pp CUDA_VISIBLE_DEVICES=$G0 STALL_S=60 END_AT=\$(( \$(date +%s) + $SECS )) \"$V/bin/python\" -u \"$W/worker90r.py\" \"\$role\" \"$JOB_OUT/plan.json\" \"$SY\" > \"$JOB_OUT/\$role.log\" 2>&1; echo \$? > \"$SY/exit.\$role\" ) &"
      echo "done"
      echo "wait"
    } > "$JOB_OUT/roles.sh"
    rc=0
    run_pg "$TMO" "$RLOG" "$W" bash "$JOB_OUT/roles.sh" || rc=$?
    RC=$rc; os=""; nbad=0
    for f in "$SY"/exit.*; do
      [ -e "$f" ] || continue; r=${f##*.}; e=$(cat "$f"); o=$(rc_name "$e" "$SY/ext_stall.$r")
      tm=$(tree_ok "$JOB_OUT/$r.log" "$r") || o="$o TREE_MISMATCH"
      echo "role $r exit $e $o" >> "$RLOG"; os="$os $r=$o"; [ "$o" = OK ] || nbad=$((nbad + 1))
      s=$(grep -m1 '^STALL ' "$JOB_OUT/$r.log" | cut -c1-300); [ -n "$s" ] && EXTRA_STATUS+=("stall_$r=$s")
    done
    fl=$(grep -hE "^(COMPILE_FAIL|RUN_FAIL|SAMPLE_FAIL|STREAMED_DIFFERS) " "$JOB_OUT"/*.log 2> /dev/null | head -3 | cut -c1-200 | tr '\n' ' ')
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
    nob=$(grep -h 'no barrier from step' "$JOB_OUT"/*.log 2> /dev/null | head -3 | tr '\n' ' ')
    EXTRA_STATUS+=("outcome=$os" "no_barrier=$nob")
    if [ $rc -eq 125 ]; then ST=RSS_KILL
    elif [ $rc -eq 124 ] || [ $rc -eq 137 ]; then ST=TIMEOUT
    elif [ $nbad -ne 0 ] || [ $rc -ne 0 ] || [ -n "$fl" ]; then ST=FAIL
    else ST=PASS; fi
    SUMM="replica (no periodic dump):$os ${fl}${nob:+ $nob}"
    ;;
  san)
    SAN_DEFAULT=/root/cudnnfe_sm90/sanitizer/cuda_sanitizer_api-linux-x86_64-13.2.87-archive/compute-sanitizer/compute-sanitizer
    SAN=${COMPUTE_SANITIZER:-$SAN_DEFAULT}
    for c in "$SAN" "$(command -v compute-sanitizer || true)" /usr/local/cuda-13.2/bin/compute-sanitizer /gbr/root/usr/local/cuda-13.2/bin/compute-sanitizer; do
      [ -n "$c" ] && [ -x "$c" ] && { SAN=$c; break; }; done
    "$SAN" --version > "$JOB_OUT/sanitizer_version.txt" 2>&1 || { ST=SETUP_FAIL; SUMM="no working compute-sanitizer ($SAN)"; exit 1; }
    extra=(--tool "$TOOL" --report-api-errors no --launch-timeout 0 --target-processes all --kernel-name-exclude regex=at6native --print-limit 0)
    [ "$TOOL" = racecheck ] && extra+=(--racecheck-report all)
    echo "sanitizer $SAN $(tail -1 "$JOB_OUT/sanitizer_version.txt"): ${extra[*]}" >> "$LOG"
    WL=$JOB_OUT/worker.log
    rc=0
    run_pg "$TMO" "$RLOG" "$W" env PYTHONPATH="$(pp_of "$TREE")" CUDA_VISIBLE_DEVICES="$G0" STALL_S=100000 PREARM_S=100000 \
      bash -c '"$7" "${@:8}" --error-exitcode 99 "$1" -u "$2" "$3" "$4" 0 "$5" > "$6" 2>&1' x "$V/bin/python" "$W/hangrepro.py" "$TREE" san "$JOB_OUT" "$WL" "$SAN" "${extra[@]}" || rc=$?
    RC=$rc
    sm=$(grep -h -E "ERROR SUMMARY|RACECHECK SUMMARY|SYNCCHECK SUMMARY" "$WL" 2> /dev/null | tail -2 | tr -s ' ' | tr '\n' ';')
    done_l=$(grep '^DONE ' "$WL" | tail -1 | cut -c1-200)
    tm=$(tree_ok "$WL" "$TREE") || done_l="$done_l TREE_MISMATCH($tm)"
    EXTRA_STATUS+=("san_tool=$TOOL" "san_rc=$rc")
    if [ $rc -eq 125 ]; then ST=RSS_KILL
    elif [ $rc -eq 124 ] || [ $rc -eq 137 ]; then ST=TIMEOUT
    elif [ $rc -eq 99 ]; then [ "$TOOL" = racecheck ] && ST=HAZARD || ST=DEVICE_ERROR
    elif [ $rc -ne 0 ] || [ -z "$done_l" ]; then ST=FAIL
    else ST=PASS; fi
    SUMM="$TOOL $TREE: rc=$rc ${sm} ${done_l}"
    ;;
esac
nvidia-smi -i "$G0" -q -d CLOCK,POWER,PERFORMANCE > "$JOB_OUT/gpu_clocks_after.txt" 2>&1 || true
b=$(busy_gpu); [ -n "$b" ] && { EXTRA_STATUS+=("foreign_after=$b"); [ "$ST" = PASS ] && { ST=FOREIGN; SUMM="after the run: $b; $SUMM"; }; }
for f in "$RLOG" "$JOB_OUT"/*.log; do
  sz=$(stat -c %s "$f" 2> /dev/null || echo 0)
  if [ "$sz" -gt 16000000 ]; then { head -c 8000000 "$f"; echo; echo "[... truncated $sz bytes ...]"; tail -c 8000000 "$f"; } > "$f.t" && mv "$f.t" "$f"; fi
done
exit 0
