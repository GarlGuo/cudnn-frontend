#!/bin/bash
# job.sh: one job of the s59sm90regress package (kit branch wip/sm90-h100-kit-b, h100_kit_b/s59sm90regress/), run by the
# h100 coordinator's dispatcher (jobs.jsonl "cmd").  s57sm90regress's job.sh with one tree (sig = 644492a92, #2908
# sm90-streaming-signal), sanitizers restricted to the SM90 streaming signal files, and no S3 (results stay on the host):
#   bash job.sh fshard <tree> <timeout_s> <rss_gb> <i> <N>          pytest -m "L0 or L1" of the ids at positions i, i+N, ...
#                                                                 of $W/collected_full_sig.txt (setup.sh's on-host
#                                                                 L0+L1 collection, sorted), in one pytest; PASS needs rc 0,
#                                                                 as many junit cases as ids, 0 failures, 0 errors
#   bash job.sh shard  <tree> <timeout_s> <rss_gb> <i> <N>          the same at -m L0 over $W/collected_sig.txt
#   bash job.sh strict <tree> <timeout_s> <rss_gb> <file>           every collected L0 case of one SM90 determinism file in
#                                                                 one pytest with CUDNN_DETERMINISM_GOLDENS set; PASS needs
#                                                                 rc 0, as many cases as collected, 0 failures, 0 errors
#                                                                 and no 'no golden' line
#   bash job.sh full   <tree> <timeout_s> <rss_gb> <file>           one test file at -m "L0 or L1" in one pytest; PASS needs
#                                                                 rc 0, > 0 cases, 0 failures, 0 errors
#   bash job.sh san    <tree> <timeout_s> <rss_gb> <tool> <i> <N> <per_id_s>
#                                                                 compute-sanitizer <tool> (memcheck | racecheck), one
#                                                                 process group per id, of the recorded L0 ids of the SM90
#                                                                 streaming signal files (test_grouped_gemm_sm90_streaming_
#                                                                 signal*.py), not warm_stress, at positions i, i+N, ...;
#                                                                 every tool --report-api-errors no --launch-timeout 0
#                                                                 --target-processes all --kernel-name-exclude
#                                                                 regex=at6native and pytest -k "not warm_stress";
#                                                                 racecheck also --racecheck-report all --print-limit 0;
#                                                                 PASS needs every id rc 0 (--error-exitcode 99)
#   CUDA_MODULE_LOADING=EAGER for every command.
#   From the dispatcher: CUDA_VISIBLE_DEVICES, JOB_ID, ATTEMPT, JOB_OUT (all optional).  Runs setup.sh <tree> first when
#   the tree is not ready on this host (idempotent, locked).
#   Each command runs under setsid in its own process group with a per-job RSS watchdog (rss_watchdog.py, cap <rss_gb>);
#   its timeout or the cap kills the whole group (kill -- -pgid).
#   Writes JOB_OUT/status.txt (line 1: PASS | FAIL | HAZARD | DEVICE_ERROR | TIMEOUT | RSS_KILL | NOT_COLLECTED |
#   SETUP_FAIL | KILLED; then key=value lines) and, when not PASS, JOB_OUT/fail_head.txt (the first failure lines).
#   No upload: JOB_OUT is copied to $W/results/<JOB_ID>_a<ATTEMPT>/ (unless it is already under $W) for report.py.
#   Exits 0 iff PASS.  No ports are opened.
set -uo pipefail
RUN=s59sm90regress
W=$HOME/cudnnfe_sm90/$RUN
KIND=${1:-}; TREE=${2:-}; TMO=${3:-}; RSS_GB=${4:-}; shift 4 2> /dev/null || { echo "usage: see the header of job.sh"; exit 2; }
case "$KIND:$TREE" in shard:sig | fshard:sig | strict:sig | full:sig | san:sig) ;; *) echo "usage: see the header of job.sh"; exit 2 ;; esac
case "$TMO$RSS_GB" in *[!0-9]* | "") echo "timeout_s and rss_gb must be integers"; exit 2 ;; esac
ID=-; SI=-; SN=-; PER=-; TOOL=-
case $KIND in
  shard | fshard) SI=${1:-}; SN=${2:-}; ID="${KIND}_$SI/$SN" ;;
  san) TOOL=${1:-}; SI=${2:-}; SN=${3:-}; PER=${4:-}; ID="${TOOL}_$SI/$SN"; case $TOOL in memcheck | racecheck) ;; *) echo "bad tool '$TOOL'"; exit 2 ;; esac
       case $PER in *[!0-9]* | "") echo "bad per-id timeout '$PER'"; exit 2 ;; esac ;;
  strict | full) ID=${1:-}; [ -n "$ID" ] || { echo "no file"; exit 2; } ;;
esac
case $KIND in shard | fshard | san) case "$SI:$SN" in *[!0-9:]* | :* | *:) echo "bad shard '$SI' '$SN'"; exit 2 ;; esac; [ "$SI" -lt "$SN" ] || { echo "bad shard $SI/$SN"; exit 2; } ;; esac
declare -A SHA=([sig]=644492a9)
JOB_ID=${JOB_ID:-local_${KIND}_$$}; ATTEMPT=${ATTEMPT:-1}
JOB_OUT=${JOB_OUT:-$W/job_out/${JOB_ID}_a$ATTEMPT}; mkdir -p "$JOB_OUT" || exit 2
LOG=$JOB_OUT/job.log; PGD=$JOB_OUT/pg; mkdir -p "$PGD"
START=$(date +%s); ST=""; RC=-; SUMM=""; PEAK_KB=0; NEXP=-; EXTRA_STATUS=()
fail_head() {  # the first failure lines of a non-PASS job (report.py prints them)
  {
    echo "== status: $ST: $SUMM"
    [ -s "$JOB_OUT/san_ids.tsv" ] && { echo "== san_ids.tsv (not PASS)"; grep -v '^PASS' "$JOB_OUT/san_ids.tsv" | head -10; }
    for f in "$JOB_OUT/run.log" "$JOB_OUT"/san_*.log "$JOB_OUT/setup.log" "$JOB_OUT/job.log"; do
      [ -s "$f" ] || continue
      echo "== $(basename "$f"): first failure lines"
      grep -m 25 -E '^(FAILED|ERROR) |^E  |Error|error:|Traceback|SETUP_FAIL|Hazard|hazard|Invalid|timeout|Killed|RSS' "$f" | cut -c1-300
    done
    [ -s "$JOB_OUT/run.log" ] && { echo "== run.log tail"; tail -n 15 "$JOB_OUT/run.log" | cut -c1-300; }
  } > "$JOB_OUT/fail_head.txt" 2> /dev/null || true
}
finish() {  # once: status.txt, fail_head.txt, the copy to $W/results, the exit code (0 iff PASS)
  local dst
  trap - EXIT
  touch "$PGD/STOP"; for f in "$PGD"/[0-9]*; do [ -e "$f" ] || continue; case $f in *.*) continue ;; esac; kill -KILL -- "-${f##*/}" 2> /dev/null || true; done
  [ -n "$ST" ] || { ST=SETUP_FAIL; SUMM=${SUMM:-job.sh stopped early (see job.log)}; }
  {
    echo "$ST"
    echo "job_id=$JOB_ID"; echo "attempt=$ATTEMPT"; echo "run=$RUN"; echo "kind=$KIND"; echo "tree=$TREE"; echo "sha=${SHA[$TREE]}"
    echo "id=$ID"; echo "n_expected=$NEXP"; echo "rc=$RC"; echo "secs=$(( $(date +%s) - START ))"
    echo "peak_rss_gb=$(awk -v k="$PEAK_KB" 'BEGIN{printf "%.1f", k / 1048576}')"; echo "rss_cap_gb=$RSS_GB"; echo "timeout_s=$TMO"
    echo "collected_md5=$(md5sum 2> /dev/null < "$W/collected_$TREE.txt" | cut -c1-32)"
    echo "collected_full_md5=$(md5sum 2> /dev/null < "$W/collected_full_$TREE.txt" | cut -c1-32)"
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
echo "$(date -u +%FT%TZ) $RUN job $JOB_ID attempt $ATTEMPT on $(hostname): $KIND $TREE timeout=${TMO}s rss_cap=${RSS_GB}G gpus=${CUDA_VISIBLE_DEVICES:-} id=$ID" | tee "$LOG"
# 0. the tree is ready on this host (setup.sh: venv + on-host collection), else set it up now
if [ "$(cat "$W/ready_$TREE" 2> /dev/null)" != "$(md5sum < "$W/MD5SUMS" 2> /dev/null | cut -c1-32)" ] || [ ! -s "$W/collected_$TREE.txt" ] || [ ! -s "$W/collected_full_$TREE.txt" ]; then
  echo "tree $TREE not ready: running setup.sh $TREE" >> "$LOG"
  SETUP_SH=$W/setup.sh; [ -f "$SETUP_SH" ] || SETUP_SH=$(cd "$(dirname "$0")" && pwd)/setup.sh
  bash "$SETUP_SH" "$TREE" >> "$JOB_OUT/setup.log" 2>&1 || { ST=SETUP_FAIL; SUMM="setup.sh $TREE failed: $(grep SETUP_FAIL "$JOB_OUT/setup.log" | tail -1)"; exit 1; }
fi
V=$HOME/venvs/cfe_pp_${SHA[$TREE]}; SRC=$W/src_$TREE
cp "$W/collected_$TREE.txt" "$JOB_OUT/collected_$TREE.txt"; cp "$W/collected_full_$TREE.txt" "$JOB_OUT/collected_full_$TREE.txt"
IDS=()
case $KIND in
  shard) mapfile -t IDS < <(awk -v i="$SI" -v n="$SN" '(NR - 1) % n == i' "$W/collected_$TREE.txt") ;;
  fshard) mapfile -t IDS < <(awk -v i="$SI" -v n="$SN" '(NR - 1) % n == i' "$W/collected_full_$TREE.txt") ;;
  san) mapfile -t IDS < <(awk -F'::' '$1 ~ /test_grouped_gemm_sm90_streaming_signal[^\/]*\.py$/ && $2 !~ /warm_stress/' "$W/collected_$TREE.txt" | awk -v i="$SI" -v n="$SN" '(NR - 1) % n == i') ;;
  strict) mapfile -t IDS < <(grep -F -- "$ID::" "$W/collected_$TREE.txt") ;;
  full) [ -f "$W/src_$TREE/test/python/$ID" ] && IDS=("$ID") ;;
esac
NEXP=${#IDS[@]}
printf '%s\n' "${IDS[@]+"${IDS[@]}"}" > "$JOB_OUT/ids.txt"
if [ "$NEXP" = 0 ]; then
  if [ "$KIND" = strict ] || [ "$KIND" = full ]; then ST=NOT_COLLECTED; SUMM="no collected id in $ID"; exit 1; fi
  ST=PASS; SUMM="empty shard (no id at $ID)"; exit 0
fi
# 1. the environment
export PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_MODULE_LOADING=EAGER
unset CUTE_DSL_ARCH CUTE_DSL_KEEP CUTE_DSL_DUMP_DIR CUTE_DSL_COMPILER_OPT QUACK_ARCH PYTHONPATH VIRTUAL_ENV CUDNN_DETERMINISM_GOLDENS CUDNN_DETERMINISM_RECORD
CTT=$((TMO - 120)); [ "$KIND" = san ] && CTT=$((PER - 120)); [ $CTT -lt 600 ] && CTT=600
export CUDNN_TEST_TIMEOUT=$CTT  # test/python/conftest.py's per-test watchdog (default 1500 s): below the run's own timeout
pgid_of() { local s; s=$(cat "/proc/$1/stat" 2> /dev/null) || return 1; s=${s##*) }; set -- $s; echo "$3"; }
run_pg() {  # run_pg <timeout s> <log> <dir> cmd...: own session / process group, RSS watchdog, TERM/KILL the whole group
            # on timeout: rc 124; watchdog kill: rc 125
  local tmo=$1 log=$2 dir=$3 pid pg t=0 rc=0 timed_out=0 pk; shift 3
  setsid bash -c 'cd "$1" && shift && exec "$@"' run_pg "$dir" "$@" >> "$log" 2>&1 < /dev/null &
  pid=$!
  pg=$(pgid_of "$pid" || true); [ -n "$pg" ] || pg=$pid
  [ "$pg" = "$(pgid_of $$ || true)" ] && pg=$pid
  rm -f "$PGD/$pg".*; echo "$log" > "$PGD/$pg"
  while kill -0 "$pid" 2> /dev/null; do
    [ -f "$PGD/$pg.rsskill" ] && break
    if [ "$t" -ge "$tmo" ]; then
      timed_out=1; echo "run_pg: timeout ${tmo}s, killing process group $pg" >> "$log"
      kill -TERM -- "-$pg" 2> /dev/null || true; sleep 30; kill -KILL -- "-$pg" 2> /dev/null || true; break
    fi
    sleep 5; t=$((t + 5))
  done
  wait "$pid" 2> /dev/null || rc=$?
  kill -KILL -- "-$pg" 2> /dev/null || true  # orphans (TreeLauncherSubreaper, the pytest child) of a finished run
  pk=$(cat "$PGD/$pg.peak" 2> /dev/null || echo 0); pk=${pk%.*}; [ "${pk:-0}" -gt "$PEAK_KB" ] 2> /dev/null && PEAK_KB=$pk
  if [ -f "$PGD/$pg.rsskill" ]; then echo "run_pg: RSS watchdog killed process group $pg: $(cat "$PGD/$pg.rsskill")" >> "$log"; rc=125
  elif [ $timed_out = 1 ]; then rc=124; fi
  rm -f "$PGD/$pg" "$PGD/$pg".*
  return $rc
}
# the per-job watchdog: this job's group only (host-level MemAvailable is the dispatcher's, host_min_avail_gb)
python3 -I "$W/rss_watchdog.py" "$PGD" "$RSS_GB" 0 2 > /dev/null 2>&1 &
junit_counts() {  # junit_counts <xml>: "tests failures errors skipped" summed over the testsuites
  python3 -I - "$1" << 'EOF'
import sys, xml.etree.ElementTree as ET
t = [0, 0, 0, 0]
try:
    for ts in ET.parse(sys.argv[1]).getroot().iter("testsuite"):
        for i, a in enumerate(("tests", "failures", "errors", "skipped")):
            t[i] += int(ts.get(a, 0))
except Exception:
    t = [-1, -1, -1, -1]
print(*t)
EOF
}
junit_skips() {  # junit_skips <xml>: one "<classname>::<name>\t<reason>" line per skipped case
  python3 -I - "$1" << 'EOF'
import sys, xml.etree.ElementTree as ET
try:
    for tc in ET.parse(sys.argv[1]).getroot().iter("testcase"):
        s = tc.find("skipped")
        if s is not None:
            print(f"{tc.get('classname')}::{tc.get('name')}\t{(s.get('message') or '').strip()[:300]}")
except Exception as e:
    print(f"junit parse error\t{e!r}")
EOF
}
RLOG=$JOB_OUT/run.log
case $KIND in
  shard | fshard | strict | full)
    envx=(); [ "$KIND" = strict ] && envx=(CUDNN_DETERMINISM_GOLDENS=$SRC/test/python/determinism_goldens)
    MARK=L0; { [ "$KIND" = full ] || [ "$KIND" = fshard ]; } && MARK="L0 or L1"
    rc=0
    run_pg "$TMO" "$RLOG" "$SRC/test/python" env ${envx[@]+"${envx[@]}"} VIRTUAL_ENV=$V PATH=$V/bin:$PATH \
      "$V/bin/python" -m pytest -p no:cacheprovider -m "$MARK" -q -rfEs --junitxml="$JOB_OUT/junit.xml" -- "${IDS[@]}" || rc=$?
    RC=$rc; read -r nt nf ne ns <<< "$(junit_counts "$JOB_OUT/junit.xml")"
    junit_skips "$JOB_OUT/junit.xml" > "$JOB_OUT/skipped.txt"
    ng=$(grep -c "no golden" "$RLOG" 2> /dev/null || true)
    SUMM="$KIND junit tests=$nt failures=$nf errors=$ne skipped=$ns (want $NEXP cases); 'no golden' lines=$ng"
    EXTRA_STATUS+=("junit_tests=$nt" "junit_failures=$nf" "junit_errors=$ne" "junit_skipped=$ns" "no_golden_lines=$ng")
    if [ $rc -eq 125 ]; then ST=RSS_KILL
    elif [ $rc -eq 124 ] || [ $rc -eq 137 ]; then ST=TIMEOUT
    elif [ "$KIND" = full ] && { [ $rc -ne 0 ] || [ "$nt" -le 0 ] || [ "$nf" != 0 ] || [ "$ne" != 0 ]; }; then ST=FAIL
    elif [ "$KIND" != full ] && { [ $rc -ne 0 ] || [ "$nt" != "$NEXP" ] || [ "$nf" != 0 ] || [ "$ne" != 0 ]; }; then ST=FAIL
    elif [ "$KIND" = strict ] && [ "$ng" != 0 ]; then ST=FAIL
    else ST=PASS; fi
    ;;
  san)
    SAN_DEFAULT=/root/cudnnfe_sm90/sanitizer/cuda_sanitizer_api-linux-x86_64-13.2.87-archive/compute-sanitizer/compute-sanitizer
    SAN=${COMPUTE_SANITIZER:-$SAN_DEFAULT}
    for c in "$SAN" "$(command -v compute-sanitizer || true)" /usr/local/cuda-13.2/bin/compute-sanitizer /gbr/root/usr/local/cuda-13.2/bin/compute-sanitizer; do
      [ -n "$c" ] && [ -x "$c" ] && { SAN=$c; break; }; done
    "$SAN" --version > "$JOB_OUT/sanitizer_version.txt" 2>&1 || { ST=SETUP_FAIL; SUMM="no working compute-sanitizer ($SAN)"; exit 1; }
    extra=(--tool "$TOOL" --report-api-errors no --launch-timeout 0 --target-processes all --kernel-name-exclude regex=at6native)
    kx=(-k "not warm_stress"); [ "$TOOL" = racecheck ] && extra+=(--racecheck-report all --print-limit 0)
    echo "sanitizer $SAN $(tail -1 "$JOB_OUT/sanitizer_version.txt"): ${extra[*]} pytest ${kx[*]+"${kx[*]}"}; per-id timeout ${PER}s, CUDNN_TEST_TIMEOUT $CTT" >> "$LOG"
    npass=0; nbad=0; k=0; worst=PASS
    : > "$JOB_OUT/san_ids.tsv"
    for tid in "${IDS[@]}"; do
      k=$((k + 1)); el=$(( $(date +%s) - START )); left=$((TMO - el - 60))
      if [ $left -lt 120 ]; then printf 'NOTRUN\t-\t-\t%s\t-\t-\n' "$tid" >> "$JOB_OUT/san_ids.tsv"; nbad=$((nbad + 1)); worst=TIMEOUT; continue; fi
      t=$PER; [ $t -gt $left ] && t=$left
      il=$JOB_OUT/san_$k.log; rc=0; t1=$(date +%s)
      run_pg "$t" "$il" "$SRC/test/python" env VIRTUAL_ENV=$V PATH=$V/bin:$PATH \
        "$SAN" "${extra[@]}" --error-exitcode 99 "$V/bin/python" -m pytest -p no:cacheprovider -m L0 ${kx[@]+"${kx[@]}"} -q -rfEs \
        --junitxml="$JOB_OUT/junit_$k.xml" -- "$tid" || rc=$?
      read -r nt nf ne ns <<< "$(junit_counts "$JOB_OUT/junit_$k.xml")"
      sm=$(grep -h -E "ERROR SUMMARY|RACECHECK SUMMARY" "$il" 2> /dev/null | tail -1 | tr -s ' ' | sed 's/^=* *//')
      if [ $rc -eq 0 ] && [ "$nt" = 1 ] && [ "$nf" = 0 ] && [ "$ne" = 0 ]; then s=PASS; npass=$((npass + 1))
      else
        nbad=$((nbad + 1))
        if [ $rc -eq 125 ]; then s=RSS_KILL; elif [ $rc -eq 124 ] || [ $rc -eq 137 ]; then s=TIMEOUT
        elif [ $rc -eq 99 ]; then [ "$TOOL" = racecheck ] && s=HAZARD || s=DEVICE_ERROR; else s=FAIL; fi
        case $worst in PASS | FAIL) worst=$s ;; esac
      fi
      printf '%s\t%s\t%ss\t%s\tjunit %s/%s/%s/%s\t%s\n' "$s" "$rc" "$(( $(date +%s) - t1 ))" "$tid" "$nt" "$nf" "$ne" "$ns" "$sm" >> "$JOB_OUT/san_ids.tsv"
      if [ "$s" = PASS ] && [ "$(stat -c %s "$il" 2> /dev/null || echo 0)" -gt 2000000 ]; then { head -c 1000000 "$il"; echo "[... truncated ...]"; tail -c 1000000 "$il"; } > "$il.t" && mv "$il.t" "$il"; fi
    done
    RC=$nbad
    SUMM="$TOOL ids passed $npass of $NEXP; SUMMARY lines: $(awk -F'\t' '{print $6}' "$JOB_OUT/san_ids.tsv" | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
    EXTRA_STATUS+=("san_tool=$TOOL" "san_pass=$npass" "san_bad=$nbad")
    if [ $nbad = 0 ]; then ST=PASS; else ST=$worst; fi
    ;;
esac
# a clean log is noise: keep the head / tail of big ones (non-PASS logs stay whole up to 64 MB)
sz=$(stat -c %s "$RLOG" 2> /dev/null || echo 0)
if [ "$ST" = PASS ] && [ "$sz" -gt 2000000 ]; then { head -c 1000000 "$RLOG"; echo; echo "[... truncated $sz bytes ...]"; tail -c 1000000 "$RLOG"; } > "$RLOG.t" && mv "$RLOG.t" "$RLOG"
elif [ "$sz" -gt 64000000 ]; then { head -c 32000000 "$RLOG"; echo; echo "[... truncated $sz bytes ...]"; tail -c 32000000 "$RLOG"; } > "$RLOG.t" && mv "$RLOG.t" "$RLOG"; fi
exit 0
