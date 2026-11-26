"""rss_watchdog.py <pg dir> <limit GB> <host min available GB> [interval s]: one per host run.

Every interval, sums the resident set of every process per registered process group (a file <pg dir>/<pgid>, written by
run_s24rc.sh's run_pg right after it starts a sanitizer under setsid) from /proc/<pid>/stat, and
  * keeps the peak in <pg dir>/<pgid>.peak (kB);
  * kills the whole group (SIGKILL) when its sum exceeds <limit GB> (or the group's own cap in kB, <pg dir>/<pgid>.limit), and writes <pg dir>/<pgid>.rsskill (reason, kB);
  * when MemAvailable drops below <host min available GB>, kills the largest registered group the same way
    (reason host_low_mem).
run_pg reports a group killed here as rc 125 (status RSS_KILL, not FAIL).  Stops when <pg dir>/STOP exists.
No third-party imports; reads /proc only."""
import os
import signal
import sys
import time

pg_dir, limit_kb, min_avail_kb = sys.argv[1], float(sys.argv[2]) * 1024 ** 2, float(sys.argv[3]) * 1024 ** 2
interval = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
page_kb = os.sysconf("SC_PAGE_SIZE") // 1024
log = open(os.path.join(pg_dir, "watchdog.log"), "a", buffering=1)
log.write(f"{time.strftime('%FT%TZ', time.gmtime())} start limit {limit_kb / 1024 ** 2:.0f} GB, host min available "
          f"{min_avail_kb / 1024 ** 2:.0f} GB, every {interval} s\n")


def registered():
    out = set()
    for name in os.listdir(pg_dir):
        if name.isdigit():
            out.add(int(name))
    return out


def group_rss(groups):
    rss = dict.fromkeys(groups, 0)
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/stat", "rb") as f:
                stat = f.read()
        except OSError:
            continue
        fields = stat[stat.rfind(b")") + 2:].split()
        # after "pid (comm) ": state ppid pgrp session ... rss is field 24 of stat, index 21 here
        pgrp = int(fields[2])
        if pgrp in rss:
            rss[pgrp] += int(fields[21]) * page_kb
    return rss


def mem_available_kb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return float(line.split()[1])
    return float("inf")


def kill(pg, kb, reason):
    marker = os.path.join(pg_dir, f"{pg}.rsskill")
    with open(marker, "w") as f:
        f.write(f"{reason} {int(kb)}\n")
    try:
        os.killpg(pg, signal.SIGKILL)
    except ProcessLookupError:
        pass
    log.write(f"{time.strftime('%FT%TZ', time.gmtime())} KILL pg {pg} ({reason}) rss {kb / 1024 ** 2:.1f} GB\n")


peaks = {}
last_host_kill = 0.0
while not os.path.exists(os.path.join(pg_dir, "STOP")):
    groups = registered()
    if groups:
        rss = group_rss(groups)
        for pg, kb in rss.items():
            if kb > peaks.get(pg, -1):
                peaks[pg] = kb
                try:
                    with open(os.path.join(pg_dir, f"{pg}.peak"), "w") as f:
                        f.write(f"{int(kb)}\n")
                except OSError:
                    pass
            try:
                with open(os.path.join(pg_dir, f"{pg}.limit")) as f:
                    cap = float(f.read().split()[0])
            except (OSError, ValueError, IndexError):
                cap = limit_kb
            if kb > cap and not os.path.exists(os.path.join(pg_dir, f"{pg}.rsskill")):
                kill(pg, kb, "group_rss_over_limit")
        if mem_available_kb() < min_avail_kb and time.time() - last_host_kill > 30:  # 30 s for the memory to come back
            last_host_kill = time.time()
            live = {pg: kb for pg, kb in rss.items() if not os.path.exists(os.path.join(pg_dir, f"{pg}.rsskill"))}
            if live:
                pg = max(live, key=live.get)
                kill(pg, live[pg], "host_low_mem")
        for pg in list(peaks):
            if pg not in groups:
                del peaks[pg]
    time.sleep(interval)
log.write(f"{time.strftime('%FT%TZ', time.gmtime())} stop\n")
