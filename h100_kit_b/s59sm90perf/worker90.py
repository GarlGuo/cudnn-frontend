"""s57sm90plainperf3: s57sm90plainperf2's worker90.py with the periodic faulthandler dump replaced by a stall watchdog
(the same file as s58sm90hangrepro worker90r.py, validated there: replica.r0-3 PASS).  plainperf2's
dump_traceback_later(600, repeat=True) walked the running main thread's frames without the GIL and SIGSEGV'd role processes
(notes/cudnn-fe-streaming/sm90_dglu_hang_investigation.md).
s57 (s54c's worker90.py; the delayed cells' producer: calibrated once by the base role after every role is ready,
shared by every role).  One process (one role = one cuDNN-FE tree) of an SM90 grouped GEMM kernel A/B on one H100: build + compile every cell,
check, then time the cells in turn with the other roles' processes (file turn barrier: one process launches at a time).

Usage: worker90.py ROLE PLAN SYNCDIR       (PLAN: {"P", "R", "roles": [..], "base": role, "cells": [..]})
Cell modes:
  ready      signal scheduler, every M tile ready before launch; R launches per pass, each after an L2 flush.
  delay<f>   a producer on another stream releases the M tiles round robin over the experts (first M tile of every expert,
             then the second, ...) in 32 groups, one group per step after a torch.cuda._sleep, spanning f x the base role's
             ready time of the cell; the GEMM runs beside it on one cluster fewer than the device holds.  Time from the
             producer's start to the GEMM's end; R/5 launches per pass.
Rows: balanced routing (cell["routed_per_expert"]), each expert padded to 256 rows; padding rows zero.
"""
import json, os, sys, time, statistics, traceback, hashlib, faulthandler

# s57sm90plainperf3 (= s58 replica): s57's worker90.py with the periodic faulthandler dump (dump_traceback_later(600, repeat=True), which
# walks the running main thread's frames without the GIL: the suspected cause of s57's SIGSEGV) replaced by a stall
# watchdog thread: no progress for STALL_S (60) s in the step loop (15 min before it) -> every thread's stack
# (sys._current_frames under the GIL, then faulthandler.dump_traceback from that thread) and exit rc 86.
import signal, threading
faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
STALL_S = float(os.environ.get("STALL_S", "60"))
_hb = [time.monotonic(), "start"]
_armed = [False]


def beat(where):
    _hb[0] = time.monotonic()
    _hb[1] = where


def _watchdog():
    wd_file = os.path.join(sys.argv[3], f"wd.{sys.argv[1]}")
    while True:
        time.sleep(1.0)
        try:
            with open(wd_file, "w") as fh:
                fh.write(f"{time.time():.0f} {_hb[1]}\n")
        except Exception:
            pass
        age = time.monotonic() - _hb[0]
        if age > (STALL_S if _armed[0] else 900.0):
            try:
                out = [f"STALL {json.dumps(dict(role=sys.argv[1], secs=round(age, 1), where=_hb[1], t=time.strftime('%H:%M:%S')))}"]
                main_id = threading.main_thread().ident
                for tid, fr in sys._current_frames().items():
                    out.append(f"--- thread {tid}{' (main)' if tid == main_id else ''}")
                    out.extend(x.rstrip() for x in traceback.format_stack(fr))
                print("\n".join(out), flush=True)
                faulthandler.dump_traceback(file=sys.stdout, all_threads=True)
                sys.stdout.flush()
            finally:
                os._exit(86)


threading.Thread(target=_watchdog, daemon=True, name="stall_watchdog").start()
# Eager module loading: a kernel loaded lazily while a persistent GEMM spins on its output can deadlock.
os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")

role, plan_path, sync = sys.argv[1], sys.argv[2], sys.argv[3]
plan = json.load(open(plan_path))
P, R = plan["P"], plan["R"]
roles = plan["roles"]
BASE = plan["base"]
cells_all = plan["cells"]
ALIGN = 256
GROUPS = 32
CHUNK = 4096


def log(*a):
    print(f"[{role} {time.strftime('%H:%M:%S')}]", *a, flush=True)


def emit(kind, **kw):
    print(kind, json.dumps(dict(role=role, **kw)), flush=True)


import torch
import cudnn
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import HardwareInfo
from cuda.bindings import driver as cuda
from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode
from cudnn.gemm.cutedsl.grouped.moe_kernel_helpers import reset_streaming_signal_workspace, streaming_signal_workspace_offsets
from cudnn.gemm.cutedsl.grouped.unfused import moe_grouped_gemm_sm90 as unfused
from cudnn.gemm.cutedsl.grouped.glu.moe_grouped_gemm_glu_sm90 import MoEGroupedGemmGluBf16KernelSm90 as Glu
from cudnn.gemm.cutedsl.grouped.dglu.moe_grouped_gemm_dglu_sm90 import MoEGroupedGemmDgluBf16KernelSm90 as Dglu

torch.cuda.set_device(0)
dev = "cuda"
torch.zeros(1, device=dev)
FULL = HardwareInfo().get_max_active_clusters(2)
emit("TREE", cudnn=cudnn.__file__, clusters=FULL, gpu=torch.cuda.get_device_name())
main_stream = torch.cuda.current_stream()
prod_stream = torch.cuda.Stream()
gemm_stream = torch.cuda.Stream()


def seed_of(*key):
    return int(hashlib.md5(repr(key).encode()).hexdigest()[:8], 16)


def t(x):
    return from_dlpack(x, assumed_align=16)


def mkl(x):
    return x.as_strided((x.shape[0], x.shape[1], 1), (x.shape[1], 1, x.shape[0] * x.shape[1]))


problems = {}


def get_problem(c):
    key = (c["kind"], c["model"], c["ep"], c["tokens"])
    if key in problems:
        return problems[key]
    routed = c["routed_per_expert"]
    padded = [-(-r // ALIGN) * ALIGN for r in routed]
    rows = max(sum(padded), ALIGN)
    starts = [0]
    for b in padded[:-1]:
        starts.append(starts[-1] + b)
    E, n, k = len(routed), c["n"], c["k"]
    g = torch.Generator(device=dev).manual_seed(seed_of(*key))

    def rnd(*shape, s=1.0):
        out = torch.empty(*shape, dtype=torch.bfloat16, device=dev)
        flat = out.view(-1, shape[-1])
        for r0 in range(0, flat.shape[0], CHUNK):
            r1 = min(flat.shape[0], r0 + CHUNK)
            flat[r0:r1] = (torch.randn(r1 - r0, shape[-1], generator=g, device=dev) * s).to(torch.bfloat16)
        return out

    def rows_rnd(width, s):
        out = torch.zeros(rows, width, dtype=torch.bfloat16, device=dev)
        for st, r in zip(starts, routed):
            out[st:st + r] = rnd(r, width, s=s) if r else out[st:st]
        return out

    p = dict(rows=rows, n=n, k=k, padded=padded, starts=starts, routed=routed)
    p["offsets"] = torch.tensor(padded, dtype=torch.int64, device=dev).cumsum(0).to(torch.int32)
    p["alpha"] = torch.ones(E, device=dev)
    p["prob"] = torch.rand(rows, 1, 1, generator=g, device=dev)
    p["a"] = rows_rnd(k, k ** -0.5)
    p["b"] = rnd(E, n, k, s=k ** -0.5)
    if c["kind"] == "dglu":
        p["c"] = rows_rnd(2 * n, 1.0)
        p["beta"] = torch.ones(E, device=dev)
    # round-robin release order of the M tiles: M tile j of every expert, j = 0, 1, ...
    first = [s // ALIGN for s in starts]
    ntiles = [b // ALIGN for b in padded]
    order = [first[e] + j for j in range(max(ntiles)) for e in range(E) if j < ntiles[e]]
    p["groups"] = [torch.tensor(gr, dtype=torch.int64, device=dev) for gr in
                   (order[i * len(order) // GROUPS:(i + 1) * len(order) // GROUPS] for i in range(min(GROUPS, len(order)))) if gr]
    problems[key] = p
    return p


def make_kernel(c, p):
    kind = c["kind"]
    E = len(p["padded"])
    sig = dict(streaming_signal_capacity=p["rows"] // ALIGN)
    if kind == "dglu":
        return Dglu(acc_dtype=cutlass.Float32, use_2cta_instrs=False, mma_tiler_mn=(128, 256), cluster_shape_mn=(2, 1), vectorized_f32=False,
                    expert_cnt=E, weight_mode=MoEWeightMode.DENSE, use_dynamic_sched=True, act_func="dswiglu", **sig), (128, 256), False
    pingpong = unfused.use_pingpong(p["n"], p["k"], None, None)
    tile = unfused.default_mma_tiler(pingpong)
    common = dict(acc_dtype=cutlass.Float32, use_2cta_instrs=False, mma_tiler_mn=tile, cluster_shape_mn=(2, 1), vectorized_f32=False,
                  generate_c=False, enable_bias=False, expert_cnt=E, weight_mode=MoEWeightMode.DENSE, use_dynamic_sched=True,
                  pingpong=pingpong, **sig)
    if kind == "glu":
        return Glu(**common, act_func="swiglu"), tile, pingpong
    return unfused.MoEGroupedGemmBf16KernelSm90(**common), tile, pingpong


def make_buffers(c, kernel, p):
    rows, n = p["rows"], p["n"]
    out = dict(ws=torch.zeros(max(kernel.get_workspace_bytes(), 128), dtype=torch.uint8, device=dev),
               c=torch.zeros(rows, n if c["kind"] != "dglu" else 2 * n, dtype=torch.bfloat16, device=dev))
    if c["kind"] == "dglu":
        out["d"] = torch.full((rows, 2 * n), float("nan"), dtype=torch.bfloat16, device=dev)
        out["dprob"] = torch.zeros(rows, 1, 1, device=dev)
    else:
        out["d"] = torch.full((rows, n // 2 if c["kind"] == "glu" else n), float("nan"), dtype=torch.bfloat16, device=dev)
    return out


def make_args(c, p, buf):
    head = [t(mkl(p["a"])), t(p["b"].permute(1, 2, 0)), cutlass.Int32(p["n"]), cutlass.Int32(p["k"]), cutlass.Int64(p["k"]), OperandMajorMode.K,
            from_dlpack(buf["ws"], assumed_align=128).iterator]
    offs = from_dlpack(p["offsets"], assumed_align=4)
    if c["kind"] == "glu":
        return head + [t(mkl(buf["c"])), t(mkl(buf["d"])), offs, t(p["alpha"]), t(p["prob"]), None]
    if c["kind"] == "plain":
        return head + [t(mkl(buf["c"])), t(mkl(buf["d"])), offs, t(p["alpha"]), None, t(p["prob"])]
    return head + [t(mkl(p["c"])), t(mkl(buf["d"])), offs, t(p["alpha"]), t(p["beta"]), t(p["prob"]), t(buf["dprob"]), cutlass.Float32(0.0), None]


def checksum(buf, p):
    x = buf["d"].index_select(0, torch.cat([torch.arange(s, s + r, device=dev) for s, r in zip(p["starts"], p["routed"])]))
    x = x.contiguous().view(torch.int32).to(torch.int64).view(-1)
    w = torch.arange(x.numel(), device=dev, dtype=torch.int64) % 65521
    return [int(x.sum()), int((x * w).sum())]


class Cell:
    pass


def build(c):
    t0 = time.time()
    try:
        p = get_problem(c)
        kernel, tile, pingpong = make_kernel(c, p)
        buf = make_buffers(c, kernel, p)
        args = make_args(c, p, buf)
        stream = cuda.CUstream(main_stream.cuda_stream)
        delayed = c["mode"].startswith("delay")
        clusters = FULL - 1 if delayed else FULL
        compiled = cute.compile(kernel, *args, clusters, stream)
        launch_args = [x for i, x in enumerate(args) if i != 5]
        x = Cell()
        x.kernel, x.buf, x.p, x.compiled, x.launch_args = kernel, buf, p, compiled, launch_args
        off = streaming_signal_workspace_offsets(kernel)
        x.signal = buf["ws"].view(torch.int32)[off["signal"] // 4: off["signal"] // 4 + p["rows"] // ALIGN]
        emit("COMPILED", cid=c["cid"], secs=round(time.time() - t0, 1), rows=p["rows"], tile=list(tile), pingpong=pingpong, clusters=clusters,
             m_tiles=p["rows"] // ALIGN, groups=len(p["groups"]))
        return x
    except Exception as ex:
        emit("COMPILE_FAIL", cid=c["cid"], error=f"{type(ex).__name__}: {ex}"[:3000], tb=traceback.format_exc()[-3000:])
        return None


def launch(x, s):
    x.compiled(*x.launch_args, cuda.CUstream(s.cuda_stream))


def produce(x, cycles, s):
    with torch.cuda.stream(s):
        for g in x.p["groups"]:
            if cycles:
                torch.cuda._sleep(cycles)
            x.signal.index_copy_(0, g, g.to(torch.int32))


mine = [c for c in cells_all if role in c["roles"]]
cells = {c["cid"]: build(c) for c in mine}
flush_buf = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
by_cid = {c["cid"]: c for c in cells_all}

# Run every producer-side kernel (sleep, index_copy_) once on its own before any streamed launch.
for cid, x in cells.items():
    if x is not None and by_cid[cid]["mode"].startswith("delay"):
        sig = torch.full_like(x.signal, -1)
        with torch.cuda.stream(prod_stream):
            torch.cuda._sleep(1000)
            for g in x.p["groups"]:
                sig.index_copy_(0, g, g.to(torch.int32))
        torch.cuda.synchronize()
        break

# check: every cell once all ready, D of the real rows (dGLU dprob is a float atomic sum: not compared)
for cid, x in list(cells.items()):
    if x is None:
        continue
    try:
        reset_streaming_signal_workspace(x.buf["ws"], x.kernel, all_complete=True)
        launch(x, main_stream)
        torch.cuda.synchronize()
        cs = checksum(x.buf, x.p)
        nan = bool(torch.isnan(x.buf["d"].index_select(0, torch.tensor([x.p["starts"][0]], device=dev)).float()).any())
        if by_cid[cid]["mode"].startswith("delay"):
            # the streamed launch must give the same bytes
            x.buf["d"].fill_(float("nan"))
            reset_streaming_signal_workspace(x.buf["ws"], x.kernel)
            torch.cuda.synchronize()
            gemm_stream.wait_stream(main_stream)
            prod_stream.wait_stream(main_stream)
            launch(x, gemm_stream)
            produce(x, 20000, prod_stream)
            torch.cuda.synchronize()
            cs2 = checksum(x.buf, x.p)
            if cs2 != cs:
                emit("STREAMED_DIFFERS", cid=cid, ready=cs, streamed=cs2)
        emit("CHECKSUM", cid=cid, rows=cs, nan=nan)
    except Exception as ex:
        emit("RUN_FAIL", cid=cid, error=f"{type(ex).__name__}: {ex}"[:2000])
        cells[cid] = None


def ready_time(x, n=10):
    ts = []
    for _ in range(n):
        flush_buf.zero_()
        reset_streaming_signal_workspace(x.buf["ws"], x.kernel, all_complete=True)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        launch(x, main_stream)
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0)
    return statistics.median(ts)


def producer_span_us(x, cycles):
    sig = torch.full_like(x.signal, -1)
    prod_stream.wait_stream(main_stream)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record(prod_stream)
    with torch.cuda.stream(prod_stream):
        for g in x.p["groups"]:
            if cycles:
                torch.cuda._sleep(cycles)
            sig.index_copy_(0, g, g.to(torch.int32))
    e1.record(prod_stream)
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000.0


# (the base role's ready time per (shape, kind) and the delayed cells' producer cycles: below, once every role is ready)
log("built + checked", len(mine), "cells", sum(1 for v in cells.values() if v), "ok")
open(f"{sync}/ready.{role}", "w").write("1")


def dead_partner():
    return [r for r in roles if r != role and os.path.exists(f"{sync}/exit.{r}")]


t_wait = time.time()
while not all(os.path.exists(f"{sync}/ready.{r}") for r in roles):
    if dead_partner() or time.time() - t_wait > 1800:
        log("partner died / ready timeout")
        break
    beat("wait ready")
    time.sleep(0.05)

# s57: the base role alone, once every role is built and checked (the GPU otherwise idle), measures the ready time of
# every (shape, kind) and calibrates the producer of every delayed cell; every role then uses the same sleep cycles
# (s54c calibrated per role, while the other roles were still compiling: the roles' producers differed).
if role == BASE:
    spans = {}
    for cid, x in cells.items():
        c = by_cid[cid]
        if x is not None and c["mode"] == "ready":
            spans["|".join([c["kind"], c["model"], str(c["ep"]), str(c["tokens"])])] = ready_time(x)
    emit("SPANS", spans=spans)
    cycles_all = {}
    for cid, x in cells.items():
        c = by_cid[cid]
        if x is None or not c["mode"].startswith("delay"):
            continue
        key = "|".join([c["kind"], c["model"], str(c["ep"]), str(c["tokens"])])
        if key not in spans:
            continue
        target = float(c["mode"][5:]) * spans[key]
        floor = statistics.median(producer_span_us(x, 0) for _ in range(5))
        probe = 200_000
        slope = (statistics.median(producer_span_us(x, probe) for _ in range(3)) - floor) / (probe * len(x.p["groups"]))
        cyc = max(0, int((target - floor) / (slope * len(x.p["groups"])))) if slope > 0 else 0
        span = statistics.median(producer_span_us(x, cyc) for _ in range(3))
        cycles_all[cid] = cyc
        emit("PRODUCER", cid=cid, target_us=target, floor_us=floor, span_us=span, cycles=cyc)
    json.dump(dict(spans=spans, cycles=cycles_all), open(f"{sync}/cycles.tmp", "w"))
    os.replace(f"{sync}/cycles.tmp", f"{sync}/cycles.json")
t_wait = time.time()
while not os.path.exists(f"{sync}/cycles.json"):
    if dead_partner() or time.time() - t_wait > 1800:
        log("no cycles.json (base died / timeout)")
        break
    beat("wait cycles")
    time.sleep(0.05)
try:
    cycles_shared = json.load(open(f"{sync}/cycles.json"))["cycles"]
except Exception:
    cycles_shared = {}
cycles_of = {}
for cid, x in cells.items():
    if x is not None and by_cid[cid]["mode"].startswith("delay"):
        if cid in cycles_shared:
            cycles_of[cid] = cycles_shared[cid]
            emit("PRODUCER_USE", cid=cid, cycles=cycles_shared[cid])
        else:
            emit("RUN_FAIL", cid=cid, error="no shared producer cycles from the base role")
            cells[cid] = None


def read_turn():
    try:
        return int(open(f"{sync}/turn").read() or 0)
    except Exception:
        return 0


def set_turn(i):
    tmp = f"{sync}/turn.tmp.{role}"
    with open(tmp, "w") as fh:
        fh.write(str(i))
    os.replace(tmp, f"{sync}/turn")


steps = []
for p_ in range(P):
    for c in cells_all:
        rs = c["roles"] if p_ % 2 == 0 else list(reversed(c["roles"]))
        for r in rs:
            steps.append((p_, c["cid"], r))
DEADLINE = float(os.environ.get("END_AT", "1e12"))
nobarrier = False
_armed[0] = True
beat("armed")
open(f"{sync}/armed.{role}", "w").write(str(os.getpid()))
for i, (p_, cid, r) in enumerate(steps):
    if time.time() > DEADLINE:
        log("deadline at step", i)
        break
    if r != role:
        continue
    t0 = time.time()
    while not nobarrier and read_turn() < i:
        if dead_partner() or time.time() - t0 > 600 or time.time() > DEADLINE:
            nobarrier = True
            log("no barrier from step", i)
            break
        beat(f"turn wait {i}")
        time.sleep(0.0003)
    x = cells.get(cid)
    if x is not None and time.time() < DEADLINE:
        try:
            c = by_cid[cid]
            ts = []
            torch.cuda.synchronize()
            if c["mode"] == "ready":
                s = [torch.cuda.Event(enable_timing=True) for _ in range(R)]
                e = [torch.cuda.Event(enable_timing=True) for _ in range(R)]
                torch.cuda._sleep(20_000_000)
                for kk in range(R):
                    flush_buf.zero_()
                    reset_streaming_signal_workspace(x.buf["ws"], x.kernel, all_complete=True)
                    s[kk].record()
                    launch(x, main_stream)
                    e[kk].record()
                torch.cuda.synchronize()
                ts = [s[kk].elapsed_time(e[kk]) * 1000.0 for kk in range(R)]
            else:
                for kk in range(max(R // 5, 3)):
                    flush_buf.zero_()
                    reset_streaming_signal_workspace(x.buf["ws"], x.kernel)
                    e0, e1, pd = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True), torch.cuda.Event()
                    e0.record(main_stream)
                    prod_stream.wait_event(e0)
                    gemm_stream.wait_event(e0)
                    launch(x, gemm_stream)
                    produce(x, cycles_of[cid], prod_stream)
                    pd.record(prod_stream)
                    gemm_stream.wait_event(pd)
                    e1.record(gemm_stream)
                    torch.cuda.synchronize()
                    ts.append(e0.elapsed_time(e1) * 1000.0)
                    beat(f"delay launch step {i}")
            emit("SAMPLE", cid=cid, pass_=p_, step=i, med=statistics.median(ts), mn=min(ts), mx=max(ts), n=len(ts), nob=nobarrier)
            beat(f"sample {i}")
        except Exception as ex:
            emit("SAMPLE_FAIL", cid=cid, pass_=p_, error=f"{type(ex).__name__}: {ex}"[:1000])
    set_turn(i + 1)
    beat(f"step {i} done")
log("done")
