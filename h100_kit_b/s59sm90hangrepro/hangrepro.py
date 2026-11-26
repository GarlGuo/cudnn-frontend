"""s58sm90hangrepro: is the s57sm90plainperf2 r0.dglu.deepseek_v3.ep4.t4096 failure (base 228f92ba, SIGSEGV during a
faulthandler periodic dump) a GPU hang of the SM90 dGLU streaming signal kernel?  One process, one tree, one mode of that
cell (dglu, deepseek_v3, EP4, T4096: 64 experts x 512 rows, n 2048, k 7168), launched back to back until the deadline.

Usage: hangrepro.py TREE MODE SECS OUTDIR
  TREE   base | old: only a label (the tree is whatever `cudnn` imports; job.sh sets PYTHONPATH for old).
  MODE   ready | delay1.0 | delay0.5   (s57's cell modes; producer cycles fixed to s57's shared calibration)
         san                           (compute-sanitizer: a few ready launches + a few launches whose producer finished
                                        before the GEMM; the sanitizer serializes kernels, so no concurrent producer)
Env: PREARM_S (900): the same limit before the loop starts (imports, compile, check; san mode throughout).
     STALL_S (60): a watchdog thread exits rc 86 after dumping every thread's stack (sys._current_frames under the GIL,
     then faulthandler.dump_traceback from the same thread) when the main loop makes no progress for STALL_S seconds.
     No faulthandler.dump_traceback_later is armed (it walks a running thread's frames without the GIL; s57's SIGSEGV).
     FH_STRESS (unset): seconds; arm faulthandler.dump_traceback_later(FH_STRESS, repeat=True) after the build, to test
     whether the periodic dump alone crashes a busy process (the suspected s57 harness artifact).
     faulthandler.register(SIGUSR1): job.sh's external stall fallback (the watchdog thread itself starved) dumps via it.
Lines: TREE, COMPILED, CHECKSUM, PRODUCER, WIN (every 30 s: launches, kernel/step time stats), BAD (checksum mismatch),
       STALL, DONE {launches, bad}.
"""
import json, os, sys, time, statistics, traceback, hashlib, faulthandler, signal, threading

STALL_S = float(os.environ.get("STALL_S", "60"))
RC_STALL = 86
CUDA_LAUNCH_BLOCKING = os.environ.get("CUDA_LAUNCH_BLOCKING")
os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")
faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)

tree, mode, secs, outdir = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
_hb = [time.monotonic(), "start"]
_armed = [False]


def beat(where):
    _hb[0] = time.monotonic()
    _hb[1] = where


def log(*a):
    print(f"[{tree} {time.strftime('%H:%M:%S')}]", *a, flush=True)


def emit(kind, **kw):
    print(kind, json.dumps(dict(tree=tree, mode=mode, **kw)), flush=True)


def _watchdog():
    wd_file = os.path.join(outdir, f"wd.{tree}.{mode}")
    while True:
        time.sleep(1.0)
        try:
            with open(wd_file, "w") as fh:
                fh.write(f"{time.time():.0f} {_hb[1]}\n")
        except Exception:
            pass
        age = time.monotonic() - _hb[0]
        # Before the loop is armed (imports, compile, check) the limit is 15 min: cute.compile holds the GIL for seconds.
        limit = STALL_S if _armed[0] else float(os.environ.get("PREARM_S", "900"))
        if age > limit:
            try:
                out = [f"STALL {json.dumps(dict(tree=tree, mode=mode, secs=round(age, 1), where=_hb[1], t=time.strftime('%H:%M:%S')))}"]
                main_id = threading.main_thread().ident
                for tid, fr in sys._current_frames().items():
                    out.append(f"--- thread {tid}{' (main)' if tid == main_id else ''}")
                    out.extend(x.rstrip() for x in traceback.format_stack(fr))
                print("\n".join(out), flush=True)
                faulthandler.dump_traceback(file=sys.stdout, all_threads=True)
                sys.stdout.flush()
            finally:
                os._exit(RC_STALL)


threading.Thread(target=_watchdog, daemon=True, name="stall_watchdog").start()

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
from cudnn.gemm.cutedsl.grouped.dglu.moe_grouped_gemm_dglu_sm90 import MoEGroupedGemmDgluBf16KernelSm90 as Dglu

ALIGN, GROUPS, CHUNK, R = 256, 32, 4096, 50
E, N, K, ROUTED = 64, 2048, 7168, 512
# s57sm90plainperf2 r0.dglu.deepseek_v3.ep4.t4096 sync/cycles.json (the base role's shared calibration)
CYCLES = {"delay1.0": 64128, "delay0.5": 21358}
KEY = ("dglu", "deepseek_v3", 4, 4096)

torch.cuda.set_device(0)
dev = "cuda"
torch.zeros(1, device=dev)
FULL = HardwareInfo().get_max_active_clusters(2)
emit("TREE", cudnn=cudnn.__file__, clusters=FULL, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
     python=sys.version.split()[0], cuda_launch_blocking=CUDA_LAUNCH_BLOCKING, stall_s=STALL_S, fh_stress=os.environ.get("FH_STRESS"))
main_stream = torch.cuda.current_stream()
prod_stream = torch.cuda.Stream()
gemm_stream = torch.cuda.Stream()


def seed_of(*key):
    return int(hashlib.md5(repr(key).encode()).hexdigest()[:8], 16)


def t(x):
    return from_dlpack(x, assumed_align=16)


def mkl(x):
    return x.as_strided((x.shape[0], x.shape[1], 1), (x.shape[1], 1, x.shape[0] * x.shape[1]))


def make_problem():
    # s57 worker90.py get_problem, same seed: the same A / B / C bytes as the failing job.
    routed = [ROUTED] * E
    padded = [-(-r // ALIGN) * ALIGN for r in routed]
    rows = max(sum(padded), ALIGN)
    starts = [0]
    for b in padded[:-1]:
        starts.append(starts[-1] + b)
    g = torch.Generator(device=dev).manual_seed(seed_of(*KEY))

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

    p = dict(rows=rows, n=N, k=K, padded=padded, starts=starts, routed=routed)
    p["offsets"] = torch.tensor(padded, dtype=torch.int64, device=dev).cumsum(0).to(torch.int32)
    p["alpha"] = torch.ones(E, device=dev)
    p["prob"] = torch.rand(rows, 1, 1, generator=g, device=dev)
    p["a"] = rows_rnd(K, K ** -0.5)
    p["b"] = rnd(E, N, K, s=K ** -0.5)
    p["c"] = rows_rnd(2 * N, 1.0)
    p["beta"] = torch.ones(E, device=dev)
    first = [s // ALIGN for s in starts]
    ntiles = [b // ALIGN for b in padded]
    order = [first[e] + j for j in range(max(ntiles)) for e in range(E) if j < ntiles[e]]
    p["groups"] = [torch.tensor(gr, dtype=torch.int64, device=dev) for gr in
                   (order[i * len(order) // GROUPS:(i + 1) * len(order) // GROUPS] for i in range(min(GROUPS, len(order)))) if gr]
    return p


class Cell:
    pass


def build(p, clusters):
    kernel = Dglu(acc_dtype=cutlass.Float32, use_2cta_instrs=False, mma_tiler_mn=(128, 256), cluster_shape_mn=(2, 1), vectorized_f32=False,
                  expert_cnt=E, weight_mode=MoEWeightMode.DENSE, use_dynamic_sched=True, act_func="dswiglu",
                  streaming_signal_capacity=p["rows"] // ALIGN)
    rows = p["rows"]
    buf = dict(ws=torch.zeros(max(kernel.get_workspace_bytes(), 128), dtype=torch.uint8, device=dev),
               d=torch.full((rows, 2 * N), float("nan"), dtype=torch.bfloat16, device=dev),
               dprob=torch.zeros(rows, 1, 1, device=dev))
    args = [t(mkl(p["a"])), t(p["b"].permute(1, 2, 0)), cutlass.Int32(N), cutlass.Int32(K), cutlass.Int64(K), OperandMajorMode.K,
            from_dlpack(buf["ws"], assumed_align=128).iterator,
            t(mkl(p["c"])), t(mkl(buf["d"])), from_dlpack(p["offsets"], assumed_align=4), t(p["alpha"]), t(p["beta"]), t(p["prob"]),
            t(buf["dprob"]), cutlass.Float32(0.0), None]
    t0 = time.time()
    compiled = cute.compile(kernel, *args, clusters, cuda.CUstream(main_stream.cuda_stream))
    x = Cell()
    x.kernel, x.buf, x.p, x.compiled = kernel, buf, p, compiled
    x.launch_args = [a for i, a in enumerate(args) if i != 5]
    off = streaming_signal_workspace_offsets(kernel)
    x.signal = buf["ws"].view(torch.int32)[off["signal"] // 4: off["signal"] // 4 + rows // ALIGN]
    emit("COMPILED", secs=round(time.time() - t0, 1), rows=rows, clusters=clusters, m_tiles=rows // ALIGN, groups=len(p["groups"]))
    return x


def launch(x, s):
    x.compiled(*x.launch_args, cuda.CUstream(s.cuda_stream))


def produce(x, cycles, s):
    with torch.cuda.stream(s):
        for g in x.p["groups"]:
            if cycles:
                torch.cuda._sleep(cycles)
            x.signal.index_copy_(0, g, g.to(torch.int32))


p = make_problem()
real_rows = torch.cat([torch.arange(s, s + r, device=dev) for s, r in zip(p["starts"], p["routed"])])
wsum = None


def checksum(x):
    global wsum
    v = x.buf["d"].index_select(0, real_rows).contiguous().view(torch.int32).to(torch.int64).view(-1)
    if wsum is None:
        wsum = torch.arange(v.numel(), device=dev, dtype=torch.int64) % 65521
    return [int(v.sum()), int((v * wsum).sum())]


delayed = mode.startswith("delay")
x = build(p, FULL - 1 if delayed else FULL)
flush_buf = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
beat("built")

# Producer-side kernels once on their own before any streamed launch (as s57).
sig = torch.full_like(x.signal, -1)
with torch.cuda.stream(prod_stream):
    torch.cuda._sleep(1000)
    for g in p["groups"]:
        sig.index_copy_(0, g, g.to(torch.int32))
torch.cuda.synchronize()

# Reference: one all-ready launch.  s57's failing job: [-6823832267924107, -2178886971186175923] on every tree and mode.
reset_streaming_signal_workspace(x.buf["ws"], x.kernel, all_complete=True)
launch(x, main_stream)
torch.cuda.synchronize()
ref = checksum(x)
nan = bool(torch.isnan(x.buf["d"][p["starts"][0]].float()).any())
emit("CHECKSUM", rows=ref, nan=nan, s57_ref=[-6823832267924107, -2178886971186175923])
beat("checked")
deadline = time.time() + secs
n_launch = 0
n_bad = 0
ts = []
win_t0 = time.time()
step = 0


def window(force=False):
    global ts, win_t0
    if ts and (force or time.time() - win_t0 >= 30):
        q = sorted(ts)
        emit("WIN", t=time.strftime("%H:%M:%S"), launches=n_launch, n=len(ts), med=round(statistics.median(ts), 1), mn=round(q[0], 1),
             p99=round(q[int(0.99 * (len(q) - 1))], 1), mx=round(q[-1], 1), bad=n_bad)
        ts = []
        win_t0 = time.time()


def check_now(where):
    """Outside the timed loop: D to NaN, one launch of this job's kind (ready, or streamed beside the producer), D compared
    with the reference (a skipped or doubled tile shows; the timed launches overwrite D in place and cannot)."""
    global n_bad
    x.buf["d"].fill_(float("nan"))
    if delayed:
        reset_streaming_signal_workspace(x.buf["ws"], x.kernel)
        torch.cuda.synchronize()
        gemm_stream.wait_stream(main_stream)
        prod_stream.wait_stream(main_stream)
        launch(x, gemm_stream)
        produce(x, CYCLES[mode], prod_stream)
    else:
        reset_streaming_signal_workspace(x.buf["ws"], x.kernel, all_complete=True)
        launch(x, main_stream)
    torch.cuda.synchronize()
    cs = checksum(x)
    if cs != ref:
        n_bad += 1
        if n_bad <= 20:
            emit("BAD", where=where, launches=n_launch, rows=cs, ref=ref)


def san_compare(where):
    global n_bad
    cs = checksum(x)
    if cs != ref:
        n_bad += 1
        emit("BAD", where=where, launches=n_launch, rows=cs, ref=ref)


if mode == "san":
    # Ready launches, then launches whose producer ran to completion first (signal reset to -1, released by the
    # producer's index_copy_, then the GEMM): the sanitizer serializes kernels, so a concurrent producer would deadlock.
    for kk in range(3):
        x.buf["d"].fill_(float("nan"))
        reset_streaming_signal_workspace(x.buf["ws"], x.kernel, all_complete=True)
        launch(x, main_stream)
        torch.cuda.synchronize()
        n_launch += 1
        san_compare(f"san_ready_{kk}")
        beat("san_ready")
    for kk in range(3):
        x.buf["d"].fill_(float("nan"))
        reset_streaming_signal_workspace(x.buf["ws"], x.kernel)
        produce(x, 0, main_stream)
        launch(x, main_stream)
        torch.cuda.synchronize()
        n_launch += 1
        san_compare(f"san_prestaged_{kk}")
        beat("san_prestaged")
    emit("DONE", launches=n_launch, bad=n_bad)
    log("done")
    sys.stdout.flush()
    os._exit(1 if n_bad else 0)

if os.environ.get("FH_STRESS"):
    faulthandler.dump_traceback_later(float(os.environ["FH_STRESS"]), repeat=True)
    log("FH_STRESS: faulthandler.dump_traceback_later", os.environ["FH_STRESS"], "repeat=True")
_armed[0] = True
beat("armed")
open(os.path.join(outdir, f"armed.{tree}.{mode}"), "w").write(str(os.getpid()))

if mode == "ready":
    s = [torch.cuda.Event(enable_timing=True) for _ in range(R)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(R)]
    while time.time() < deadline:
        torch.cuda._sleep(20_000_000)
        for kk in range(R):
            flush_buf.zero_()
            reset_streaming_signal_workspace(x.buf["ws"], x.kernel, all_complete=True)
            s[kk].record()
            launch(x, main_stream)
            e[kk].record()
        torch.cuda.synchronize()
        n_launch += R
        ts.extend(s[kk].elapsed_time(e[kk]) * 1000.0 for kk in range(R))
        beat(f"ready step {step}")
        step += 1
        if step % 20 == 0:
            check_now(f"ready step {step}")
        window()
elif delayed:
    cyc = CYCLES[mode]
    emit("PRODUCER", cycles=cyc)
    while time.time() < deadline:
        flush_buf.zero_()
        reset_streaming_signal_workspace(x.buf["ws"], x.kernel)
        e0, e1, pd = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True), torch.cuda.Event()
        e0.record(main_stream)
        prod_stream.wait_event(e0)
        gemm_stream.wait_event(e0)
        launch(x, gemm_stream)
        produce(x, cyc, prod_stream)
        pd.record(prod_stream)
        gemm_stream.wait_event(pd)
        e1.record(gemm_stream)
        torch.cuda.synchronize()
        n_launch += 1
        ts.append(e0.elapsed_time(e1) * 1000.0)
        beat(f"{mode} launch {n_launch}")
        if n_launch % 500 == 0:
            check_now(f"{mode} launch {n_launch}")
        window()
else:
    raise SystemExit(f"bad mode {mode}")

check_now("final")
window(force=True)
emit("DONE", launches=n_launch, bad=n_bad)
log("done")
sys.stdout.flush()
os._exit(1 if n_bad else 0)
