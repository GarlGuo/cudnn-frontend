"""Lock-step kernel timing across the 4 GPUs of a GB200 tray (one torchrun rank per GPU, each rank its own shape).

Every rank fires the same arm at the same time (host gloo barrier), after an idle gate of GATE_MS (default 20 ms) with
the GPU drained, so the GPUs that share the tray's power budget see the same neighbours in every arm.  Call
pin_rank_gpu() before importing torch."""
import os, statistics, time


def pin_rank_gpu():
    lr = os.environ.get("LOCAL_RANK")
    if lr is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = lr


def run(arms, pairs, emit, budget, reps, flush, t_start):
    """arms: {name: (prep, [launches])} in the same order on every rank.  One "pass" line per pass: per-arm median over
    2 * reps samples (forward then backward order, rotating start)."""
    import torch
    import torch.distributed as dist
    dist.init_process_group(backend="gloo")
    gate = float(os.environ.get("GATE_MS", "20")) / 1000.0
    names = list(arms)

    def fire(name):
        prep, launches = arms[name][0], arms[name][1]
        prep()
        flush.zero_()
        torch.cuda.synchronize()
        dist.barrier()
        time.sleep(gate)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(50000)  # ~25 us so the start event is not enqueued behind the host
        s.record()
        for f in launches:
            f()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) * 1000.0

    def keep_going():
        flag = torch.tensor([1 if time.time() - t_start < budget else 0], dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    for n in names:
        fire(n)
    emit(dict(kind="arms", arms=names, setup_s=round(time.time() - t_start, 1), gate_ms=gate * 1000))
    p = 0
    while keep_going():
        k = p % len(names)
        order = names[k:] + names[:k]
        samples = {n: [] for n in names}
        for n in order + order[::-1]:
            for _ in range(reps):
                samples[n].append(fire(n))
        med = {n: statistics.median(v) for n, v in samples.items()}
        emit(dict(kind="pass", p=p, us=med, ratios={f"{x}/{y}": med[x] / med[y] for x, y in pairs}))
        p += 1
    emit(dict(kind="done", passes=p, s=round(time.time() - t_start, 1)))
    dist.barrier()
    dist.destroy_process_group()
