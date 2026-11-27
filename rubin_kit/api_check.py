#!/usr/bin/env python3
"""API-level check of the library's MXFP8 wgrad kernel selection on this GPU (SM107 Rubin, or SM100/SM103 as a smoke test).

python api_check.py [--shapes model:ep:tokens,...] [--modes 0,1] [--out FILE]
Env: FE_PYTHON (the tree's python/ root; the compiled module is not needed). Uses one GPU (CUDA_VISIBLE_DEVICES).

For each benchmark GEMM (default: the 10 shapes x dW1 / dW2), with CLC (mode 0) and the atomic counter (mode 1):
  - builds MXFP8 operands as lsw_rubin.py does (balanced experts padded to 256 rows, random E8M0 scales);
  - runs the public wrapper grouped_gemm_wgrad_wrapper_sm100 with dense output, so the API picks the kernel, and with
    discrete output, which runs the stock kernel;
  - checks that the picked kernel class is the expected one (EXPECTED below, which must also agree with the library rule
    prefer_over_dense) and that the dense output equals the discrete output byte for byte.
Then host-only cases at the rule's thresholds (no launch). Prints one line per check and a SUMMARY line.
"""

import argparse, json, os, sys, types

MODELS = {
    "deepseek_v3": dict(hidden=7168, experts=256, top_k=8, intermediate=2048),
    "qwen35_122b": dict(hidden=3072, experts=256, top_k=8, intermediate=1024),
    "qwen38_2p4t": dict(hidden=8192, experts=512, top_k=10, intermediate=2048),
    "qwen35_397b": dict(hidden=4096, experts=512, top_k=10, intermediate=1024),
    "glm53_flash": dict(hidden=4096, experts=288, top_k=8, intermediate=2048),
}
SHAPES = "deepseek_v3:4:8192,qwen35_122b:4:8192,qwen35_397b:4:8192,glm53_flash:4:8192,qwen38_2p4t:4:8192,deepseek_v3:8:8192,qwen35_122b:8:8192,qwen35_397b:8:8192,glm53_flash:8:8192,qwen38_2p4t:8:8192"
# (model, EP, GEMM) -> A reuse expected on (SM100/SM103, SM107); everything else gets the global-descriptor kernel.
REUSE = {
    ("qwen38_2p4t", 4, "dW1"): (True, True),
    ("qwen35_397b", 4, "dW1"): (True, True),
    ("deepseek_v3", 4, "dW1"): (False, True),
    ("qwen35_122b", 4, "dW1"): (False, True),
    ("glm53_flash", 4, "dW1"): (False, True),
    ("qwen38_2p4t", 8, "dW1"): (False, True),
    ("qwen35_397b", 8, "dW1"): (False, True),
}


def install_cudnn(python_root):
    class _AnyMeta(type):
        def __getattr__(cls, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return f"{cls.__name__}.{name}"

    class _Anything(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return _AnyMeta(name, (), {})

    package = types.ModuleType("cudnn")
    package.__path__ = [os.path.join(python_root, "cudnn")]
    sys.modules["cudnn"] = package
    for name in ("_compiled_module", "_pybind_module"):
        stub = _Anything("cudnn." + name)
        sys.modules["cudnn." + name] = stub
        setattr(package, name, stub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=SHAPES)
    ap.add_argument("--modes", default="0,1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    install_cudnn(os.environ["FE_PYTHON"])
    import torch
    from cudnn.api_base import get_device_type
    from cudnn.gemm.cutedsl.grouped.wgrad import _blockscaled_api
    from cudnn.gemm.cutedsl.grouped.wgrad.api import grouped_gemm_wgrad_wrapper_sm100

    rubin = get_device_type() == "rubin"
    dense_cls, reuse_cls = _blockscaled_api._dense_output_variants(rubin)
    out = open(args.out, "a") if args.out else None
    ok = fail = 0

    def report(passed, rec):
        nonlocal ok, fail
        ok, fail = ok + passed, fail + (not passed)
        rec["ok"] = bool(passed)
        print(("API OK   " if passed else "API FAIL ") + json.dumps(rec), flush=True)
        if out:
            out.write(json.dumps(rec) + "\n")
            out.flush()

    picked = []
    original = _blockscaled_api.GroupedGemmWgradBlockScaledAPI._select_kernel

    def spy(self):
        kernel = original(self)
        picked.append(kernel)
        return kernel

    _blockscaled_api.GroupedGemmWgradBlockScaledAPI._select_kernel = spy
    g = torch.Generator(device="cuda").manual_seed(3)
    cap = torch.cuda.get_device_capability()
    print(f"device {torch.cuda.get_device_name()} {cap} type {get_device_type()} dense {dense_cls.__name__} reuse {reuse_cls.__name__}", flush=True)

    for shape in args.shapes.split(","):
        model, ep, tokens = shape.split(":")
        ep, tokens = int(ep), int(tokens)
        md = MODELS[model]
        H, E, topk, I = md["hidden"], md["experts"], md["top_k"], md["intermediate"]
        el = E // ep
        per = -(-(tokens * topk * ep // E) // 256) * 256
        rows = el * per
        offsets = (torch.arange(1, el + 1, dtype=torch.int32) * per).cuda()
        for gname, (m, n) in {"dW1": (2 * I, H), "dW2": (H, I)}.items():
            a = (torch.randn(m, rows, generator=g, device="cuda") * 2).to(torch.float8_e4m3fn)
            b = (torch.randn(n, rows, generator=g, device="cuda") * 2).to(torch.float8_e4m3fn).t()
            cols = -(-(rows // 32) // 4) * 4
            sfa = torch.randint(124, 131, (m, cols), generator=g, device="cuda", dtype=torch.uint8).view(torch.float8_e8m0fnu)
            sfb = torch.randint(124, 131, (n, cols), generator=g, device="cuda", dtype=torch.uint8).view(torch.float8_e8m0fnu)
            expect_reuse = REUSE.get((model, ep, gname), (False, False))[1 if rubin else 0]
            expected = reuse_cls if expect_reuse else dense_cls
            rule = reuse_cls.prefer_over_dense(m, n, ab_bits=8, c_bits=16, has_global_scale=False, tokens=rows, experts=el)
            for mode in (int(x) for x in args.modes.split(",")):
                common = dict(
                    a_tensor=a,
                    b_tensor=b,
                    sfa_tensor=sfa,
                    sfb_tensor=sfb,
                    offsets_tensor=offsets,
                    acc_dtype=torch.float32,
                    wgrad_dtype=torch.bfloat16,
                    mma_tiler_mn=(256, 256),
                    cluster_shape_mn=(2, 1),
                    sf_vec_size=32,
                    atomic_counter_claim=bool(mode),
                )
                rec = dict(model=model, ep=ep, gemm=gname, m=m, n=n, rows_per_expert=per, experts=el, counter=mode, expected=expected.__name__, rule=rule)
                try:
                    dense_out = torch.full((el, m, n), float("nan"), dtype=torch.bfloat16, device="cuda")
                    start = len(picked)
                    grouped_gemm_wgrad_wrapper_sm100(output_mode="dense", wgrad_tensor=dense_out, **common)
                    rec["picked"] = [k.__name__ for k in picked[start:]]
                    stock_out = torch.full((el, m, n), float("nan"), dtype=torch.bfloat16, device="cuda")
                    ptrs = torch.tensor([e.data_ptr() for e in stock_out], dtype=torch.int64, device="cuda")
                    grouped_gemm_wgrad_wrapper_sm100(output_mode="discrete", wgrad_ptrs=ptrs, **common)
                    torch.cuda.synchronize()
                    rec["diff_bytes"] = int((dense_out.view(torch.uint16) != stock_out.view(torch.uint16)).sum().item())
                    rec["nan"] = int(torch.isnan(dense_out).sum().item())
                    del dense_out, stock_out
                    picks_ok = len(rec["picked"]) > 0 and all(p == expected.__name__ for p in rec["picked"])
                    report(picks_ok and rule == expect_reuse and rec["diff_bytes"] == 0 and rec["nan"] == 0, rec)
                except Exception as exc:  # noqa: BLE001
                    rec["error"] = repr(exc)[-500:]
                    report(False, rec)
            del a, b, sfa, sfb
            torch.cuda.empty_cache()

    # Host-only: the rule at its thresholds (mean tokens per expert vs the A ring; work units per M tile).
    capacity = reuse_cls.A_STAGES * reuse_cls.K_TILE
    for n, units in ((1536, 2), (2048, 2), (2304, 3), (3072, 3), (4096, 4)):
        for mean, short in ((capacity, True), (capacity + 1, False)):
            got = reuse_cls.prefer_over_dense(4096, n, ab_bits=8, c_bits=16, has_global_scale=False, tokens=2 * mean - (mean > capacity), experts=2)
            report(got == (short and units >= 3), dict(kind="threshold", n=n, units_per_m_tile=units, mean_tokens=mean, capacity=capacity, reuse=got))
    print(f"SUMMARY ok {ok} fail {fail}", flush=True)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
