# wgrad timing benches (not for upstream)

Run from test/python of this tree (bench_wgrad_signal.py imports the signal test helpers), with this tree's python/ on
PYTHONPATH. Same seed (default --seed 0) and routing on every machine: per EP rank a 4096-token microbatch, top-k,
log-normal (sigma 1) expert counts floored to 256 tokens (BF16) / 128 (MXFP8).

Cells: --shape deepseek_v3|qwen35, --ep 4|8, --precision bf16|mxfp8 (FC1 and FC2 both by default).

1. CLC vs atomic_counter_claim (both arms in one process, outputs must be byte-equal):
   python ../../benchmark/gemm/wgrad_rubin/bench_wgrad_acc.py --shape deepseek_v3 --ep 8 --precision bf16 --out acc.json
2. Streaming signal cost with every tile ready (signal off vs on, counter path, byte-equal):
   python ../../benchmark/gemm/wgrad_rubin/bench_wgrad_signal.py --shape deepseek_v3 --ep 8 --precision bf16 --mode ready --out sig.json
   Optional overlap with a delayed producer: --mode overlap --windows 0.5,1.0

Optional third arm for the streaming signal (not in any PR): `v7_window.patch` applies on
`sm100-wgrad-streaming-input-signal-v7` (`git apply`). The TMA warp keeps a 32-token-tile ready window across work
tiles, so it loads `signal[]` about once per 32 token tiles instead of once per expert.
