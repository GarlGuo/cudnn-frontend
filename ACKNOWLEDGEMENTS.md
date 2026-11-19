# Acknowledgements

## CuTe DSL OSS kernels

Several open-source kernels in this repository originated in NVIDIA's CuTe DSL kernel
library. We gratefully acknowledge the contributors who helped develop, bring up,
and test these kernels.

Contributors include:

- Rachit Garg <rachitg@nvidia.com>
- Jack Yang <jackyang@nvidia.com>
- Hao Sheng <hsheng@nvidia.com>
- Alex Li <alel@nvidia.com>
- Aragorn Guan <aragorng@nvidia.com>
- Bangyu Shen <bangyus@nvidia.com>
- Caleb Du <cadu@nvidia.com>
- Xiao Song <xiaos@nvidia.com>
- Siddhartha Raman <sraman@nvidia.com>

This acknowledgement covers the CuTe DSL kernel work now represented by these
modules in `python/cudnn/`:

- `gemm/cutedsl/dense/amax`
- `gemm/cutedsl/dense/dsrelu`
- `gemm/cutedsl/dense/srelu`
- `gemm/cutedsl/dense/swiglu`
- `gemm/cutedsl/grouped/dglu`
- `gemm/cutedsl/grouped/dsrelu`
- `gemm/cutedsl/grouped/dswiglu`
- `gemm/cutedsl/grouped/glu`
- `gemm/cutedsl/grouped/glu_hadamard`
- `gemm/cutedsl/grouped/quant`
- `gemm/cutedsl/grouped/srelu`
- `gemm/cutedsl/grouped/swiglu`
- `gemm/cutedsl/grouped/wgrad`
- `gemm/cutedsl/discrete_grouped/dswiglu`
- `gemm/cutedsl/discrete_grouped/swiglu`
- `rmsnorm_rht_amax`

Thank you also to the broader CUTLASS/CuTe DSL and infrastructure teams who
supported the original kernel development.

## Streaming signal grouped GEMM scheduling

The streaming input-block and output-chunk tile schedulers let the MoE grouped GEMMs run concurrently with the
kernels that produce their inputs and consume their outputs, connected only by device-side readiness signals.
We gratefully acknowledge the contributors who designed, developed, and validated this work.

Contributors include:

- Wentao Guo
- Phuong Nguyen
- Zhongbo Zhu
- Subhadeep Bhattacharya
- Fei Wu
- Jimmy Zhang

This acknowledgement covers these modules in `python/cudnn/`:

- `gemm/cutedsl/grouped/moe_streaming_input_block_tile_scheduler.py`
- `gemm/cutedsl/grouped/moe_streaming_output_chunk_tile_scheduler.py`
- `gemm/cutedsl/grouped/moe_signal_helpers.py`
- `gemm/cutedsl/grouped/moe_tail_tiles.py`
- the streaming signal options of `gemm/cutedsl/grouped/unfused` and `gemm/cutedsl/grouped/glu`
