# marlin

`marlin` implements the Marlin mixed-precision matrix-multiplication kernel for weight-only quantized LLM inference, principally FP16/BF16 activations multiplied by packed 4-bit weights. Its distinctive contribution is a GPU work-partitioning and data-movement scheme designed to retain near-ideal quantization speedups at medium autoregressive batch sizes.

## Original source files

From [`attention.rs`](https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src):

- `marlin/marlin.cuh`
- `marlin/marlin_cuda_kernel.cuh`
- `marlin/marlin_dtypes.cuh`
- `marlin/marlin_gptq_cuda_kernel.cuh`
- `marlin/marlin_matmul_awq_bf16.cu`
- `marlin/marlin_matmul_awq_f16.cu`
- `marlin/marlin_matmul_bf16.cu`
- `marlin/marlin_matmul_f16.cu`
- `marlin/marlin_repack.cu`

Original upstream: [IST-DASLab/marlin](https://github.com/IST-DASLab/marlin).

## Mathematics

See [MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models](https://arxiv.org/abs/2408.11743). It explains the mixed-precision matrix multiplication, quantized-weight representation, striped partitioning, and data-movement strategy underlying the kernel.
