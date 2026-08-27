# gemm

`gemm` implements general and quantized matrix multiplication used by transformer linear layers and expert networks. It should capture the mathematical operation `D = A × B + C` while expressing GPU tiling, MMA use, and datatype specialization directly in Rust rather than reproducing CUTLASS template machinery.

## Original source files

From [`attention.rs`](https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src):

- `flashinfer_bmm_fp8.cu`
- `fp8_gemm_cutlass.cu`
- `fp8_matmul.cu`
- `gguf_gemm.cu`
- `mxfp4_gemm.cu`
- `mxfp4_gemm_cutlass.cu`
- `mxfp4_gemm_wmma.cu`
- `nvfp4_gemm.cu`
- `nvfp4_gemm_cutlass.cu`
- `nvfp4_gemm_flashinfer.cu`
- `nvfp4_gemm_flashinfer_sm103.cu`
- `trtllm/trtllm_batched_gemm_runner.cu`
- `trtllm/trtllm_cutlass_heuristic.cpp`

## Mathematics

For the basic operation and GPU decomposition, see the [NVIDIA CUTLASS GEMM model](https://docs.nvidia.com/cutlass/media/docs/cpp/gemm_api.html). CUTLASS describes GEMM as hierarchical tiled evaluation of matrix multiply-accumulate.
