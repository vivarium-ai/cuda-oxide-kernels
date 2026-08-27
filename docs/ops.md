# ops

`ops` contains small, generally reusable GPU operations used by inference models: fused activation/gating, rotary position embedding, masking, and related elementwise or indexing operations. These are useful first ports because they exercise the Rust/CUDA surface without requiring a large attention or GEMM implementation.

## Original source files

From [`attention.rs`](https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src):

- `fused_rope.cu`
- `gptoss_swiglu.cu`
- `mask.cu`
- `silu_and_mul.cu`

## Mathematics

- Rotary position embedding: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- SiLU/Swish activation: [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
