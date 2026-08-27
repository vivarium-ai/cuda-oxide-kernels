# sampling

`sampling` implements GPU-side token selection from model logits, including top-k and nucleus/top-p sampling and the reductions/sorts required to perform them efficiently. It moves generation-time sampling off the host and avoids transferring full logits distributions back to the CPU.

## Original source files

From [`attention.rs`](https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src):

- `fast_topk.cu`
- `gpu_sampling.cu`
- `gpu_sampling.cuh`
- `sort.cu`

## Mathematics

See [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751), which defines nucleus (top-p) sampling and discusses top-k sampling in the context of neural text generation.
