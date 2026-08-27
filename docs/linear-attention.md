# linear-attention

`linear-attention` contains recurrent or state-space-like alternatives to quadratic softmax attention, currently centered on Gated DeltaNet/GDN and Mamba-style state-cache operations. These kernels update a compact recurrent state instead of storing and attending over every previous token.

## Original source files

From [`attention.rs`](https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src):

- `gdn.cu`
- `gdn_flashinfer_prefill.cu`
- `mamba_scatter_kernel.cu`

## Mathematics

See [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464), which derives the gated delta update rule and its parallel formulation.
