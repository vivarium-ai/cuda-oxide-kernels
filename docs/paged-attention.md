# paged-attention

`paged-attention` implements attention over a KV cache stored in non-contiguous fixed-size pages/blocks. It preserves ordinary scaled dot-product attention semantics while translating logical token positions through a block table, allowing the inference runtime to manage KV memory without requiring large contiguous allocations.

## Original source files

From [`attention.rs`](https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src):

- `paged_attention_v1.cu`
- `paged_attention_v2.cu`
- `pagedattention.cuh`
- `prefill_paged_attn.cu`
- `prefill_paged_attn_opt.cu`

Closely coupled cache files:

- `copy_blocks_kernel.cu`
- `reshape_and_cache_kernel.cu`

## Mathematics / design

- [vLLM PagedAttention kernel walkthrough](https://docs.vllm.ai/en/latest/design/paged_attention/)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
