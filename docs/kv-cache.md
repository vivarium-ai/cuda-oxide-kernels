# kv-cache

`kv-cache` manages the key/value tensors retained across autoregressive decoding, including paged block copies, layout conversion, and insertion of newly computed K/V vectors. These kernels are primarily memory-layout and indexing operations rather than new attention mathematics.

## Original source files

From [`attention.rs`](https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src):

- `copy_blocks_kernel.cu`
- `reshape_and_cache_kernel.cu`
- `flash/flash_reshape_cache.cuh`

## Mathematics / design

See the [vLLM PagedAttention design document](https://docs.vllm.ai/en/latest/design/paged_attention/) and [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180). They explain why the KV cache is paged and how the attention computation addresses it.
