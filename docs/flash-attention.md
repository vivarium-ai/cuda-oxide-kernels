# flash-attention

`flash-attention` implements exact scaled dot-product attention using tiled, IO-aware algorithms that avoid materializing the full attention matrix in GPU global memory. This crate covers both the native FlashAttention-style kernels in `attention.rs` and the FlashAttention-3-derived implementation wrapped by `flashattn.rs`.

## Original source files

From [`attention.rs`](https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src):

- `flash/flash_decode.cu`
- `flash/flash_decode_paged.cuh`
- `flash/flash_decode_paged_fp8.cuh`
- `flash/flash_instantiate.cu`
- `flash/flash_prefill_paged.cuh`
- `flash/flash_prefill_paged_fp8.cuh`
- `flash/flash_prefill_tq3.cuh`
- `flash/flash_prefill_tq4.cuh`
- `flash/flash_reshape_cache.cuh`
- `flash/flash_sm_compat.cuh`
- `flash/flash_turboquant.cuh`
- `flash/flash_turboquant_lowbit.cuh`

From [`flashattn.rs`](https://github.com/guoqingbao/flashattn.rs/tree/main/kernels):

- `block.h`
- `copy_sm90_bulk_reduce.hpp`
- `epilogue_fwd.hpp`
- `flash.h`
- `flash_api_dispatch.cu`
- `flash_api_impl.h`
- `flash_api_sm80.cu`
- `flash_api_sm90.cu`
- `flash_fwd_combine.cu`
- `flash_fwd_combine_kernel.h`
- `flash_fwd_combine_launch_template.h`
- `flash_fwd_kernel_sm80.h`
- `flash_fwd_kernel_sm90.h`
- `flash_fwd_launch_template.h`
- `flash_prepare_scheduler.cu`
- `heuristics.h`
- `mainloop_fwd_sm80.hpp`
- `mainloop_fwd_sm90_tma_gmma_ws.hpp`
- `mask.h`
- `named_barrier.hpp`
- `pack_gqa.h`
- `paged_kv.h`
- `rotary.h`
- `seqlen.h`
- `sm90_pipeline_no_cluster.hpp`
- `softmax.h`
- `static_switch.h`
- `tile_scheduler.hpp`
- `tile_size.h`
- `utils.h`
- `instantiations/*.cu`

The backward-only `epilogue_bwd.hpp` and `mainloop_bwd_*` files can be omitted for an inference-only first port.

## Mathematics

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)
