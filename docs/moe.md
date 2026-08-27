# moe

`moe` implements Mixture-of-Experts routing and expert execution: selecting the highest-scoring experts for each token, grouping/routing token rows, and evaluating the selected expert MLPs efficiently. The crate includes both routing/top-k kernels and specialized GEMM/GEMV paths for dense and quantized expert weights.

## Original source files

From [`attention.rs`](https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src):

- `flashinfer_moe_adapter.cu`
- `fp8_moe_cutlass.cu`
- `moe_gemm.cu`
- `moe_gemm_gguf.cu`
- `moe_gemm_wmma.cu`
- `moe_gemv.cu`
- `moe_gguf_small_m.cu`
- `moe_w2_pack.cu`
- `moe_w2_unpack.cu`
- `moe_wmma_gguf.cu`
- `nvfp4_moe_cutlass.cu`
- `moe/cub_helpers.h`
- `moe/moe_utils.cuh`
- `moe/topk_softmax_kernels.cu`
- `deepseek_v4/ds_moe.cu`
- `trtllm/trtllm_fused_moe_dev_kernel.cu`
- `trtllm/trtllm_fused_moe_runner.cu`
- `trtllm/trtllm_fused_moe_routing_common.cu`
- `trtllm/trtllm_fused_moe_routing_custom_block.cu`
- `trtllm/trtllm_fused_moe_routing_custom_cluster.cu`
- `trtllm/trtllm_fused_moe_routing_deepseek.cu`
- `trtllm/trtllm_fused_moe_routing_llama4.cu`
- `trtllm/trtllm_fused_moe_routing_renormalize.cu`

## Mathematics

See [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) for the basic sparse-gating formulation. Modern LLM routing variants differ in balancing and normalization, but retain the same token-to-selected-experts structure.
