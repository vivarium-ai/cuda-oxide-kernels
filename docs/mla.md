# mla

`mla` implements Multi-head Latent Attention and its paged/sparse variants. MLA compresses the key/value state into a lower-dimensional latent representation, reducing KV-cache memory and memory bandwidth during inference while reconstructing the information needed for attention.

## Original source files

From [`attention.rs`](https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src):

- `concat_and_cache_mla_kernel.cu`
- `dsa_lightning_indexer.cu`
- `flashinfer_mla.cu`
- `flashinfer_sparse_mla_dsv4.cu`
- `flashmla_sparse_mla.cu`
- `mla_paged_attention.cu`
- `mla_sparse_attention.cu`
- `deepseek_v4/ds_compressor.cu`
- `deepseek_v4/ds_fp8_kv_pack.cu`
- `deepseek_v4/ds_hc.cu`
- `deepseek_v4/ds_indexer.cu`
- `deepseek_v4/ds_sparse_attn.cu`

## Mathematics

MLA was introduced in [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434). For a paper focused specifically on the mechanism and KV-cache compression, see [Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs](https://arxiv.org/abs/2502.14837).
