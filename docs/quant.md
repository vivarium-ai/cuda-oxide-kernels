# quant

`quant` contains conversion, scaling, packing, and dequantization support for reduced-precision weights and activations, including FP8, GPTQ, MXFP4, and NVFP4. The crate should describe formats and transformations; GEMM execution using those formats belongs in `gemm` or a specialized kernel crate.

## Original source files

From [`attention.rs`](https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src):

- `flashinfer_fp8_qquant.cu`
- `gptq_cuda_kernel.cu`
- `mlx_nvfp4_utils.cu`
- `mxfp4_quant.cu`
- `nvfp4_quant.cu`
- `nvfp4_quant_flashinfer.cu`
- `update_kvscales.cu`
- `moe_w2_pack.cu`
- `moe_w2_unpack.cu`
- `attention/dtype_e8m0.cuh`
- `attention/dtype_fp8.cuh`

## Mathematics

- FP8 representation and scaling: [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
- GPTQ: [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
