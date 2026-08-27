# kernel-core

`kernel-core` provides the low-level GPU abstractions shared by the other crates: scalar/device datatypes, packed vector operations, reductions, matrix multiply-accumulate (MMA), asynchronous memory copies, synchronization, and work/tile scheduling. It is infrastructure rather than a single mathematical kernel.

## Original source files

From [`attention.rs`](https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src):

- `attention/attention_dtypes.h`
- `attention/attention_generic.cuh`
- `attention/attention_utils.cuh`
- `attention/dtype_bfloat16.cuh`
- `attention/dtype_e8m0.cuh`
- `attention/dtype_float16.cuh`
- `attention/dtype_float32.cuh`
- `attention/dtype_fp8.cuh`
- `cuda_compat.h`
- `flash/flash_sm_compat.cuh`

From [`flashattn.rs`](https://github.com/guoqingbao/flashattn.rs/tree/main/kernels):

- `block.h`
- `copy_sm90_bulk_reduce.hpp`
- `named_barrier.hpp`
- `sm90_pipeline_no_cluster.hpp`
- `static_switch.h`
- `tile_scheduler.hpp`
- `tile_size.h`
- `utils.h`

## Technical reference

There is no single kernel mathematics paper for this crate. The closest primary reference is the [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/), especially its sections on warp reductions, matrix operations/Tensor Cores, asynchronous copies, barriers, and scheduling.
