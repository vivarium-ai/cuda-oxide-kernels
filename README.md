# Ports of C++/Python kernels to cuda-oxide

The purpose of this repository is to organize and facilitate the porting of various existing C++/Python CUDA kernels to Rust using [cuda-oxide](https://github.com/nvlabs/cuda-oxide).

The ports are organized as reasonably contained Rust crates that can be used individually.

In addition, this repository contains Docker files, scripts, and GitHub CI workflows to coherently organize the disparate components and provide consistency across dev, test, build, and deploy functions.

## Code Organization

The following is the toplevel organization of this repository:

```
.
├── .github     # GitHub workflows
├── crates      # Rust crates for cuda-oxide kernels
├── docker      # Dockerfiles for various dev, test, build, deploy containers
├── docs        # Documentation related to porting
├── Makefile    # Convenience tasks and documentation
├── README.md   # This file
├── scripts     # Various development functions as repeatable scripts
└── upstream    # Git submodules for code being ported and related components
```

## Kernels

These are the Rust CUDA kernels included in this repository:

- [`flash-attention`](/docs/flash-attention.md)
- [`gemm`](/docs/gemm.md)
- [`kernel-core`](/docs/kernel-core.md)
- [`kv-cache`](/docs/kv-cache.md)
- [`linear-attention`](/docs/linear-attention.md)
- [`marlin`](/docs/marlin.md)
- [`mla`](/docs/mla.md)
- [`moe`](/docs/moe.md)
- [`ops`](/docs/ops.md)
- [`paged-attention`](/docs/paged-attention.md)
- [`quant`](/docs/quant.md)
- [`sampling`](/docs/sampling.md)

## License

This project is licensed under the [MIT License](/LICENSE).
