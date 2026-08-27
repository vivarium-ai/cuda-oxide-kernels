FROM ghcr.io/vivarium-ai/cuda-oxide-dev-sv:amd64

ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
ENV PATH=/usr/local/cargo/bin:${PATH}

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --profile minimal \
    && rustup toolchain install nightly-2026-04-03 \
    && rustup component add \
        rust-src \
        rustc-dev \
        rust-analyzer \
        clippy \
        llvm-tools \
        --toolchain nightly-2026-04-03 \
    && rustup default nightly-2026-04-03

RUN rustc --version
