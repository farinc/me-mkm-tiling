FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential cmake pkg-config \
    libhwloc-dev libssl-dev libtirpc-dev libpam-dev libncurses-dev \
    libdb-dev \
    m4 dos2unix \
    libopenmpi-dev openmpi-bin \
    petsc-dev slepc-dev \
    && rm -rf /var/lib/apt/lists/*

USER vscode
WORKDIR /home/vscode

ENV RUSTUP_HOME=/home/vscode/.rustup \
    CARGO_HOME=/home/vscode/.cargo \
    PATH=/home/vscode/.cargo/bin:/home/vscode/.local/bin:$PATH

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile default \
    && rustup component add rust-src

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN uv tool install nbstripout
