FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential cmake git pkg-config \
    libhwloc-dev libssl-dev libtirpc-dev libpam-dev libncurses-dev \
    libdb-dev \
    m4 sudo curl ca-certificates dos2unix \
    libopenmpi-dev openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash vscode \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/vscode \
    && chmod 0440 /etc/sudoers.d/vscode

USER vscode
WORKDIR /home/vscode

ENV RUSTUP_HOME=/home/vscode/.rustup \
    CARGO_HOME=/home/vscode/.cargo \
    PATH=/home/vscode/.cargo/bin:/home/vscode/.local/bin:$PATH

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile default \
    && rustup component add rust-src

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN uv tool install nbstripout
