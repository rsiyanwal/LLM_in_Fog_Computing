#!/usr/bin/env bash
set -euo pipefail

# deps (curl dev for -hf path; OpenBLAS for BLAS build)
sudo apt-get update
sudo apt-get install -y git build-essential cmake pkg-config ninja-build \
    libopenblas-dev libcurl4-openssl-dev ca-certificates zlib1g-dev

# keep OpenBLAS from over-threading vs. llama -t
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

mkdir -p "$HOME/edge-llm-bench/bin"
cd "$HOME/edge-llm-bench"

# clone/update
if [ ! -d llama.cpp ]; then
  git clone https://github.com/ggml-org/llama.cpp
fi
cd llama.cpp
git pull --ff-only || true

# -------- BLAS build (OpenBLAS) --------
rm -rf build-blas
cmake -S . -B build-blas -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS \
  -DLLAMA_CURL=ON
cmake --build build-blas -j4
install -Dm755 build-blas/bin/llama-cli "$HOME/edge-llm-bench/bin/llama_blas"

# -------- Pure CPU/NEON build (no BLAS) --------
rm -rf build-pure
cmake -S . -B build-pure -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_BLAS=OFF \
  -DLLAMA_CURL=ON
cmake --build build-pure -j4
install -Dm755 build-pure/bin/llama-cli "$HOME/edge-llm-bench/bin/llama_pure"

# sanity
"$HOME/edge-llm-bench/bin/llama_blas" -h | head -n 5 || true
"$HOME/edge-llm-bench/bin/llama_pure" -h | head -n 5 || true
echo "OK: built llama_blas and llama_pure"
