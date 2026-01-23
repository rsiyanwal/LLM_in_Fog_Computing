sudo apt-get update
sudo apt-get install -y git build-essential cmake pkg-config ninja-build libopenblas-dev
# optional: keep OpenBLAS from over-threading against llama.cpp's -t
echo 'export OPENBLAS_NUM_THREADS=1' >> ~/.bashrc
source ~/.bashrc
