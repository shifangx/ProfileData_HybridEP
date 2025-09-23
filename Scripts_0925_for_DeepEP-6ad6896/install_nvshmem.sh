#!/bin/bash
set -euxo pipefail
source set_env.sh
echo "NODE_TYPE: $NODE_TYPE"

mkdir -p /home
cd /home
mkdir -p /home/dpsk_a2a
cd /home/dpsk_a2a
export WORKSPACE=/home/dpsk_a2a
cd ${WORKSPACE}
rm ./nvshmem* -rf

git clone -b v2.4.4 https://github.com/NVIDIA/gdrcopy.git

wget https://developer.download.nvidia.com/compute/redist/nvshmem/3.3.9/source/nvshmem_src_cuda12-all-all-3.3.9.tar.gz
tar -xvf nvshmem_src_cuda12-all-all-3.3.9.tar.gz
rm nvshmem_src_cuda12-all-all-3.3.9.tar.gz
mv nvshmem_src nvshmem

## create link for the dependency of IBGDA
arch=$(uname -m)
if [ "$arch" = "x86_64" ]; then
    if [ ! -e /usr/lib/x86_64-linux-gnu/libmlx5.so ]; then
        ln -s /usr/lib/x86_64-linux-gnu/libmlx5.so.1 /usr/lib/x86_64-linux-gnu/libmlx5.so
    else
        rm -f /usr/lib/x86_64-linux-gnu/libmlx5.so
        ln -s /usr/lib/x86_64-linux-gnu/libmlx5.so.1 /usr/lib/x86_64-linux-gnu/libmlx5.so
    fi
elif [ "$arch" = "aarch64" ]; then
    if [ ! -e /usr/lib/aarch64-linux-gnu/libmlx5.so ]; then
        ln -s /usr/lib/aarch64-linux-gnu/libmlx5.so.1 /usr/lib/aarch64-linux-gnu/libmlx5.so
    else
        rm -f /usr/lib/aarch64-linux-gnu/libmlx5.so
        ln -s /usr/lib/aarch64-linux-gnu/libmlx5.so.1 /usr/lib/aarch64-linux-gnu/libmlx5.so
    fi
else
    echo "Unsupported architecture: $arch"
fi

# set ENV
export CUDA_HOME=/usr/local/cuda
export CPATH=/usr/local/mpi/include:${CPATH:-}
export LD_LIBRARY_PATH=/usr/local/mpi/lib:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/usr/local/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
if [ "$NODE_TYPE" == "H100" ]; then
    export CUDA_ARCH=90a
elif [[ "$NODE_TYPE" == "GB200" || "$NODE_TYPE" == "B200" ]]; then
    export CUDA_ARCH=100a
else
    echo "Unsupported node type: $NODE_TYPE"
    exit 1
fi
export GDRCOPY_HOME=${WORKSPACE}/gdrcopy

## Build nvshmem_src
cd ${WORKSPACE}/nvshmem
NVSHMEM_SHMEM_SUPPORT=0 \
NVSHMEM_UCX_SUPPORT=0 \
NVSHMEM_USE_NCCL=0 \
NVSHMEM_IBGDA_SUPPORT=1 \
NVSHMEM_PMIX_SUPPORT=0 \
NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
NVSHMEM_USE_GDRCOPY=1 \
cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=${WORKSPACE}/nvshmem/install -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} && cd build && make install -j
