#!/bin/bash
set -euxo pipefail

export DeepEP_SOURC_CODE_DIR=$(realpath "$PWD/../DeepEP/")
export WORKSPACE=/home/dpsk_a2a
source set_env.sh
echo "NODE_TYPE: $NODE_TYPE"
cd ${WORKSPACE}
rm ./DeepEP -rf
cp ${DeepEP_SOURC_CODE_DIR} ./  -rf

# set ENV
export CUDA_HOME=/usr/local/cuda
export CPATH=/usr/local/mpi/include:${CPATH:-}
export LD_LIBRARY_PATH=/usr/local/mpi/lib:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/usr/local/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
if [ "$NODE_TYPE" == "H100" ]; then
    export TORCH_CUDA_ARCH_LIST=9.0a
elif [[ "$NODE_TYPE" == "GB200" || "$NODE_TYPE" == "B200" ]]; then
    export TORCH_CUDA_ARCH_LIST=10.0a
else
    echo "Unsupported node type: $NODE_TYPE"
    exit 1
fi

## Build deepep
## make sure nvshmem has already installed in ${NVSHMEM_DIR}
cd ${WORKSPACE}/DeepEP
NVSHMEM_DIR=${WORKSPACE}/nvshmem/install python setup.py develop
NVSHMEM_DIR=${WORKSPACE}/nvshmem/install python setup.py install

