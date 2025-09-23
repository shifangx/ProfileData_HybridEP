#!/bin/bash
set -euxo pipefail

export WORKSPACE=$(realpath "$PWD")
export PYTORCH_DOCKER_IMAGE=${WORKSPACE}/../docker/pytorch:25.05-py3.sqsh
export NVSHMEM_DOCKER_IMAGE=${WORKSPACE}/../docker/nvshmem-pytorch:25.05-py3.sqsh
export DEEPPEP_DOCKER_IMAGE=${WORKSPACE}/../docker/deepep-pytorch:25.05-py3.sqsh
mkdir -p ${WORKSPACE}/../docker
if [[ ! -f ${PYTORCH_DOCKER_IMAGE} ]]; then
    echo "download pytorch docker image"
    srun -N 1 --container-image=nvcr.io/nvidia/pytorch:25.05-py3 --container-mounts  /lustre:/lustre --container-save=${PYTORCH_DOCKER_IMAGE} --container-workdir="$PWD" hostname
fi
if [[ ! -f ${NVSHMEM_DOCKER_IMAGE} ]]; then
    echo "install nvshmem in docker image"
    srun -N 1 --container-image=${PYTORCH_DOCKER_IMAGE} --container-mounts  /lustre:/lustre --container-save=${NVSHMEM_DOCKER_IMAGE} --container-workdir="$PWD" bash install_nvshmem.sh
fi

echo "install deepep in docker image"
srun -N 1 --container-image=${NVSHMEM_DOCKER_IMAGE} --container-mounts  /lustre:/lustre --container-save=${DEEPPEP_DOCKER_IMAGE} --container-workdir="$PWD" bash install_deepep.sh
