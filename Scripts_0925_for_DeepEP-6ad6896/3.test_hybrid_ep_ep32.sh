#!/bin/bash
set -euxo pipefail

export NSYS_PROFILE=0
export WORKSPACE=$(realpath "$PWD")
export WORLD_SIZE=8
mkdir -p ${WORKSPACE}/../logs

export NSYS_PROFILE=0
export NUM_OF_RANKS_PER_NODE=32
srun -N ${WORLD_SIZE} --container-image=${WORKSPACE}/../docker/deepep-pytorch:25.05-py3.sqsh --container-mounts  /lustre:/lustre  --container-workdir="$PWD" bash test_hybrid_ep.sh | tee ${WORKSPACE}/../logs/test_hybrid_ep_nvl72_N_${WORLD_SIZE}.log
python ${WORKSPACE}/parse_log.py ${WORKSPACE}/../logs/test_mnnvlink_hybridep_N_${WORLD_SIZE}.log | tee ${WORKSPACE}/../logs/test_mnnvlink_hybridep_N_${WORLD_SIZE}_parsed.log

export NSYS_PROFILE=1
srun -N ${WORLD_SIZE} --container-image=${WORKSPACE}/../docker/deepep-pytorch:25.05-py3.sqsh --container-mounts  /lustre:/lustre  --container-workdir="$PWD" bash test_hybrid_ep.sh
