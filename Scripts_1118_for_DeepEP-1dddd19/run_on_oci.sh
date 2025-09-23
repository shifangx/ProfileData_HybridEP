#!/bin/bash
#SBATCH -A coreai_devtech_all
#SBATCH -J coreai_devtech_all-FW:hybrid_ep
#SBATCH -p batch
#SBATCH -t 00:10:00
#SBATCH --nodes=16
#SBATCH --gres=gpu:4
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
set -x

RANKS_PER_PHYSICAL_NODE=4
export HIDDEN_DIM=7168
export MAX_NUM_OF_TOKENS_PER_RANK=8192
export NUM_OF_TOKENS_PER_RANK=8192
export NUM_LOCAL_EXPERTS=4
export NUM_OF_RANKS_PER_NODE=$((SLURM_NNODES * RANKS_PER_PHYSICAL_NODE))
export TOPK=8
export PAD_MULTIPLE=32
export NVLINK_DOMAIN_SIZE=72
export NUM_OF_STAGES_DISPATCH_API=10
export NUM_OF_STAGES_G2S_COMBINE_API=10
export NUM_OF_STAGES_S2G_COMBINE_API=2
export WORLD_SIZE=${SLURM_NNODES}

BASE_PATH=/lustre/fsw/portfolios/coreai/users/tongliu/deepep
OUTPUT_PATH=$BASE_PATH/output
SCRIPT_PATH=$BASE_PATH/tests/test_hybrid_ep.py
CONT=nvcr.io/nvidia/pytorch:25.05-py3
export PYTHONPATH=$BASE_PATH:$PYTHONPATH

if [ "${PROFILE}" = 1 ]; then
    PROFILE_CMD="nsys profile --sample=none --cpuctxsw=none -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-memory-usage true -f true -x true -o ${OUTPUT_PATH}/report_EP$((NUM_LOCAL_EXPERTS * NUM_OF_RANKS_PER_NODE))"
else
    PROFILE_CMD=""
fi

run_cmd="${PROFILE_CMD} python ${SCRIPT_PATH}"

srun --mpi=pmix  ${SEGMENT}  --container-image=${CONT} \
    --container-mounts=/lustre/:/lustre/ \
    --output="log/hybrid_ep_%j.log" \
    --no-container-mount-home ${run_cmd}
