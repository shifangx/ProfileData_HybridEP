#!/bin/bash
set -euxo pipefail

export WORKSPACE=$(realpath "$PWD")

NUM_RANKS="${1:-64}"
WORLD_SIZE=$((NUM_RANKS/4))
echo "NUM_RANKS: $NUM_RANKS"
echo "WORLD_SIZE: $WORLD_SIZE"

NUM_SMS="${2:-32}"
echo "NUM_SMS: $NUM_SMS"

sed -i "s/NUM_OF_RANKS_PER_NODE = 32/NUM_OF_RANKS_PER_NODE = ${NUM_RANKS}/g" ./test_mnnvlink_hybridep.py
sed -i "s/NUM_OF_RANKS_PER_NODE = 32/NUM_OF_RANKS_PER_NODE = ${NUM_RANKS}/g" ../DeepEP/csrc/kernels/hybrid_ep_backend_configs.hpp

sed -i "s/NUM_OF_BLOCKS_PREPROCESSING_API = 32/NUM_OF_BLOCKS_PREPROCESSING_API = ${NUM_SMS}/g" ../DeepEP/csrc/kernels/hybrid_ep_backend_configs.hpp
sed -i "s/NUM_OF_BLOCKS_DISPATCH_API = 32/NUM_OF_BLOCKS_DISPATCH_API = ${NUM_SMS}/g" ../DeepEP/csrc/kernels/hybrid_ep_backend_configs.hpp
sed -i "s/NUM_OF_BLOCKS_COMBINE_API = 32/NUM_OF_BLOCKS_COMBINE_API = ${NUM_SMS}/g" ../DeepEP/csrc/kernels/hybrid_ep_backend_configs.hpp
