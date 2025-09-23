#!/bin/bash

PARTITION=${PARTITION:-"batch"}
if hostname | grep -q ptyche; then
  PARTITION="36x2-a01r"
  GPU_OPTION=""
else
  GPU_OPTION="--gpus-per-node=4"
fi

echo "PARTITION: ${PARTITION}"
export WORLD_SIZE=16
salloc -A coreai_devtech_all -J coreai_devtech_all-test:test ${GPU_OPTION} -p ${PARTITION} -N ${WORLD_SIZE} --segment=${WORLD_SIZE} --exclusive --mem 0 --time 4:00:00


