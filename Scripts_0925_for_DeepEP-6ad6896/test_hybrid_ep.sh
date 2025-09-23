#!/bin/bash

export WORKSPACE=$(realpath "$PWD")
mkdir -p ${WORKSPACE}/../logs
TEST_NAME=test_mnnvlink_hybridep
echo "test ${TEST_NAME}, host: $HOSTNAME"
NSYS_PROFILE=${NSYS_PROFILE:-false}
if [[ $NSYS_PROFILE == 1 ]]; then
    nsys profile --sample=none --cpuctxsw=none -t cuda,nvtx --capture-range=cudaProfilerApi  -f true -x true -o ${WORKSPACE}/../logs/${TEST_NAME}_FP8_N_${WORLD_SIZE}_r_${RANK}  python ${WORKSPACE}/${TEST_NAME}.py --nsys-profile  --use-fp8 
    nsys stats --force-export=true ${WORKSPACE}/../logs/${TEST_NAME}_FP8_N_${WORLD_SIZE}_r_${RANK}.nsys-rep 2>&1 |tee ${WORKSPACE}/../logs/${TEST_NAME}_FP8_N_${WORLD_SIZE}_r_${RANK}.txt
    nsys profile --sample=none --cpuctxsw=none -t cuda,nvtx --capture-range=cudaProfilerApi -f true -x true -o ${WORKSPACE}/../logs/${TEST_NAME}_N_${WORLD_SIZE}_r_${RANK}  python ${WORKSPACE}/${TEST_NAME}.py --nsys-profile 
    nsys stats --force-export=true ${WORKSPACE}/../logs/${TEST_NAME}_N_${WORLD_SIZE}_r_${RANK}.nsys-rep 2>&1 |tee ${WORKSPACE}/../logs/${TEST_NAME}_N_${WORLD_SIZE}_r_${RANK}.txt
else
    python ${WORKSPACE}/${TEST_NAME}.py --use-fp8 
    python ${WORKSPACE}/${TEST_NAME}.py 
fi
