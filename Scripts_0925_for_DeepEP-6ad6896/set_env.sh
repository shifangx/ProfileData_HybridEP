#!/bin/bash
set -euxo pipefail

HOSTNAME=$(hostname)
if [[ $HOSTNAME == *"cw"* ]]; then
    export NODE_TYPE="H100"
    export NUM_GPUS=8
elif [[ $HOSTNAME == *"prenyx"* ]]; then
    export NODE_TYPE="B200"
    export NUM_GPUS=8
elif [[ $HOSTNAME == *"ptyche"* ]]; then
    export NODE_TYPE="GB200"
    export NUM_GPUS=4
elif [[ $HOSTNAME == *"nvl"* ]]; then
    export NODE_TYPE="GB200"
    export NUM_GPUS=4
fi
echo "Please change the following variables according to your cluster!!!"
echo "HOSTNAME: $HOSTNAME"
echo "NODE_TYPE: $NODE_TYPE"
echo "NUM_GPUS: $NUM_GPUS"

