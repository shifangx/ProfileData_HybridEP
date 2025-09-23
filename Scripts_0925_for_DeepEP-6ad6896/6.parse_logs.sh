#!/bin/bash
set -euxo pipefail

export WORKSPACE=$(realpath "$PWD")
for log_filename in ${WORKSPACE}/../logs/*.log; do
    python ${WORKSPACE}/parse_log.py "$log_filename"
done
