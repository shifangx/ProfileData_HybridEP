#!/bin/bash
set -euxo pipefail

export WORKSPACE=$(realpath "$PWD")

# git clone deepep
cd ${WORKSPACE}/../
rm DeepEP -rf
git clone https://github.com/deepseek-ai/DeepEP.git
cd DeepEP
git checkout 6ad6896ddcc4f02531a008851006af1e191d9cd4
