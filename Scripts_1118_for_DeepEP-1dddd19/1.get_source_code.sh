#!/bin/bash
set -euxo pipefail

export WORKSPACE=$(realpath "$PWD/../")

# git clone deepep
cd ${WORKSPACE}
git clone https://github.com/deepseek-ai/DeepEP.git
cd DeepEP
git checkout 1dddd194c26911c35b4f53a148617dd73de0ffc9
