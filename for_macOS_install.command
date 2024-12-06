#!/bin/bash

# Preparatory work
# Install git
# Install Conda: please see https://docs.conda.io/en/latest/miniconda.html

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
echo "=============$CURRENT_DIR============="
cd "$CURRENT_DIR"

# create env. you could also use virtualenv
conda create -n f5-tts python=3.10

# activate env
source ~/.zshrc
conda activate f5-tts
# print current env, make sure the f5-tts env has been activated.
conda info --envs

pip install -e .
pip install -r requirements.txt