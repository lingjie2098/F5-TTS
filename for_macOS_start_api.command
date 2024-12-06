#!/bin/bash

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
echo "=============$CURRENT_DIR============="
cd "$CURRENT_DIR"

# use CPU-only
export PYTORCH_ENABLE_MPS_FALLBACK=1

# activate env
source ~/.zshrc
conda activate f5-tts
# print current env, make sure the f5-tts env has been activated.
conda info --envs

python3.10 api.py