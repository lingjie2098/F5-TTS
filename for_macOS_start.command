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

f5-tts_infer-gradio								# Launch a Gradio app (web interface)
# f5-tts_infer-gradio --port 7860 --host 0.0.0.0 	# Specify the port/host
# f5-tts_infer-gradio --share  					# Launch a share link