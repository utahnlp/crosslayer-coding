#!/bin/bash

#source ~/software/pkg/miniconda3/etc/profile.d/conda.sh
#conda activate /uufs/chpc.utah.edu/common/home/u0879092/scr/scr_envs/clt

# add work dir to python path
# export CODE_DIR="/uufs/chpc.utah.edu/common/home/u1472283/scr/crosslayer-coding"
export CODE_DIR="/uufs/chpc.utah.edu/common/home/u0879092/scr/transcoders/open-clts/crosslayer-coding"
export PYTHONPATH="$CODE_DIR:$PYTHONPATH"

# Params
export DATA_DIR="$CODE_DIR/data"
export MODEL_NAME="allenai/OLMo-2-0425-1B-Instruct"
# export DATASET_NAME="allenai/olmo-mix-1124"
export DATASET_NAME=(
    "allenai/olmo-mix-1124"
    "allenai/dolmino-mix-1124"
)
export CONTEXT_SIZE=4096
export BATCH_SIZE=16
export NUM_TOKENS=1000000
export CHUNK_TOKEN_THRESHOLD=100000
export ACTIVATION_DTYPE="float32"

# uncomment to add dataset offset
# export DATASET_SKIP=0


python $CODE_DIR/scripts/generate_activations.py \
    --model-name $MODEL_NAME \
    --mlp-input-template "model.layers.{}.mlp.input" \
    --mlp-output-template "model.layers.{}.mlp.output" \
    --dataset-path ${DATASET_NAME[@]}\
    --context-size $CONTEXT_SIZE \
    --inference-batch-size $BATCH_SIZE \
    --prepend-bos \
    --target-total-tokens $NUM_TOKENS \
    --activation-dir $DATA_DIR/activations \
    --compression "gzip" \
    --chunk-token-threshold $CHUNK_TOKEN_THRESHOLD \
    --activation-dtype $ACTIVATION_DTYPE \
    --compute-norm-stats #\
    # --dataset-skip $DATASET_SKIP
