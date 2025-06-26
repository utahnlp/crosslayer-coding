#!/bin/bash

#SBATCH --job-name generate_activations
#SBATCH --account rai
#SBATCH --partition rai-gpu-grn
#SBATCH --qos rai-gpu-grn
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=4-00:00:00
#SBATCH --mem=264GB
#SBATCH --requeue
#SBATCH -o log_generate_activations_%j


# virtual environment
export PYENV_ROOT="$SCR_DIR/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv activate 3.12

# add work dir to python path
export CODE_DIR="/uufs/chpc.utah.edu/common/home/u1472283/scr/crosslayer-coding"
export PYTHONPATH="$CODE_DIR:$PYTHONPATH"


# Params
export DATA_DIR="$CODE_DIR/data"
export MODEL_NAME="allenai/OLMo-2-0425-1B-Instruct"
export DATASET_NAME="allenai/olmo-mix-1124"
export CONTEXT_SIZE=4096
export BATCH_SIZE=16
export NUM_TOKENS=1000000
export CHUNK_TOKEN_THRESHOLD=100000
export ACTIVATION_DTYPE="float32"


python $CODE_DIR/scripts/generate_activations.py \
    --model-name $MODEL_NAME \
    --mlp-input-template "model.layers.{}.mlp.input" \
    --mlp-output-template "model.layers.{}.mlp.output" \
    --dataset-path $DATASET_NAME \
    --context-size $CONTEXT_SIZE \
    --inference-batch-size $BATCH_SIZE \
    --prepend-bos \
    --target-total-tokens $NUM_TOKENS \
    --activation-dir $DATA_DIR/activations \
    --compression "gzip" \
    --chunk-token-threshold $CHUNK_TOKEN_THRESHOLD \
    --activation-dtype $ACTIVATION_DTYPE \
    --compute-norm-stats

