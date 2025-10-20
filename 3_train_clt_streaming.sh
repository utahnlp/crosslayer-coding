#!/bin/bash

#SBATCH --job-name train_clt
#SBATCH --account rai
##SBATCH --partition rai-gpu-grn
##SBATCH --qos rai-gpu-grn
#SBATCH --partition=rai-gpu-grn 
#SBATCH --qos=rai-gpu-grn-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=1:00:00
#SBATCH --mem=100GB
#SBATCH --requeue
#SBATCH -o log_train_clt_%j


# virtual environment
export USER="oliver"
export USER="nate"

if [ "$USER" = "oliver" ]; then
    # virtual environment for Oliver
    export PYENV_ROOT="$SCR_DIR/.pyenv"
    command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
    pyenv activate 3.12
    # add work dir to python path
    export CODE_DIR="/uufs/chpc.utah.edu/common/home/u1472283/scr/crosslayer-coding"
    export PYTHONPATH="$CODE_DIR:$PYTHONPATH"

elif [ "$USER" = "nate" ]; then
    # virtual environment for Nate
    source ~/software/pkg/miniconda3/etc/profile.d/conda.sh
    conda activate /scratch/rai/vast1/u0879092/envs/clt
    
    export CODE_DIR="/scratch/rai/vast1/u0879092/clts/crosslayer-coding"
     # export PYTHONHOME=/uufs/chpc.utah.edu/common/home/u0879092/scr/scr_envs/clts
fi

# export PYTHONHOME=/uufs/chpc.utah.edu/common/home/u0879092/scr/scr_envs/clts


# weights and biases will cache >100gb of artifacts, need to point to storage dir
export WANDB_DIR="$CODE_DIR/wandb"
export WANDB_CACHE_DIR="$WANDB_DIR/cache"
export WANDB_ARTIFACT_DIR="$WANDB_DIR/artifacts"
export WANDB_DATA_DIR="$WANDB_DIR/data"

# CLT Architecture
export CLT_FEATURES=100
export BATCHTOPK_K=10

# Reduced Precision
# export CLT_DTYPE="float32"
# export PRECISION="fp32"
# export ACTIVATION_DTYPE="float32"
export MODEL_DTYPE="float32"
export CLT_DTYPE="bfloat16"
export PRECISION="bf16"
export ACTIVATION_DTYPE="bfloat16"

# Training Hyperparams
export DATASET_SIZE=1000000
export BATCH_SIZE=32
# export TRAINING_STEPS=$(($DATASET_SIZE/$BATCH_SIZE)) # set to DATASET_SIZE / BATCH_SIZE above
export TRAINING_STEPS=5000
export LEARNING_RATE=0.0001

# Logging And Evaluation
export LOG_INTERVAL=1000
export EVAL_INTERVAL=1000
export CHECKPOINT_INTERVAL=1000

# Paths
export DATA_DIR="$CODE_DIR/data"
export MODEL_NAME="allenai/OLMo-2-0425-1B-Instruct"
export DATASET_NAME="olmo-mix-1124"
export DATASET_SPLIT="train"
export ACTIVATION_PATH="$DATA_DIR/activations/$MODEL_NAME/${DATASET_NAME}_${DATASET_SPLIT}_${DATASET_SIZE}_${ACTIVATION_DTYPE}"
export OUT_DIR="$DATA_DIR/clt/$MODEL_NAME/${DATASET_NAME}_${DATASET_SPLIT}_${DATASET_SIZE}_${ACTIVATION_DTYPE}/${CLT_FEATURES}feats_${BATCH_SIZE}batchsize"

# Streaming
export DATASET_NAME="allenai/olmo-mix-1124"
export CONTEXT_SIZE=4096
export INFERENCE_BATCH_SIZE=4
export NUM_TOKENS=1000000
export CHUNK_TOKEN_THRESHOLD=1000


torchrun \
     $CODE_DIR/scripts/train_clt.py \
    --activation-source streaming \
    --output-dir $OUT_DIR \
    --model-name $MODEL_NAME \
    --distributed \
    --activation-path $ACTIVATION_PATH \
    --num-features $CLT_FEATURES \
    --activation-fn batchtopk \
    --batchtopk-k $BATCHTOPK_K \
    --clt-dtype $CLT_DTYPE \
    --learning-rate $LEARNING_RATE \
    --training-steps $TRAINING_STEPS \
    --train-batch-size-tokens $BATCH_SIZE \
    --precision $PRECISION \
    --no-apply-sparsity-penalty-to-batchtopk \
    --seed 42 \
    --activation-dtype $ACTIVATION_DTYPE \
    --compute-sparsity-diagnostics \
    --log-interval $LOG_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --checkpoint-interval $CHECKPOINT_INTERVAL \
    --mlp-input-template "model.layers.{}.mlp.input" \
    --mlp-output-template "model.layers.{}.mlp.output" \
    --model-dtype $MODEL_DTYPE \
    --dataset-path $DATASET_NAME \
    --context-size $CONTEXT_SIZE \
    --inference-batch-size $INFERENCE_BATCH_SIZE \
    --prepend-bos \
    --debug-anomaly \
    --enable-wandb \
    --wandb-project "cross-layer-transcoders" \
    --wandb-entity "utah-clt" \
    # --resume_from_checkpoint_dir $CHECKPOINT_DIR \
    # --resume_step 1234
