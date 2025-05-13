# %% [markdown]
# # Tutorial: Aux Loss Factor Sweep for CLT Training with BatchTopK
#
# This script performs a hyperparameter sweep for `aux_loss_factor`
# when training a Cross-Layer Transcoder (CLT) using BatchTopK activation.
# It will:
# 1. Configure the CLT model, activation generation, and base training parameters.
# 2. Generate activations locally (with manifest) if not already present.
# 3. Loop through a predefined list of `aux_loss_factor` values.
# 4. For each `aux_loss_factor`:
#    a. Update the training configuration and WandB run name.
#    b. Train the CLT model.
#    c. Log results to WandB.

# %% [markdown]
# ## 1. Imports and Setup

# %%
import torch
import os
import time
import sys
import traceback
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from clt.config import CLTConfig, TrainingConfig, ActivationConfig
    from clt.activation_generation.generator import ActivationGenerator
    from clt.training.trainer import CLTTrainer
except ImportError as e:
    print(f"ImportError: {e}")
    print("Please ensure the 'clt' library is installed or the clt directory is in your PYTHONPATH.")
    raise

# Device setup
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# Base model for activation extraction
BASE_MODEL_NAME = "EleutherAI/pythia-70m"

# %% [markdown]
# ## 2. Static Configurations (CLT, Activation Generation)

# %%
# --- CLT Architecture Configuration ---
# This configuration is fixed for all runs in this sweep.
num_layers = 6
d_model = 512
expansion_factor = 32
clt_num_features = d_model * expansion_factor
batchtopk_k = 500

clt_config = CLTConfig(
    num_features=clt_num_features,
    num_layers=num_layers,
    d_model=d_model,
    activation_fn="batchtopk",
    batchtopk_k=batchtopk_k,
    batchtopk_straight_through=True,
)
print("CLT Configuration (BatchTopK - Fixed for Sweep):")
print(clt_config)

# --- Activation Generation Configuration ---
# Activations are generated once and reused for all sweep runs.
activation_dir = "./tutorial_activations_local_1M_pythia"  # Reusing existing activations if possible
dataset_name = "monology/pile-uncopyrighted"
activation_config = ActivationConfig(
    model_name=BASE_MODEL_NAME,
    mlp_input_module_path_template="gpt_neox.layers.{}.mlp.input",
    mlp_output_module_path_template="gpt_neox.layers.{}.mlp.output",
    dataset_path=dataset_name,
    dataset_split="train",
    dataset_text_column="text",
    context_size=128,
    inference_batch_size=192,
    exclude_special_tokens=True,
    prepend_bos=True,
    streaming=True,
    target_total_tokens=1_000_000,  # Keep small for tutorial/sweep example
    activation_dir=activation_dir,
    output_format="hdf5",
    compression="gzip",
    chunk_token_threshold=8_000,
    activation_dtype="float32",
    compute_norm_stats=True,
)
print("\nActivation Generation Configuration (Fixed for Sweep):")
print(activation_config)

# Expected path for activations, used in TrainingConfig
expected_activation_path = os.path.join(
    activation_config.activation_dir,
    activation_config.model_name,
    f"{os.path.basename(activation_config.dataset_path)}_{activation_config.dataset_split}",
)

# %% [markdown]
# ## 3. Generate Activations (One-Time Step)
# Generate the activation dataset if it doesn\'t already exist.

# %%
print("Step 3: Generating/Verifying Activations (including manifest)...")
metadata_path = os.path.join(expected_activation_path, "metadata.json")
manifest_path = os.path.join(expected_activation_path, "index.bin")

if os.path.exists(metadata_path) and os.path.exists(manifest_path):
    print(f"Activations and manifest already found at: {expected_activation_path}")
    print("Skipping generation.")
else:
    print(f"Activations or manifest not found. Generating them now at: {expected_activation_path}")
    try:
        generator = ActivationGenerator(
            cfg=activation_config,
            device=device,
        )
        generation_start_time = time.time()
        generator.generate_and_save()
        generation_end_time = time.time()
        print(f"Activation generation complete in {generation_end_time - generation_start_time:.2f}s.")
    except Exception as gen_err:
        print(f"[ERROR] Activation generation failed: {gen_err}")
        traceback.print_exc()
        raise

# %% [markdown]
# ## 4. Hyperparameter Sweep for Aux Loss Factor

# %%
# --- Define Aux Loss Factors for Sweep ---
AUX_LOSS_FACTORS_TO_SWEEP: List[float] = [1 / 64, 1 / 128, 1 / 256, 1 / 512]
BASE_LR = 1e-4
BASE_BATCH_SIZE = 1024
# K_INT is already defined in clt_config.batchtopk_k

print(f"\nStarting sweep over aux_loss_factor values: {AUX_LOSS_FACTORS_TO_SWEEP}")

for current_aux_loss_factor in AUX_LOSS_FACTORS_TO_SWEEP:
    print(f"\n--- Running for aux_loss_factor: {current_aux_loss_factor:.4e} ---")

    # --- Determine WandB Run Name (using config values) ---
    # Format aux_loss_factor for readability in run name, e.g., 1over64
    if current_aux_loss_factor == 1 / 64:
        aux_str = "1over64"
    elif current_aux_loss_factor == 1 / 128:
        aux_str = "1over128"
    elif current_aux_loss_factor == 1 / 256:
        aux_str = "1over256"
    elif current_aux_loss_factor == 1 / 512:
        aux_str = "1over512"
    else:
        aux_str = f"{current_aux_loss_factor:.0e}"

    wdb_run_name = (
        f"{clt_config.num_features}-width-"
        f"batchtopk-k{clt_config.batchtopk_k}-"
        f"{BASE_BATCH_SIZE}-batch-"
        f"{BASE_LR:.0e}-lr-"  # Use .0e for LR as well for consistency if preferred
        f"aux_{aux_str}-"
        f"{clt_config.activation_fn}-"
    )
    print("Generated WandB run name: " + wdb_run_name)

    # --- Training Configuration (Dynamic for sweep) ---
    training_config = TrainingConfig(
        learning_rate=BASE_LR,
        training_steps=1000,  # Keep steps low for tutorial sweep
        seed=42,
        activation_source="local_manifest",
        activation_path=expected_activation_path,
        activation_dtype="float32",
        train_batch_size_tokens=BASE_BATCH_SIZE,
        sampling_strategy="sequential",
        normalization_method="auto",
        sparsity_lambda=0.0,
        sparsity_lambda_schedule="linear",
        sparsity_c=0.0,
        preactivation_coef=0,
        aux_loss_factor=current_aux_loss_factor,  # Key parameter for the sweep
        apply_sparsity_penalty_to_batchtopk=False,
        optimizer="adamw",
        lr_scheduler="linear_final20",
        optimizer_beta2=0.98,
        log_interval=10,
        eval_interval=50,
        diag_every_n_eval_steps=1,
        max_features_for_diag_hist=1000,
        checkpoint_interval=500,  # Trainer saves checkpoints
        dead_feature_window=200,
        enable_wandb=True,
        wandb_project="clt-hp-sweeps-pythia-70m",  # Target project
        wandb_run_name=wdb_run_name,  # Dynamic run name
    )
    print("\nTraining Configuration for current sweep run:")
    print(training_config)

    # --- Training the CLT ---
    # Unique log_dir for each sweep run to avoid conflicts
    run_specific_log_dir = os.path.join(
        "clt_training_logs",
        f"clt_pythia_batchtopk_aux_sweep_{wdb_run_name.replace('/', '_').replace(':', '_')}_{int(time.time())}",
    )
    os.makedirs(run_specific_log_dir, exist_ok=True)
    print(f"Logs and checkpoints for this run will be saved to: {run_specific_log_dir}")

    try:
        print("Creating CLTTrainer instance...")
        trainer = CLTTrainer(
            clt_config=clt_config,
            training_config=training_config,
            log_dir=run_specific_log_dir,
            device=device,
            distributed=False,
        )
        print("CLTTrainer instance created successfully.")

        print(f"Beginning training for aux_loss_factor: {current_aux_loss_factor:.4e}...")
        start_train_time = time.time()
        # The train method will handle model saving according to checkpoint_interval and at the end.
        trained_clt_model = trainer.train(eval_every=training_config.eval_interval)
        end_train_time = time.time()
        print(
            f"Training for aux_loss_factor {current_aux_loss_factor:.4e} finished in {end_train_time - start_train_time:.2f} seconds."
        )
        print(f"Final model and logs saved in: {run_specific_log_dir}/final/")

    except Exception as train_err:
        print(f"[ERROR] Training failed for aux_loss_factor {current_aux_loss_factor:.4e}: {train_err}")
        traceback.print_exc()
        print("Skipping to next aux_loss_factor if any.")
        continue  # Continue to the next item in the sweep

# %% [markdown]
# ## 5. Sweep Complete

# %%
print("\nHyperparameter sweep for aux_loss_factor is complete!")
print("Check WandB for detailed results for each run in project 'clt-hp-sweeps-pythia-70m'.")
