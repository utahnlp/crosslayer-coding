# %% [markdown]
# # Sweep: JumpReLU Sparsity-Parameter Grid Search (Remote Activation Store)
#
# This script performs a grid sweep over the key sparsity‐related hyper-parameters when
# training a Cross-Layer Transcoder (CLT) with **JumpReLU** activation,
# **using a remote activation store**.
#
# Parameters explored
# • sparsity_c  ∈ {0.10, 0.30, 1.00}
# • sparsity_lambda ∈ {0, 1e-5, 3e-5, 1e-4}
# • jumprelu_threshold ∈ {0.01, 0.03, 0.05}
#
# Constraint: we skip the too-strong corner (c = 1.0, lambda = 1e-4).
#
# To keep exactly 36 runs we add three extra replicates of the mid-grid setting
#   (c = 0.30, lambda = 3e-5, threshold = 0.03) with seeds 43, 44, 45.
#
# The script:
# 1. Checks if the remote activation server is running.
# 2. Iterates over the run list, building `CLTConfig` and `TrainingConfig` for each.
# 3. Launches training via `CLTTrainer`, saving logs/checkpoints per run.
# 4. **Deletes the local `clt_training_logs` directory at the end of each run.**
# 5. **Clears WandB cache, CUDA cache (if applicable), and runs garbage collection after each run.**
#
# The overall structure mirrors `tutorials/aux_loss_factor_sweep.py`.
# **IMPORTANT**: Ensure the remote activation server is running and populated with the
# specified `dataset_id` before running this script.

# %% [markdown]
# ## 1. Imports and Setup

# %%
import torch
import os
import time
import sys
import traceback
from typing import List, Tuple
import logging
import requests  # For server health check
from urllib.parse import urljoin  # For server health check
import shutil  # For deleting directories
import gc  # For garbage collection

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s")

# Disable HF tokenizer parallelism to avoid noisy warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure project root is on PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from clt.config import CLTConfig, TrainingConfig  # ActivationConfig removed
    from clt.training.trainer import CLTTrainer
except ImportError as e:
    print(f"ImportError: {e}")
    print("Please ensure the 'clt' package is available or the clt directory is on PYTHONPATH.")
    raise

# Device selection
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

# Base model for activation extraction (used by the server when generating activations)
BASE_MODEL_NAME = "EleutherAI/pythia-70m"

# --- Server Configuration (from 4-remote-server-training-test.py) --- #
SERVER_HOST = "34.41.125.189"  # Or your server IP/hostname
SERVER_PORT = 8000
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
HEALTH_CHECK_URL = urljoin(SERVER_URL, "/api/v1/health")
# This is the dataset ID the server should have generated and stored.
# It should match the dataset_id used when populating the server.
REMOTE_DATASET_ID = "EleutherAI/pythia-70m/pile-uncopyrighted_train"


# %% [markdown]
# ## 2. Static Configurations (CLT architecture)

# %%
# --- CLT architecture (constant across sweep except for JumpReLU threshold) ---
NUM_LAYERS = 6
D_MODEL = 512
EXPANSION_FACTOR = 32  # Using the same expansion factor as original jumprelu example
CLT_NUM_FEATURES = D_MODEL * EXPANSION_FACTOR

# --- Activation generation config (shared) ---
# No longer needed here, as activations are fetched from remote server.
# activation_dir = "./tutorial_activations_local_1M_pythia"
# dataset_name = "monology/pile-uncopyrighted"

# activation_config = ActivationConfig(
#     model_name=BASE_MODEL_NAME,
#     mlp_input_module_path_template="gpt_neox.layers.{}.mlp.input",
#     mlp_output_module_path_template="gpt_neox.layers.{}.mlp.output",
#     dataset_path=dataset_name,
#     dataset_split="train",
#     dataset_text_column="text",
#     context_size=128,
#     inference_batch_size=192,
#     exclude_special_tokens=True,
#     prepend_bos=True,
#     streaming=True,
#     target_total_tokens=1_000_000,  # small for tutorial
#     activation_dir=activation_dir,
#     output_format="hdf5",
#     compression="gzip",
#     chunk_token_threshold=8_000,
#     activation_dtype="float32",
#     compute_norm_stats=True,
# )
# print("Activation Generation Configuration (shared):")
# print(activation_config)

# expected_activation_path = os.path.join(
#     activation_config.activation_dir,
#     activation_config.model_name,
#     f"{os.path.basename(activation_config.dataset_path)}_{activation_config.dataset_split}",
# )

# %% [markdown]
# ## 3. Server Health Check
# Ensure the remote activation server is running and accessible.

# %%
print(f"Checking remote server health at: {HEALTH_CHECK_URL}...")
try:
    response = requests.get(HEALTH_CHECK_URL, timeout=10)  # 10 second timeout
    if response.status_code == 200 and response.json().get("status") == "ok":
        print(f"✅ Remote server is running and healthy at {SERVER_URL}")
        # Optionally check if the specific dataset_id is available
        datasets_url = urljoin(SERVER_URL, "/api/v1/datasets")
        available_datasets_response = requests.get(datasets_url, timeout=10)
        if available_datasets_response.status_code == 200:
            available_ids = [ds["dataset_id"] for ds in available_datasets_response.json()]
            if REMOTE_DATASET_ID in available_ids:
                print(f"   ✅ Dataset '{REMOTE_DATASET_ID}' is available on the server.")
            else:
                print(f"   ⚠️ Dataset '{REMOTE_DATASET_ID}' NOT FOUND on the server. Available: {available_ids}")
                print("      Please ensure the server is populated with the correct dataset.")
                # sys.exit(1) # Optionally exit if dataset not found
        else:
            print(
                f"   ⚠️ Could not verify available datasets. Server responded with {available_datasets_response.status_code}"
            )
    else:
        print(f"❌ Server health check FAILED or gave unexpected status: {response.status_code} - {response.text}")
        print(f"   Please ensure the activation server is running at {SERVER_URL} and is accessible.")
        sys.exit(1)
except requests.exceptions.ConnectionError:
    print(f"❌ ConnectionError: Could not connect to the server at {SERVER_URL}.")
    print("   Please start the activation server and ensure it is accessible.")
    sys.exit(1)
except requests.exceptions.Timeout:
    print(f"❌ Timeout: Connection to the server at {SERVER_URL} timed out.")
    sys.exit(1)
except Exception as e:
    print(f"❌ An unexpected error occurred during server health check: {e}")
    traceback.print_exc()
    sys.exit(1)


# %% [markdown]
# ## 4. Sweep Definition
# Local activation generation is removed. Activations will be fetched from the remote server.
# %%
SPARSITY_CS: List[float] = [0.10, 0.30, 1.00]
SPARSITY_LAMBDAS: List[float] = [0.0, 1e-5, 3e-5, 1e-4]
JUMPRELU_THRESHOLDS: List[float] = [0.01, 0.03, 0.05]

# Forbidden combo
FORBIDDEN: Tuple[float, float] = (1.00, 1e-4)

# Replicate runs for variance estimation
REPLICATE_C = 0.30
REPLICATE_LAMBDA = 3e-5
REPLICATE_THRESHOLD = 0.03
REPLICATE_SEEDS = [43, 44, 45]  # seed 42 will already be in factorial grid

RunSpec = Tuple[float, float, float, int]  # (c, lambda, threshold, seed)
run_specs: List[RunSpec] = []

# Main factorial grid
for c_val in SPARSITY_CS:
    for lam in SPARSITY_LAMBDAS:
        if (c_val, lam) == FORBIDDEN:
            continue  # skip too-strong corner
        for theta in JUMPRELU_THRESHOLDS:
            run_specs.append((c_val, lam, theta, 42))  # default seed

# Add replicates
for seed in REPLICATE_SEEDS:
    run_specs.append((REPLICATE_C, REPLICATE_LAMBDA, REPLICATE_THRESHOLD, seed))

assert len(run_specs) == 36, f"Expected 36 runs, got {len(run_specs)}"
print(f"Prepared {len(run_specs)} runs in sweep.")

# %% [markdown]
# ## 5. Run Sweep

# %%
BASE_LR = 1e-4
BASE_BATCH_SIZE = 1024
PARENT_LOG_DIR = "clt_training_logs"  # Define the parent log directory name

for idx, (c_val, lam_val, theta_val, seed_val) in enumerate(run_specs, start=1):
    print(
        "\n=== Run {}/{}: C={}, λ={}, θ={}, seed={} ===".format(
            idx, len(run_specs), c_val, lam_val, theta_val, seed_val
        )
    )
    run_had_error = False  # Flag to indicate if the current run had an error

    # --- Build CLTConfig (varies by theta) ---
    clt_config = CLTConfig(
        num_features=CLT_NUM_FEATURES,
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        activation_fn="jumprelu",
        jumprelu_threshold=theta_val,
    )

    # --- Determine WandB run name fragment ---
    def fmt_float(f: float) -> str:
        # Helper to get tidy strings like 0p30 or 3e-05
        if f >= 0.01:
            return str(f).replace(".", "p")
        return f"{f:.0e}"

    wandb_run_name = (
        f"{CLT_NUM_FEATURES}-width-"
        f"C{fmt_float(c_val)}_L{fmt_float(lam_val)}_T{fmt_float(theta_val)}-"
        f"{BASE_BATCH_SIZE}-batch-"
        f"{BASE_LR:.0e}-lr-"
        f"seed{seed_val}"
    )

    # --- TrainingConfig (varies by c & λ & seed) ---
    training_config = TrainingConfig(
        learning_rate=BASE_LR,
        training_steps=1000,
        seed=seed_val,
        # Key changes for remote activation store:
        activation_source="remote",
        remote_config={
            "server_url": SERVER_URL,
            "dataset_id": REMOTE_DATASET_ID,
            "timeout": 120,  # From 4-remote-server-training-test.py
            "max_retries": 3,  # From 4-remote-server-training-test.py
            "prefetch_batches": 4,  # From 4-remote-server-training-test.py
        },
        # activation_path is not used for remote source
        activation_dtype="float32",  # Server is expected to provide this dtype
        train_batch_size_tokens=BASE_BATCH_SIZE,
        sampling_strategy="sequential",  # Remote store typically handles shuffling/sampling
        normalization_method="auto",  # Remote store handles providing normalized or raw based on metadata
        sparsity_lambda=lam_val,
        sparsity_lambda_schedule="linear",
        sparsity_c=c_val,
        preactivation_coef=3e-6,
        aux_loss_factor=0.0,
        optimizer="adamw",
        lr_scheduler="linear_final20",
        optimizer_beta2=0.98,
        log_interval=10,
        eval_interval=50,
        diag_every_n_eval_steps=1,
        max_features_for_diag_hist=1000,
        checkpoint_interval=500,
        dead_feature_window=200,
        enable_wandb=True,
        wandb_project="clt-hp-sweeps-pythia-70m",
        wandb_run_name=wandb_run_name,
    )

    print("TrainingConfig:")
    print(training_config)

    # --- Unique log directory per run ---
    log_dir = os.path.join(PARENT_LOG_DIR, f"clt_pythia_jumprelu_sparsity_sweep_{wandb_run_name.replace('/', '_')}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logs and checkpoints for this run will be saved to: {log_dir}")

    # --- Run training ---
    try:
        trainer = CLTTrainer(
            clt_config=clt_config,
            training_config=training_config,
            log_dir=log_dir,
            device=device,
            distributed=False,
        )
        print("CLTTrainer created. Starting training …")
        t_start = time.time()
        trainer.train(eval_every=training_config.eval_interval)
        print(f"Run finished in {time.time() - t_start:.1f}s. Final checkpoint in {log_dir}/final/")
    except Exception as err_info:  # Renamed to err_info to avoid conflict
        print(f"[ERROR] Run failed: {err_info}")
        traceback.print_exc()
        print("Continuing to next run …")
        run_had_error = True  # Set the flag
        # Ensure finally block is hit before continue
    finally:
        print(f"\nAttempting to delete base log directory after run {idx}: {PARENT_LOG_DIR}")
        if os.path.exists(PARENT_LOG_DIR):
            try:
                shutil.rmtree(PARENT_LOG_DIR)
                print(f"Successfully deleted {PARENT_LOG_DIR}")
            except OSError as e:
                print(f"Error deleting {PARENT_LOG_DIR}: {e.strerror}")
        else:
            print(f"{PARENT_LOG_DIR} does not exist at end of run {idx}, no need to delete.")

        # --- Clear caches --- #
        print("Attempting to clear caches...")
        try:
            wandb_cache_dir = os.path.expanduser("~/.cache/wandb")
            if os.path.exists(wandb_cache_dir):
                print(f"Attempting to remove wandb cache directory: {wandb_cache_dir}")
                shutil.rmtree(wandb_cache_dir)
                print("Wandb cache directory removed.")
        except OSError as e:
            print(f"[Warning] Failed to remove wandb cache directory {wandb_cache_dir}: {e}")
            traceback.print_exc()

        if device == "cuda":  # device is defined globally
            print("Emptying CUDA cache...")
            torch.cuda.empty_cache()

        print("Running garbage collection...")
        gc.collect()
        print("Cache clearing attempt complete.")
        # --- End Cache Clearing --- #

        if run_had_error:  # Check the flag
            continue

# %% [markdown]
# ## 6. Sweep Complete

# %%
print("\nJumpReLU sparsity-parameter sweep (using remote activation store) complete! Check WandB for results.")
