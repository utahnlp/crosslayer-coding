# %% [markdown]
# # Tutorial 7: GPT-2 Hyperparameter Sweep (Remote, Norm=None, Part 2)
#
# This tutorial demonstrates splitting a hyperparameter sweep across multiple scripts,
# fetching activations from a remote server.
# This script focuses on training GPT-2 CLTs with `normalization_method="none"` and the second
# half of the `sparsity_c` values.
#
# It assumes activations for 'gpt2/pile-uncopyrighted_train' are available on the server
# at http://34.41.125.189:8000.
#
# We will:
# 1. Configure CLT and Training parameters (setting norm="none", source="remote").
# 2. Define the second subset of hyperparameters.
# 3. Configure the trainer to use the `RemoteActivationStore`.
# 4. Train CLT models for the assigned hyperparameter subset.
# 5. Save the trained CLT models.

# %% [markdown]
# ## 1. Imports and Setup
#
# First, let's import the necessary components from the library and set up basic parameters like the device.

# %%
import torch
import os
import time
import sys
import traceback
import copy
from transformers import AutoModelForCausalLM  # Moved import
import requests  # Ensure requests is imported

# Import components from the clt library
# (Ensure the 'clt' directory is in your Python path or installed)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# logging.basicConfig(level=logging.DEBUG)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from clt.config import CLTConfig, TrainingConfig, ActivationConfig

    # ActivationGenerator not needed
    from clt.training.trainer import CLTTrainer

    # CrossLayerTranscoder not explicitly needed

except ImportError as e:
    print(f"ImportError: {e}")
    print("Please ensure the 'clt' library is installed or the clt directory is in your PYTHONPATH.")

# Device setup
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# Base model for activation extraction
BASE_MODEL_NAME = "gpt2"
# %%

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
print(model)

# %% [markdown]
# ## 2. Configuration
#
# Define CLT, Activation (reference), and Training configs.
# Key change: `TrainingConfig.normalization_method="none"`.

# %%
# --- CLT Architecture Configuration ---
gpt2_num_layers = 12
gpt2_d_model = 768
expansion_factor = 4
clt_num_features = gpt2_d_model * expansion_factor
clt_config = CLTConfig(
    num_features=clt_num_features,
    num_layers=gpt2_num_layers,
    d_model=gpt2_d_model,
    activation_fn="relu",
    jumprelu_threshold=0.03,
)
print("CLT Configuration:")
print(clt_config)

# --- Activation Generation Configuration (Reference) ---
activation_dir = "./tutorial_activations_local_1M"
dataset_name = "monology/pile-uncopyrighted"
activation_config = ActivationConfig(
    model_name=BASE_MODEL_NAME,
    mlp_input_module_path_template="transformer.h.{}.ln_2.input",
    mlp_output_module_path_template="transformer.h.{}.mlp.output",
    dataset_path=dataset_name,
    dataset_split="train",
    dataset_text_column="text",
    context_size=128,
    inference_batch_size=192,
    exclude_special_tokens=True,
    prepend_bos=True,
    streaming=True,
    target_total_tokens=1_000_000,
    activation_dir=activation_dir,
    output_format="hdf5",
    compression="gzip",
    chunk_token_threshold=8_000,
    activation_dtype="float32",
)
print("\nReference Activation Generation Configuration (Assumed):")
print(activation_config)

# --- Base Training Configuration ---
# Define dataset ID expected on the server
dataset_id = f"{activation_config.model_name.replace('/', '_')}/{os.path.basename(activation_config.dataset_path)}_{activation_config.dataset_split}"
print(f"Expecting dataset_id on server: {dataset_id}")

_lr = 1e-4
_batch_size = 1024
_default_sparsity_lambda = 0.003
_default_sparsity_c = 0.03
SERVER_URL = "http://34.41.125.189:8000"  # From tutorial 4

base_training_config = TrainingConfig(
    learning_rate=_lr,
    training_steps=1000,
    seed=42,
    # --- Activation Source: Remote --- #
    activation_source="remote",
    # activation_path=expected_activation_path, # Removed for remote source
    remote_config={
        "server_url": SERVER_URL,
        "dataset_id": dataset_id,
        "timeout": 120,
        "max_retries": 3,
        "prefetch_batches": 16,
    },
    # --------------------------------- #
    activation_dtype="float32",
    train_batch_size_tokens=_batch_size,
    sampling_strategy="random_chunk",  # Keep random chunk for sweep
    normalization_method="none",  # Set normalization method for this script
    sparsity_lambda=_default_sparsity_lambda,
    sparsity_c=_default_sparsity_c,
    preactivation_coef=3e-6,
    optimizer="adamw",
    lr_scheduler="linear_final20",
    log_interval=10,
    eval_interval=50,
    checkpoint_interval=100,
    dead_feature_window=200,
    enable_wandb=True,
    wandb_project="clt-hp-sweeps-gpt2-norm-none",  # Specific project for gpt2 norm=none
    wandb_run_name="placeholder-run-name",
)
print("\nBase Training Configuration (Normalization=None, Source=Remote):")
print(base_training_config)

# %% [markdown]
# ## 3. Verify Server Connection (Optional)
#
# Before starting the sweep, we can optionally check if the server is reachable.

# %%
print(f"\nStep 1: Verifying Server Connection at {SERVER_URL}...")

health_check_url = f"{SERVER_URL}/api/v1/health"
try:
    response = requests.get(health_check_url, timeout=10)
    if response.status_code == 200 and response.json().get("status") == "ok":
        print(f"✅ Server connection successful and healthy at {SERVER_URL}")
    else:
        print(f"⚠️ Server health check failed or gave unexpected status: {response.status_code} - {response.text}")
        print("   Training will likely fail. Ensure the server is running and accessible.")
except requests.exceptions.RequestException as req_err:
    print(f"[ERROR] Failed to connect to server at {SERVER_URL}: {req_err}")
    print("   Please ensure the clt_server is running and accessible at the specified URL.")
    sys.exit(1)  # Exit if connection fails

# %% [markdown]
# ## 4. Hyperparameter Sweep (Remote, Norm=None, Part 2)
#
# Loop through the second half of `sparsity_c` values and all `sparsity_lambda` values.

# %%

# --- Define Hyperparameter Ranges for this script ---
all_sparsity_c_values = [0.01, 0.03, 0.09, 0.27, 0.81, 2.43]
split_point = len(all_sparsity_c_values) // 2
sparsity_c_values = all_sparsity_c_values[split_point:]  # Second half

sparsity_lambda_values = [1e-5, 3e-5, 9e-5, 2.7e-4, 8.1e-4, 2.43e-3]

print("\nStarting Hyperparameter Sweep (GPT-2, Remote, Norm=None, Part 2)...")
print(f"Sweeping over sparsity_c: {sparsity_c_values}")
print(f"Sweeping over sparsity_lambda: {sparsity_lambda_values}")

# --- Sweep Loop ---
sweep_results: dict = {}
log_base_dir = f"clt_training_logs/norm_none_part2"
os.makedirs(log_base_dir, exist_ok=True)

for sc in sparsity_c_values:
    for sl in sparsity_lambda_values:
        run_start_time = time.time()
        print(f"\n--- Starting Run: sparsity_c={sc:.2f}, sparsity_lambda={sl:.1e} ---")

        training_config = copy.deepcopy(base_training_config)
        training_config.sparsity_c = sc
        training_config.sparsity_lambda = sl

        wdb_run_name = (
            f"{clt_config.num_features}-width-"
            f"{training_config.train_batch_size_tokens}-batch-"
            f"{training_config.learning_rate:.1e}-lr-"
            f"{sl:.1e}-slambda-"
            f"{sc:.2f}-sc-"
            f"norm_{training_config.normalization_method}"  # Will be "none"
        )
        training_config.wandb_run_name = wdb_run_name
        print(f"Generated WandB run name: {wdb_run_name}")

        log_dir = os.path.join(log_base_dir, f"sweep_sc_{sc:.2f}_sl_{sl:.1e}_{int(run_start_time)}")
        os.makedirs(log_dir, exist_ok=True)
        print(f"Logs and checkpoints will be saved to: {log_dir}")

        try:
            print("\nCreating CLTTrainer instance for this run...")
            print(f"- Activation Source: {training_config.activation_source}")
            # Safely access remote config details
            if training_config.remote_config:
                print(f"- Reading activations from server: {training_config.remote_config.get('server_url')}")
                print(f"- Reading dataset_id: {training_config.remote_config.get('dataset_id')}")
            else:
                print("- Error: Remote config not found in training config!")
            print(f"- Training Config: {vars(training_config)}")

            trainer = CLTTrainer(
                clt_config=clt_config,
                training_config=training_config,
                log_dir=log_dir,
                device=device,
                distributed=False,
            )
            print("CLTTrainer instance created successfully.")

            print("\nBeginning training for this run...")
            start_train_time = time.time()
            trained_clt_model = trainer.train(eval_every=training_config.eval_interval)
            end_train_time = time.time()
            print(
                f"\nTraining for run (sc={sc:.2f}, sl={sl:.1e}) finished in {end_train_time - start_train_time:.2f} seconds."
            )
            print(f"Final model for this run saved automatically in: {log_dir}")

        except Exception as e:
            print(f"\n[ERROR] Failed during run (sc={sc:.2f}, sl={sl:.1e}): {e}")
            traceback.print_exc()
            print("Continuing to the next run...")
            continue

        run_end_time = time.time()
        print(f"--- Finished Run (sc={sc:.2f}, sl={sl:.1e}) in {run_end_time - run_start_time:.2f}s ---")

# %% [markdown]
# ## 5. Post-Sweep Analysis
#
# Analyze results in `{log_base_dir}` alongside other scripts.

# %% [markdown]
# ## 6. Loading a Trained Model
#
# Models saved in `{log_base_dir}`.

# %%
print("\nHyperparameter Sweep (GPT-2, Remote, Norm=None, Part 2) Complete!")  # Updated print
print(f"Logs for each run are saved in subdirectories within: {log_base_dir}")

# %%
