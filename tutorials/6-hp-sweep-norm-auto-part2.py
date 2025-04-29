# %% [markdown]
# # Tutorial 6: GPT-2 Hyperparameter Sweep (Local Manifest, Norm=Auto, Part 2)
#
# This tutorial demonstrates splitting a hyperparameter sweep across multiple scripts,
# using locally generated activations accessed via a manifest file.
# This script focuses on training GPT-2 CLTs with `normalization_method="auto"` and the second
# half of the `sparsity_c` values.
#
# It first generates activations locally for 'gpt2' on 'monology/pile-uncopyrighted'
# if they don't exist, including computing normalization stats and creating a manifest.
#
# We will:
# 1. Configure CLT, Activation generation, and Training parameters (setting norm="auto", source="local_manifest").
# 2. Define the second subset of hyperparameters.
# 3. Generate activations locally (if needed).
# 4. Train CLT models for the assigned hyperparameter subset using the local manifest.
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

import shutil  # Import shutil for directory removal

# Import components from the clt library
# (Ensure the 'clt' directory is in your Python path or installed)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# logging.basicConfig(level=logging.DEBUG)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from clt.config import CLTConfig, TrainingConfig, ActivationConfig
    from clt.activation_generation.generator import ActivationGenerator  # Added import

    # from clt.activation_generation.generator import ActivationGenerator # Removed unused import
    from clt.training.trainer import CLTTrainer

    # from clt.models.clt import CrossLayerTranscoder # Removed unused import

    # We no longer need to import specific stores here, trainer handles it
    # from clt.training.local_activation_store import LocalActivationStore
    # from clt.training.data import StreamingActivationStore
except ImportError as e:
    print(f"ImportError: {e}")
    print("Please ensure the 'clt' library is installed or the clt directory is in your PYTHONPATH.")
    # You might need to adjust your PYTHONPATH, e.g.,
    # import sys
    # sys.path.append('path/to/your/project/root')
    # from clt.config import ... etc.

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
# We define three configuration objects:
# - `CLTConfig`: Specifies the architecture of the Cross-Layer Transcoder itself.
# - `ActivationConfig`: Specifies parameters used for generating the activation dataset (for reference).
# - `TrainingConfig`: Specifies parameters for the training process, including `normalization_method="auto"`.

# %%
# --- CLT Architecture Configuration ---
gpt2_num_layers = 12
gpt2_d_model = 768
expansion_factor = 32
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

# --- Activation Generation Configuration ---
# Settings for generating activations locally, including norm stats for "auto" method
activation_dir = "./tutorial_activations_local_1M"
dataset_name = "monology/pile-uncopyrighted"
activation_config = ActivationConfig(
    # Model Source
    model_name=BASE_MODEL_NAME,
    mlp_input_module_path_template="transformer.h.{}.ln_2.input",
    mlp_output_module_path_template="transformer.h.{}.mlp.output",
    model_dtype=None,
    # Dataset Source
    dataset_path=dataset_name,
    dataset_split="train",
    dataset_text_column="text",
    # Generation Parameters
    context_size=128,
    inference_batch_size=192,  # Adjust based on GPU memory
    exclude_special_tokens=True,
    prepend_bos=True,
    # Dataset Handling
    streaming=True,
    dataset_trust_remote_code=False,
    cache_path=None,
    # Generation Output Control
    target_total_tokens=1_000_000,  # Generate 1M tokens
    # Storage Parameters
    activation_dir=activation_dir,
    output_format="hdf5",  # Ensure HDF5 for manifest
    compression="gzip",
    chunk_token_threshold=32_000,  # Keeping user's value
    activation_dtype="float32",
    # Normalization - Crucial for "auto" mode during training
    compute_norm_stats=True,
    # NNsight args (defaults are usually fine)
    nnsight_tracer_kwargs={},
    nnsight_invoker_args={},
)
print("\nActivation Generation Configuration:")
print(activation_config)

# --- Base Training Configuration ---
# Define the path where the generated activations are expected
expected_activation_path = os.path.join(
    activation_config.activation_dir,
    activation_config.model_name.replace("/", "_"),  # Handle potential slashes in model names
    f"{os.path.basename(activation_config.dataset_path)}_{activation_config.dataset_split}",
)
print(f"Expecting activations at: {expected_activation_path}")

_lr = 1e-4
_batch_size = 1024
_default_sparsity_lambda = 0.003
_default_sparsity_c = 0.03
# SERVER_URL = "http://34.41.125.189:8000" # Removed - Not needed for local

base_training_config = TrainingConfig(
    learning_rate=_lr,
    training_steps=1000,
    seed=42,
    # --- Activation Source: Local Manifest --- #
    activation_source="local_manifest",
    activation_path=expected_activation_path,  # Point to generated data directory
    # remote_config={ ... }, # Removed remote config block
    # --------------------------------------- #
    activation_dtype="float32",
    train_batch_size_tokens=_batch_size,
    sampling_strategy="random_chunk",  # Keep random chunk for sweep
    normalization_method="auto",  # Set normalization method for this script
    sparsity_lambda=_default_sparsity_lambda,  # Default, will be overridden
    sparsity_c=_default_sparsity_c,  # Default, will be overridden
    preactivation_coef=3e-6,
    optimizer="adamw",
    lr_scheduler="linear_final20",
    log_interval=10,
    eval_interval=50,
    checkpoint_interval=100,
    dead_feature_window=200,
    enable_wandb=True,
    wandb_project="clt-hp-sweeps-gpt2",  # Specific project for gpt2 norm=auto
    wandb_run_name="placeholder-run-name",  # Will be set in loop
)
print("\nBase Training Configuration (Normalization=Auto, Source=Local Manifest):")
print(base_training_config)

# %% [markdown]
# ## 3. Generate Activations (One-Time Step, if needed)
#
# Before the sweep, generate the activation dataset locally using the configuration defined above.
# This creates HDF5 chunks, metadata, normalization stats, and the `index.bin` manifest file.

# %%
print("\nStep 3: Generating/Verifying Activations (including manifest)...")

# Check if activations *and manifest* already exist
metadata_path = os.path.join(expected_activation_path, "metadata.json")
manifest_path = os.path.join(expected_activation_path, "index.bin")  # Check for manifest
norm_stats_path = os.path.join(expected_activation_path, "norm_stats.json")  # Check for norm stats

if (
    os.path.exists(metadata_path)
    and os.path.exists(manifest_path)
    and (not activation_config.compute_norm_stats or os.path.exists(norm_stats_path))
):
    print(f"Activations, manifest, and required stats already found at: {expected_activation_path}")
    print("Skipping generation. Delete the directory to regenerate.")
else:
    print(f"Activations, manifest, or stats not found. Generating them now at: {expected_activation_path}")
    try:
        # Instantiate the generator with the ActivationConfig
        generator = ActivationGenerator(
            cfg=activation_config,
            device=device,  # Pass the device determined earlier
        )
        # Run the generation process
        generation_start_time = time.time()
        generator.generate_and_save()  # This now saves index.bin and norm_stats.json
        generation_end_time = time.time()
        print(f"Activation generation complete in {generation_end_time - generation_start_time:.2f}s.")
    except Exception as gen_err:
        print(f"[ERROR] Activation generation failed: {gen_err}")
        traceback.print_exc()
        # sys.exit(1) # Exit if generation fails critical step
        raise  # Re-raise the exception to halt execution

# %% [markdown]
# ## 4. Hyperparameter Sweep (Local Manifest, Norm=Auto, Part 2)
#
# Loop through the second half of `sparsity_c` values and all `sparsity_lambda` values,
# training using the locally generated activations via the manifest.

# %%

# --- Define Hyperparameter Ranges for this script ---
# Split sparsity_c values (second half)
all_sparsity_c_values = [0.01, 0.03, 0.09, 0.27, 0.81, 2.43]
split_point = len(all_sparsity_c_values) // 2
sparsity_c_values = all_sparsity_c_values[split_point:]  # Second half

# Use all lambda values
sparsity_lambda_values = [1e-5, 3e-5, 9e-5, 2.7e-4, 8.1e-4, 2.43e-3]

print("\nStarting Hyperparameter Sweep (GPT-2, Local Manifest, Norm=Auto, Part 2)...")
print(f"Sweeping over sparsity_c: {sparsity_c_values}")
print(f"Sweeping over sparsity_lambda: {sparsity_lambda_values}")

# --- Sweep Loop ---
sweep_results: dict = {}
log_base_dir = f"clt_training_logs/norm_auto_part2"  # Specific log base directory

# --- Delete existing log directory for this script before starting ---
if os.path.exists(log_base_dir):
    print(f"Deleting existing log directory: {log_base_dir}")
    shutil.rmtree(log_base_dir)
# ------------------------------------------------------------------

os.makedirs(log_base_dir, exist_ok=True)

for sc in sparsity_c_values:
    for sl in sparsity_lambda_values:
        run_start_time = time.time()
        print(f"\n--- Starting Run: sparsity_c={sc:.2f}, sparsity_lambda={sl:.1e} ---")

        # --- Configure for this specific run ---
        training_config = copy.deepcopy(base_training_config)  # Start with base config
        training_config.sparsity_c = sc
        training_config.sparsity_lambda = sl
        # Normalization method is already set to "auto" in base_training_config

        # Update WandB run name
        wdb_run_name = (
            f"{clt_config.num_features}-width-"
            f"{training_config.train_batch_size_tokens}-batch-"
            f"{training_config.learning_rate:.1e}-lr-"
            f"{sl:.1e}-slambda-"
            f"{sc:.2f}-sc-"
            f"norm_{training_config.normalization_method}"  # Will be "auto"
        )
        training_config.wandb_run_name = wdb_run_name
        print(f"Generated WandB run name: {wdb_run_name}")

        # Define a unique directory for logs and checkpoints for this run
        # Use the specific base directory for this script
        log_dir = os.path.join(log_base_dir, f"sweep_sc_{sc:.2f}_sl_{sl:.1e}_{int(run_start_time)}")
        os.makedirs(log_dir, exist_ok=True)
        print(f"Logs and checkpoints will be saved to: {log_dir}")

        # --- Instantiate and Run Trainer ---
        try:
            print("\nCreating CLTTrainer instance for this run...")
            print(f"- Using device: {device}")
            print(f"- CLT config: {vars(clt_config)}")
            print(f"- Activation Source: {training_config.activation_source}")
            print(f"- Reading activations from: {training_config.activation_path}")
            # Safely access remote config details - REMOVED
            # if training_config.remote_config: ...
            print(f"- Current Training Config: {vars(training_config)}")

            trainer = CLTTrainer(
                clt_config=clt_config,
                training_config=training_config,
                log_dir=log_dir,
                device=device,
                distributed=False,
            )
            print("CLTTrainer instance created successfully.")

            # Start training for this run
            print("\nBeginning training for this run...")
            print(f"Training for {training_config.training_steps} steps.")
            print(f"Normalization method set to: {training_config.normalization_method}")

            start_train_time = time.time()
            trained_clt_model = trainer.train(eval_every=training_config.eval_interval)
            end_train_time = time.time()
            print(
                f"Training for run (sc={sc:.2f}, sl={sl:.1e}) finished in {end_train_time - start_train_time:.2f} seconds."
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
# ## 5. Post-Sweep Analysis (Optional)
#
# After this script completes, the `{log_base_dir}` directory will contain subdirectories
# for each hyperparameter combination run by this script. Analyze these results
# (e.g., WandB logs, saved checkpoints) alongside the results from the other sweep scripts.

# %% [markdown]
# ## 6. Loading a Trained Model (Example)
#
# Refer to the combined analysis or the log directories created by this script
# (within `{log_base_dir}`) to load specific models trained here.

# %%
print("\nHyperparameter Sweep (GPT-2, Local Manifest, Norm=Auto, Part 2) Complete!")
print(f"Logs for each run are saved in subdirectories within: {log_base_dir}")

# %%
