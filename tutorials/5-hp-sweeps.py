# %% [markdown]
# # Tutorial 1: End-to-End CLT Training on GPT-2 Small
#
# This tutorial demonstrates the complete process of training a Cross-Layer Transcoder (CLT)
# using the `clt` library. We will:
# 1. Configure the CLT model, activation generation, and training parameters.
# 2. Generate activations locally (with manifest) using the ActivationGenerator.
# 3. Configure the trainer to use the locally stored activations via the manifest.
# 4. Train the CLT model.
# 5. Save the trained CLT model.
#
# Note: Some datasets may have sequences exceeding the model's context size,
# causing extraction errors. We use a dataset with appropriate sequence lengths.

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

# Import components from the clt library
# (Ensure the 'clt' directory is in your Python path or installed)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# logging.basicConfig(level=logging.DEBUG)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from clt.config import CLTConfig, TrainingConfig, ActivationConfig
    from clt.activation_generation.generator import ActivationGenerator
    from clt.training.trainer import CLTTrainer
    from clt.models.clt import CrossLayerTranscoder

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
BASE_MODEL_NAME = "EleutherAI/pythia-70m"  # Using GPT-2 small
# %%
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)

print(model)

# %% [markdown]
# ## 2. Configuration
#
# We define three configuration objects:
# - `CLTConfig`: Specifies the architecture of the Cross-Layer Transcoder itself (number of features, layers matching the base model, activation function, etc.).
# - `ActivationConfig`: Specifies parameters for generating the activation dataset (source model, dataset, tokenization, storage format, *including manifest generation*, etc.).
# - `TrainingConfig`: Specifies parameters for the training process (learning rate, number of steps, loss coefficients, *how to access activations via manifest*, etc.).
#
# **Note**: For this tutorial, we use small values for features, steps, and generated tokens for speed.

# %%
# --- CLT Architecture Configuration ---
# We need to match the base model's dimensions
gpt2_num_layers = 6
gpt2_d_model = 512
expansion_factor = 4

# For the tutorial, let's use a smaller number of features than d_model
clt_num_features = gpt2_d_model * expansion_factor
clt_config = CLTConfig(
    num_features=clt_num_features,
    num_layers=gpt2_num_layers,  # Must match the base model
    d_model=gpt2_d_model,  # Must match the base model
    # clt_dtype="bfloat16", # Configured via TrainingConfig.activation_dtype now
    activation_fn="relu",  # As described in the paper
    jumprelu_threshold=0.03,  # Default value from paper
)
print("CLT Configuration:")
print(clt_config)

# --- Activation Generation Configuration ---
# Define where activations will be stored and how they should be generated
# Use a small number of target tokens for the tutorial
activation_dir = "./tutorial_activations_local_1M_pythia"
# Fix SyntaxError: remove parenthesis around string assignment
dataset_name = "monology/pile-uncopyrighted"  # "NeelNanda/pile-10k" is smaller if needed
activation_config = ActivationConfig(
    # Model Source
    model_name=BASE_MODEL_NAME,
    mlp_input_module_path_template="gpt_neox.layers.{}.mlp.input",  # We include the layernorm for linearity
    mlp_output_module_path_template="gpt_neox.layers.{}.mlp.output",  # Default for GPT2
    model_dtype=None,  # Use default precision
    # Dataset Source
    dataset_path=dataset_name,
    dataset_split="train",
    dataset_text_column="text",
    # Generation Parameters
    context_size=128,
    inference_batch_size=192,
    exclude_special_tokens=True,
    prepend_bos=True,
    # Dataset Handling
    streaming=True,
    dataset_trust_remote_code=False,
    cache_path=None,
    # Generation Output Control
    target_total_tokens=1_000_000,  # Generate 1M tokens for the tutorial
    # Storage Parameters
    activation_dir=activation_dir,
    output_format="hdf5",
    compression="gzip",
    chunk_token_threshold=8_000,
    activation_dtype="float32",  # Explicitly set desired storage precision
    # Normalization
    compute_norm_stats=True,
    # NNsight args (defaults are usually fine)
    nnsight_tracer_kwargs={},
    nnsight_invoker_args={},
)
print("\nActivation Generation Configuration:")
print(activation_config)

# --- Base Training Configuration ---
# Configure the training loop, pointing it to the pre-generated activations
# Define the path where the generated activations are expected
expected_activation_path = os.path.join(
    activation_config.activation_dir,
    activation_config.model_name,
    f"{os.path.basename(activation_config.dataset_path)}_{activation_config.dataset_split}",
)

# --- Default Hyperparameters (will be overridden in sweep) ---
_lr = 1e-4
_batch_size = 1024
_default_sparsity_lambda = 0.003  # Keep a default reference if needed elsewhere
_default_sparsity_c = 0.03  # Keep a default reference if needed elsewhere

base_training_config = TrainingConfig(  # Renamed to base_training_config
    # Training loop parameters
    learning_rate=_lr,
    training_steps=1000,  # Reduced steps for tutorial
    seed=42,  # Added seed for reproducibility
    # Activation source - use local manifest-based store
    activation_source="local_manifest",  # Changed from 'local'
    activation_path=expected_activation_path,  # Point to generated data directory
    activation_dtype="float32",  # Specify dtype for loading/training
    # Training batch size
    train_batch_size_tokens=_batch_size,
    sampling_strategy="random_chunk",
    # Normalization for training (use stored stats)
    normalization_method="auto",  # Use stats from norm_stats.json generated earlier
    # Loss function coefficients (will be overridden)
    sparsity_lambda=_default_sparsity_lambda,
    sparsity_c=_default_sparsity_c,
    preactivation_coef=3e-6,
    # Optimizer & Scheduler
    optimizer="adamw",
    lr_scheduler="linear_final20",
    # Logging & Checkpointing
    log_interval=10,
    eval_interval=50,
    checkpoint_interval=100,
    dead_feature_window=200,  # Reduced window for tutorial
    # WandB (Optional)
    enable_wandb=True,
    wandb_project="clt-hp-sweeps-pythia",
    wandb_run_name="placeholder-run-name",  # Will be set in loop
    # Fields removed (now in ActivationConfig or implicitly handled):
    # model_name, model_dtype, mlp_*, dataset_*, streaming, context_size,
    # inference_batch_size, prepend_bos, exclude_special_tokens, cache_path,
    # n_batches_in_buffer (only used by streaming), normalization_estimation_batches
)
print("\nBase Training Configuration:")
print(base_training_config)

# %% [markdown]
# ## 3. Generate Activations (One-Time Step)
#
# Before training, we need to generate the activation dataset using the configuration defined above.
# The generator will now create HDF5 chunks, `metadata.json`, `norm_stats.json` (if enabled),
# and importantly, the `index.bin` manifest file required by `LocalActivationStore`.
#
# **Example Command:**
# ```bash
# python scripts/generate_activations.py --config path/to/activation_config.yaml
# ```
# (Using a config file is recommended for complex generation setups)
#
# For this tutorial notebook, we will *simulate* this step by directly calling the generator class.
# **Note:** This generation step can take some time depending on the model, dataset, and `target_total_tokens`.

# %%
print("\nStep 1: Generating/Verifying Activations (including manifest)...")

# Check if activations *and manifest* already exist
metadata_path = os.path.join(expected_activation_path, "metadata.json")
manifest_path = os.path.join(expected_activation_path, "index.bin")  # Check for manifest

if os.path.exists(metadata_path) and os.path.exists(manifest_path):
    print(f"Activations and manifest already found at: {expected_activation_path}")
    print("Skipping generation. Delete the directory to regenerate.")
else:
    print(f"Activations or manifest not found. Generating them now at: {expected_activation_path}")
    try:
        # Instantiate the generator with the ActivationConfig
        generator = ActivationGenerator(
            cfg=activation_config,
            device=device,  # Pass the device determined earlier
        )
        # Run the generation process
        generation_start_time = time.time()
        generator.generate_and_save()  # This now saves index.bin too
        generation_end_time = time.time()
        print(f"Activation generation complete in {generation_end_time - generation_start_time:.2f}s.")
    except Exception as gen_err:
        print(f"[ERROR] Activation generation failed: {gen_err}")
        traceback.print_exc()
        raise

# %% [markdown]
# ## 4. Hyperparameter Sweep for Training
#
# Now we will loop through different values for `sparsity_c` and `sparsity_lambda`,
# training a separate CLT model for each combination using the pre-generated activations.

# %%

# --- Define Hyperparameter Ranges ---
sparsity_c_values = [0.01, 0.03, 0.09, 0.27, 0.81, 2.43]
sparsity_lambda_values = [1e-5, 3e-5, 9e-5, 2.7e-4, 8.1e-4, 2.43e-3]

print("\nStarting Hyperparameter Sweep...")
print(f"Sweeping over sparsity_c: {sparsity_c_values}")
print(f"Sweeping over sparsity_lambda: {sparsity_lambda_values}")

# --- Sweep Loop ---
sweep_results: dict = {}

for sc in sparsity_c_values:
    for sl in sparsity_lambda_values:
        run_start_time = time.time()
        print(f"\n--- Starting Run: sparsity_c={sc:.2f}, sparsity_lambda={sl:.1e} ---")

        # --- Configure for this specific run ---
        training_config = copy.deepcopy(base_training_config)  # Start with base config
        training_config.sparsity_c = sc
        training_config.sparsity_lambda = sl

        # Update WandB run name
        wdb_run_name = (
            f"{clt_config.num_features}-width-"
            f"{training_config.train_batch_size_tokens}-batch-"
            f"{training_config.learning_rate:.1e}-lr-"
            f"{sl:.1e}-slambda-"  # Use current sl
            f"{sc:.2f}-sc"  # Use current sc
        )
        training_config.wandb_run_name = wdb_run_name
        print(f"Generated WandB run name: {wdb_run_name}")

        # Define a unique directory for logs and checkpoints for this run
        log_dir = f"clt_training_logs/sweep_sc_{sc:.2f}_sl_{sl:.1e}_{int(time.time())}"
        os.makedirs(log_dir, exist_ok=True)
        print(f"Logs and checkpoints will be saved to: {log_dir}")

        # --- Instantiate and Run Trainer ---
        try:
            print("\nCreating CLTTrainer instance for this run...")
            print(f"- Using device: {device}")
            print(f"- CLT config: {vars(clt_config)}")
            print(f"- Activation Source: {training_config.activation_source}")
            print(f"- Reading activations from: {training_config.activation_path}")
            print(f"- Current Training Config: {vars(training_config)}")  # Print specific config

            trainer = CLTTrainer(
                clt_config=clt_config,
                training_config=training_config,  # Use the modified config
                log_dir=log_dir,  # Use the run-specific log dir
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
                f"\nTraining for run (sc={sc:.2f}, sl={sl:.1e}) finished in {end_train_time - start_train_time:.2f} seconds."
            )

            # --- Saving the model for this run ---
            # The trainer automatically saves the final model in log_dir/clt_final.pt
            print(f"Final model for this run saved automatically in: {log_dir}")
            # Optional: manual save if needed
            # final_model_path = os.path.join(log_dir, f"clt_sc_{sc:.2f}_sl_{sl:.1e}_final_manual.pt")
            # print(f"Manually saving final model to: {final_model_path}")
            # trained_clt_model.save(final_model_path)

        except Exception as e:
            print(f"\n[ERROR] Failed during run (sc={sc:.2f}, sl={sl:.1e}): {e}")
            traceback.print_exc()
            print("Continuing to the next run...")
            continue  # Move to the next iteration of the inner loop

        run_end_time = time.time()
        print(f"--- Finished Run (sc={sc:.2f}, sl={sl:.1e}) in {run_end_time - run_start_time:.2f}s ---")

# %% [markdown]
# ## 5. Post-Sweep Analysis (Optional)
#
# After the sweep completes, the `clt_training_logs` directory will contain subdirectories
# for each hyperparameter combination. You can analyze the results stored in these directories
# (e.g., WandB logs, saved checkpoints, evaluation metrics) to determine the best
# performing hyperparameters.

# %% [markdown]
# ## 6. Loading a Specific Trained Model (Example)
#
# If you want to load a specific model from the sweep later, you'll need its
# corresponding config values (especially `sparsity_c` and `sparsity_lambda` used for its run)
# and the path to its saved state dict within its `log_dir`.

# %%
# --- Example: Loading one model from the sweep ---
# Let's say we want to load the model for the first run (adjust as needed)
example_sc = sparsity_c_values[0]
example_sl = sparsity_lambda_values[0]

# Find the log directory associated with this run (requires inspecting log dir names or storing them)
# This is a simple search, might need refinement if timestamps cause issues
example_log_dir_pattern = f"sweep_sc_{example_sc:.2f}_sl_{example_sl:.1e}"
found_log_dir = None
if os.path.exists("clt_training_logs"):
    for dirname in os.listdir("clt_training_logs"):
        if dirname.startswith(example_log_dir_pattern):
            found_log_dir = os.path.join("clt_training_logs", dirname)
            break

if found_log_dir and os.path.exists(os.path.join(found_log_dir, "clt_final.pt")):
    example_model_path = os.path.join(found_log_dir, "clt_final.pt")
    print(f"\nExample: Loading the trained model from: {example_model_path}")

    # 1. Re-create the config (or load it) - using the base CLT config is fine here
    #    The CLTConfig doesn't change during the sweep.
    loaded_clt_config = CLTConfig(
        num_features=clt_config.num_features,
        num_layers=clt_config.num_layers,
        d_model=clt_config.d_model,
        activation_fn=clt_config.activation_fn,
        jumprelu_threshold=clt_config.jumprelu_threshold,
    )

    # 2. Load the model structure and state dict
    loaded_clt_model = CrossLayerTranscoder(
        config=loaded_clt_config,
        process_group=None,
        device=torch.device(device),
    )
    loaded_clt_model.load(example_model_path)

    print("Example model loaded successfully.")
    print(f"Loaded model is on device: {next(loaded_clt_model.parameters()).device}")
else:
    print(
        f"\nCould not find log directory or final model for example run (sc={example_sc:.2f}, sl={example_sl:.1e}). Skipping load example."
    )


# %% [markdown]
# ## 7. Next Steps
#
# This tutorial demonstrated sweeping hyperparameters for CLT training after generating
# activations once. This approach is efficient for exploring parameter spaces without
# repeatedly running the costly activation generation step.
#
# With the results from the sweep (logs in `clt_training_logs` and WandB), you can:
# - **Identify Optimal Hyperparameters**: Find the `sparsity_c` and `sparsity_lambda` that provide the best trade-off between sparsity and reconstruction loss for your needs.
# - **Analyze Trained Models**: Examine the features learned by models trained with different sparsity pressures.
# - **Build and Evaluate Replacement Models**: Use the best-performing CLT to create and test a sparse replacement model.

# %%
print("\nHyperparameter Sweep Tutorial Complete!")
print("You have trained multiple Cross-Layer Transcoders with varying sparsity parameters.")
print("Logs for each run are saved in subdirectories within: clt_training_logs")

# %%
