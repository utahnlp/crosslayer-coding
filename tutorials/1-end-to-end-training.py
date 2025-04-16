# %% [markdown]
# # Tutorial 1: End-to-End CLT Training on GPT-2 Small
#
# This tutorial demonstrates the complete process of training a Cross-Layer Transcoder (CLT)
# using the `clt` library. We will:
# 1. Configure the CLT model, activation generation, and training parameters.
# 2. (Simulate) Generate activations locally using the ActivationGenerator.
# 3. Configure the trainer to use the locally stored activations.
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
except ImportError as e:
    print(f"ImportError: {e}")
    print(
        "Please ensure the 'clt' library is installed or the clt directory is in your PYTHONPATH."
    )
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
BASE_MODEL_NAME = "gpt2"  # Using GPT-2 small

# %% [markdown]
# ## 2. Configuration
#
# We define three configuration objects:
# - `CLTConfig`: Specifies the architecture of the Cross-Layer Transcoder itself (number of features, layers matching the base model, activation function, etc.).
# - `ActivationConfig`: Specifies parameters for generating the activation dataset (source model, dataset, tokenization, storage format, etc.).
# - `TrainingConfig`: Specifies parameters for the training process (learning rate, number of steps, loss coefficients, how to access activations, etc.).
#
# **Note**: For this tutorial, we use small values for features, steps, and generated tokens for speed.

# %%
# --- CLT Architecture Configuration ---
# We need to match the base model's dimensions
gpt2_num_layers = 12
gpt2_d_model = 768
expansion_factor = 4

# For the tutorial, let's use a smaller number of features than d_model
clt_num_features = gpt2_d_model * expansion_factor
clt_config = CLTConfig(
    num_features=clt_num_features,
    num_layers=gpt2_num_layers,  # Must match the base model
    d_model=gpt2_d_model,  # Must match the base model
    # clt_dtype="bfloat16",
    activation_fn="relu",  # As described in the paper
    jumprelu_threshold=0.03,  # Default value from paper
)
print("CLT Configuration:")
print(clt_config)

# --- Activation Generation Configuration ---
# Define where activations will be stored and how they should be generated
# Use a small number of target tokens for the tutorial
activation_dir = "./tutorial_activations"
dataset_name = (
    "monology/pile-uncopyrighted"  # "NeelNanda/pile-10k" is smaller if needed
)
activation_config = ActivationConfig(
    # Model Source
    model_name=BASE_MODEL_NAME,
    mlp_input_module_path_template="transformer.h.{}.ln_2.input",  # We include the layernorm for linearity
    mlp_output_module_path_template="transformer.h.{}.mlp.output",  # Default for GPT2
    model_dtype=None,  # Use default precision
    # Dataset Source
    dataset_path=dataset_name,
    dataset_split="train",
    dataset_text_column="text",
    # Generation Parameters
    context_size=128,
    inference_batch_size=256,
    exclude_special_tokens=True,
    prepend_bos=True,
    # Dataset Handling
    streaming=True,
    dataset_trust_remote_code=False,
    cache_path=None,
    # Generation Output Control
    target_total_tokens=2_000_000,  # Generate 2M tokens for the tutorial (~100GB)
    # Storage Parameters
    activation_dir=activation_dir,
    output_format="hdf5",
    compression="gzip",
    chunk_token_threshold=10_000,
    # Normalization
    compute_norm_stats=True,
    # NNsight args (defaults are usually fine)
    nnsight_tracer_kwargs={},
    nnsight_invoker_args={},
)
print("\nActivation Generation Configuration:")
print(activation_config)

# --- Training Configuration ---
# Configure the training loop, pointing it to the pre-generated activations
# Define the path where the generated activations are expected
expected_activation_path = os.path.join(
    activation_config.activation_dir,
    activation_config.model_name,
    f"{os.path.basename(activation_config.dataset_path)}_{activation_config.dataset_split}",
)

training_config = TrainingConfig(
    # Training loop parameters
    learning_rate=3e-4,
    training_steps=1000,  # Reduced steps for tutorial
    # Activation source - use the generated data
    activation_source="local",
    activation_path=expected_activation_path,  # Point to generated data
    # Training batch size
    train_batch_size_tokens=1024,
    # Normalization for training (use stored stats)
    normalization_method="auto",  # Use stats from norm_stats.json generated earlier
    # Loss function coefficients
    sparsity_lambda=0.00003,
    sparsity_c=1.0,
    preactivation_coef=3e-6,
    # Optimizer & Scheduler
    optimizer="adamw",
    lr_scheduler="linear",
    # Logging & Checkpointing
    log_interval=10,
    eval_interval=50,
    checkpoint_interval=100,
    dead_feature_window=200,  # Reduced window for tutorial
    # WandB (Optional)
    enable_wandb=True,
    wandb_project="clt-tutorial",
    # Fields removed (now in ActivationConfig or implicitly handled):
    # model_name, model_dtype, mlp_*, dataset_*, streaming, context_size,
    # inference_batch_size, prepend_bos, exclude_special_tokens, cache_path,
    # n_batches_in_buffer (only used by streaming), normalization_estimation_batches
)
print("\nTraining Configuration:")
print(training_config)

# %% [markdown]
# ## 3. Generate Activations (One-Time Step)
#
# Before training, we need to generate the activation dataset using the configuration defined above.
# This is typically done by running the `scripts/generate_activations.py` script from the command line.
#
# **Example Command:**
# ```bash
# python scripts/generate_activations.py --model_name gpt2 --mlp_input_template "transformer.h.{}.mlp.input" --mlp_output_template "transformer.h.{}.mlp.output" --dataset_path monology/pile-uncopyrighted --context_size 128 --target_total_tokens 1000000 --activation_dir ./tutorial_activations --output_format hdf5 --compute_norm_stats
# ```
#
# For this tutorial notebook, we will *simulate* this step by directly calling the generator class.
# **Note:** This generation step can take some time depending on the model, dataset, and `target_total_tokens`.

# %%
print("\nStep 1: Generating/Verifying Activations...")

# Check if activations already exist
metadata_path = os.path.join(expected_activation_path, "metadata.json")
if os.path.exists(metadata_path):
    print(f"Activations already found at: {expected_activation_path}")
    print("Skipping generation. Delete the directory to regenerate.")
else:
    print(f"Activations not found. Generating them now at: {expected_activation_path}")
    try:
        # Instantiate the generator with the ActivationConfig
        generator = ActivationGenerator(
            activation_config=activation_config,
            device=device,  # Pass the device determined earlier
        )
        # Run the generation process
        generation_start_time = time.time()
        generator.generate_and_save()
        generation_end_time = time.time()
        print(
            f"Activation generation complete in {generation_end_time - generation_start_time:.2f}s."
        )
    except Exception as gen_err:
        print(f"[ERROR] Activation generation failed: {gen_err}")
        traceback.print_exc()
        raise

# %% [markdown]
# ## 4. Training the CLT from Local Activations
#
# Now we instantiate the `CLTTrainer`. Because `TrainingConfig.activation_source` is set to `'local'`,
# the trainer will automatically create a `MappedActivationStore` internally, pointing to the
# `TrainingConfig.activation_path` where we just generated the data.
#
# If `TrainingConfig.normalization_method` is set to `'auto'`, the `MappedActivationStore` will look for
# the `norm_stats.json` file (created during generation) and apply normalization accordingly.

# %%
print("\nInitializing CLTTrainer for training from local activations...")

# Define a directory for logs and checkpoints (can be different from activation dir)
log_dir = f"clt_training_logs/clt_gpt2_local_train_{int(time.time())}"
os.makedirs(log_dir, exist_ok=True)
print(f"Logs and checkpoints will be saved to: {log_dir}")

# Instantiate the trainer
try:
    print("\nCreating CLTTrainer instance...")
    print(f"- Using device: {device}")
    print(f"- CLT config: {vars(clt_config)}")
    print(f"- Activation Source: {training_config.activation_source}")
    print(f"- Reading activations from: {training_config.activation_path}")

    trainer = CLTTrainer(
        clt_config=clt_config,
        training_config=training_config,
        log_dir=log_dir,
        device=device,
    )
    print("CLTTrainer instance created successfully.")
except Exception as e:
    print(f"\n[ERROR] Failed to initialize CLTTrainer: {e}")
    traceback.print_exc()

    print("\nPlease fix the issues above before continuing.")
    raise

# Start training
print("\nBeginning training using MappedActivationStore...")
print(f"Training for {training_config.training_steps} steps.")
print(f"Normalization method set to: {training_config.normalization_method}")

try:
    start_train_time = time.time()
    trained_clt_model = trainer.train(eval_every=training_config.eval_interval)
    end_train_time = time.time()
    print(f"\nTraining finished in {end_train_time - start_train_time:.2f} seconds.")
except Exception as train_err:
    print(f"\n[ERROR] Training failed: {train_err}")
    traceback.print_exc()
    raise

# %% [markdown]
# ## 5. Saving and Loading the Trained Model
#
# The trainer automatically saves the final model (`clt_final.pt`) and checkpoints in the `log_dir`. The `CrossLayerTranscoder` model has `save` and `load` methods.

# %%
# The trainer already saved the final model, but we can also save it manually
final_model_path = os.path.join(log_dir, "clt_trained_final_manual.pt")
print(f"\nManually saving final model to: {final_model_path}")
trained_clt_model.save(final_model_path)

# Verify saved files in the log directory
print(f"\nContents of log directory ({log_dir}):")
print(os.listdir(log_dir))

# --- Loading the model ---
# To load the model later, you need the configuration it was trained with
# and the path to the saved state dict.

print(f"\nLoading the trained model from: {final_model_path}")

# 1. Re-create the config (or load it from where you saved it)
loaded_clt_config = CLTConfig(
    num_features=clt_config.num_features,  # Use the same params
    num_layers=clt_config.num_layers,
    d_model=clt_config.d_model,
    activation_fn=clt_config.activation_fn,
    jumprelu_threshold=clt_config.jumprelu_threshold,
)

# 2. Load the model structure and state dict
loaded_clt_model = CrossLayerTranscoder(loaded_clt_config)
loaded_clt_model.load(final_model_path, device=torch.device(device))

print("Model loaded successfully.")
print(f"Loaded model is on device: {next(loaded_clt_model.parameters()).device}")

# %% [markdown]
# ## 6. Next Steps
#
# This tutorial demonstrated the workflow of pre-generating activations and then training from them.
# This separation allows:
# - Generating activations once and reusing them for multiple training runs/experiments.
# - Generating activations on powerful hardware and training on less powerful hardware.
# - Easier sharing of processed activation datasets.
#
# With a trained `CrossLayerTranscoder`, you can now:
# - **Analyze Features**: Examine the learned encoder/decoder weights and feature activations.
# - **Build Replacement Models**: Integrate the CLT into the original base model to create a model that uses the sparse CLT features.
# - **Evaluate Performance**: Measure the trade-off between sparsity (L0 norm) and reconstruction accuracy (MSE loss).

# %%
print("\nTutorial Complete!")
print("You have trained a Cross-Layer Transcoder!")
print(f"The trained model and logs are saved in: {log_dir}")

# %%
