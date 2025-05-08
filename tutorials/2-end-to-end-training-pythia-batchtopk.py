# %% [markdown]
# # Tutorial 2: End-to-End CLT Training with BatchTopK Activation
#
# This tutorial demonstrates training a Cross-Layer Transcoder (CLT)
# using the **BatchTopK** activation function. We will:
# 1. Configure the CLT model for BatchTopK, activation generation, and training parameters.
# 2. Generate activations locally (with manifest) using the ActivationGenerator.
# 3. Configure the trainer to use the locally stored activations via the manifest.
# 4. Train the CLT model using BatchTopK activation.
# 5. Save the trained CLT model.
#
# BatchTopK enforces sparsity globally across all features from all layers for each
# sample in a batch, controlled by a single `k` or `frac` parameter.

# %% [markdown]
# ## 1. Imports and Setup
#
# First, let's import the necessary components and set up the device.

# %%
import torch
import os
import time
import sys
import traceback

# Need transformers for the import check below
from transformers import AutoModelForCausalLM


# Import components from the clt library
# (Ensure the 'clt' directory is in your Python path or installed)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
# ## 2. Configuration
#
# We configure the CLT, Activation Generation, and Training.
# Key change: `CLTConfig.activation_fn` is set to `"batchtopk"`.

# %%
# --- CLT Architecture Configuration ---
num_layers = 6
d_model = 512
expansion_factor = 32
clt_num_features = d_model * expansion_factor

# Recommended sparsity fraction for BatchTopK
batchtopk_sparsity_fraction = 0.005  # Keep top 0.2% (590 with expansion factor 32) of features across all layers

clt_config = CLTConfig(
    num_features=clt_num_features,
    num_layers=num_layers,
    d_model=d_model,
    activation_fn="batchtopk",  # Use BatchTopK activation
    batchtopk_k=None,  # Specify k or frac
    batchtopk_frac=batchtopk_sparsity_fraction,  # Keep top 2% features globally
    batchtopk_straight_through=True,  # Use STE for gradients
    # jumprelu_threshold is not used for batchtopk
)
print("CLT Configuration (BatchTopK):")
print(clt_config)

# --- Activation Generation Configuration ---
# Same as before - generate activations to train on
# Use the same directory as the first tutorial, since generation is independent of CLT activation fn
activation_dir = "./tutorial_activations_local_1M_pythia"  # Point back to original activations
dataset_name = "monology/pile-uncopyrighted"
activation_config = ActivationConfig(
    # Model Source
    model_name=BASE_MODEL_NAME,
    mlp_input_module_path_template="gpt_neox.layers.{}.mlp.input",
    mlp_output_module_path_template="gpt_neox.layers.{}.mlp.output",
    model_dtype=None,
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
    target_total_tokens=1_000_000,  # Keep it small for tutorial
    # Storage Parameters
    activation_dir=activation_dir,
    output_format="hdf5",
    compression="gzip",
    chunk_token_threshold=8_000,
    activation_dtype="float32",
    # Normalization
    compute_norm_stats=True,
    # NNsight args
    nnsight_tracer_kwargs={},
    nnsight_invoker_args={},
)
print("Activation Generation Configuration:")
print(activation_config)

# --- Training Configuration ---
expected_activation_path = os.path.join(
    activation_config.activation_dir,
    activation_config.model_name,
    f"{os.path.basename(activation_config.dataset_path)}_{activation_config.dataset_split}",
)

# --- Determine WandB Run Name (using config values) ---
_lr = 1e-4
_batch_size = 1024
_k_frac = clt_config.batchtopk_frac  # Use frac for name

wdb_run_name = (
    f"{clt_config.num_features}-width-"
    f"batchtopk-kfrac{_k_frac:.3f}-"  # Indicate BatchTopK and frac
    f"{_batch_size}-batch-"
    f"{_lr:.1e}-lr"
    # Sparsity lambda/c less relevant when apply_sparsity_penalty_to_batchtopk=False
)
print("\nGenerated WandB run name: " + wdb_run_name)

training_config = TrainingConfig(
    # Training loop parameters
    learning_rate=_lr,
    training_steps=1000,  # Reduced steps for tutorial
    seed=42,
    # Activation source
    activation_source="local_manifest",
    activation_path=expected_activation_path,
    activation_dtype="float32",
    # Training batch size
    train_batch_size_tokens=_batch_size,
    sampling_strategy="sequential",
    # Normalization
    normalization_method="none",  # Use pre-calculated stats
    # Loss function coefficients
    sparsity_lambda=0.0,  # Disable standard sparsity penalty
    sparsity_lambda_schedule="linear",
    sparsity_c=0.0,  # Disable standard sparsity penalty
    preactivation_coef=0,  # Disable preactivation loss (AuxK handles dead latents)
    aux_loss_factor=1 / 32,  # Enable AuxK loss with typical factor from paper
    apply_sparsity_penalty_to_batchtopk=False,  # Ensure standard sparsity penalty is off for BatchTopK
    # Optimizer & Scheduler
    optimizer="adamw",
    lr_scheduler="linear_final20",
    optimizer_beta2=0.98,
    # Logging & Checkpointing
    log_interval=10,
    eval_interval=50,
    checkpoint_interval=500,
    dead_feature_window=1000,
    # WandB (Optional)
    enable_wandb=True,
    wandb_project="clt-hp-sweeps-pythia-70m",
    wandb_run_name=wdb_run_name,
)
print("\nTraining Configuration (BatchTopK):")
print(training_config)


# %% [markdown]
# ## 3. Generate Activations (One-Time Step)
#
# Generate the activation dataset, including the manifest file (`index.bin`). This step is the same
# as in the previous tutorial, just saving to a different directory (`activation_dir`).

# %%
print("Step 1: Generating/Verifying Activations (including manifest)...")

metadata_path = os.path.join(expected_activation_path, "metadata.json")
manifest_path = os.path.join(expected_activation_path, "index.bin")

if os.path.exists(metadata_path) and os.path.exists(manifest_path):
    print(f"Activations and manifest already found at: {expected_activation_path}")
    print("Skipping generation. Delete the directory to regenerate.")
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
# ## 4. Training the CLT with BatchTopK Activation
#
# Instantiate the `CLTTrainer` using the configurations defined above.
# The trainer will use the `LocalActivationStore` (driven by the manifest) and the CLT model
# will use the BatchTopK activation function internally based on `clt_config`.

# %%
print("Initializing CLTTrainer for training with BatchTopK...")

log_dir = f"clt_training_logs/clt_pythia_batchtopk_train_{int(time.time())}"
os.makedirs(log_dir, exist_ok=True)
print(f"Logs and checkpoints will be saved to: {log_dir}")

try:
    print("Creating CLTTrainer instance...")
    print(f"- Using device: {device}")
    print(f"- CLT config (BatchTopK): {vars(clt_config)}")
    print(f"- Activation Source: {training_config.activation_source}")
    print(f"- Reading activations from: {training_config.activation_path}")

    trainer = CLTTrainer(
        clt_config=clt_config,
        training_config=training_config,
        log_dir=log_dir,
        device=device,
        distributed=False,
    )
    print("CLTTrainer instance created successfully.")
except Exception as e:
    print(f"[ERROR] Failed to initialize CLTTrainer: {e}")
    traceback.print_exc()
    raise

# Start training
print("Beginning training using BatchTopK activation...")
print(f"Training for {training_config.training_steps} steps.")
print(f"Normalization method set to: {training_config.normalization_method}")
print(
    f"Standard sparsity penalty applied to BatchTopK activations: {training_config.apply_sparsity_penalty_to_batchtopk}"
)

try:
    start_train_time = time.time()
    trained_clt_model = trainer.train(eval_every=training_config.eval_interval)
    end_train_time = time.time()
    print(f"Training finished in {end_train_time - start_train_time:.2f} seconds.")
except Exception as train_err:
    print(f"[ERROR] Training failed: {train_err}")
    traceback.print_exc()
    raise

# %% [markdown]
# ## 5. Saving and Loading the Trained Model
#
# The saving and loading process remains the same. Ensure you use the correct `CLTConfig` (the one with `activation_fn="batchtopk"`) when loading the model.

# %%
# Trainer saves automatically, but we can save manually too
final_model_path = os.path.join(log_dir, "clt_batchtopk_final_manual.pt")
print("\nManually saving final BatchTopK model to: " + final_model_path)

print(f"\nContents of log directory ({log_dir}):")
print(os.listdir(log_dir))

# --- Loading the model ---
print("\nLoading the trained BatchTopK model from: " + final_model_path)

# 1. Re-create the config used for training
loaded_clt_config = CLTConfig(
    num_features=clt_config.num_features,
    num_layers=clt_config.num_layers,
    d_model=clt_config.d_model,
    activation_fn="batchtopk",  # Important: Match activation function
    batchtopk_frac=clt_config.batchtopk_frac,  # Include BatchTopK specific params
    batchtopk_k=clt_config.batchtopk_k,
    batchtopk_straight_through=clt_config.batchtopk_straight_through,
)

# 2. Load the model structure and state dict
loaded_clt_model = CrossLayerTranscoder(
    config=loaded_clt_config,
    process_group=None,
    device=torch.device(device),
)
loaded_clt_model.load(final_model_path)

print("BatchTopK model loaded successfully.")
print(f"Loaded model is on device: {next(loaded_clt_model.parameters()).device}")

# %% [markdown]
# ## 6. Next Steps
#
# This tutorial showed how to train a CLT using the global BatchTopK activation. The main difference lies in setting the `activation_fn` and related `batchtopk_*` parameters in `CLTConfig`, and potentially disabling the standard sparsity loss via `TrainingConfig`.
#
# You can now analyze the features learned by this BatchTopK-based CLT.

# %%
print("\nBatchTopK Tutorial Complete!")
print(f"The trained BatchTopK CLT model and logs are saved in: {log_dir}")

# %%
