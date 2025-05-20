# %% [markdown]
# # Tutorial 2: End-to-End CLT Training with TopK Activation
#
# This tutorial demonstrates training a Cross-Layer Transcoder (CLT)
# using the **TopK** activation function. We will:
# 1. Configure the CLT model for TopK, activation generation, and training parameters.
# 2. Generate activations locally (with manifest) using the ActivationGenerator.
# 3. Configure the trainer to use the locally stored activations via the manifest.
# 4. Train the CLT model using TopK activation.
# 5. Save and load the final trained model (which will be a TopK model).

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
import json
from torch.distributions.normal import Normal  # For post-hoc sweep
import logging  # Import logging

# Configure logging to show INFO level messages for the notebook
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s")

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

# For post-hoc sweep N(0,1) assumption
std_normal = Normal(0, 1)

# %% [markdown]
# ## 2. Configuration
#
# We configure the CLT, Activation Generation, and Training.
# Key change: `CLTConfig.activation_fn` is set to `"topk"`.

# %%
# --- CLT Architecture Configuration ---
num_layers = 6
d_model = 512
expansion_factor = 32
clt_num_features = d_model * expansion_factor

topk_k_val = 200  # Can be a float for fraction or int for count

clt_config = CLTConfig(
    num_features=clt_num_features,
    num_layers=num_layers,
    d_model=d_model,
    activation_fn="topk",  # Use TopK activation
    topk_k=topk_k_val,  # Specify k for TopK
    topk_straight_through=True,  # Use STE for gradients
    # jumprelu_threshold is not used for topk
)
print("CLT Configuration (TopK):")
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
_k_val_for_name = clt_config.topk_k  # Use topk_k for name

wdb_run_name = (
    f"{clt_config.num_features}-width-"
    f"topk-k{_k_val_for_name}-"  # Indicate TopK and k
    f"{_batch_size}-batch-"
    f"{_lr:.1e}-lr"
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
    normalization_method="auto",  # Use pre-calculated stats
    # Loss function coefficients
    sparsity_lambda=0.0,  # Standard sparsity penalty (adjust as needed for TopK)
    sparsity_lambda_schedule="linear",
    sparsity_c=0.0,  # Standard sparsity penalty (adjust as needed for TopK)
    preactivation_coef=0,
    aux_loss_factor=1 / 32,  # Enable AuxK loss with typical factor from paper
    # Optimizer & Scheduler
    optimizer="adamw",
    lr_scheduler="linear_final20",
    optimizer_beta2=0.98,
    # Logging & Checkpointing
    log_interval=10,
    eval_interval=50,
    diag_every_n_eval_steps=1,  # run diagnostics every eval
    max_features_for_diag_hist=1000,  # optional cap per layer
    checkpoint_interval=500,
    dead_feature_window=200,
    # WandB (Optional)
    enable_wandb=True,
    wandb_project="clt-hp-sweeps-pythia-70m",
    wandb_run_name=wdb_run_name,
)
print("\nTraining Configuration (TopK):")
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
# ## 4. Training the CLT with TopK Activation
#
# Instantiate the `CLTTrainer` using the configurations defined above.
# The trainer will use the `LocalActivationStore` (driven by the manifest) and the CLT model
# will use the TopK activation function internally based on `clt_config`.

# %%
print("Initializing CLTTrainer for training with TopK...")

log_dir = f"clt_training_logs/clt_pythia_topk_train_{int(time.time())}"
os.makedirs(log_dir, exist_ok=True)
print(f"Logs and checkpoints will be saved to: {log_dir}")

try:
    print("Creating CLTTrainer instance...")
    print(f"- Using device: {device}")
    print(f"- CLT config (TopK): {vars(clt_config)}")
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
print("Beginning training using TopK activation...")
print(f"Training for {training_config.training_steps} steps.")
print(f"Normalization method set to: {training_config.normalization_method}")

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
# ## 5. Saving and Loading the Final Trained Model
#
# The `CLTTrainer` automatically saves the final model and its configuration (cfg.json)
# in the `log_dir/final/` directory. Unlike BatchTopK, TopK models are typically
# not converted to JumpReLU by the trainer at the end of training.
# Here, we'll also demonstrate a manual save of the model state and its config as Python dict,
# and then load it back. This manually saved model will be the one returned by trainer.train(),
# so it will be a TopK model.

# %%
# The trained_clt_model is what trainer.train() returned.
# For TopK, this model remains a TopK model.
final_model_state_path = os.path.join(log_dir, "clt_final_manual_state.pt")
final_model_config_path = os.path.join(log_dir, "clt_final_manual_config.json")

print(f"\nManually saving final model state to: {final_model_state_path}")
print(f"Manually saving final model config to: {final_model_config_path}")

torch.save(trained_clt_model.state_dict(), final_model_state_path)
with open(final_model_config_path, "w") as f:
    # The config on trained_clt_model will reflect 'topk'
    json.dump(trained_clt_model.config.__dict__, f, indent=4)

print(f"\nContents of log directory ({log_dir}):")
for item in os.listdir(log_dir):
    print(f"- {item}")

# --- Loading the manually saved model ---
print("\nLoading the manually saved model...")

# 1. Load the saved configuration
with open(final_model_config_path, "r") as f:
    loaded_config_dict_manual = json.load(f)
loaded_clt_config_manual = CLTConfig(**loaded_config_dict_manual)

print(f"Loaded manual config, activation_fn: {loaded_clt_config_manual.activation_fn}")

# 2. Instantiate model with this loaded config and load state dict
loaded_clt_model_manual = CrossLayerTranscoder(
    config=loaded_clt_config_manual,
    process_group=None,  # Assuming non-distributed for this load
    device=torch.device(device),
)
loaded_clt_model_manual.load_state_dict(torch.load(final_model_state_path, map_location=device))
loaded_clt_model_manual.eval()  # Set to evaluation mode

print("Manually saved model loaded successfully.")
print(f"Loaded model is on device: {next(loaded_clt_model_manual.parameters()).device}")


# %% [markdown]
# ## 6. Next Steps
#
# This tutorial showed how to train a CLT using the token-wise TopK activation
# and save/load the trained TopK model.

# %%
print("\nTopK Tutorial Complete!")
print(f"Logs and checkpoints will be saved to: {log_dir}")

# %%
# The following cells for loading a specific BatchTopK checkpoint are removed
# as they are not relevant to this TopK tutorial and the section was cut.
# weights = torch.load(batchtopk_checkpoint_path)

# %%
# weights.keys()
# %%
import torch

checkpoint_path = (
    "clt_training_logs/clt_pythia_topk_train_1747350938/trainer_state_latest.pt"  # Or the specific step's file
)
# checkpoint_path = "tutorials/clt_training_logs/clt_pythia_topk_train_1747350938/step_X/trainer_state.pt" # For distributed
try:
    data = torch.load(checkpoint_path, map_location="cpu")
    print(data.keys())
    if "wandb_run_id" in data:
        print(f"WandB Run ID found: {data['wandb_run_id']}")
    else:
        print("WandB Run ID NOT found in checkpoint.")
except FileNotFoundError:
    print(f"Checkpoint file not found: {checkpoint_path}")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
# %%
