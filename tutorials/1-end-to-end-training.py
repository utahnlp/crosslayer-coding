# %% [markdown]
# # Tutorial 1: End-to-End CLT Training on GPT-2 Small
#
# This tutorial demonstrates the complete process of training a Cross-Layer Transcoder (CLT)
# using the `clt` library. We will:
# 1. Configure the CLT model and training parameters.
# 2. Train a CLT model using the streaming data processing pipeline with HuggingFace datasets.
# 3. Save the trained CLT model.
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

# Import components from the clt library
# (Ensure the 'clt' directory is in your Python path or installed)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from clt.config import CLTConfig, TrainingConfig
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
BASE_MODEL_NAME = "gpt2"  # Using GPT-2 small

# %% [markdown]
# ## 2. Configuration
#
# We define two configuration objects:
# - `CLTConfig`: Specifies the architecture of the Cross-Layer Transcoder itself (number of features, layers matching the base model, activation function, etc.).
# - `TrainingConfig`: Specifies parameters for the training process (learning rate, batch size, number of steps, loss coefficients, etc.).
#
# **Note**: For this tutorial, we use a small number of features (`num_features`) and training steps (`training_steps`) for faster execution. For a production-quality CLT, these values would be significantly larger.
#
# **Important**: The total number of tokens processed per batch is calculated as `batch_size * context_size`. We use a smaller `batch_size` of 6 and `context_size` of 128 for a total of 768 tokens per batch, optimized for MPS memory usage. The `num_features` (set to 1536) defines the dimension of the hidden layer in the CLT and is unrelated to batch construction.

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
    activation_fn="jumprelu",  # As described in the paper
    jumprelu_threshold=0.03,  # Default value from paper
)
print("CLT Configuration:")
print(clt_config)

# --- Training Configuration ---
# Using smaller values for a quick tutorial run
training_config = TrainingConfig(
    # Model parameters
    model_name=BASE_MODEL_NAME,  # Model to extract activations from
    # model_dtype="bfloat16",
    # Dataset parameters
    dataset_path="monology/pile-uncopyrighted",  # Common dataset with lots of text
    dataset_split="train",
    dataset_text_column="text",
    streaming=True,
    dataset_trust_remote_code=False,  # Standard dataset, no special code needed
    cache_path=None,  # Path to cache downloaded data (optional)
    # Activation extraction parameters
    context_size=128,
    store_batch_size_prompts=10,  # Extraction batch size
    batch_size=6,  # Training batch size
    exclude_special_tokens=True,  # currently not doing anything
    prepend_bos=True,
    # Buffer and normalization parameters
    n_batches_in_buffer=4,  # Reduced buffer size to conserve memory
    # Using 'none' normalization to avoid extraction issues in tutorial
    # In production, 'estimated_mean_std' would be preferred
    normalization_method="estimated_mean_std",
    normalization_estimation_batches=10,  # Number of batches used for statistics estimation
    # Training parameters
    learning_rate=3e-5,
    training_steps=1000,
    sparsity_lambda=5,  # Default unknown
    sparsity_c=1.0,  # Default value from paper: 1.0
    preactivation_coef=3e-6,  # Default value: 3e-6
    optimizer="adamw",
    lr_scheduler="linear",
    eval_interval=10,
    log_interval=10,
    dead_feature_window=500,
    enable_wandb=True,
    wandb_project="clt-tutorial",
)
print("\nTraining Configuration:")
print(training_config)

# %% [markdown]
# ## 3. Training the CLT with Streaming Data
#
# Now we'll train the CLT using the streaming data pipeline. The `CLTTrainer` will:
#
# 1. Create an `ActivationExtractorCLT` to extract activations from the dataset
# 2. Initialize an `ActivationStore` to manage the streaming activations
# 3. Automatically estimate normalization statistics (mean and std) from the initial batches
# 4. Train the model using the streamed activations, with periodic checkpointing
#
# Using streaming allows us to train on datasets much larger than would fit in memory.

# %%
print("\nInitializing CLTTrainer...")

# Define a directory for logs and checkpoints
log_dir = f"clt_training_logs/clt_gpt2_tutorial_{int(time.time())}"
os.makedirs(log_dir, exist_ok=True)
print(f"Logs and checkpoints will be saved to: {log_dir}")

# Instantiate the trainer
try:
    print("\nStep 1: Creating CLTTrainer instance...")
    print(f"- Using device: {device}")
    print(f"- CLT config: {vars(clt_config)}")
    print(f"- Model name: {training_config.model_name}")
    print(f"- Dataset: {training_config.dataset_path}")

    trainer = CLTTrainer(
        clt_config=clt_config,
        training_config=training_config,
        log_dir=log_dir,
        device=device,
    )
    print("CLTTrainer instance created successfully.")
except Exception as e:
    import traceback

    print(f"\n[ERROR] Failed to initialize CLTTrainer: {e}")
    traceback.print_exc()

    print("\nDebug information:")
    print("1. Check if nnsight is properly installed:")
    try:
        import nnsight

        print(f"   - nnsight version: {nnsight.__version__}")
    except ImportError:
        print("   - nnsight is not installed or cannot be imported")

    print("2. Check if the training dataset is accessible:")
    try:
        from datasets import load_dataset

        dataset_info = load_dataset(
            training_config.dataset_path,
            split=training_config.dataset_split,
            streaming=True,
        )
        print(f"   - Dataset accessible: {training_config.dataset_path}")
    except Exception as ds_err:
        print(f"   - Dataset error: {ds_err}")

    print("3. Check model accessibility:")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(training_config.model_name)
        print(f"   - Model tokenizer accessible: {training_config.model_name}")
    except Exception as model_err:
        print(f"   - Model error: {model_err}")

    print("\nPlease fix the issues above before continuing.")
    raise

# Start training
print("\nStep 2: Setup phase...")
if training_config.normalization_method == "estimated_mean_std":
    print("Estimating normalization statistics from the dataset...")
    print("This process analyzes activation distributions to standardize inputs.")
    print(
        f"Using {training_config.normalization_estimation_batches} batches for estimation."
    )
else:
    print(f"Using normalization method: {training_config.normalization_method}")

print("\nStep 3: Beginning training...")
print(
    f"Training for {training_config.training_steps} steps with batch size {training_config.batch_size}."
)
print("The process will load the model and create the activation pipeline.")

try:
    start_train_time = time.time()
    trained_clt_model = trainer.train(eval_every=100)  # Evaluate every 100 steps
    end_train_time = time.time()
    print(f"\nTraining finished in {end_train_time - start_train_time:.2f} seconds.")
except Exception as train_err:
    print(f"\n[ERROR] Training failed: {train_err}")
    import traceback

    traceback.print_exc()
    raise

# %% [markdown]
# ## 4. Saving and Loading the Trained Model
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
# ## 5. Next Steps and Benefits of Streaming
#
# Using the new streaming implementation provides several benefits:
#
# - **Memory Efficiency**: Process datasets much larger than system memory
# - **Improved Statistics**: Normalization stats can be estimated from large datasets
# - **Checkpointing**: The system can resume training from checkpoints
# - **Scalability**: Easily train on diverse HuggingFace datasets
#
# With a trained `CrossLayerTranscoder`, you can now:
# - **Analyze Features**: Examine the learned encoder/decoder weights and feature activations.
# - **Build Replacement Models**: Integrate the CLT into the original base model to create a model that uses the sparse CLT features.
# - **Evaluate Performance**: Measure the trade-off between sparsity (L0 norm) and reconstruction accuracy (MSE loss).

# %%
print("\nTutorial Complete!")
print(
    "You have trained a Cross-Layer Transcoder using streaming data from HuggingFace."
)
print(f"The trained model and logs are saved in: {log_dir}")
