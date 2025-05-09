# %% [markdown]
# # Tutorial 2: Training with Remote Activation Storage
#
# This tutorial demonstrates training a CLT using activations stored and served
# by the `ActivationStorageServer`. We will:
#
# 1. Configure the CLT model, activation generation, and training parameters for remote storage.
# 2. Start the `ActivationStorageServer` locally as a background process.
# 3. Generate activations and send them to the running server.
# 4. Configure the trainer to use the `RemoteActivationStore` to fetch batches from the server.
# 5. Train the CLT model briefly.
# 6. Shut down the server process.
#
# **Prerequisites:**
# - Ensure dependencies for both the `clt` library and the `clt_server` are installed.
#   ```bash
#   pip install -r requirements.txt # In project root
#   pip install -r clt_server/requirements.txt # In project root
#   ```
# - PyTorch should be installed separately.

# %% [markdown]
# ## 1. Imports and Setup
#
# Import components and set up paths, device, and server URL.

# %%
import torch
import os
import time
import sys
import traceback
import requests
from urllib.parse import urljoin

import logging

# --- Set general INFO level for our application ---
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)  # Keep our code logging at INFO

# --- QUIET DOWN noisy libraries ---
# Set level for 'requests' library (often includes urllib3 logs)
logging.getLogger("requests").setLevel(logging.WARNING)
# Set level specifically for 'urllib3' connection pool logs
logging.getLogger("urllib3").setLevel(logging.WARNING)
# Set level for nnsight (can be noisy)
logging.getLogger("nnsight").setLevel(logging.WARNING)

# --- Ensure handlers are configured ---
if not root_logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)  # Our handler shows INFO and above
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
else:
    # Ensure existing handlers are at least INFO level
    for handler in root_logger.handlers:
        if handler.level > logging.INFO:
            handler.setLevel(logging.INFO)

# --- Optional: Print current levels ---
print(f"Root logger level: {logging.getLevelName(root_logger.level)}")
print(f"Requests logger level: {logging.getLevelName(logging.getLogger('requests').level)}")
print(f"Urllib3 logger level: {logging.getLevelName(logging.getLogger('urllib3').level)}")
print(f"Nnsight logger level: {logging.getLevelName(logging.getLogger('nnsight').level)}")

# --- Path Setup --- #
# Assume this script is run from the `tutorials` directory
tutorial_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(tutorial_dir)
server_dir = os.path.join(project_root, "clt_server")
server_main_script = os.path.join(server_dir, "main.py")

if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Imports from clt library --- #
try:
    from clt.config import CLTConfig, TrainingConfig
    from clt.training.trainer import CLTTrainer
except ImportError as e:
    print(f"ImportError: {e}")
    print("Please ensure the 'clt' library is installed or the project root is in your PYTHONPATH.")
    raise

# --- Device Setup --- #
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# --- Server Configuration --- #
SERVER_HOST = "34.41.125.189"  # Run on local server for now
SERVER_PORT = 8000  # Use a different port than default 8000 just in case
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
HEALTH_CHECK_URL = urljoin(SERVER_URL, "/api/v1/health")
# Ensure server uses a temporary directory for this tutorial
SERVER_STORAGE_DIR = os.path.join(project_root, "temp_tutorial_server_data")

# --- Base Model --- #
BASE_MODEL_NAME = "EleutherAI/pythia-70m"  # Using GPT-2 small

server_process = None  # Global variable to hold the server process

# %% [markdown]
# ## 2. Configuration for Remote Workflow
#
# We configure the components similarly to Tutorial 1, but with key changes for remote storage:
# - `ActivationConfig`: Must specify `remote_server_url`. `output_format` should be `hdf5`.
# - `TrainingConfig`: Set `activation_source='remote'` and provide `remote_config` pointing to the server.
#
# We use small values for features, steps, tokens, etc., for speed.

# %%
# --- CLT Architecture Configuration ---
# (Same as Tutorial 1 for simplicity)
num_layers = 6
d_model = 512
expansion_factor = 4  # Use smaller expansion factor for speed

clt_num_features = d_model * expansion_factor
clt_config = CLTConfig(
    num_features=clt_num_features,
    num_layers=num_layers,
    d_model=d_model,
    activation_fn="relu",
    normalization_method="auto",
)
print("CLT Configuration:")
print(clt_config)

# --- Activation Generation Configuration (Remote) ---
dataset_id = "EleutherAI/pythia-70m/pile-uncopyrighted_train"
training_config = TrainingConfig(
    # Training loop parameters
    learning_rate=1e-4,
    training_steps=1000,  # Very few steps for tutorial
    train_batch_size_tokens=1024,
    gradient_clip_val=1.0,
    # >> Key change: Activation source is remote <<
    activation_source="remote",
    # >> Key change: Provide remote config <<
    remote_config={
        "server_url": SERVER_URL,
        "dataset_id": dataset_id,  # Match generator output
        # Added timeout parameters for remote connections
        "timeout": 120,  # 2 minutes timeout for batch fetching
        "max_retries": 3,  # Number of retries for failed batch fetches
        "prefetch_batches": 4,  # Prefetch more batches to handle potential timeouts
    },
    # Normalization (Remote store handles fetching based on 'auto')
    normalization_method="auto",
    activation_dtype="float32",
    # Loss function coefficients
    sparsity_lambda=0.00001,
    sparsity_c=1.0,
    preactivation_coef=3e-6,
    # Optimizer & Scheduler
    optimizer="adamw",
    lr_scheduler="cosine",
    lr_scheduler_params={"eta_min": 1e-4},
    # Logging & Checkpointing
    log_interval=10,
    eval_interval=50,
    checkpoint_interval=100,
    dead_feature_window=200,
    # WandB (Optional)
    enable_wandb=True,
    wandb_project="clt-testing",
)
print("\nTraining Configuration (Remote):")
print(training_config)

# %% [markdown]
# ## 3. Preliminary Health Check (Optional)
#
# Before starting a new server, let's check if one is already running at the configured URL.

# %%
# --- Optional: Check if a server is already running ---
print(f"Checking for existing server at: {HEALTH_CHECK_URL}...")
try:
    response = requests.get(HEALTH_CHECK_URL, timeout=5)  # 5 second timeout
    if response.status_code == 200 and response.json().get("status") == "ok":
        print(f"✅ Server already running and healthy at {SERVER_URL}")
        print(
            "   -> If you intended to start a new local server, please stop the existing one or change the SERVER_PORT."
        )
    else:
        print(
            f"⚠️ Server responded at {SERVER_URL}, but health check failed or gave unexpected status: {response.status_code} - {response.text}"
        )
except requests.exceptions.ConnectionError:
    print(f"ℹ️ No server detected at {SERVER_URL} (Connection Error). This is expected if starting fresh.")
except requests.exceptions.Timeout:
    print(f"ℹ️ No server detected at {SERVER_URL} (Connection Timeout).")
except Exception as e:
    print(f"⚠️ An unexpected error occurred during health check: {e}")


# %%
# --- Cell 3: Train Model --- #
print("\nStep 3: Initializing CLTTrainer for training from remote server...")
log_dir = f"clt_training_logs/clt_gpt2_remote_train_{int(time.time())}"
os.makedirs(log_dir, exist_ok=True)
print(f"Logs and checkpoints will be saved to: {log_dir}")

trainer = None  # Initialize trainer to None
try:
    print("\nCreating CLTTrainer instance...")
    print(f"- Activation Source: {training_config.activation_source}")
    # Check if remote_config exists before accessing keys
    if training_config.remote_config:
        print(f"- Reading activations from server: {training_config.remote_config.get('server_url')}")
        print(f"- Dataset ID on server: {training_config.remote_config.get('dataset_id')}")
    else:
        print("- Error: remote_config is missing in TrainingConfig!")
        raise ValueError("remote_config must be set for remote training source.")

    trainer = CLTTrainer(
        clt_config=clt_config,
        training_config=training_config,
        log_dir=log_dir,
        device=device,
        distributed=False,
    )
    print("CLTTrainer instance created successfully.")

    print("\nBeginning training using RemoteActivationStore...")
    print(f"Training for {training_config.training_steps} steps.")
    start_train_time = time.time()
    trained_clt_model = trainer.train(eval_every=training_config.eval_interval)
    end_train_time = time.time()
    print(f"\nTraining finished in {end_train_time - start_train_time:.2f} seconds.")

except Exception as train_err:
    print(f"\n[ERROR] Remote training failed: {train_err}")
    traceback.print_exc()
    # Note: Trainer might not be fully initialized if error occurred early
    # Ensure the trainer's activation store is closed if it exists
    if trainer and hasattr(trainer, "activation_store") and trainer.activation_store:
        if hasattr(trainer.activation_store, "close") and callable(trainer.activation_store.close):
            print("Attempting to close activation store...")
            trainer.activation_store.close()
    # Important: Stop the server if training fails
    raise

# %%
