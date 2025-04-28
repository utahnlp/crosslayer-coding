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
import subprocess
import requests
from urllib.parse import urljoin

import logging

# --- Set general INFO level for our application ---
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)  # Set to DEBUG for more verbose output

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
    handler.setLevel(logging.DEBUG)  # Handler should also show DEBUG
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
else:
    # Ensure existing handlers are at least INFO level
    for handler in root_logger.handlers:
        if handler.level > logging.DEBUG:
            handler.setLevel(logging.DEBUG)

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
    from clt.config import CLTConfig, TrainingConfig, ActivationConfig
    from clt.activation_generation.generator import ActivationGenerator
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
SERVER_HOST = "34.41.125.189"
SERVER_PORT = 8000
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
gpt2_num_layers = 6
gpt2_d_model = 512
expansion_factor = 4  # Use smaller expansion factor for speed

clt_num_features = gpt2_d_model * expansion_factor
clt_config = CLTConfig(
    num_features=clt_num_features,
    num_layers=gpt2_num_layers,
    d_model=gpt2_d_model,
    activation_fn="relu",
)
print("CLT Configuration:")
print(clt_config)

# --- Activation Generation Configuration (Remote) ---
activation_dir = "./tutorial_activations"  # Still needed for potential local fallback/temp files
dataset_name = "monology/pile-uncopyrighted"  # "NeelNanda/pile-10k" is smaller if needed
activation_config = ActivationConfig(
    # Model Source
    model_name=BASE_MODEL_NAME,
    mlp_input_module_path_template="gpt_neox.layers.{}.mlp.input",
    mlp_output_module_path_template="gpt_neox.layers.{}.mlp.output",
    # Dataset Source
    dataset_path=dataset_name,
    dataset_split="train",
    dataset_text_column="text",
    # Generation Parameters
    context_size=128,
    inference_batch_size=192,  # Smaller batch size if needed
    activation_dtype="float32",
    exclude_special_tokens=True,
    prepend_bos=True,
    # Dataset Handling
    streaming=True,
    dataset_trust_remote_code=False,
    cache_path=None,
    # Generation Output Control
    target_total_tokens=1_000_000,  # Very small token count for tutorial speed
    # Storage Parameters
    activation_dir=activation_dir,
    output_format="hdf5",  # MUST be hdf5 for current remote implementation
    compression="gzip",
    chunk_token_threshold=10_000,  # Small chunk size (reduced from 10k)
    # >> Key change: Point to the server <<
    remote_server_url=SERVER_URL,
    # Normalization
    compute_norm_stats=True,
    model_dtype=None,
)
print("\nActivation Generation Configuration (Remote):")
print(activation_config)

# --- Training Configuration (Remote) ---
# Construct the dataset_id used by the server/generator
dataset_id = f"{activation_config.model_name}/{os.path.basename(activation_config.dataset_path)}_{activation_config.dataset_split}"

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


# %% [markdown]
# ## 5. Generate Activations (Send to Server)
#
# Now we run the `ActivationGenerator`. Because `ActivationConfig.remote_server_url` is set
# and we call `set_storage_type('remote')`, the generator will:
# 1. Generate activations as usual.
# 2. Save each chunk temporarily as an HDF5 file.
# 3. Send the HDF5 file bytes to the server's `/chunks` endpoint.
# 4. Send the final `metadata.json` and `norm_stats.json` to the server.

# %%
# --- Cell 2: Generate Activations --- #
print("\nStep 2: Generating Activations and Sending to Server...")

try:
    generator = ActivationGenerator(
        cfg=activation_config,
        device=device,
    )
    # Explicitly set storage type to remote
    generator.set_storage_type("remote")

    generation_start_time = time.time()
    generator.generate_and_save()
    generation_end_time = time.time()
    print(f"Activation generation and sending complete in {generation_end_time - generation_start_time:.2f}s.")
except Exception as gen_err:
    print(f"[ERROR] Remote activation generation failed: {gen_err}")
    traceback.print_exc()
    raise
