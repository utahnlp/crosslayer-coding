import torch
import os
import time
import sys
import traceback

# Import components from the clt library
os.environ["TOKENIZERS_PARALLELISM"] = "false"

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from clt.config import CLTConfig, TrainingConfig, ActivationConfig
    from clt.activation_generation.generator import ActivationGenerator
    from clt.training.trainer import CLTTrainer
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

# --- CLT Architecture Configuration ---
num_layers = 6
d_model = 512
expansion_factor = 32
clt_num_features = d_model * expansion_factor

batchtopk_sparsity_fraction = 0.005

clt_config = CLTConfig(
    num_features=clt_num_features,
    num_layers=num_layers,
    d_model=d_model,
    activation_fn="batchtopk",
    batchtopk_k=None,
    batchtopk_frac=batchtopk_sparsity_fraction,
    batchtopk_straight_through=True,
)
print("CLT Configuration (BatchTopK):")
print(clt_config)

# --- Activation Generation Configuration ---
activation_dir = "./activations_local_10M_pythia"
dataset_name = "monology/pile-uncopyrighted"
activation_config = ActivationConfig(
    model_name=BASE_MODEL_NAME,
    mlp_input_module_path_template="gpt_neox.layers.{}.mlp.input",
    mlp_output_module_path_template="gpt_neox.layers.{}.mlp.output",
    model_dtype=None,
    dataset_path=dataset_name,
    dataset_split="train",
    dataset_text_column="text",
    context_size=128,
    inference_batch_size=192,
    exclude_special_tokens=True,
    prepend_bos=True,
    streaming=True,
    dataset_trust_remote_code=False,
    cache_path=None,
    target_total_tokens=10_000_000,
    activation_dir=activation_dir,
    output_format="hdf5",
    compression="gzip",
    chunk_token_threshold=32_000,
    activation_dtype="float32",
    compute_norm_stats=True,
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

_lr = 1e-4
_batch_size = 1024
_k_frac = clt_config.batchtopk_frac

wdb_run_name = (
    f"{clt_config.num_features}-width-" f"batchtopk-kfrac{_k_frac:.3f}-" f"{_batch_size}-batch-" f"{_lr:.1e}-lr"
)
print("\nGenerated WandB run name: " + wdb_run_name)

training_config = TrainingConfig(
    learning_rate=_lr,
    training_steps=10000,
    seed=42,
    activation_source="local_manifest",
    activation_path=expected_activation_path,
    activation_dtype="float32",
    train_batch_size_tokens=_batch_size,
    sampling_strategy="sequential",
    normalization_method="none",
    sparsity_lambda=0.0,
    sparsity_lambda_schedule="linear",
    sparsity_c=0.0,
    preactivation_coef=0,
    aux_loss_factor=1 / 32,
    apply_sparsity_penalty_to_batchtopk=False,
    optimizer="adamw",
    lr_scheduler="linear_final20",
    optimizer_beta2=0.98,
    log_interval=10,
    eval_interval=50,
    checkpoint_interval=1000,
    dead_feature_window=1000,
    enable_wandb=True,
    wandb_project="clt-hp-sweeps-pythia-70m",
    wandb_run_name=wdb_run_name,
)
print("\nTraining Configuration (BatchTopK):")
print(training_config)

# --- Generate Activations (One-Time Step) ---
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

# --- Training the CLT with BatchTopK Activation ---
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

# --- Saving the Trained Model ---
final_model_path = os.path.join(log_dir, "clt_batchtopk_final.pt")  # Changed from _manual
# trainer.save_model(trained_clt_model, final_model_path) # Trainer saves automatically via checkpoints and at the end.
# If you want an explicit final save *after* training finishes and returns the model, you could do this:
# if trained_clt_model:
#     torch.save(trained_clt_model.state_dict(), final_model_path)
#     print(f"Manually saved final BatchTopK model to: {final_model_path}")
# else:
#     print("Training did not complete successfully, model not saved manually.")

print(f"\nContents of log directory ({log_dir}):")
try:
    print(os.listdir(log_dir))
except FileNotFoundError:
    print(f"Log directory {log_dir} not found. This might happen if training failed very early.")


print("\nBatchTopK Training Script Complete!")
print(f"The trained BatchTopK CLT model and logs are saved in: {log_dir}")
