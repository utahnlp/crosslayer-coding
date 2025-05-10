import torch
import os
import json
import argparse
import sys
import logging
from typing import Dict

# Ensure the project root is in the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from clt.config import CLTConfig
    from clt.models.clt import CrossLayerTranscoder
    from clt.training.data import BaseActivationStore  # For type hinting
    from clt.training.local_activation_store import LocalActivationStore  # Default store for this script

    # Add other store types if needed, e.g., RemoteActivationStore
except ImportError as e:
    print(f"ImportError: {e}")
    print("Please ensure the 'clt' library is installed or the clt directory is in your PYTHONPATH.")
    raise

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main(args):
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load the BatchTopK CLTConfig
    if not os.path.exists(args.config_path):
        logger.error(f"BatchTopK config file not found at: {args.config_path}")
        return
    with open(args.config_path, "r") as f:
        config_dict_batchtopk = json.load(f)

    # Ensure the loaded config is indeed for BatchTopK for loading the checkpoint
    if config_dict_batchtopk.get("activation_fn") != "batchtopk":
        logger.warning(
            f"Warning: Config at {args.config_path} has activation_fn='{config_dict_batchtopk.get('activation_fn')}'. "
            f"Proceeding as if it's a BatchTopK config for loading the checkpoint, "
            f"but ensure this is intended and the checkpoint matches."
        )
        # Override to ensure BatchTopK for initial loading if needed, though ideally cfg.json matches.
        # config_dict_batchtopk['activation_fn'] = 'batchtopk'
        # It is better if the user provides the correct config file that was used to save the batchtopk checkpoint.

    try:
        clt_config_batchtopk = CLTConfig(**config_dict_batchtopk)
        logger.info(f"Loaded BatchTopK CLTConfig: {clt_config_batchtopk}")
    except Exception as e:
        logger.error(f"Error creating CLTConfig from {args.config_path}: {e}")
        return

    # 2. Instantiate BatchTopK model and load state_dict
    logger.info("Instantiating CLT model with BatchTopK configuration...")
    model = CrossLayerTranscoder(
        config=clt_config_batchtopk, process_group=None, device=device  # Assuming non-distributed for conversion script
    )

    if not os.path.exists(args.batchtopk_checkpoint_path):
        logger.error(f"BatchTopK checkpoint file not found at: {args.batchtopk_checkpoint_path}")
        return

    logger.info(f"Loading BatchTopK model state from: {args.batchtopk_checkpoint_path}")
    try:
        # Load checkpoint, handling potential sharded format if it\'s a directory
        if os.path.isdir(args.batchtopk_checkpoint_path):
            from torch.distributed.checkpoint import load_state_dict as dist_load_state_dict
            from torch.distributed.checkpoint.filesystem import FileSystemReader

            logger.info(f"Checkpoint path is a directory, attempting distributed load into a single model.")
            state_dict_to_populate = model.state_dict()  # Get state_dict from the instantiated model
            dist_load_state_dict(
                state_dict=state_dict_to_populate,
                storage_reader=FileSystemReader(args.batchtopk_checkpoint_path),
                no_dist=True,  # Load sharded into a non-distributed model structure
            )
            model.load_state_dict(state_dict_to_populate)

        else:  # Standard single-file checkpoint
            model.load_state_dict(torch.load(args.batchtopk_checkpoint_path, map_location=device))

        model.eval()
        logger.info("BatchTopK model loaded and set to eval mode.")
    except Exception as e:
        logger.error(f"Error loading BatchTopK model state: {e}")
        return

    # 3. Initialize ActivationStore
    logger.info(f"Initializing LocalActivationStore from: {args.activation_data_path}")
    if not os.path.exists(args.activation_data_path):
        logger.error(f"Activation data path not found: {args.activation_data_path}")
        return

    try:
        # For simplicity, this script defaults to LocalActivationStore.
        # Users can modify this to use other store types if needed.
        activation_store = LocalActivationStore(
            dataset_path=args.activation_data_path,
            train_batch_size_tokens=args.estimation_batch_size_tokens,  # Use a batch size for estimation
            device=device,
            dtype=args.activation_dtype or clt_config_batchtopk.expected_input_dtype or "float32",
            rank=0,
            world=1,
            seed=args.seed,
            sampling_strategy="sequential",  # Sequential is fine for estimation
            normalization_method="auto",  # Or "none" if data is not normalized
        )
        logger.info("Activation store initialized.")
    except Exception as e:
        logger.error(f"Error initializing activation store: {e}")
        return

    # 4. Estimate Theta and Convert Model
    logger.info(
        f"Starting theta estimation using {args.num_batches_for_theta_estimation} batches "
        f"with scale_factor={args.scale_factor} and default_theta_value={args.default_theta_value}."
    )
    try:
        data_iterator = iter(activation_store)
        estimated_thetas = model.estimate_theta_posthoc(
            data_iter=data_iterator,
            num_batches=args.num_batches_for_theta_estimation,
            default_theta_value=args.default_theta_value,
            scale_factor=args.scale_factor,
            device=device,  # Pass device to ensure buffers are on correct device
        )
        logger.info(
            f"Theta estimation and conversion to JumpReLU complete. Estimated theta tensor shape: {estimated_thetas.shape}"
        )
        logger.info(f"Model config after conversion: {model.config}")
    except Exception as e:
        logger.error(f"Error during estimate_theta_posthoc: {e}")
        if hasattr(e, "__traceback__"):
            import traceback

            traceback.print_tb(e.__traceback__)
        return
    finally:
        if hasattr(activation_store, "close") and callable(getattr(activation_store, "close")):
            activation_store.close()

    # 5. Save the Converted JumpReLU Model and its Config
    logger.info(f"Saving converted JumpReLU model state to: {args.output_model_path}")
    os.makedirs(os.path.dirname(args.output_model_path), exist_ok=True)
    torch.save(model.state_dict(), args.output_model_path)

    logger.info(f"Saving converted JumpReLU model config to: {args.output_config_path}")
    os.makedirs(os.path.dirname(args.output_config_path), exist_ok=True)
    with open(args.output_config_path, "w") as f:
        # model.config is updated inplace by convert_to_jumprelu_inplace (called by estimate_theta_posthoc)
        json.dump(model.config.__dict__, f, indent=4)

    # 6. Perform a quick L0 check on the converted model
    logger.info("Performing L0 check on the converted JumpReLU model...")
    # Re-initialize data_iterator for the L0 check if it was exhausted or if you want a fresh sample
    # For this script, we will try to get one batch. If store is exhausted, dummy_l0 might be based on zeros.
    sample_batch_for_l0_inputs: Dict[int, torch.Tensor] = {}
    try:
        # Need to re-open the store or ensure it can be iterated again if it was fully consumed.
        # For simplicity, re-create the iterator. If the store was exhausted, this might still yield nothing.
        # A more robust solution might involve a resettable store or loading a specific small dataset for this check.
        activation_store_for_l0_check = LocalActivationStore(
            dataset_path=args.activation_data_path,
            train_batch_size_tokens=args.estimation_batch_size_tokens,
            device=device,
            dtype=args.activation_dtype or clt_config_batchtopk.expected_input_dtype or "float32",
            rank=0,
            world=1,
            seed=args.seed + 1,  # Use a different seed or ensure sequential to get different data if possible
            sampling_strategy="sequential",
            normalization_method="auto",
        )
        data_iterator_for_l0_check = iter(activation_store_for_l0_check)
        sample_inputs_l0, _ = next(data_iterator_for_l0_check)
        sample_batch_for_l0_inputs = sample_inputs_l0
        if hasattr(activation_store_for_l0_check, "close") and callable(
            getattr(activation_store_for_l0_check, "close")
        ):
            activation_store_for_l0_check.close()
    except StopIteration:
        logger.warning("Activation store exhausted. L0 check will use zero input.")
    except Exception as e_l0_fetch:
        logger.warning(f"Error fetching batch for L0 check: {e_l0_fetch}. L0 check will use zero input.")

    model_for_l0_check = model  # Always use the converted model, no more overriding k for L0 check here

    avg_empirical_l0, expected_l0 = run_quick_l0_checks_script(
        model_for_l0_check, sample_batch_for_l0_inputs, args.num_tokens_for_l0_check_script
    )
    logger.info(
        f"  Average Empirical L0 (Layer 0, {args.num_tokens_for_l0_check_script} random tokens): {avg_empirical_l0:.2f}"
    )
    logger.info(f"  Expected L0 (N(0,1) assumption, all layers): {expected_l0:.2f}")

    logger.info("Conversion script finished successfully.")


def run_quick_l0_checks_script(
    model: CrossLayerTranscoder, sample_batch_inputs: Dict[int, torch.Tensor], num_tokens_to_check: int
) -> tuple[float, float]:
    """Helper function for L0 checks within the script."""
    model.eval()  # Ensure model is in eval mode
    avg_empirical_l0_layer0 = float("nan")
    std_normal_dist = torch.distributions.normal.Normal(0, 1)  # Create Normal distribution object

    if not sample_batch_inputs or 0 not in sample_batch_inputs or sample_batch_inputs[0].numel() == 0:
        logger.warning(
            "run_quick_l0_checks_script received empty or invalid sample_batch_inputs for layer 0. Empirical L0 will be NaN."
        )
        # Keep avg_empirical_l0_layer0 as NaN
    else:
        layer0_inputs_all_tokens = sample_batch_inputs[0].to(device=model.device, dtype=model.dtype)

        if layer0_inputs_all_tokens.dim() == 3:  # B, S, D
            num_tokens_in_batch = layer0_inputs_all_tokens.shape[0] * layer0_inputs_all_tokens.shape[1]
            layer0_inputs_flat = layer0_inputs_all_tokens.reshape(num_tokens_in_batch, model.config.d_model)
        elif layer0_inputs_all_tokens.dim() == 2:  # Already [num_tokens, d_model]
            num_tokens_in_batch = layer0_inputs_all_tokens.shape[0]
            layer0_inputs_flat = layer0_inputs_all_tokens
        else:
            logger.warning(
                f"run_quick_l0_checks_script received unexpected input shape {layer0_inputs_all_tokens.shape} for layer 0. Empirical L0 will be NaN."
            )
            layer0_inputs_flat = None

        if layer0_inputs_flat is not None and num_tokens_in_batch > 0:
            num_to_sample = min(num_tokens_to_check, num_tokens_in_batch)
            indices = torch.randperm(num_tokens_in_batch, device=model.device)[:num_to_sample]
            selected_tokens_for_l0 = layer0_inputs_flat[indices]

            if selected_tokens_for_l0.numel() > 0:
                acts_layer0_selected = model.encode(selected_tokens_for_l0, layer_idx=0)
                l0_per_token_selected = (acts_layer0_selected > 1e-6).sum(dim=1).float()
                avg_empirical_l0_layer0 = l0_per_token_selected.mean().item()
            else:
                logger.warning("No tokens selected for empirical L0 check after sampling. Empirical L0 will be NaN.")
        elif layer0_inputs_flat is None:
            pass
        else:
            logger.warning("Batch for L0 check contains no tokens for layer 0. Empirical L0 will be NaN.")

    expected_l0 = float("nan")
    if hasattr(model, "log_threshold") and model.log_threshold is not None:
        theta = model.log_threshold.exp().cpu()
        p_fire = 1.0 - std_normal_dist.cdf(theta.float())
        expected_l0 = p_fire.sum().item()
    else:
        logger.warning("Model does not have log_threshold. Cannot compute expected_l0.")

    return avg_empirical_l0_layer0, expected_l0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a BatchTopK CLT model to JumpReLU with post-hoc theta estimation."
    )

    parser.add_argument(
        "--batchtopk_checkpoint_path",
        type=str,
        required=True,
        help="Path to the saved BatchTopK model checkpoint (.pt file or sharded directory).",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the JSON config file corresponding to the BatchTopK model.",
    )
    parser.add_argument(
        "--activation_data_path",
        type=str,
        required=True,
        help="Path to the activation data manifest directory (for LocalActivationStore).",
    )
    parser.add_argument(
        "--output_model_path",
        type=str,
        required=True,
        help="Path to save the converted JumpReLU model's state_dict (.pt).",
    )
    parser.add_argument(
        "--output_config_path",
        type=str,
        required=True,
        help="Path to save the converted JumpReLU model's config (.json).",
    )

    parser.add_argument(
        "--num_batches_for_theta_estimation",
        type=int,
        default=100,
        help="Number of batches to use for theta estimation.",
    )
    parser.add_argument(
        "--estimation_batch_size_tokens",
        type=int,
        default=1024,
        help="Number of tokens per batch for theta estimation.",
    )
    parser.add_argument(
        "--scale_factor", type=float, default=1.0, help="Scaling factor to apply to estimated theta values."
    )
    parser.add_argument(
        "--default_theta_value", type=float, default=1e6, help="Default theta value for features that never activated."
    )

    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (e.g., 'cpu', 'cuda', 'cuda:0'). Auto-detects if None."
    )
    parser.add_argument(
        "--activation_dtype",
        type=str,
        default="float32",
        help="Data type for loading activations (e.g., 'float16', 'bfloat16', 'float32').",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for activation store sampling (if applicable)."
    )
    parser.add_argument(
        "--num_tokens_for_l0_check_script",
        type=int,
        default=100,
        help="Number of random tokens to sample from a batch for the empirical L0 check in this script.",
    )
    # Note: clt_dtype for the model will be taken from the loaded config_dict_batchtopk initially.

    args = parser.parse_args()
    main(args)

# Example Usage:
# python scripts/convert_batchtopk_to_jumprelu.py \\
#   --batchtopk_checkpoint_path clt_training_logs/clt_pythia_batchtopk_train_XYZ/final \\
#   --config_path clt_training_logs/clt_pythia_batchtopk_train_XYZ/cfg.json \\
#   --activation_data_path ./tutorial_activations_local_1M_pythia/EleutherAI/pythia-70m/monology/pile-uncopyrighted_train \\
#   --output_model_path clt_training_logs/clt_pythia_batchtopk_train_XYZ/final_jumprelu_sf1.5/clt_model_jumprelu.pt \\
#   --output_config_path clt_training_logs/clt_pythia_batchtopk_train_XYZ/final_jumprelu_sf1.5/cfg_jumprelu.json \\
#   --num_batches_for_theta_estimation 50 \\
#   --scale_factor 1.5 \\
#   --device cuda
