import torch
import torch.optim as optim
from typing import Dict, Optional, Union, Any
from tqdm import tqdm  # type: ignore
import os
import json
import time
import sys
import logging  # Add logging import
import torch.distributed as dist  # Import torch.distributed
from torch.distributed import ProcessGroup  # Import ProcessGroup
from dataclasses import asdict  # Add dataclasses import

from clt.config import CLTConfig, TrainingConfig
from clt.models.clt import CrossLayerTranscoder
from clt.training.data.base_store import BaseActivationStore

# Import the new manifest-based stores
from clt.training.data.local_activation_store import LocalActivationStore
from clt.training.data.remote_activation_store import RemoteActivationStore

from clt.training.losses import LossManager

from .evaluator import CLTEvaluator  # Import the new evaluator
from .wandb_logger import WandBLogger, DummyWandBLogger
from .checkpointing import CheckpointManager
from .distributed_utils import average_shared_parameter_grads  # Add this import
from clt.training.data.activation_store_factory import create_activation_store  # Add this import
from .metric_utils import MetricLogger  # Add this import
from .diagnostics import compute_sparsity_diagnostics  # Add this import

# Get logger for this module
logger = logging.getLogger(__name__)


class CLTTrainer:
    """Trainer for Cross-Layer Transcoder models."""

    # Add type hint for the activation store attribute
    activation_store: BaseActivationStore
    # Model type hint
    model: CrossLayerTranscoder

    def __init__(
        self,
        clt_config: CLTConfig,
        training_config: TrainingConfig,
        log_dir: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        distributed: bool = False,  # Add distributed flag
    ):
        """Initialize the CLT trainer.

        Args:
            clt_config: Configuration for the CLT model
            training_config: Configuration for training
            log_dir: Directory to save logs and checkpoints
            device: Device to use for training (ignored if distributed)
            distributed: Whether to use distributed training
        """
        self.clt_config = clt_config
        self.training_config = training_config
        self.distributed = distributed

        # Initialize distributed training if enabled
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.process_group: Optional[ProcessGroup] = None  # For tensor parallelism

        if self.distributed:
            if not dist.is_initialized():
                # Default backend, consider NCCL for NVIDIA GPUs
                dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank))  # Get local rank if available

            # Set device based on local_rank when distributed
            if torch.cuda.is_available():
                self.device = torch.device(f"cuda:{self.local_rank}")
                torch.cuda.set_device(self.device)
            else:
                # Fallback for CPU distributed testing (not typical)
                self.device = torch.device("cpu")
                logger.warning("Distributed training requested but CUDA not available. Using CPU.")
            # Set the process group for tensor parallelism (using WORLD for now)
            self.process_group = dist.group.WORLD
        else:
            # Original device handling for non-distributed case
            _device_input = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            self.device = torch.device(_device_input) if isinstance(_device_input, str) else _device_input
            # Process group is None when not distributed
            self.process_group = None

        # Set up log directory - only rank 0 creates it
        self.log_dir = log_dir or f"clt_train_{int(time.time())}"
        if not self.distributed or self.rank == 0:
            os.makedirs(self.log_dir, exist_ok=True)

        # Record start time
        self.start_time = time.time()

        # Initialize model, passing device and process group for direct initialization
        # self.process_group is correctly set to None if not distributed
        self.model = CrossLayerTranscoder(clt_config, process_group=self.process_group, device=self.device)

        # Initialize optimizer - works on local parameters
        # Explicitly type the kwargs dict for clarity and linting
        optimizer_kwargs: Dict[str, Any] = {"lr": training_config.learning_rate}
        beta1 = training_config.optimizer_beta1  # Could be None
        beta2 = training_config.optimizer_beta2  # Could be None

        # Only add 'betas' if at least one is specified
        if beta1 is not None or beta2 is not None:
            # Get defaults if one is None
            # Default Adam/AdamW betas are (0.9, 0.999)
            final_beta1 = beta1 if beta1 is not None else 0.9
            final_beta2 = beta2 if beta2 is not None else 0.999
            optimizer_kwargs["betas"] = (final_beta1, final_beta2)
            logger.info(f"Rank {self.rank}: Using optimizer betas: ({final_beta1}, {final_beta2})")

        if training_config.optimizer == "adam":
            self.optimizer: Any = optim.Adam(self.model.parameters(), **optimizer_kwargs)
        else:  # "adamw"
            self.optimizer = optim.AdamW(self.model.parameters(), **optimizer_kwargs)

        # Initialize scheduler
        self.scheduler: Optional[Any] = None
        scheduler_type = training_config.lr_scheduler
        # Get scheduler params from config, default to empty dict if None
        scheduler_params = training_config.lr_scheduler_params or {}

        if scheduler_type == "linear":
            # Default params for LinearLR
            default_linear_params = {
                "start_factor": 1.0,
                "end_factor": 0.1,
                # total_iters is always training_steps for this setup
            }
            # Update defaults with user-provided params
            final_params = {**default_linear_params, **scheduler_params}
            # Ensure total_iters is not overridden by user params
            final_params.pop("total_iters", None)

            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                total_iters=training_config.training_steps,
                **final_params,  # Pass start_factor, end_factor, etc.
            )
            logger.info(
                f"Rank {self.rank}: Using LinearLR scheduler with params: {final_params}, total_iters={training_config.training_steps}"
            )

        elif scheduler_type == "cosine":
            # Default params for CosineAnnealingLR
            default_cosine_params = {
                # T_max defaults to training_steps
                "eta_min": 0,  # Default minimum LR
            }
            # Update defaults with user-provided params
            final_params = {**default_cosine_params, **scheduler_params}
            # Set T_max explicitly, allowing override but defaulting to training_steps
            t_max = final_params.pop("T_max", training_config.training_steps)

            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=t_max, **final_params  # Pass eta_min, etc.
            )
            logger.info(
                f"Rank {self.rank}: Using CosineAnnealingLR scheduler with params: {final_params}, T_max={t_max}"
            )

        elif scheduler_type == "linear_final20":
            # This scheduler keeps LR constant for the initial fraction of training
            # and then linearly decays it to 0 over the remaining steps (default 20%).
            # The fraction can be customized via lr_scheduler_params["decay_start_frac"].
            decay_start_frac = scheduler_params.get("decay_start_frac", 0.8)  # 0.8 means last 20% decays
            assert 0.0 < decay_start_frac < 1.0, "decay_start_frac must be between 0 and 1"
            total_steps = training_config.training_steps
            decay_start_step = int(decay_start_frac * total_steps)

            def lr_lambda(current_step: int):
                if current_step < decay_start_step:
                    return 1.0  # Keep LR constant
                # Linearly decay from 1 -> 0 over the remaining steps
                remaining = total_steps - current_step
                decay_steps = total_steps - decay_start_step
                return max(remaining / decay_steps, 0.0)

            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
            logger.info(
                "Rank %d: Using linear_final20 LR scheduler with decay_start_frac=%s (start step %d of %d)",
                self.rank,
                decay_start_frac,
                decay_start_step,
                total_steps,
            )

        # Add elif blocks here for other potential schedulers

        # Initialize activation store.
        # For tensor parallelism we need every rank to process the *same* batch because
        # activations are sharded **across the feature dimension**, not across tokens.
        # Passing the per-rank/world sharded parameters here would make each rank see a
        # different subset of tokens and lead to inconsistent batch sizes which breaks
        # collective ops such as all_gather in ColumnParallelLinear.

        # Pass the actual rank and world size from the distributed setup
        activation_store_rank = self.rank
        activation_store_world = self.world_size

        self.activation_store = create_activation_store(
            training_config=self.training_config,
            clt_config=self.clt_config,
            device=self.device,
            rank=activation_store_rank,  # Pass the actual rank
            world_size=activation_store_world,  # Pass the actual world size
            start_time=self.start_time,
            shard_data=not self.distributed,  # ---> ADDED: False if distributed (TP), True otherwise <----
        )

        # Pass normalisation statistics (if available) so the loss can be computed in
        # the *original* scale even when inputs/targets are stored normalised.
        mean_tg_stats = getattr(self.activation_store, "mean_tg", {})  # type: ignore[arg-type]
        std_tg_stats = getattr(self.activation_store, "std_tg", {})  # type: ignore[arg-type]

        self.loss_manager = LossManager(
            training_config,
            mean_tg=mean_tg_stats,
            std_tg=std_tg_stats,
        )

        # Initialize Evaluator - Pass norm stats here too
        self.evaluator = CLTEvaluator(
            model=self.model,
            device=self.device,
            start_time=self.start_time,
            mean_tg=mean_tg_stats,  # Pass the same stats
            std_tg=std_tg_stats,  # Pass the same stats
        )

        # Initialize dead neuron counters (replicated for now, consider sharding later if needed)
        self.n_forward_passes_since_fired = torch.zeros(
            (clt_config.num_layers, clt_config.num_features),
            device=self.device,
            dtype=torch.long,
        )

        # Initialize WandB logger - only on rank 0
        if not self.distributed or self.rank == 0:
            self.wandb_logger: Union[WandBLogger, DummyWandBLogger] = WandBLogger(
                training_config=training_config, clt_config=clt_config, log_dir=self.log_dir
            )
        else:
            # Dummy logger for non-rank-0 processes
            self.wandb_logger = DummyWandBLogger()

        # Initialize CheckpointManager
        self.checkpoint_manager = CheckpointManager(
            model=self.model,
            activation_store=self.activation_store,
            wandb_logger=self.wandb_logger,
            log_dir=self.log_dir,
            distributed=self.distributed,
            rank=self.rank,
            device=self.device,
        )

        # Initialize MetricLogger
        self.metric_logger = MetricLogger(
            distributed=self.distributed,
            rank=self.rank,
            log_dir=self.log_dir,
            wandb_logger=self.wandb_logger,
            training_config=self.training_config,
            world_size=self.world_size,
        )

    @property
    def dead_neurons_mask(self) -> torch.Tensor:
        """Boolean mask indicating dead neurons based on inactivity window."""
        # Ensure counter is initialized
        if not hasattr(self, "n_forward_passes_since_fired") or self.n_forward_passes_since_fired is None:
            # Return an all-false mask if counter doesn't exist yet
            return torch.zeros(
                (self.clt_config.num_layers, self.clt_config.num_features),
                dtype=torch.bool,
                device=self.device,
            )
        return self.n_forward_passes_since_fired > self.training_config.dead_feature_window

    def train(self, eval_every: int = 1000) -> CrossLayerTranscoder:
        """Train the CLT model.

        Args:
            eval_every: Evaluate model every N steps

        Returns:
            Trained CLT model (local shard)
        """
        # Print startup message from rank 0 only
        if not self.distributed or self.rank == 0:
            print(f"Starting CLT training on {self.device}...")
            print(
                f"Model has {self.clt_config.num_features} features per layer "
                f"and {self.clt_config.num_layers} layers"
            )
            print(f"Training for {self.training_config.training_steps} steps.")
            print(f"Logging to {self.log_dir}")
            if self.distributed:
                print(f"Distributed training with {self.world_size} processes (Tensor Parallelism)")

            # Check if using normalization and notify user
            if self.training_config.normalization_method == "estimated_mean_std":
                print("\n>>> NORMALIZATION PHASE <<<")
                print("Normalization statistics are being estimated from dataset activations.")
                print("This may take some time, but happens only once before training begins.")
                print(f"Using {self.training_config.normalization_estimation_batches} batches for estimation.\n")

            # Make sure we flush stdout to ensure prints appear immediately,
            # especially important in Jupyter/interactive environments
            sys.stdout.flush()
            # Wait for 1 second to ensure output is displayed before training starts
            time.sleep(1)
            print("\n>>> TRAINING PHASE <<<")
            sys.stdout.flush()

        # After the existing startup messages
        if self.distributed:
            print("\n!!! DIAGNOSTIC INFO !!!")
            print(f"Rank {self.rank}: Process group type: {type(self.process_group)}")
            print(f"Rank {self.rank}: RowParallelLinear _reduce does NOT divide by world_size")
            print(f"Rank {self.rank}: Using weight regularization in sparsity penalty")
            print(f"Rank {self.rank}: Averaging replicated parameter gradients")
            # Check if activation store has rank/world attributes before accessing
            store_rank = getattr(self.activation_store, "rank", "N/A")
            store_world = getattr(self.activation_store, "world", "N/A")
            print(f"Rank {self.rank}: Data sharding: rank={store_rank}, world={store_world}")
            print(f"Rank {self.rank}: Batch size tokens: {self.training_config.train_batch_size_tokens}")
            print(f"Rank {self.rank}: Sparsity lambda: {self.training_config.sparsity_lambda}")

            # Check if activation store actually loaded correctly
            batch_avail = next(iter(self.activation_store), None)
            print(f"Rank {self.rank}: First batch available: {batch_avail is not None}")

            # Force torch to compile/execute our code by running a tiny forward/backward pass
            dummy = torch.ones(1, device=self.device, requires_grad=True)
            dummy_out = dummy * 2
            dummy_out.backward()
            print("!!! END DIAGNOSTIC !!!\n")

        # Create progress bar only on rank 0
        pbar: Union[tqdm, range]
        if not self.distributed or self.rank == 0:
            pbar = tqdm(
                range(self.training_config.training_steps),
                desc="Training CLT",
                leave=True,
            )
        else:
            pbar = range(self.training_config.training_steps)

        step = 0
        try:
            for step in pbar:
                # Refresh progress bar on rank 0
                step_start_time = time.monotonic()  # Start timing the step
                if isinstance(pbar, tqdm):
                    pbar.refresh()

                try:
                    # Get batch directly from the iterator (handles distributed sampling internally)
                    batch_get_start_time = time.monotonic()
                    inputs, targets = next(self.activation_store)
                    batch_get_duration = time.monotonic() - batch_get_start_time
                    logger.debug(f"Rank {self.rank} Step {step}: Getting batch took {batch_get_duration:.4f}s")

                    # logging to diagnose batch size mismatch
                    tok_cnt = next(iter(inputs.values())).shape[0]  # number of rows (=tokens) in this batch
                    # Only run the all_gather diagnostic when running in distributed mode
                    if self.distributed and self.world_size > 1 and dist.is_initialized():
                        tok_cnt_t = torch.tensor([tok_cnt], device=self.device)
                        gathered = [torch.zeros_like(tok_cnt_t) for _ in range(self.world_size)]
                        dist.all_gather(gathered, tok_cnt_t)
                        if self.rank == 0:
                            print("Batch token-count per rank:", [int(x.item()) for x in gathered])

                except StopIteration:
                    # Rank 0 prints message
                    if not self.distributed or self.rank == 0:
                        print("Activation store exhausted. Training finished early.")
                    if self.distributed:
                        dist.barrier()  # Ensure all ranks see this
                    break  # Exit training loop if data runs out
                except Exception as e:
                    # Rank 0 prints message
                    if not self.distributed or self.rank == 0:
                        print(f"\nRank {self.rank}: Error getting batch at step {step}: {e}. Skipping step.")
                    # Maybe barrier here too? If one rank fails, others might hang?
                    # Let's continue for now, assuming store handles internal errors.
                    continue

                # --- Check for empty batch --- (Optional but good practice)
                # This check should ideally happen *before* moving data potentially
                if not inputs or not targets or not any(v.numel() > 0 for v in inputs.values()):
                    if not self.distributed or self.rank == 0:
                        print(f"\nRank {self.rank}: Warning: Received empty batch at step {step}. Skipping.")
                    continue

                # --- BEGIN: One-time Normalization Check ---
                if step == 0 and (not self.distributed or self.rank == 0):
                    logger.info("--- Running Post-Normalization Check (First Batch) ---")
                    norm_applied = getattr(self.activation_store, "apply_normalization", None)
                    if isinstance(self.activation_store, (LocalActivationStore, RemoteActivationStore)):
                        logger.info(f"ActivationStore reports apply_normalization={norm_applied}")

                    for li in range(self.clt_config.num_layers):
                        mean_in, std_in, mean_tg, std_tg = float("nan"), float("nan"), float("nan"), float("nan")
                        try:
                            if li in inputs and inputs[li].numel() > 0:
                                input_tensor = inputs[li].float()
                                mean_in = input_tensor.mean().item()
                                std_in = input_tensor.std().item()
                            if li in targets and targets[li].numel() > 0:
                                target_tensor = targets[li].float()
                                mean_tg = target_tensor.mean().item()
                                std_tg = target_tensor.std().item()

                            if not (
                                torch.isnan(torch.tensor(mean_in)) and torch.isnan(torch.tensor(mean_tg))
                            ):  # Log if at least one value is valid
                                logger.info(
                                    f"  Layer {li:>2}: Input Mean={mean_in:+.4f}, Std={std_in:.4f} | Target Mean={mean_tg:+.4f}, Std={std_tg:.4f}"
                                )
                        except Exception as e:
                            logger.error(f"  Layer {li}: Error during normalization check: {e}")
                    logger.info("--- End Post-Normalization Check ---")
                # --- END: One-time Normalization Check ---

                # --- Forward pass and compute loss --- (All ranks)
                self.optimizer.zero_grad()

                # Compute feature activations **once** per step to avoid redundant encoder forward passes.
                feature_activations_batch = self.model.get_feature_activations(inputs)

                # Compute total loss using the pre-computed activations
                loss, loss_dict = self.loss_manager.compute_total_loss(
                    self.model,
                    inputs,
                    targets,
                    step,
                    self.training_config.training_steps,
                    precomputed_activations=feature_activations_batch,
                    dead_neuron_mask=self.dead_neurons_mask,
                )

                # --- Update Dead Neuron Counters --- (All ranks, counter is replicated)
                # We need *full* feature activations *after* non-linearity
                if hasattr(self, "n_forward_passes_since_fired"):
                    with torch.no_grad():
                        for layer_idx, layer_acts in feature_activations_batch.items():
                            # Ensure layer index is within bounds of the counter tensor
                            if layer_idx < self.n_forward_passes_since_fired.shape[0]:
                                if layer_acts.numel() > 0:
                                    # layer_acts shape: [batch_tokens, num_features]
                                    fired_mask_per_token = layer_acts > 1e-6
                                    fired_features_this_layer = fired_mask_per_token.any(dim=0)

                                    if fired_features_this_layer.shape[0] == self.n_forward_passes_since_fired.shape[1]:
                                        self.n_forward_passes_since_fired[layer_idx] += 1
                                        self.n_forward_passes_since_fired[layer_idx][fired_features_this_layer] = 0
                                    else:
                                        if not self.distributed or self.rank == 0:  # Only rank 0 logs warning
                                            print(
                                                f"Rank {self.rank}: Warning: Shape mismatch for dead neuron update at layer {layer_idx}. "
                                                f"Acts shape: {layer_acts.shape}, Fired mask: {fired_features_this_layer.shape}, "
                                                f"Counter: {self.n_forward_passes_since_fired.shape}"
                                            )

                # --- Backward pass --- (All ranks, handles communication implicitly)
                if torch.isnan(loss):
                    if not self.distributed or self.rank == 0:
                        print(
                            f"\nRank {self.rank}: Warning: NaN loss encountered at step {step}. "
                            f"Skipping backward pass and optimizer step."
                        )
                else:
                    try:
                        loss.backward()

                        # --- Synchronise gradients of replicated parameters --- #
                        if self.distributed and self.world_size > 1:  # Guard before calling the static/util function
                            average_shared_parameter_grads(self.model, self.world_size)  # New call

                        # --- Gradient clipping --- #
                        if self.training_config.gradient_clip_val is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.training_config.gradient_clip_val,
                            )
                    except RuntimeError as e:
                        if not self.distributed or self.rank == 0:
                            print(
                                f"\nRank {self.rank}: Error during backward pass at step {step}: {e}. Skipping optimizer step."
                            )
                        continue

                    # --- Optimizer step --- (Applied to local parameters using local gradients)
                    self.optimizer.step()

                    # --- Invalidate Caches --- #
                    if hasattr(self.model, "_cached_decoder_norms"):
                        self.model._cached_decoder_norms = None

                # --- Scheduler step --- (All ranks)
                if self.scheduler:
                    self.scheduler.step()

                # --- Update progress bar --- (Rank 0 only)
                if isinstance(pbar, tqdm):
                    description = (
                        f"Loss: {loss_dict.get('total', float('nan')):.4f} "
                        f"(R: {loss_dict.get('reconstruction', float('nan')):.4f} "
                        f"S: {loss_dict.get('sparsity', float('nan')):.4f} "
                        f"P: {loss_dict.get('preactivation', float('nan')):.4f})"
                    )
                    pbar.set_description(description)
                    # Force update to display progress
                    if step % 1 == 0:  # Update every step
                        pbar.refresh()
                        sys.stdout.flush()

                # --- Log metrics --- (Rank 0 logs to WandB/file)
                # self._log_metrics(step, loss_dict) # Old call
                current_lr_for_log = self.scheduler.get_last_lr()[0] if self.scheduler else None
                current_lambda_for_log = self.loss_manager.get_current_sparsity_lambda()
                self.metric_logger.log_training_step(
                    step, loss_dict, current_lr=current_lr_for_log, current_sparsity_lambda=current_lambda_for_log
                )

                step_duration = time.monotonic() - step_start_time
                logger.debug(
                    f"Rank {self.rank} Step {step}: Main logic (incl. batch get, fwd, bwd, optim) took {step_duration:.4f}s"
                )

                # --- Evaluation & Checkpointing ---
                eval_interval = self.training_config.eval_interval
                checkpoint_interval = self.training_config.checkpoint_interval

                save_checkpoint_flag = (step > 0 and step % checkpoint_interval == 0) or (  # Avoid checkpoint at step 0
                    step == self.training_config.training_steps - 1
                )
                run_eval_flag = (step % eval_interval == 0) or (step == self.training_config.training_steps - 1)

                # --- Evaluation (all ranks participate to match collectives) ---
                # In tensor-parallel mode the model forward includes collective ops (all_reduce/all_gather).
                # If only rank 0 performed the forward pass these collectives would block on the other ranks
                # resulting in NCCL timeouts.  Therefore, *every* rank must execute the evaluation forward pass.
                # We still only log / store the resulting metrics on rank 0.
                if run_eval_flag:
                    if self.distributed:
                        dist.barrier()  # Sync before evaluation starts so that all ranks enter together

                    # Compute evaluation metrics on all ranks to keep collective ops aligned
                    current_dead_mask = self.dead_neurons_mask.detach().clone()
                    eval_metrics = self.evaluator.compute_metrics(
                        inputs,
                        targets,
                        dead_neuron_mask=current_dead_mask,
                    )

                    # --- Log per-layer standard deviation of pre-activations ---
                    # This requires getting the pre-activations first.
                    # _encode_all_layers returns: preactivations_dict, original_shapes_info, device, dtype
                    preactivations_eval_dict, _, _, _ = self.model._encode_all_layers(inputs)
                    layerwise_preact_std_dev: Dict[str, float] = {}
                    if preactivations_eval_dict:
                        for layer_idx, preact_tensor in preactivations_eval_dict.items():
                            if preact_tensor.numel() > 0:
                                # Calculate std dev of all elements in the preactivation tensor for this layer
                                std_dev_val = preact_tensor.std().item()
                                layerwise_preact_std_dev[f"layer_{layer_idx}"] = std_dev_val
                            else:
                                layerwise_preact_std_dev[f"layer_{layer_idx}"] = float("nan")  # Or 0.0

                    # Add to eval_metrics for WandB logging
                    if layerwise_preact_std_dev:
                        eval_metrics["layerwise/preactivation_std_dev"] = layerwise_preact_std_dev
                    # --- End logging pre-activation std dev ---

                    if not self.distributed or self.rank == 0:
                        # Store evaluation metrics (for saving to JSON) - Handled by MetricLogger
                        # self.metrics["eval_metrics"].append({"step": step, **eval_metrics})

                        # --- Update Progress Bar Postfix --- (Restore this block)
                        l0_str = f"AvgL0: {eval_metrics.get('sparsity/avg_l0', 0.0):.2f}"
                        ev_str = f"EV: {eval_metrics.get('reconstruction/explained_variance', 0.0):.3f}"
                        avg_density_mean = eval_metrics.get("sparsity/feature_density_mean")
                        dens_str = f"Dens: {avg_density_mean:.3f}" if avg_density_mean is not None else "Dens: N/A"
                        eval_dead_str = f"Dead(Eval): {eval_metrics.get('dead_features/total_eval', 0)}"
                        eval_msg = f"{l0_str}, {ev_str}, {dens_str}, {eval_dead_str}"

                        if isinstance(pbar, tqdm):
                            pbar.set_postfix_str(eval_msg)
                            pbar.refresh()

                        # --- Log evaluation metrics (now done by MetricLogger) ---
                        self.metric_logger.log_evaluation_metrics(step, eval_metrics)

                    # Optionally compute and log sparsity diagnostics (can be slow).
                    # IMPORTANT: every rank must execute compute_sparsity_diagnostics because it
                    # internally calls `get_decoder_norms`, which performs distributed all-reduces.
                    # We therefore compute the diagnostics on **all** ranks to keep the NCCL
                    # collectives aligned, but we still only merge the returned values into
                    # `eval_metrics` and log them on rank 0.
                    if self.training_config.compute_sparsity_diagnostics:
                        sparsity_diag_metrics = compute_sparsity_diagnostics(
                            model=self.model,
                            training_config=self.training_config,
                            feature_activations=feature_activations_batch,
                        )

                        # Only rank 0 merges & logs the diagnostics, preventing duplicate WandB entries.
                        if (not self.distributed) or (self.rank == 0):
                            if sparsity_diag_metrics:
                                eval_metrics.update(sparsity_diag_metrics)
                                self.wandb_logger.log_evaluation(step, eval_metrics)

                    # Ensure all ranks finish evaluation before proceeding
                    if self.distributed:
                        dist.barrier()

                # --- Checkpointing (All ranks participate) ---
                if save_checkpoint_flag:
                    self.checkpoint_manager._save_checkpoint(step)

            # --- Explicitly delete tensors at the very end of the loop iteration --- #
            # Do this on all ranks
            try:
                del inputs
                del targets
                if "loss" in locals() and loss is not None:
                    del loss
                if "feature_activations_batch" in locals():
                    del feature_activations_batch
            except NameError:
                pass

        except KeyboardInterrupt:
            if not self.distributed or self.rank == 0:
                print("\nTraining interrupted by user.")
        finally:
            if isinstance(pbar, tqdm):
                pbar.close()
            if not self.distributed or self.rank == 0:
                print(f"Training loop finished at step {step}.")

        # Sync before final save attempt
        if self.distributed:
            dist.barrier()

        # --- Save final model and metrics --- (Rank 0 handles metrics/store, all ranks save model state)
        final_checkpoint_dir = os.path.join(self.log_dir, "final")
        final_store_path = os.path.join(final_checkpoint_dir, "activation_store_final.pt")

        # All ranks save final model state
        try:
            # final_model_state_dict = self.model.state_dict() # Not strictly needed here if CheckpointManager handles it
            self.checkpoint_manager._save_checkpoint(
                step=self.training_config.training_steps
            )  # step value indicates final
        except Exception as e:
            print(f"Rank {self.rank}: Warning: Failed to save final distributed model state: {e}")

        # Rank 0 saves store, metrics, logs artifact
        if not self.distributed or self.rank == 0:
            print(f"Saving final activation store state to {final_store_path}...")
            os.makedirs(final_checkpoint_dir, exist_ok=True)
            try:
                # Check if the store has a close method before calling (for compatibility)
                if hasattr(self.activation_store, "close") and callable(getattr(self.activation_store, "close")):
                    self.activation_store.close()
            except Exception as e:
                print(f"Rank 0: Warning: Failed to close activation store: {e}")

            print("Saving final metrics...")
            self.metric_logger._save_metrics_to_disk()  # Final save

            # --- Save CLT Config to JSON ---
            # The config saved here will now reflect the configuration *during training* (e.g. BatchTopK)
            # The user will need to run estimate_theta_posthoc and then save the converted JumpReLU model themselves.
            config_save_path = os.path.join(self.log_dir, "cfg.json")
            print(f"Saving CLT configuration (as trained) to {config_save_path}...")
            try:
                config_dict_as_trained = asdict(self.clt_config)
                # Remove accesses to attributes no longer in TrainingConfig
                # model_name = None
                # if (
                #     hasattr(self.training_config, "generation_config")
                #     and self.training_config.generation_config is not None
                #     and "model_name" in self.training_config.generation_config
                # ):
                #     model_name = self.training_config.generation_config["model_name"]
                #     config_dict_as_trained["model_name"] = model_name
                # elif (
                #     hasattr(self.training_config, "activation_config")
                #     and hasattr(self.training_config.activation_config, "model_name")
                #     and self.training_config.activation_config.model_name is not None
                # ):
                #     config_dict_as_trained["model_name"] = self.training_config.activation_config.model_name

                # Attempt to add model_name if available in clt_config itself
                if hasattr(self.clt_config, "model_name") and self.clt_config.model_name:
                    config_dict_as_trained["model_name"] = self.clt_config.model_name

                if hasattr(self.training_config, "normalization_method"):
                    config_dict_as_trained["normalization_method"] = self.training_config.normalization_method
                if hasattr(self.training_config, "activation_dtype"):
                    config_dict_as_trained["expected_input_dtype"] = self.training_config.activation_dtype

                # Add hook templates if available directly in clt_config
                if hasattr(self.clt_config, "mlp_input_template") and self.clt_config.mlp_input_template:
                    config_dict_as_trained["mlp_input_template"] = self.clt_config.mlp_input_template
                if hasattr(self.clt_config, "mlp_output_template") and self.clt_config.mlp_output_template:
                    config_dict_as_trained["mlp_output_template"] = self.clt_config.mlp_output_template

                with open(config_save_path, "w") as f:
                    json.dump(config_dict_as_trained, f, indent=2)
                print(f"Successfully saved training configuration to {config_save_path}")
                if self.clt_config.activation_fn == "batchtopk":
                    print(
                        "NOTE: Model was trained with BatchTopK. Run estimate_theta_posthoc() on the saved model to convert to JumpReLU and finalize theta values."
                    )

            except Exception as e:
                print(f"Rank 0: Warning: Failed to save CLT configuration to JSON: {e}")
            # --- End Save CLT Config ---

            # Log final checkpoint directory as artifact
            self.wandb_logger.log_artifact(artifact_path=final_checkpoint_dir, artifact_type="model", name="clt_final")

            # Finish WandB logging
            self.wandb_logger.finish()
            print(f"Training completed! Final checkpoint saved to {final_checkpoint_dir}")

        # --- Close the activation store (stops prefetch thread if applicable) --- #
        if hasattr(self.activation_store, "close") and callable(getattr(self.activation_store, "close")):
            self.activation_store.close()

        # Clean up distributed process group
        if self.distributed:
            dist.destroy_process_group()

        return self.model
