import torch

# import torch.optim as optim # Unused
from typing import Dict, Optional, Any, Callable, Union, Tuple
from tqdm import tqdm  # type: ignore
import time
import sys
import logging
import torch.distributed as dist
from torch.distributed import ProcessGroup

from clt.config import CLTConfig, TrainingConfig
from clt.models.clt import CrossLayerTranscoder
from clt.training.data.base_store import BaseActivationStore
from clt.training.losses import LossManager
from clt.logging.metric_logger import MetricLogger
from clt.training.distributed_utils import average_shared_parameter_grads

# Assuming LocalActivationStore and RemoteActivationStore might be checked with isinstance
from clt.training.data.local_activation_store import LocalActivationStore
from clt.training.data.remote_activation_store import RemoteActivationStore


logger = logging.getLogger(__name__)

# Callback signature:
# (step, inputs, targets, feature_activations_batch) -> Optional[str] (eval_message)
EvalCheckpointCallback = Callable[
    [int, Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor]], Optional[str]
]


class TrainingLoop:
    def __init__(
        self,
        model: CrossLayerTranscoder,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],  # torch.optim.lr_scheduler._LRScheduler
        scaler: torch.cuda.amp.GradScaler,
        activation_store: BaseActivationStore,
        loss_manager: LossManager,
        clt_config: CLTConfig,
        training_config: TrainingConfig,
        device: torch.device,
        distributed: bool,
        rank: int,
        world_size: int,
        process_group: Optional[ProcessGroup],
        metric_logger: MetricLogger,
        initial_step: int,
        n_forward_passes_since_fired: torch.Tensor,  # The actual tensor
        eval_checkpoint_callback: EvalCheckpointCallback,
        use_cuda_amp: bool,
        autocast_dtype: torch.dtype,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.activation_store = activation_store
        self.loss_manager = loss_manager
        self.clt_config = clt_config
        self.training_config = training_config
        self.device = device
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        self.process_group = process_group
        self.metric_logger = metric_logger
        self.initial_step = initial_step
        self.n_forward_passes_since_fired = n_forward_passes_since_fired
        self.eval_checkpoint_callback = eval_checkpoint_callback
        self.use_cuda_amp = use_cuda_amp
        self.autocast_dtype = autocast_dtype

    def get_dead_neuron_mask(self) -> torch.Tensor:
        """Boolean mask indicating dead neurons based on inactivity window."""
        if self.n_forward_passes_since_fired is None:
            return torch.zeros(
                (self.clt_config.num_layers, self.clt_config.num_features),
                dtype=torch.bool,
                device=self.device,
            )
        return self.n_forward_passes_since_fired > self.training_config.dead_feature_window

    def run(self) -> Tuple[CrossLayerTranscoder, int]:
        pbar: Union[tqdm, range]
        if not self.distributed or self.rank == 0:
            pbar = tqdm(
                range(self.initial_step, self.training_config.training_steps),
                desc="Training CLT",
                leave=True,
                initial=self.initial_step,
                total=self.training_config.training_steps,
            )
        else:
            pbar = range(self.initial_step, self.training_config.training_steps)

        current_step_in_loop = self.initial_step
        try:
            for step in pbar:
                current_step_in_loop = step
                step_start_time = time.monotonic()
                if isinstance(pbar, tqdm):
                    pbar.refresh()

                try:
                    batch_get_start_time = time.monotonic()
                    inputs, targets = next(self.activation_store)
                    batch_get_duration = time.monotonic() - batch_get_start_time
                    logger.debug(f"Rank {self.rank} Step {step}: Getting batch took {batch_get_duration:.4f}s")

                    if inputs and next(iter(inputs.values())).numel() > 0:  # Check if inputs is not empty and has data
                        tok_cnt = next(iter(inputs.values())).shape[0]
                        if self.distributed and self.world_size > 1 and dist.is_initialized():
                            tok_cnt_t = torch.tensor([tok_cnt], device=self.device)
                            gathered_tok_cnts = [torch.zeros_like(tok_cnt_t) for _ in range(self.world_size)]
                            dist.all_gather(gathered_tok_cnts, tok_cnt_t, group=self.process_group)
                            if self.rank == 0:
                                logger.debug(
                                    f"Batch token-count per rank (Step {step}): {[int(x.item()) for x in gathered_tok_cnts]}"
                                )
                    else:  # Handle case where inputs might be empty or values are empty tensors
                        if not self.distributed or self.rank == 0:
                            logger.warning(
                                f"Rank {self.rank} Step {step}: Inputs dictionary is empty or contains empty tensors before token count diagnostic."
                            )

                except StopIteration:
                    if not self.distributed or self.rank == 0:
                        print("Activation store exhausted. Training finished early.")
                    if self.distributed and dist.is_initialized():
                        dist.barrier(group=self.process_group)
                    break
                except Exception as e:
                    if not self.distributed or self.rank == 0:
                        print(f"\nRank {self.rank}: Error getting batch at step {step}: {e}. Skipping step.")
                    continue

                if not inputs or not targets or not any(v.numel() > 0 for v in inputs.values()):
                    if not self.distributed or self.rank == 0:
                        print(f"\nRank {self.rank}: Warning: Received empty batch at step {step}. Skipping.")
                    continue

                if step == 0 and (not self.distributed or self.rank == 0):
                    logger.info("--- Running Post-Normalization Check (First Batch) ---")
                    norm_applied = "Unknown"
                    if isinstance(self.activation_store, (LocalActivationStore, RemoteActivationStore)):
                        norm_applied = getattr(self.activation_store, "apply_normalization", "Attribute_Not_Found")
                    logger.info(
                        f"ActivationStore (type: {type(self.activation_store).__name__}) reports apply_normalization={norm_applied}"
                    )

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

                            if not (torch.isnan(torch.tensor(mean_in)) and torch.isnan(torch.tensor(mean_tg))):
                                logger.info(
                                    f"  Layer {li:>2}: Input Mean={mean_in:+.4f}, Std={std_in:.4f} | Target Mean={mean_tg:+.4f}, Std={std_tg:.4f}"
                                )
                        except Exception as e_norm_check:
                            logger.error(f"  Layer {li}: Error during normalization check: {e_norm_check}")
                    logger.info("--- End Post-Normalization Check ---")

                self.optimizer.zero_grad(set_to_none=True)

                feature_activations_batch: Dict[int, torch.Tensor] = {}  # Initialize
                loss = torch.tensor(0.0, device=self.device)  # Initialize
                loss_dict: Dict[str, float] = {}  # Initialize

                with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.use_cuda_amp):
                    feature_activations_batch = self.model.get_feature_activations(inputs)
                    current_dead_mask_for_loss = self.get_dead_neuron_mask()
                    loss, loss_dict = self.loss_manager.compute_total_loss(
                        self.model,
                        inputs,
                        targets,
                        step,
                        self.training_config.training_steps,
                        precomputed_activations=feature_activations_batch,
                        dead_neuron_mask=current_dead_mask_for_loss,
                    )

                if self.n_forward_passes_since_fired is not None:
                    with torch.no_grad():
                        for layer_idx, layer_acts in feature_activations_batch.items():
                            if layer_idx < self.n_forward_passes_since_fired.shape[0]:
                                if layer_acts.numel() > 0:
                                    fired_mask_per_token = layer_acts > 1e-6
                                    fired_features_this_layer = fired_mask_per_token.any(dim=0)
                                    if fired_features_this_layer.shape[0] == self.n_forward_passes_since_fired.shape[1]:
                                        self.n_forward_passes_since_fired[layer_idx] += 1
                                        self.n_forward_passes_since_fired[layer_idx][fired_features_this_layer] = 0
                                    else:
                                        if not self.distributed or self.rank == 0:
                                            print(
                                                f"Rank {self.rank}: Warning: Shape mismatch for dead neuron update at layer {layer_idx}. "
                                                f"Acts shape: {layer_acts.shape}, Fired mask: {fired_features_this_layer.shape}, "
                                                f"Counter: {self.n_forward_passes_since_fired.shape}"
                                            )
                if torch.isnan(loss):
                    if not self.distributed or self.rank == 0:
                        print(
                            f"\nRank {self.rank}: Warning: NaN loss encountered at step {step}. "
                            f"Skipping backward pass and optimizer step."
                        )
                else:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    if self.distributed and self.world_size > 1 and dist.is_initialized():
                        average_shared_parameter_grads(self.model, self.world_size)
                    if self.training_config.gradient_clip_val is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.training_config.gradient_clip_val,
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if hasattr(self.model, "_cached_decoder_norms"):
                        self.model._cached_decoder_norms = None

                if (
                    self.distributed
                    and self.world_size > 1
                    and self.n_forward_passes_since_fired is not None
                    and dist.is_initialized()
                ):
                    dist.all_reduce(self.n_forward_passes_since_fired, op=dist.ReduceOp.MIN, group=self.process_group)

                if self.scheduler:
                    self.scheduler.step()

                if isinstance(pbar, tqdm):
                    description = (
                        f"Loss: {loss_dict.get('total', float('nan')):.4f} "
                        f"(R: {loss_dict.get('reconstruction', float('nan')):.4f} "
                        f"S: {loss_dict.get('sparsity', float('nan')):.4f} "
                        f"P: {loss_dict.get('preactivation', float('nan')):.4f})"
                    )
                    pbar.set_description(description)
                    if step % 1 == 0:
                        pbar.refresh()
                        sys.stdout.flush()

                current_lr_for_log = (
                    self.scheduler.get_last_lr()[0] if self.scheduler else self.training_config.learning_rate
                )
                current_lambda_for_log = self.loss_manager.get_current_sparsity_lambda()
                self.metric_logger.log_training_step(
                    step, loss_dict, current_lr=current_lr_for_log, current_sparsity_lambda=current_lambda_for_log
                )

                step_duration = time.monotonic() - step_start_time
                logger.debug(f"Rank {self.rank} Step {step}: Main loop iteration took {step_duration:.4f}s")

                eval_msg = self.eval_checkpoint_callback(step, inputs, targets, feature_activations_batch)
                if eval_msg and isinstance(pbar, tqdm):
                    pbar.set_postfix_str(eval_msg)
                    pbar.refresh()  # Refresh to show postfix immediately

                # Explicitly delete tensors
                del inputs
                del targets
                del loss
                del feature_activations_batch

        except KeyboardInterrupt:
            if not self.distributed or self.rank == 0:
                print("\nTraining interrupted by user.")
        finally:
            if isinstance(pbar, tqdm):
                pbar.close()
            if not self.distributed or self.rank == 0:
                print(f"Training loop part finished at step {current_step_in_loop}.")

        return self.model, current_step_in_loop
