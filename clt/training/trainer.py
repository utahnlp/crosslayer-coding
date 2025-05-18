import torch
import torch.optim as optim
from typing import Dict, Optional, Union, Any

# from tqdm import tqdm  # type: ignore # No longer directly used by CLTTrainer
import os
import json
import time
import sys
import logging
import torch.distributed as dist
from dataclasses import asdict
import numpy as np
import random

from clt.config import CLTConfig, TrainingConfig
from clt.models.clt import CrossLayerTranscoder
from clt.training.data.base_store import BaseActivationStore  # Keep for type hint
from clt.training.data.activation_store_factory import create_activation_store
from clt.training.losses import LossManager
from .evaluator import CLTEvaluator
from clt.logging.wandb_logger import WandBLogger, DummyWandBLogger
from clt.logging.metric_logger import MetricLogger
from clt.logging.factory import setup_loggers
from clt.checkpointing.engine import CheckpointManager
from .distributed_utils import initialize_distributed_env
from clt.training.training_loop import TrainingLoop, EvalCheckpointCallback

logger = logging.getLogger(__name__)


class CLTTrainer:
    activation_store: BaseActivationStore
    model: CrossLayerTranscoder
    wandb_logger: Union[WandBLogger, DummyWandBLogger]
    metric_logger: MetricLogger
    training_loop: TrainingLoop  # Add type hint for training_loop
    optimizer: torch.optim.Optimizer  # Add type hint
    scheduler: Optional[Any]  # Add type hint
    scaler: torch.cuda.amp.GradScaler  # Add type hint
    evaluator: CLTEvaluator  # Add type hint
    checkpoint_manager: CheckpointManager  # Add type hint
    n_forward_passes_since_fired: torch.Tensor  # Add type hint

    def __init__(
        self,
        clt_config: CLTConfig,
        training_config: TrainingConfig,
        log_dir: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        distributed: bool = False,
        resume_from_checkpoint_path: Optional[str] = None,
    ):
        self.clt_config = clt_config
        self.training_config = training_config
        self.raw_resume_from_checkpoint_path = resume_from_checkpoint_path
        self.loaded_atomic_checkpoint_state: Optional[Dict[str, Any]] = None

        dist_env_info = initialize_distributed_env(use_distributed_flag=distributed, manual_device_override=device)
        self.rank = dist_env_info.rank
        self.world_size = dist_env_info.world_size
        self.local_rank = dist_env_info.local_rank
        self.device = dist_env_info.device
        self.process_group = dist_env_info.process_group
        self.distributed = dist_env_info.is_distributed

        logger.info(
            f"CLTTrainer initialized with: Rank={self.rank}, WorldSize={self.world_size}, "
            f"LocalRank={self.local_rank}, Device={self.device}, DistributedActive={self.distributed}"
        )
        if self.process_group and self.distributed and dist.is_initialized():
            logger.info(f"Process group backend: {dist.get_backend(self.process_group)}")

        self.mixed_precision = self.training_config.precision.lower()
        self.use_cuda_amp = torch.cuda.is_available() and self.mixed_precision in {"fp16", "bf16"}
        self.autocast_dtype = (
            torch.float16
            if self.mixed_precision == "fp16"
            else torch.bfloat16 if self.mixed_precision == "bf16" else torch.float32
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.mixed_precision == "fp16" and self.use_cuda_amp))

        logger.info(
            f"Rank {self.rank}: Mixed precision mode: {self.mixed_precision}, use_cuda_amp: {self.use_cuda_amp}, autocast_dtype: {self.autocast_dtype}"
        )
        logger.info(f"Rank {self.rank}: GradScaler enabled: {self.scaler.is_enabled()}")

        self.log_dir = log_dir or f"clt_train_{int(time.time())}"
        if not self.distributed or self.rank == 0:
            os.makedirs(self.log_dir, exist_ok=True)

        self.start_time = time.time()

        if self.training_config.seed is not None:
            torch.manual_seed(self.training_config.seed + self.rank)
            np.random.seed(self.training_config.seed + self.rank)
            random.seed(self.training_config.seed + self.rank)
            logger.info(
                f"Rank {self.rank}: Set manual seed to {self.training_config.seed + self.rank} for torch, numpy, random."
            )
        else:
            logger.warning(f"Rank {self.rank}: No seed provided. Using default torch seeding.")

        self.model = CrossLayerTranscoder(clt_config, process_group=self.process_group, device=self.device)

        if self.training_config.fp16_convert_weights:
            if self.mixed_precision == "fp16":
                logger.warning(
                    f"Rank {self.rank}: 'fp16_convert_weights=True' is set with 'precision=fp16'. "
                    "GradScaler expects FP32 optimizer parameters. Model weights will NOT be converted to FP16 "
                    "before optimizer initialization. Autocast will still use FP16 for computations."
                )
            else:
                logger.info(
                    f"Rank {self.rank}: Converting model weights and buffers to FP16 (fp16_convert_weights=True, precision={self.mixed_precision})."
                )
                self.model.half()
                for name, buf in self.model.named_buffers():
                    if buf.dtype == torch.float16 and "norm" in name.lower():
                        logger.info(f"Rank {self.rank}: Converting buffer '{name}' from FP16 back to FP32.")
                        buf.data = buf.data.float()

        optimizer_kwargs: Dict[str, Any] = {"lr": training_config.learning_rate}
        beta1 = training_config.optimizer_beta1
        beta2 = training_config.optimizer_beta2
        if beta1 is not None or beta2 is not None:
            final_beta1 = beta1 if beta1 is not None else 0.9
            final_beta2 = beta2 if beta2 is not None else 0.999
            optimizer_kwargs["betas"] = (final_beta1, final_beta2)
            logger.info(f"Rank {self.rank}: Using optimizer betas: ({final_beta1}, {final_beta2})")

        if training_config.optimizer == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), **optimizer_kwargs)
        else:
            self.optimizer = optim.AdamW(self.model.parameters(), **optimizer_kwargs)

        self.scheduler = None
        scheduler_type = training_config.lr_scheduler
        scheduler_params = training_config.lr_scheduler_params or {}
        if scheduler_type == "linear":
            default_linear_params = {"start_factor": 1.0, "end_factor": 0.1}
            final_params = {**default_linear_params, **scheduler_params}
            final_params.pop("total_iters", None)
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer, total_iters=training_config.training_steps, **final_params
            )
            logger.info(
                f"Rank {self.rank}: Using LinearLR: {final_params}, total_iters={training_config.training_steps}"
            )
        elif scheduler_type == "cosine":
            default_cosine_params = {"eta_min": 0}
            final_params = {**default_cosine_params, **scheduler_params}
            t_max = final_params.pop("T_max", training_config.training_steps)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max, **final_params)
            logger.info(f"Rank {self.rank}: Using CosineAnnealingLR: {final_params}, T_max={t_max}")
        elif scheduler_type == "linear_final20":
            decay_start_frac = scheduler_params.get("decay_start_frac", 0.8)
            assert 0.0 < decay_start_frac < 1.0, "decay_start_frac must be between 0 and 1"
            total_steps = training_config.training_steps
            decay_start_step = int(decay_start_frac * total_steps)

            def lr_lambda(current_step: int):
                if current_step < decay_start_step:
                    return 1.0
                remaining = total_steps - current_step
                decay_steps = total_steps - decay_start_step
                return max(remaining / decay_steps, 0.0)

            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
            logger.info(f"Rank {self.rank}: Using linear_final20 LR scheduler: decay_start_frac={decay_start_frac}")

        self.activation_store = create_activation_store(
            training_config=self.training_config,
            clt_config=self.clt_config,
            device=self.device,
            rank=self.rank,
            world_size=self.world_size,
            start_time=self.start_time,
            shard_data=not self.distributed,
        )

        mean_tg_stats = getattr(self.activation_store, "mean_tg", {})
        std_tg_stats = getattr(self.activation_store, "std_tg", {})

        self.loss_manager = LossManager(
            training_config,
            mean_tg=mean_tg_stats,
            std_tg=std_tg_stats,
        )

        self.evaluator = CLTEvaluator(
            model=self.model,
            device=self.device,
            start_time=self.start_time,
            mean_tg=mean_tg_stats,
            std_tg=std_tg_stats,
            training_config=self.training_config,
        )

        self.n_forward_passes_since_fired = torch.zeros(
            (clt_config.num_layers, clt_config.num_features),
            device=self.device,
            dtype=torch.long,
        )

        placeholder_wandb_logger = DummyWandBLogger(
            training_config=training_config, clt_config=clt_config, log_dir=self.log_dir, resume_wandb_id=None
        )

        self.checkpoint_manager = CheckpointManager(
            model=self.model,
            activation_store=self.activation_store,
            wandb_logger=placeholder_wandb_logger,
            log_dir=self.log_dir,
            distributed=self.distributed,
            rank=self.rank,
            device=self.device,
            world_size=self.world_size,
        )

        if self.raw_resume_from_checkpoint_path:
            logger.info(
                f"Rank {self.rank}: Attempting to resume from checkpoint: {self.raw_resume_from_checkpoint_path}"
            )
            self.loaded_atomic_checkpoint_state = self.checkpoint_manager.load_checkpoint(
                checkpoint_path=self.raw_resume_from_checkpoint_path,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
            )
            if self.loaded_atomic_checkpoint_state and self.loaded_atomic_checkpoint_state.get("step", 0) > 0:
                loaded_step_for_log = self.loaded_atomic_checkpoint_state.get("step", -1)
                logger.info(
                    f"Rank {self.rank}: Checkpoint loaded. Trainer state applied for step {loaded_step_for_log}."
                )
                loaded_n_passes = self.loaded_atomic_checkpoint_state.get("n_forward_passes_since_fired")
                if loaded_n_passes is not None:
                    self.n_forward_passes_since_fired.data = loaded_n_passes.to(self.device)
                    logger.info(f"Rank {self.rank}: n_forward_passes_since_fired state loaded and applied.")
            else:
                logger.warning(
                    f"Rank {self.rank}: Checkpoint load attempt finished. Trainer state might not be fully restored or starting fresh."
                )

        loaded_wandb_run_id_for_init: Optional[str] = None
        if self.loaded_atomic_checkpoint_state:
            loaded_wandb_run_id_for_init = self.loaded_atomic_checkpoint_state.get("wandb_run_id")
            if loaded_wandb_run_id_for_init:
                logger.info(
                    f"Rank {self.rank}: Found WandB run ID {loaded_wandb_run_id_for_init} in loaded state. Attempting to resume WandB run."
                )

        self.wandb_logger, self.metric_logger = setup_loggers(
            training_config=self.training_config,
            clt_config=self.clt_config,
            log_dir=self.log_dir,
            rank=self.rank,
            distributed=self.distributed,
            world_size=self.world_size,
            resume_wandb_id=loaded_wandb_run_id_for_init,
        )

        self.checkpoint_manager.wandb_logger = self.wandb_logger

        initial_step_for_loop = 0
        if self.loaded_atomic_checkpoint_state:
            initial_step_for_loop = self.loaded_atomic_checkpoint_state.get("step", 0) + 1
            if initial_step_for_loop <= 0:  # Handles if step was 0 or not found
                initial_step_for_loop = 0

        self.training_loop = TrainingLoop(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            activation_store=self.activation_store,
            loss_manager=self.loss_manager,
            clt_config=self.clt_config,
            training_config=self.training_config,
            device=self.device,
            distributed=self.distributed,
            rank=self.rank,
            world_size=self.world_size,
            process_group=self.process_group,
            metric_logger=self.metric_logger,
            initial_step=initial_step_for_loop,
            n_forward_passes_since_fired=self.n_forward_passes_since_fired,
            eval_checkpoint_callback=self._eval_checkpoint_callback,
            use_cuda_amp=self.use_cuda_amp,
            autocast_dtype=self.autocast_dtype,
        )

    @property
    def dead_neurons_mask(self) -> torch.Tensor:
        """Delegates to TrainingLoop's get_dead_neuron_mask."""
        return self.training_loop.get_dead_neuron_mask()

    def _eval_checkpoint_callback(
        self,
        step: int,
        inputs: Dict[int, torch.Tensor],
        targets: Dict[int, torch.Tensor],
        feature_activations_batch: Dict[int, torch.Tensor],
    ) -> Optional[str]:
        """Callback for TrainingLoop to handle evaluation and checkpointing."""
        eval_msg = None
        eval_interval = self.training_config.eval_interval
        checkpoint_interval = self.training_config.checkpoint_interval

        save_checkpoint_flag = (step > 0 and step % checkpoint_interval == 0) or (
            step == self.training_config.training_steps - 1
        )
        run_eval_flag = (step % eval_interval == 0) or (step == self.training_config.training_steps - 1)

        if run_eval_flag:
            if self.distributed and dist.is_initialized():
                dist.barrier(group=self.process_group)  # Ensure all ranks are ready for eval

            # Get current dead mask from the training_loop instance for consistency
            current_dead_mask = self.training_loop.get_dead_neuron_mask().detach().clone()
            eval_metrics = self.evaluator.compute_metrics(
                inputs,
                targets,
                dead_neuron_mask=current_dead_mask,
                feature_activations_batch=feature_activations_batch,
                autocast_dtype=self.autocast_dtype,
                use_cuda_amp=self.use_cuda_amp,
            )

            if not self.distributed or self.rank == 0:
                l0_str = f"AvgL0: {eval_metrics.get('sparsity/avg_l0', 0.0):.2f}"
                ev_str = f"EV: {eval_metrics.get('reconstruction/explained_variance', 0.0):.3f}"
                avg_density_mean = eval_metrics.get("sparsity/feature_density_mean")
                dens_str = f"Dens: {avg_density_mean:.3f}" if avg_density_mean is not None else "Dens: N/A"
                eval_dead_str = f"Dead(Eval): {eval_metrics.get('dead_features/total_eval', 0)}"
                eval_msg = f"{l0_str}, {ev_str}, {dens_str}, {eval_dead_str}"
                self.metric_logger.log_evaluation_metrics(step, eval_metrics)

            if self.distributed and dist.is_initialized():
                dist.barrier(group=self.process_group)  # Ensure all ranks complete eval before next step/checkpoint

        if save_checkpoint_flag:
            self.checkpoint_manager.save_checkpoint(
                step=step,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                n_forward_passes_since_fired=self.n_forward_passes_since_fired,  # Pass the tensor from CLTTrainer
            )
        return eval_msg

    def train(self) -> CrossLayerTranscoder:
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

            if self.training_config.normalization_method == "estimated_mean_std":
                print("\n>>> NORMALIZATION PHASE <<<")
                print("Normalization statistics are being estimated from dataset activations.")
                print("This may take some time, but happens only once before training begins.")
                print(f"Using {self.training_config.normalization_estimation_batches} batches for estimation.\n")

            sys.stdout.flush()
            time.sleep(1)
            print("\n>>> TRAINING PHASE <<<")
            sys.stdout.flush()

        if self.distributed:
            print("\n!!! DIAGNOSTIC INFO !!!")
            print(f"Rank {self.rank}: Process group type: {type(self.process_group)}")
            if dist.is_initialized():  # Check before calling get_backend
                print(f"Rank {self.rank}: Process group backend: {dist.get_backend(self.process_group)}")
            print(f"Rank {self.rank}: RowParallelLinear _reduce does NOT divide by world_size")
            print(f"Rank {self.rank}: Using weight regularization in sparsity penalty")
            print(
                f"Rank {self.rank}: Averaging replicated parameter gradients (handled in TrainingLoop)"
            )  # Updated comment
            store_rank = getattr(self.activation_store, "rank", "N/A")
            store_world = getattr(self.activation_store, "world", "N/A")
            print(f"Rank {self.rank}: Data sharding: rank={store_rank}, world={store_world}")
            print(f"Rank {self.rank}: Batch size tokens: {self.training_config.train_batch_size_tokens}")
            print(f"Rank {self.rank}: Sparsity lambda: {self.training_config.sparsity_lambda}")

            if hasattr(self.activation_store, "__iter__") and hasattr(self.activation_store, "__next__"):
                try:
                    # Try to get a batch for diagnostics, then reset if possible
                    # This part is tricky as not all stores might support easy reset
                    original_store_state = None
                    if hasattr(self.activation_store, "get_state"):  # For stores that can save/load state
                        original_store_state = self.activation_store.get_state()

                    batch_avail = next(iter(self.activation_store), None)
                    print(f"Rank {self.rank}: First batch available (for diagnostics): {batch_avail is not None}")

                    if hasattr(self.activation_store, "reset_iterator"):
                        self.activation_store.reset_iterator()
                    elif original_store_state is not None and hasattr(self.activation_store, "set_state"):
                        self.activation_store.set_state(original_store_state)
                    # Else: cannot easily reset, diagnostic might have consumed a batch

                except Exception as e_diag_batch:
                    print(f"Rank {self.rank}: Could not check first batch for diagnostics: {e_diag_batch}")
            else:
                print(f"Rank {self.rank}: Activation store is not iterable for diagnostic batch check.")

            dummy = torch.ones(1, device=self.device, requires_grad=True)
            dummy_out = dummy * 2
            dummy_out.backward()
            print("!!! END DIAGNOSTIC !!!\n")

        if self.training_config.debug_anomaly:
            torch.autograd.set_detect_anomaly(True)
            if not self.distributed or self.rank == 0:
                logger.info("PyTorch Anomaly Detection ENABLED.")

        # --- Main training execution delegated to TrainingLoop --- #
        trained_model, last_completed_step = self.training_loop.run()

        # --- Final operations after training loop --- #
        if self.distributed and dist.is_initialized():
            dist.barrier(group=self.process_group)

        final_checkpoint_dir = os.path.join(self.log_dir, "final")

        try:
            final_trainer_state_for_checkpoint = {
                "step": last_completed_step,  # Use the actual last completed step from TrainingLoop
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "scaler_state_dict": self.scaler.state_dict() if self.scaler and self.scaler.is_enabled() else None,
                "n_forward_passes_since_fired": (
                    self.n_forward_passes_since_fired.cpu() if self.n_forward_passes_since_fired is not None else None
                ),
                "wandb_run_id": self.wandb_logger.get_current_wandb_run_id(),
                "torch_rng_state": torch.get_rng_state(),
                "numpy_rng_state": np.random.get_state(),
                "python_rng_state": random.getstate(),
            }
            # Use the internal _save_checkpoint which expects the full trainer state dict
            self.checkpoint_manager._save_checkpoint(
                step=last_completed_step,
                trainer_state_to_save=final_trainer_state_for_checkpoint,
            )
        except Exception as e:
            print(f"Rank {self.rank}: Warning: Failed to save final model/trainer state: {e}")

        if not self.distributed or self.rank == 0:
            print(f"Saving final activation store state (handled by CheckpointManager's _save_checkpoint)...")
            os.makedirs(final_checkpoint_dir, exist_ok=True)  # Ensure dir exists for other saves

            # Activation store is saved as part of the checkpoint directory by CheckpointManager on rank 0
            # No need for separate save here unless specifically desired outside the checkpoint structure

            print("Saving final metrics...")
            self.metric_logger._save_metrics_to_disk()

            config_save_path = os.path.join(self.log_dir, "cfg.json")
            print(f"Saving CLT configuration (as trained) to {config_save_path}...")
            try:
                config_dict_as_trained = asdict(self.clt_config)
                if hasattr(self.clt_config, "model_name") and self.clt_config.model_name:
                    config_dict_as_trained["model_name"] = self.clt_config.model_name
                if hasattr(self.training_config, "normalization_method"):
                    config_dict_as_trained["normalization_method"] = self.training_config.normalization_method
                if hasattr(self.training_config, "activation_dtype"):
                    config_dict_as_trained["expected_input_dtype"] = self.training_config.activation_dtype
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

            self.wandb_logger.log_artifact(artifact_path=final_checkpoint_dir, artifact_type="model", name="clt_final")
            self.wandb_logger.finish()
            print(f"Training completed! Final checkpoint saved to {final_checkpoint_dir}")

        if hasattr(self.activation_store, "close") and callable(getattr(self.activation_store, "close")):
            try:
                self.activation_store.close()
            except Exception as e_close:
                print(f"Rank {self.rank}: Error closing activation store at end of training: {e_close}")

        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()

        return trained_model  # Return the model received from TrainingLoop


# Remove parts of the old train method that are now in TrainingLoop or _eval_checkpoint_callback
# - Progress bar (pbar) initialization and updates
# - The main for loop `for step in pbar:`
# - Batch fetching and processing inside the loop
# - Loss computation, backward pass, optimizer step
# - Dead neuron updates inside the loop (now in TrainingLoop, counter passed from CLTTrainer)
# - Metric logging per step (now in TrainingLoop)
# - Evaluation and checkpointing logic within the loop (now in _eval_checkpoint_callback)
# - Explicit tensor deletion inside the loop
# - KeyboardInterrupt handling for the loop (now in TrainingLoop)
# - tqdm.close() (now in TrainingLoop)
