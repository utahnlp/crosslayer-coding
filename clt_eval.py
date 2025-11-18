import argparse
from collections import defaultdict
from pathlib import Path
from typing import Literal, Optional, Dict, Any, List, Dict, Union
import os
import json
import time
import sys
import logging  # Add logging import
from dataclasses import asdict  # Add dataclasses import
import numpy as np  # For numpy RNG state
import random  # For python RNG state

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist  # Import torch.distributed
from torch.distributed import ProcessGroup  # Import ProcessGroup
from tqdm import tqdm  # type: ignore

from clt.config import CLTConfig, TrainingConfig, ActivationConfig
from clt.models.clt import CrossLayerTranscoder
from clt.training.data.base_store import BaseActivationStore

# Import the new manifest-based stores
from clt.training.data.local_activation_store import LocalActivationStore
from clt.training.data.remote_activation_store import RemoteActivationStore

# Import the streaming activation store
from clt.training.data.streaming_activation_store import StreamingActivationStore
from clt.nnsight.extractor import ActivationExtractorCLT

from clt.training.losses import LossManager
from clt.training.evaluator import _format_elapsed_time  # Import the new evaluator
from clt.training.wandb_logger import WandBLogger, DummyWandBLogger
from clt.training.checkpointing import CheckpointManager
from clt.training.distributed_utils import average_shared_parameter_grads  # Add this import
from clt.training.data.activation_store_factory import create_activation_store  # Add this import
from clt.training.metric_utils import MetricLogger  # Add this import
from clt.training.diagnostics import compute_sparsity_diagnostics  # Add this import
from clt.training.profiler import TrainingProfiler, CUDAMemoryProfiler, DistributedProfiler  # Add profiler imports
import datetime as dt  # For distributed init timeout

from transformers import AutoConfig
import transformers  # Import the library itself to check version
import sys  # Import sys to check path
from clt.config import CLTConfig, TrainingConfig, ActivationConfig
from clt.training.trainer import CLTTrainer

# Get logger for this module
logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(
            self,
            clt_config: CLTConfig,
            training_config: TrainingConfig,
            activation_config = None,
            log_dir: Optional[str] = None,
            device: Optional[Union[str, torch.device]] = None,
            distributed: bool = False,  # Add distributed flag
            resume_from_checkpoint_path: Optional[str] = None,  # For resuming
        ):
        
        self.clt_config = clt_config
        self.training_config = training_config
        self.distributed = distributed
        self.loaded_trainer_state: Optional[Dict[str, Any]] = None  # Store loaded state here

        # Initialize distributed training if enabled
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.process_group: Optional[ProcessGroup] = None  # For tensor parallelism

        if self.distributed:
            if not dist.is_initialized():
                # Default backend, consider NCCL for NVIDIA GPUs
                dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", 
                                        timeout=dt.timedelta(hours=2))
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
            # world_size is 1 if not distributed, rank is 0 (already initialized)

        # --- Mixed Precision Setup ---
        self.mixed_precision = self.training_config.precision.lower()  # fp32, fp16, bf16

        self.autocast_enabled = False
        self.autocast_dtype = torch.float32  # Default for fp32

        if self.mixed_precision == "fp16":
            if torch.cuda.is_available():
                self.autocast_enabled = True
                self.autocast_dtype = torch.float16
                logger.info(f"Rank {self.rank}: Enabling CUDA autocast with float16.")
            elif self.device.type == "mps":
                self.autocast_enabled = True
                self.autocast_dtype = torch.float16  # MPS supports float16
                logger.info(f"Rank {self.rank}: Enabling MPS autocast with float16.")
        elif self.mixed_precision == "bf16":
            if torch.cuda.is_available():
                self.autocast_enabled = True
                self.autocast_dtype = torch.bfloat16
                logger.info(f"Rank {self.rank}: Enabling CUDA autocast with bfloat16.")
            elif self.device.type == "mps":
                self.autocast_enabled = True
                self.autocast_dtype = torch.bfloat16  # MPS supports bfloat16
                logger.info(f"Rank {self.rank}: Enabling MPS autocast with bfloat16.")
        # If self.mixed_precision is "fp32", autocast_enabled remains False, autocast_dtype remains float32

        # Set up log directory - only rank 0 creates it
        self.log_dir = log_dir or f"clt_eval_{int(time.time())}"
        if not self.distributed or self.rank == 0:
            os.makedirs(self.log_dir, exist_ok=True)

        # Record start time
        self.start_time = time.time()

        self.model = CrossLayerTranscoder(
            clt_config,
            process_group=self.process_group,
            device=self.device,
            profiler=self.profiler if self.training_config.enable_profiling else None,
        )

        # --- Optionally convert model to FP16 (Step 8) ---
        # If precision is "fp16", GradScaler is used, which expects FP32 optimizer parameters.
        # Therefore, if precision is "fp16", we do not convert model to .half() before optimizer init,
        # regardless of fp16_convert_weights. Autocast handles FP16 computations.
        # fp16_convert_weights will apply if precision is not "fp16" (e.g., "bf16" or "fp32" training
        # where user still wants to store/use fp16 weights directly without GradScaler for fp16).
        if self.training_config.fp16_convert_weights:
            if self.mixed_precision == "fp16":
                logger.warning(
                    f"Rank {self.rank}: 'fp16_convert_weights=True' is set with 'precision=fp16'. "
                    "GradScaler expects FP32 optimizer parameters. Model weights will NOT be converted to FP16 "
                    "before optimizer initialization to ensure GradScaler compatibility. "
                    "Autocast will still use FP16 for computations."
                )
            else:  # mixed_precision is 'bf16' or 'fp32' (autocast_dtype is bfloat16 or float32, GradScaler is disabled for fp16)
                logger.info(
                    f"Rank {self.rank}: Converting model weights and buffers to FP16 (fp16_convert_weights=True, precision={self.mixed_precision})."
                )
                self.model.half()  # Converts all parameters and buffers
                # Keep LayerNorm buffers in FP32 for stability
                for name, buf in self.model.named_buffers():
                    if buf.dtype == torch.float16 and "norm" in name.lower():
                        logger.info(f"Rank {self.rank}: Converting buffer '{name}' from FP16 back to FP32.")
                        buf.data = buf.data.float()

        # Initialize activation store.
        # For tensor parallelism we need every rank to process the *same* batch because
        # activations are sharded **across the feature dimension**, not across tokens.
        # Passing the per-rank/world sharded parameters here would make each rank see a
        # different subset of tokens and lead to inconsistent batch sizes which breaks
        # collective ops such as all_gather in ColumnParallelLinear.

        # Pass the actual rank and world size from the distributed setup
        activation_store_rank = self.rank
        activation_store_world = self.world_size
        if self.distributed and getattr(self.training_config, "activation_source", None) == "streaming":
            self.activation_store = StreamingActivationStore(
                activation_cfg=activation_config,
                activation_extractor=ActivationExtractorCLT(
                    model_name=activation_config.model_name,
                    mlp_input_module_path_template=activation_config.mlp_input_module_path_template,
                    mlp_output_module_path_template=activation_config.mlp_output_module_path_template,
                    device=self.device,
                    model_dtype=getattr(activation_config, "model_dtype", None),
                    context_size=getattr(activation_config, "context_size", None),
                    inference_batch_size=getattr(activation_config, "inference_batch_size", None),
                    exclude_special_tokens=getattr(activation_config, "exclude_special_tokens", None),
                    prepend_bos=getattr(activation_config, "prepend_bos", None),
                    nnsight_tracer_kwargs=getattr(activation_config, "nnsight_tracer_kwargs", None),
                    nnsight_invoker_args=getattr(activation_config, "nnsight_invoker_args", None),
                ),
                train_batch_size_tokens=self.training_config.train_batch_size_tokens,
                device=self.device,
                # Use the configured activation dtype for the *data*;
                # autocast controls compute separately.
                dtype=self.training_config.activation_dtype,
                rank=self.rank,
                world=self.world_size,
                seed=self.training_config.seed or 42,
                sampling_strategy=getattr(self.training_config, "sampling_strategy", "sequential"),
                normalization_method=getattr(self.training_config, "normalization_method", "none"),
                shard_data=False,
            )
        else:
            self.activation_store = create_activation_store(
                training_config=self.training_config,
                clt_config=self.clt_config,
                activation_config=activation_config,
                device=self.device,
                rank=activation_store_rank,  # Pass the actual rank
                world_size=activation_store_world,  # Pass the actual world size
                start_time=self.start_time,
                shard_data=not self.distributed,  # ---> ADDED: False if distributed (TP), True otherwise <----
            )

        # Pass normalisation statistics (if available) so the loss can be computed in
        # the *original* scale even when inputs/targets are stored normalised.
        self.mean_tg = getattr(self.activation_store, "mean_tg", {})  # type: ignore[arg-type]
        self.std_tg = getattr(self.activation_store, "std_tg", {})  # type: ignore[arg-type]
        
        # Validate normalization method
        valid_norm_methods = ["none", "mean_std", "sqrt_d_model"]
        if training_config.normalization_method not in valid_norm_methods:
            raise ValueError(
                f"Invalid normalization_method: {training_config.normalization_method}. "
                f"Must be one of {valid_norm_methods}"
            )
        self.normalization_method = training_config.normalization_method
        self.d_model = clt_config.d_model

        # Initialize dead neuron counters (replicated for now, consider sharding later if needed)
        self.n_forward_passes_since_fired = torch.zeros(
            (clt_config.num_layers, clt_config.num_features),
            device=self.device,
            dtype=torch.long,
            requires_grad=False,  # Explicitly disable gradient tracking
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
            keep_n_checkpoints=self.training_config.keep_n_checkpoints
        )

        # ------------------------------------------------------------------
        #  Load checkpoint (if any) *before* real WandB logger init
        # ------------------------------------------------------------------
        if resume_from_checkpoint_path:
            logger.info(
                f"Rank {self.rank}: Attempting to resume training from checkpoint: {resume_from_checkpoint_path}"
            )
            # load_checkpoint loads model and activation store state internally
            # and returns the trainer_state dictionary (optimizer, scheduler, step, etc.)
            self.loaded_trainer_state = self.checkpoint_manager.load_checkpoint(resume_from_checkpoint_path)
            if self.loaded_trainer_state:
                loaded_step_for_log = self.loaded_trainer_state.get("step", -1)
                logger.info(
                    f"Rank {self.rank}: Successfully loaded checkpoint. Trainer state recovered for step {loaded_step_for_log}."
                )
                # ADDED: Detailed logging of loaded_trainer_state keys and step value
                logger.info(f"Rank {self.rank}: Keys in loaded_trainer_state: {list(self.loaded_trainer_state.keys())}")
                logger.info(
                    f"Rank {self.rank}: Value of 'step' in loaded_trainer_state: {self.loaded_trainer_state.get('step')}"
                )
                logger.info(
                    f"Rank {self.rank}: Value of 'wandb_run_id' in loaded_trainer_state: {self.loaded_trainer_state.get('wandb_run_id')}"
                )
            else:
                logger.warning(
                    f"Rank {self.rank}: Checkpoint loaded, but trainer state (optimizer, step, etc.) was empty or not found. Starting from scratch."
                )
                # loaded_trainer_state will be None or empty, so train() will start fresh


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
        # Detach to prevent any computation graph references
        return (self.n_forward_passes_since_fired > self.training_config.dead_feature_window).detach()



    def evaluate(self):
        
        # Print startup message from rank 0 only
        if not self.distributed or self.rank == 0:
            logger.info(f"Starting CLT training on {self.device}...")
            logger.info(
                f"Model has {self.clt_config.num_features} features per layer "
                f"and {self.clt_config.num_layers} layers"
            )
            logger.info(f"Training for {self.training_config.training_steps} steps.")
            logger.info(f"Logging to {self.log_dir}")
            if self.distributed:
                logger.info(f"Distributed training with {self.world_size} processes (Tensor Parallelism)")

            # Check if using normalization and notify user
            if self.training_config.normalization_method == "mean_std":
                logger.info("\n>>> NORMALIZATION CONFIGURATION <<<")
                logger.info("Using mean/std normalization with pre-calculated statistics from norm_stats.json")
            elif self.training_config.normalization_method == "sqrt_d_model":
                logger.info("\n>>> NORMALIZATION CONFIGURATION <<<")
                logger.info("Using sqrt(d_model) normalization (EleutherAI-style)")
            elif self.training_config.normalization_method == "none":
                logger.info("\n>>> NORMALIZATION CONFIGURATION <<<")
                logger.info("No normalization will be applied to activations")

            # Make sure we flush stdout to ensure prints appear immediately,
            # especially important in Jupyter/interactive environments
            sys.stdout.flush()
            # Wait for 1 second to ensure output is displayed before training starts
            time.sleep(1)
            logger.info("\n>>> TRAINING PHASE <<<")
            sys.stdout.flush()

        # After the existing startup messages
        if self.distributed:
            logger.info("\n!!! DIAGNOSTIC INFO !!!")
            logger.info(f"Rank {self.rank}: Process group type: {type(self.process_group)}")
            logger.info(f"Rank {self.rank}: RowParallelLinear _reduce does NOT divide by world_size")
            logger.info(f"Rank {self.rank}: Using weight regularization in sparsity penalty")
            logger.info(f"Rank {self.rank}: Averaging replicated parameter gradients")
            # Check if activation store has rank/world attributes before accessing
            store_rank = getattr(self.activation_store, "rank", "N/A")
            store_world = getattr(self.activation_store, "world", "N/A")
            logger.info(f"Rank {self.rank}: Data sharding: rank={store_rank}, world={store_world}")
            logger.info(f"Rank {self.rank}: Batch size tokens: {self.training_config.train_batch_size_tokens}")
            logger.info(f"Rank {self.rank}: Sparsity lambda: {self.training_config.sparsity_lambda}")

            # Avoid prefetching a real batch here; it can desync the streaming iterator across ranks.
            if self.activation_store is None:
                logger.warning("Activation store is None (unexpected).")

            # Force torch to compile/execute our code by running a tiny forward/backward pass
            dummy = torch.ones(1, device=self.device, requires_grad=True)
            dummy_out = dummy * 2
            dummy_out.backward()
            logger.info("!!! END DIAGNOSTIC !!!\n")

        # --- Enable Anomaly Detection (if configured) ---
        if self.training_config.debug_anomaly:
            torch.autograd.set_detect_anomaly(True)
            if not self.distributed or self.rank == 0:
                logger.info("PyTorch Anomaly Detection ENABLED.")
    
        # Create progress bar only on rank 0
        pbar: Union[tqdm, range]

        initial_step = 0
        if self.loaded_trainer_state:  # Check if we are resuming and state was loaded
            initial_step = self.loaded_trainer_state.get("step", 0) + 1  # Start from next step
            if initial_step > 0:
                logger.info(f"Rank {self.rank}: Resuming training from step {initial_step}")
                try:
                    self.optimizer.load_state_dict(self.loaded_trainer_state["optimizer_state_dict"])
                    logger.info(f"Rank {self.rank}: Optimizer state loaded.")
                    if self.scheduler and self.loaded_trainer_state.get("scheduler_state_dict"):
                        self.scheduler.load_state_dict(self.loaded_trainer_state["scheduler_state_dict"])
                        logger.info(f"Rank {self.rank}: Scheduler state loaded.")
                    if self.scaler and self.scaler.is_enabled() and self.loaded_trainer_state.get("scaler_state_dict"):
                        self.scaler.load_state_dict(self.loaded_trainer_state["scaler_state_dict"])
                        logger.info(f"Rank {self.rank}: GradScaler state loaded.")

                    loaded_n_passes = self.loaded_trainer_state.get("n_forward_passes_since_fired")
                    if loaded_n_passes is not None:
                        self.n_forward_passes_since_fired.data = loaded_n_passes.to(self.device)
                        logger.info(f"Rank {self.rank}: n_forward_passes_since_fired state loaded.")

                    # Restore RNG states
                    if "torch_rng_state" in self.loaded_trainer_state:
                        torch.set_rng_state(
                            self.loaded_trainer_state["torch_rng_state"].cpu()
                        )  # Ensure it's on CPU before loading
                        logger.info(f"Rank {self.rank}: PyTorch RNG state loaded.")
                    if "numpy_rng_state" in self.loaded_trainer_state:
                        np.random.set_state(self.loaded_trainer_state["numpy_rng_state"])
                        logger.info(f"Rank {self.rank}: NumPy RNG state loaded.")
                    if "python_rng_state" in self.loaded_trainer_state:
                        random.setstate(self.loaded_trainer_state["python_rng_state"])
                        logger.info(f"Rank {self.rank}: Python RNG state loaded.")

                except KeyError as e:
                    logger.error(
                        f"Rank {self.rank}: KeyError when loading trainer state ({e}). Some states might not be restored correctly. Check checkpoint compatibility."
                    )
                except Exception as e:
                    logger.error(
                        f"Rank {self.rank}: Error loading trainer state: {e}. Training might not resume correctly."
                    )
            else:
                logger.info(
                    f"Rank {self.rank}: Loaded step is {initial_step - 1}, starting training from step 0 or continuing if step was 0."
                )
                initial_step = 0  # Ensure we don't start from 1 if loaded step was 0

        pbar = None
        if not self.distributed or self.rank == 0:
            pbar = tqdm()

        results = defaultdict(list)
        while True:
            try:
                inputs, targets = next(self.activation_store)
                inputs = {k: v.detach() for k, v in inputs.items()}
                targets = {k: v.detach() for k, v in targets.items()}
            except StopIteration:
                break

            # logging to diagnose batch size mismatch
            tok_cnt = next(iter(inputs.values())).shape[0]  # number of rows (=tokens) in this batch
            # Only run the all_gather diagnostic when running in distributed mode
            if self.distributed and self.world_size > 1 and dist.is_initialized():
                with self.dist_profiler.profile_op("batch_size_all_gather"):
                    tok_cnt_t = torch.tensor([tok_cnt], device=self.device)
                    gathered = [torch.zeros_like(tok_cnt_t) for _ in range(self.world_size)]
                    dist.all_gather(gathered, tok_cnt_t)

            # --- Check for empty batch --- (Optional but good practice)
            # This check should ideally happen *before* moving data potentially
            if not inputs or not targets or not any(v.numel() > 0 for v in inputs.values()):
                if not self.distributed or self.rank == 0:
                    logger.warning(f"Rank {self.rank}: Warning: Received empty batch at step {step}. Skipping.")
                continue

            # Wrap the entire evaluation logic in no_grad to prevent graph pollution
            with torch.no_grad():
                current_dead_mask = self.dead_neurons_mask.detach().clone()
                # Detach inputs and targets to prevent retaining computation graph
                inputs_detached = {k: v.detach() for k, v in inputs.items()}
                targets_detached = {k: v.detach() for k, v in targets.items()}
                eval_metrics = self.compute_metrics(
                    inputs_detached,  # These inputs are from the current training batch
                    targets_detached,  # These targets are from the current training batch
                    dead_neuron_mask=current_dead_mask,
                )
                self.print_evaluation_report('EVAL', eval_metrics, {})

            exit()
            

            if pbar is not None:
                pbar.update()


    @staticmethod
    def _log_density(density: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """Computes log10 density, adding epsilon for numerical stability."""
        # .detach().cpu() is removed as subsequent processing might need gradients or specific device
        return torch.log10(density + eps)

    @staticmethod
    def _calculate_aggregate_metric(
        per_layer_data: Dict[str, List[float]],
    ) -> Optional[float]:
        """Helper to calculate the mean of a metric across all layers' features."""
        all_values: List[float] = []  # Type hint for clarity
        for layer_key in per_layer_data:
            all_values.extend(per_layer_data[layer_key])
        if not all_values:
            return None
        return float(np.mean(all_values))

    @staticmethod
    def _calculate_aggregate_histogram_data(
        per_layer_data: Dict[str, List[float]],
    ) -> List[float]:
        """Helper to flatten metric data from all layers for an aggregate histogram."""
        all_values: List[float] = []
        for layer_key in per_layer_data:
            all_values.extend(per_layer_data[layer_key])
        return all_values

    @torch.no_grad()
    def compute_metrics(
        self,
        inputs: Dict[int, torch.Tensor],
        targets: Dict[int, torch.Tensor],
        dead_neuron_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Compute all evaluation metrics for the given batch with structured keys.

        Args:
            inputs: Dictionary mapping layer indices to input activations.
            targets: Dictionary mapping layer indices to target activations.
            dead_neuron_mask: Optional mask indicating dead neurons based on trainer's window.

        Returns:
            Dictionary containing computed metrics structured for WandB logging.
            Metrics are organized into 'reconstruction', 'sparsity', 'dead_features',
            'layerwise'.
        """
        mem_before_eval = 0.0
        if torch.cuda.is_available() and self.device.type == "cuda":
            mem_before_eval = torch.cuda.memory_allocated(self.device) / (1024**2)
            elapsed_str = _format_elapsed_time(time.time() - self.start_time)
            logger.debug(f"Eval - Start [{elapsed_str}]. Mem: {mem_before_eval:.2f} MB")

        # Ensure data is on the correct device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        targets = {k: v.to(self.device) for k, v in targets.items()}

        # Get model outputs (reconstructions and feature activations)
        reconstructions = self.model(inputs)
        feature_activations = self.model.get_feature_activations(inputs)

        # --- Compute Metrics ---
        sparsity_metrics = self._compute_sparsity(feature_activations)
        reconstruction_metrics = self._compute_reconstruction_metrics(targets, reconstructions)
        density_metrics = self._compute_feature_density(feature_activations)
        # Compute layerwise dead features based on the provided mask
        dead_neuron_metrics = self._compute_dead_neuron_metrics(dead_neuron_mask)

        # --- Calculate Aggregate Metrics & Histograms ---
        log_feature_density_layerwise = density_metrics.get("layerwise/log_feature_density", {})
        consistent_activation_heuristic_layerwise = density_metrics.get("layerwise/consistent_activation_heuristic", {})

        # Calculate aggregate mean values
        feature_density_mean = self._calculate_aggregate_metric(log_feature_density_layerwise)
        consistent_activation_heuristic_mean = self._calculate_aggregate_metric(
            consistent_activation_heuristic_layerwise
        )

        # Calculate aggregate histogram data (flattened lists)
        log_feature_density_agg_hist_data = self._calculate_aggregate_histogram_data(log_feature_density_layerwise)
        consistent_activation_heuristic_agg_hist_data = self._calculate_aggregate_histogram_data(
            consistent_activation_heuristic_layerwise
        )

        # Add aggregate metrics to sparsity section
        if feature_density_mean is not None:
            sparsity_metrics["sparsity/feature_density_mean"] = feature_density_mean
        if consistent_activation_heuristic_mean is not None:
            sparsity_metrics["sparsity/consistent_activation_heuristic_mean"] = consistent_activation_heuristic_mean
        # Add aggregate histogram data
        if log_feature_density_agg_hist_data:
            sparsity_metrics["sparsity/log_feature_density_agg_hist"] = log_feature_density_agg_hist_data
        if consistent_activation_heuristic_agg_hist_data:
            sparsity_metrics["sparsity/consistent_activation_heuristic_agg_hist"] = (
                consistent_activation_heuristic_agg_hist_data
            )

        # Calculate total dead features from layerwise eval data
        total_dead_eval = sum(dead_neuron_metrics.get("layerwise/dead_features", {}).values())
        dead_neuron_metrics["dead_features/total_eval"] = total_dead_eval

        # --- Combine results into structured dictionary ---
        all_metrics = {
            **reconstruction_metrics,
            **sparsity_metrics,
            **density_metrics,  # Contains the layerwise density/heuristic data
            **dead_neuron_metrics,  # Contains layerwise dead features and total eval dead features
        }
        # Explicitly delete intermediate tensors to potentially free memory sooner
        del reconstructions
        del feature_activations
        # Optionally empty cache, though it has a performance cost
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

        if torch.cuda.is_available() and self.device.type == "cuda":
            mem_after_eval = torch.cuda.memory_allocated(self.device) / (1024**2)
            elapsed_str = _format_elapsed_time(time.time() - self.start_time)
            logger.debug(
                f"Eval - End [{elapsed_str}]. Mem: {mem_after_eval:.2f} MB (+{mem_after_eval - mem_before_eval:.2f} MB)"
            )

        return all_metrics

    def _compute_sparsity(self, activations: Dict[int, torch.Tensor]) -> Dict[str, Any]:
        """Compute L0 sparsity metrics with structured keys.

        Args:
            activations: Dictionary mapping layer indices to feature activations.

        Returns:
            Dictionary with L0 stats under 'sparsity/' and 'layerwise/l0/' keys.
        """
        if not activations or not any(v.numel() > 0 for v in activations.values()):
            if not self.model.world_size > 1 or self.model.rank == 0:
                logger.warning("Warning: Received empty activations for sparsity computation. " "Returning zeros.")
            return {
                "sparsity/total_l0": 0.0,
                "sparsity/avg_l0": 0.0,
                "sparsity/sparsity_fraction": 1.0,  # Renamed from 'sparsity'
                "layerwise/l0": {f"layer_{i}": 0.0 for i in range(self.model.config.num_layers)},
            }

        per_layer_l0_dict = {}
        total_l0 = 0.0
        num_valid_layers = 0

        for layer_idx, layer_activations in activations.items():
            # layer_activations shape: [num_tokens, num_features]
            if layer_activations.numel() == 0 or layer_activations.shape[0] == 0:
                per_layer_l0_dict[f"layer_{layer_idx}"] = 0.0
                continue

            # Count active features per token, then average across tokens
            active_count_per_token = (layer_activations != 0).float().sum(dim=-1)
            avg_active_this_layer = active_count_per_token.mean().item()

            per_layer_l0_dict[f"layer_{layer_idx}"] = avg_active_this_layer
            total_l0 += avg_active_this_layer
            num_valid_layers += 1

        avg_l0 = total_l0 / num_valid_layers if num_valid_layers > 0 else 0.0
        # Use total avg L0 across layers for sparsity fraction calculation
        total_possible_features_per_token = self.model.config.num_features
        sparsity_fraction = (
            1.0 - (avg_l0 / total_possible_features_per_token) if total_possible_features_per_token > 0 else 1.0
        )
        sparsity_fraction = max(0.0, min(1.0, sparsity_fraction))

        return {
            "sparsity/total_l0": total_l0,
            "sparsity/avg_l0": avg_l0,
            "sparsity/sparsity_fraction": sparsity_fraction,  # Renamed for clarity
            "layerwise/l0": per_layer_l0_dict,
        }

    def _compute_reconstruction_metrics(
        self,
        targets: Dict[int, torch.Tensor],
        reconstructions: Dict[int, torch.Tensor],
    ) -> Dict[str, float]:
        """Compute explained variance and MSE with structured keys.

        If normalisation statistics were provided, this method first *de-normalises*
        both the targets and reconstructions before computing metrics.

        Args:
            targets: Dictionary mapping layer indices to target activations.
            reconstructions: Dictionary mapping layer indices to reconstructed activations.

        Returns:
            Dictionary with 'reconstruction/explained_variance' and 'reconstruction/normalized_mean_reconstruction_error'.
        """
        total_explained_variance = 0.0
        total_nmse = 0.0
        num_layers = 0
        
        # For layerwise metrics
        layerwise_nmse = {}
        layerwise_explained_variance = {}

        for layer_idx, target_act in targets.items():
            if layer_idx not in reconstructions:
                continue

            recon_act = reconstructions[layer_idx]

            # --- De-normalise based on normalization method ---
            target_act_denorm = target_act
            recon_act_denorm = recon_act
            
            if self.normalization_method == "mean_std" and layer_idx in self.mean_tg and layer_idx in self.std_tg:
                # Standard denormalization: x * std + mean
                mean = self.mean_tg[layer_idx].to(recon_act.device, recon_act.dtype)
                std = self.std_tg[layer_idx].to(recon_act.device, recon_act.dtype)
                # Ensure broadcast shape
                target_act_denorm = target_act * std + mean
                recon_act_denorm = recon_act * std + mean
            elif self.normalization_method == "sqrt_d_model" and self.d_model is not None:
                # sqrt_d_model denormalization: x / sqrt(d_model)
                sqrt_d_model = (self.d_model ** 0.5)
                target_act_denorm = target_act / sqrt_d_model
                recon_act_denorm = recon_act / sqrt_d_model
            # --- End De-normalisation ---

            # Ensure shapes match (flatten if necessary) and up-cast to float32 for numerically stable metrics
            target_flat = target_act_denorm.view(-1, target_act_denorm.shape[-1]).float()
            recon_flat = recon_act_denorm.view(-1, recon_act_denorm.shape[-1]).float()

            if target_flat.shape != recon_flat.shape or target_flat.numel() == 0:
                continue

            # Calculate MSE (de-normalized)
            mse_layer = F.mse_loss(recon_flat, target_flat, reduction="mean").item()

            # Calculate Explained Variance (EV) - uses de-normalized values
            target_variance_layer = torch.var(target_flat, dim=0, unbiased=False).mean().item()
            error_variance_layer = torch.var(target_flat - recon_flat, dim=0, unbiased=False).mean().item()

            explained_variance_layer = 0.0
            if target_variance_layer > 1e-9:  # Avoid division by zero or near-zero
                explained_variance_layer = 1.0 - (error_variance_layer / target_variance_layer)
            else:
                # If target variance is zero, EV is 1 if error is also zero, else 0 or undefined.
                # Let's be consistent: if target var is ~0, EV is 1 if error var is also ~0, else 0.
                explained_variance_layer = 1.0 if error_variance_layer < 1e-9 else 0.0
            total_explained_variance += explained_variance_layer

            # Calculate NMSE for the layer (de-normalized)
            nmse_layer = 0.0
            if target_variance_layer > 1e-9:
                nmse_layer = mse_layer / target_variance_layer
            elif mse_layer < 1e-9:  # Target variance is zero and MSE is also zero
                nmse_layer = 0.0
            else:  # Target variance is zero but MSE is non-zero (implies error, NMSE is effectively infinite)
                nmse_layer = float("inf")  # Or a large number, or handle as NaN depending on preference
            total_nmse += nmse_layer
            
            # Store layerwise metrics
            layerwise_nmse[f"layer_{layer_idx}"] = nmse_layer
            layerwise_explained_variance[f"layer_{layer_idx}"] = explained_variance_layer

            num_layers += 1

        avg_explained_variance = total_explained_variance / num_layers if num_layers > 0 else 0.0
        avg_normalized_mean_reconstruction_error = total_nmse / num_layers if num_layers > 0 else 0.0

        # Clamp EV between 0 and 1 for robustness
        avg_explained_variance = max(0.0, min(1.0, avg_explained_variance))

        # avg_normalized_mean_reconstruction_error can be inf, handle this if it needs to be bounded or logged carefully.
        # For now, log as is.

        return {
            "reconstruction/explained_variance": avg_explained_variance,
            "reconstruction/normalized_mean_reconstruction_error": avg_normalized_mean_reconstruction_error,
            "layerwise/normalized_mse": layerwise_nmse,
            "layerwise/explained_variance": layerwise_explained_variance,
        }

    def _compute_feature_density(self, activations: Dict[int, torch.Tensor]) -> Dict[str, Any]:
        """Compute feature density metrics with structured keys for layerwise data.

        Args:
            activations: Dictionary mapping layer indices to feature activations.

        Returns:
            Dictionary containing per-layer dictionaries under
            'layerwise/log_feature_density' and 'layerwise/consistent_activation_heuristic'.
        """
        if not activations or not any(v.numel() > 0 for v in activations.values()):
            return {
                "layerwise/log_feature_density": {},
                "layerwise/consistent_activation_heuristic": {},
            }

        per_layer_log_density: Dict[str, list[float]] = {}
        per_layer_heuristic: Dict[str, list[float]] = {}

        for layer_idx, layer_activations in activations.items():
            # layer_activations: [batch_tokens, num_features]
            if layer_activations.numel() == 0:
                continue

            num_tokens, num_features = layer_activations.shape
            if num_tokens == 0:
                continue

            # Use a small threshold for numerical stability
            act_bool = (layer_activations > 1e-6).float()

            # Feature Density: Fraction of tokens each feature is active for
            # Shape: [num_features]
            feature_density_tensor = act_bool.mean(dim=0)
            # Apply log10 transformation
            log_feature_density_tensor = Evaluator._log_density(feature_density_tensor)
            log_feature_density_this_layer = log_feature_density_tensor.tolist()
            per_layer_log_density[f"layer_{layer_idx}"] = log_feature_density_this_layer

            # Consistent Activation Heuristic:
            # [num_features], number of tokens each feature fired for
            tokens_feature_active = act_bool.sum(dim=0)  # [num_features]

            # Calculate heuristic per feature: total activations / num prompts active
            # Add small epsilon to denominator to avoid division by zero
            # Use act_bool.any(dim=0) instead of (tokens_feature_active > 0).float() for clarity
            prompts_feature_active_mask = act_bool.any(dim=0)  # Check if feature fired at least once
            denominator = prompts_feature_active_mask.float() + 1e-9  # [num_features]
            heuristic_this_layer = (tokens_feature_active / denominator).tolist()
            per_layer_heuristic[f"layer_{layer_idx}"] = heuristic_this_layer

        # Return per-layer dictionaries containing lists of per-feature values
        return {
            "layerwise/log_feature_density": per_layer_log_density,
            "layerwise/consistent_activation_heuristic": per_layer_heuristic,
        }

    def _compute_dead_neuron_metrics(self, dead_neuron_mask: Optional[torch.Tensor]) -> Dict[str, Any]:
        """Compute layerwise dead neuron metrics based on the provided mask.

        Args:
            dead_neuron_mask: Optional mask indicating dead neurons.

        Returns:
            Dictionary with 'layerwise/dead_features/layer_{i}'.
            Total count is calculated later in compute_metrics.
        """
        dead_neuron_metrics: Dict[str, Any] = {
            # "dead_features/total": 0, # Total calculated later
            "layerwise/dead_features": {},
        }
        if dead_neuron_mask is not None:
            # Ensure mask is on the correct device
            dead_neuron_mask = dead_neuron_mask.to(self.device)
            # Validate mask shape matches model config
            expected_shape = (
                self.model.config.num_layers,
                self.model.config.num_features,
            )
            if dead_neuron_mask.shape == expected_shape:
                # total_dead = dead_neuron_mask.sum().item() # No longer needed here
                # dead_neuron_metrics["dead_features/total"] = total_dead
                per_layer_dead_dict = {}
                for layer_idx in range(dead_neuron_mask.shape[0]):
                    per_layer_dead_dict[f"layer_{layer_idx}"] = dead_neuron_mask[layer_idx].sum().item()
                dead_neuron_metrics["layerwise/dead_features"] = per_layer_dead_dict
            else:
                if not self.model.world_size > 1 or self.model.rank == 0:
                    logger.warning(
                        f"Warning: Received dead_neuron_mask with unexpected shape {dead_neuron_mask.shape}. Expected {expected_shape}. Skipping dead neuron eval metrics."
                    )
        return dead_neuron_metrics

    def print_evaluation_report(
        self,
        step: int,
        metrics: Dict[str, Any],
        detailed_metrics: Dict[str, Any],
        current_training_config: Optional[TrainingConfig] = None,
        current_clt_config: Optional[CLTConfig] = None,
    ):
        if not self.model.world_size > 1 or self.model.rank == 0:
            logger.info(
                "\n======================================================================="
                "\n--- Model Evaluation Report ---"
            )
            logger.info(f"Evaluation at Step: {step}")
            logger.info(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("--- Overall Performance ---")
            logger.info(f"  Total Reconstruction Loss: {metrics.get('reconstruction/total_loss', float('nan')):.4f}")
            logger.info(f"  Total Sparsity Loss: {metrics.get('sparsity/total_loss', float('nan')):.4f}")
            logger.info(
                f"  Explained Variance (Avg): {metrics.get('reconstruction/explained_variance', float('nan')):.4f}"
            )
            logger.info(f"  NMSE (Avg): {metrics.get('reconstruction/nmse', float('nan')):.4f}")
            logger.info(f"  L0 Norm (Avg per token): {metrics.get('sparsity/avg_l0', float('nan')):.2f}")
            logger.info(f"  Sparsity Fraction: {metrics.get('sparsity/sparsity_fraction', float('nan')):.4f}")

            # Layer-wise details
            if metrics.get("layerwise/reconstruction_loss"):
                logger.info("--- Layer-wise Reconstruction Loss ---")
                for layer, loss in metrics["layerwise/reconstruction_loss"].items():
                    logger.info(f"  {layer}: {loss:.4f}")

            if metrics.get("layerwise/explained_variance"):
                logger.info("--- Layer-wise Explained Variance ---")
                for layer, ev in metrics["layerwise/explained_variance"].items():
                    logger.info(f"  {layer}: {ev:.4f}")

            if metrics.get("layerwise/nmse"):
                logger.info("--- Layer-wise NMSE ---")
                for layer, nmse_val in metrics["layerwise/nmse"].items():
                    logger.info(f"  {layer}: {nmse_val:.4f}")

            if metrics.get("layerwise/l0"):
                logger.info("--- Layer-wise L0 Norm (Avg per token) ---")
                for layer, l0 in metrics["layerwise/l0"].items():
                    logger.info(f"  {layer}: {l0:.2f}")

            # Feature density details
            if detailed_metrics.get("feature_density_per_layer"):
                logger.info("Feature Density Per Layer:")
                for layer, density in detailed_metrics["feature_density_per_layer"].items():
                    logger.info(f"  Layer {layer}: {density:.4f}")

            # Dead features details
            if detailed_metrics.get("dead_features_per_layer_eval"):
                logger.info("Dead Features Per Layer (Evaluation Batch):")
                for layer, count in detailed_metrics["dead_features_per_layer_eval"].items():
                    logger.info(f"  Layer {layer}: {count}")

            # Active features details
            if detailed_metrics.get("active_features_per_layer_eval"):
                logger.info("Active Features Per Layer (Evaluation Batch):")
                for layer, count in detailed_metrics["active_features_per_layer_eval"].items():
                    logger.info(f"  Layer {layer}: {count}")

            # Overall dead/active features
            logger.info(f"Total Dead Features (Eval Batch): {detailed_metrics.get('dead_features/total_eval', 0)}")
            logger.info(f"Total Active Features (Eval Batch): {detailed_metrics.get('active_features/total_eval', 0)}")

            if current_training_config:
                logger.info("--- Training Configuration ---")
                logger.info(f"  Learning Rate: {current_training_config.learning_rate}")
                logger.info(f"  Sparsity Lambda: {current_training_config.sparsity_lambda}")
                # Add other relevant training config details

            if current_clt_config:
                logger.info("--- CLT Model Configuration ---")
                logger.info(f"  Activation Function: {current_clt_config.activation_fn}")
                logger.info(f"  Number of Features: {current_clt_config.num_features}")
                # Add other relevant CLT config details
            logger.info("=======================================================================\n")




# Setup basic logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_model_dimensions(model_name: str) -> tuple[Optional[int], Optional[int]]:
    """Attempt to dynamically get num_layers and d_model from model_name."""
    if AutoConfig is None:
        logger.warning(
            "Transformers library not found. Cannot dynamically detect model dimensions."
            " Falling back to gpt2 defaults (12 layers, 768 hidden size) if not otherwise specified."
            " Install transformers (`pip install transformers`) for auto-detection."
        )
        return None, None  # Indicate failure to auto-detect

    try:
        if transformers and hasattr(transformers, "__version__"):
            logger.info(f"Transformers library version: {transformers.__version__}")
        if sys:
            logger.info(f"Python sys.path: {sys.path}")

        logger.info(f"Attempting to load config for model_name: '{model_name}'")
        config = AutoConfig.from_pretrained(model_name)
        logger.info(f"Loaded config object: type={type(config)}")
        if hasattr(config, "to_dict"):
            # Log only a few key attributes to avoid excessively long log messages
            # if the config is huge. Relevant ones might be 'model_type', 'architectures'.
            config_dict_summary = {
                k: v
                for k, v in config.to_dict().items()
                if k in ["model_type", "architectures", "num_hidden_layers", "n_layer", "hidden_size", "n_embd"]
            }
            logger.info(f"Config content summary: {config_dict_summary}")
            # If still debugging, can log the full dict, but be wary of verbosity:
            # logger.debug(f"Full config content: {config.to_dict()}")
        elif hasattr(config, "__dict__"):
            logger.info(f"Config content (vars): {vars(config)}")
        else:
            logger.info(f"Config object does not have to_dict or __dict__ methods. Content: {config}")

        num_layers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", None)
        d_model = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)
        logger.info(f"Attempted to get dimensions: num_layers={num_layers}, d_model={d_model}")
        if num_layers is None or d_model is None:
            logger.warning(
                f"Could not automatically determine num_layers or d_model for {model_name}. "
                "Will rely on defaults or error out if not sufficient."
            )
            return None, None
        logger.info(f"Detected model dimensions for {model_name}: {num_layers} layers, {d_model} hidden size.")
        return num_layers, d_model
    except Exception as e:
        logger.warning(
            f"Failed to get model dimensions for {model_name}: {e}. "
            "Will rely on defaults or error out if not sufficient."
        )
        return None, None


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a Cross-Layer Transcoder (CLT) from local or remote activations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Core Training Parameters ---
    core_group = parser.add_argument_group("Core Training Parameters")
    core_group.add_argument(
        "--activation-source",
        type=str,
        choices=["local_manifest", "remote", "streaming"],
        required=True,
        help="Source of activations: 'local_manifest' or 'remote' server.",
    )
    core_group.add_argument(
        "--output-dir",
        type=str,
        default=f"clt_train_{int(time.time())}",
        help="Directory to save logs, checkpoints, and final model. If resuming, this might be overridden by --resume_from_checkpoint_dir.",
    )
    core_group.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Base model name or path (e.g., 'gpt2', 'EleutherAI/pythia-70m'). Used for activation generation context and CLT dimension inference.",
    )
    core_group.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cuda', 'cpu', 'mps'). Auto-detected if None.",
    )
    core_group.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed training (requires torchrun/appropriate launcher).",
    )
    core_group.add_argument(
        "--resume_from_checkpoint_dir",
        type=str,
        default=None,
        help="Path to the output directory of a previous run to resume from. Will attempt to load 'latest' or a specific step if --resume_step is also given.",
    )
    core_group.add_argument(
        "--resume_step",
        type=int,
        default=None,
        help="Optional specific step to resume from. Used in conjunction with --resume_from_checkpoint_dir.",
    )

    # --- Local Activation Source Parameters ---
    local_group = parser.add_argument_group(
        "Local Activation Source Parameters (if --activation-source=local_manifest)"
    )
    local_group.add_argument(
        "--activation-path",
        type=str,
        default=None,  # Required if local_manifest, checked in main
        help="Path to the directory containing pre-generated activations (e.g., .../index.bin, metadata.json).",
    )

    # --- Remote Activation Server Parameters ---
    remote_group = parser.add_argument_group("Remote Activation Server Parameters (if --activation-source=remote)")
    remote_group.add_argument(
        "--server-url",
        type=str,
        default=None,  # Required if remote, checked in main
        help="URL of the remote activation storage server (e.g., 'http://localhost:8000').",
    )
    remote_group.add_argument(
        "--dataset-id",
        type=str,
        default=None,  # Required if remote, checked in main
        help="Unique identifier for the dataset on the remote server (e.g., 'gpt2/pile-10k_train').",
    )
    remote_group.add_argument(
        "--remote-timeout",
        type=int,
        default=120,
        help="Timeout in seconds for fetching batches from the remote server.",
    )
    remote_group.add_argument(
        "--remote-max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed batch fetch requests.",
    )
    remote_group.add_argument(
        "--remote-prefetch-batches",
        type=int,
        default=16,  # Default from train_clt_remote
        help="Number of batches to prefetch from the server.",
    )

    # --- CLT Model Architecture (CLTConfig) ---
    clt_group = parser.add_argument_group("CLT Model Architecture (CLTConfig)")
    clt_group.add_argument(
        "--num-features",
        type=int,
        required=True,
        help="Number of features per layer in the CLT.",
    )
    # num_layers and d_model are derived from the base model if not explicitly set
    clt_group.add_argument(
        "--activation-fn",
        type=str,
        choices=["jumprelu", "relu", "batchtopk", "topk"],
        default="jumprelu",
        help="Activation function for the CLT.",
    )
    clt_group.add_argument(
        "--jumprelu-threshold",
        type=float,
        default=0.03,
        help="Threshold for JumpReLU activation (if used).",
    )
    clt_group.add_argument(
        "--batchtopk-k",
        type=int,
        default=None,
        help="Absolute k for BatchTopK activation (if used). Only one of k or frac.",
    )
    clt_group.add_argument(
        "--batchtopk-frac",
        type=float,
        default=None,
        help="Fraction of features to keep for BatchTopK (if used). Only one of k or frac.",
    )
    clt_group.add_argument(
        "--disable-batchtopk-straight-through",
        action="store_true",  # If flag is present, disable is true. Default behavior is enabled.
        help="Disable straight-through estimator for BatchTopK. (BatchTopK default is True).",
    )
    clt_group.add_argument(
        "--topk-mode",
        type=str,
        choices=["global", "per_layer"],
        default="global",
        help="How to apply top-k selection: 'global' (across all layers) or 'per_layer' (each layer independently).",
    )
    clt_group.add_argument(
        "--topk-k",
        type=float,  # As per CLTConfig, topk_k can be a float (fraction) or int (count)
        default=None,
        help="Number or fraction of features to keep for TopK activation (if used). If < 1, treated as fraction; if >= 1, treated as int count.",
    )
    clt_group.add_argument(
        "--disable-topk-straight-through",
        action="store_true",
        help="Disable straight-through estimator for TopK. (TopK default is True).",
    )
    clt_group.add_argument(
        "--clt-dtype",
        type=str,
        default=None,
        help="Optional data type for the CLT model parameters (e.g., 'float16', 'bfloat16').",
    )
    clt_group.add_argument(
        "--decoder-tying",
        type=str,
        choices=["none", "per_source", "per_target"],
        default="none",
        help="Decoder weight sharing strategy: 'none' (default), 'per_source' (tied per source layer), or 'per_target' (tied per target layer, EleutherAI style).",
    )
    clt_group.add_argument(
        "--enable-feature-offset",
        action="store_true",
        help="Enable per-feature bias (theta_bias) applied after encoding.",
    )
    clt_group.add_argument(
        "--enable-feature-scale",
        action="store_true",
        help="Enable per-feature scale (theta_scale) applied after encoding.",
    )
    clt_group.add_argument(
        "--skip-connection",
        action="store_true",
        help="Enable skip connection from input to output.",
    )

    # --- Training Hyperparameters (TrainingConfig) ---
    train_group = parser.add_argument_group("Training Hyperparameters (TrainingConfig)")
    train_group.add_argument("--learning-rate", type=float, default=3e-4, help="Optimizer learning rate.")
    train_group.add_argument(
        "--training-steps",
        type=int,
        default=50000,
        help="Total number of training steps.",
    )
    train_group.add_argument(
        "--train-batch-size-tokens",
        type=int,
        default=4096,
        help="Target number of tokens per training batch.",
    )
    train_group.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp32",  # Default to fp32 as per TrainingConfig
        help="Training precision: 'fp32', 'fp16' (mixed precision with AMP), or 'bf16' (mixed precision with AMP).",
    )
    train_group.add_argument(
        "--fp16-convert-weights",
        action="store_true",
        help="If --precision is fp16, also convert model weights to fp16. Saves memory but model parameters remain fp32 by default with AMP. Default is False.",
    )
    train_group.add_argument(
        "--debug-anomaly",
        action="store_true",
        help="Enable PyTorch autograd anomaly detection for debugging NaN issues. Default is False.",
    )
    train_group.add_argument(
        "--normalization-method",
        type=str,
        choices=["none", "mean_std", "sqrt_d_model"],
        default="mean_std",
        help=(
            "Normalization method for activations. "
            "'none': No normalization. "
            "'mean_std': Standard (x - mean) / std normalization using pre-calculated stats. "
            "'sqrt_d_model': EleutherAI-style x * sqrt(d_model) normalization."
        ),
    )
    train_group.add_argument(
        "--sparsity-lambda",
        type=float,
        default=1e-3,
        help="Coefficient for the L1 sparsity penalty.",
    )
    train_group.add_argument(
        "--sparsity-lambda-schedule",
        type=str,
        choices=["linear", "delayed_linear"],
        default="linear",
        help="Schedule for applying sparsity lambda.",
    )
    train_group.add_argument(
        "--sparsity-lambda-delay-frac",
        type=float,
        default=0.1,
        help="Fraction of steps to delay lambda increase for 'delayed_linear' schedule.",
    )
    train_group.add_argument(
        "--sparsity-c",
        type=float,
        default=1.0,
        help="Constant shaping the sparsity penalty (typically 1.0).",
    )
    train_group.add_argument(
        "--preactivation-coef",
        type=float,
        default=3e-6,
        help="Coefficient for the pre-activation MSE loss term.",
    )
    train_group.add_argument(
        "--aux-loss-factor",
        type=float,
        default=None,
        help="Coefficient for the auxiliary reconstruction loss (e.g., for dead latents). If None, loss is not applied.",
    )
    train_group.add_argument(
        "--apply-sparsity-penalty-to-batchtopk",
        action=argparse.BooleanOptionalAction,  # Allows --apply-sparsity-penalty-to-batchtopk or --no-apply-sparsity-penalty-to-batchtopk
        default=True,  # Matches TrainingConfig default
        help="Apply standard L1 sparsity penalty to BatchTopK activations. Default is True. Use --no-apply-sparsity-penalty-to-batchtopk to disable.",
    )
    train_group.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "adamw"],
        default="adamw",
        help="Optimizer algorithm.",
    )
    train_group.add_argument(
        "--optimizer-beta1",
        type=float,
        default=None,
        help="Optimizer beta1 value (if using Adam/AdamW).",
    )
    train_group.add_argument(
        "--optimizer-beta2",
        type=float,
        default=None,
        help="Optimizer beta2 value (if using Adam/AdamW).",
    )
    train_group.add_argument(
        "--lr-scheduler",
        type=str,
        choices=["linear", "cosine", "linear_final20", "none"],
        default="linear",
        help=(
            "Learning rate scheduler type. 'linear_final20' keeps LR constant until the last 20% "
            "of steps then decays linearly to 0 ('none' to disable)."
        ),
    )
    train_group.add_argument(
        "--gradient-clip-val",
        type=float,
        default=None,
        help="Value for gradient clipping. If None, no clipping.",
    )
    train_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    train_group.add_argument(
        "--activation-dtype",
        type=str,
        default="float32",  # Consistent default
        help="Data type to process/load activations as (e.g., 'float32', 'bfloat16').",
    )
    train_group.add_argument(
        "--dead-feature-window",
        type=int,
        default=1000,
        help="Number of steps of inactivity before a feature is considered 'dead' for evaluation.",
    )
    train_group.add_argument(
        "--compute-sparsity-diagnostics",
        action="store_true",
        help="Enable computation of detailed sparsity diagnostics during evaluation.",
    )
    train_group.add_argument(
        "--enable-profiling",
        action="store_true",
        help="Enable detailed performance profiling to identify bottlenecks.",
    )

    # --- Sampling Strategy ---
    sampling_group = parser.add_argument_group("Sampling Strategy (TrainingConfig)")
    sampling_group.add_argument(
        "--sampling-strategy",
        type=str,
        choices=["sequential", "random_chunk"],
        default="sequential",
        help="Sampling strategy for manifest-based stores: 'sequential' or 'random_chunk'.",
    )

    # --- Logging & Checkpointing ---
    log_group = parser.add_argument_group("Logging & Checkpointing (TrainingConfig)")
    log_group.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Log training metrics every N steps.",
    )
    log_group.add_argument(
        "--eval-interval",
        type=int,
        default=1000,
        help="Run evaluation metrics computation every N steps.",
    )
    log_group.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Save a training checkpoint every N steps.",
    )
    log_group.add_argument(
        "--keep-n-checkpoints",
        type=int,
        default=3,
        help="How many checkpoints to keep, deleting oldest first",
    )
    log_group.add_argument("--enable-wandb", action="store_true", help="Enable Weights & Biases logging.")
    log_group.add_argument("--wandb-project", type=str, default=None, help="WandB project name.")
    log_group.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB entity (username or team).",
    )
    log_group.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Custom name for the WandB run. Auto-generated if None.",
    )
    log_group.add_argument("--wandb-tags", nargs="+", default=None, help="List of tags for the WandB run.")

    # --- Streaming ---
    streaming_group = parser.add_argument_group("Streaming")
    streaming_group.add_argument(
        "--mlp-input-template",
        type=str,
        required=True,
        help="NNsight path template for MLP inputs.",
    )
    streaming_group.add_argument(
        "--mlp-output-template",
        type=str,
        required=True,
        help="NNsight path template for MLP outputs.",
    )
    streaming_group.add_argument(
        "--model-dtype",
        type=str,
        default=None,
        help="Optional model dtype (e.g., 'float16').",
    )
    streaming_group.add_argument("--dataset-path", type=str, required=True, help="Dataset name or path.")
    streaming_group.add_argument("--dataset-split", type=str, default="train", help="Dataset split.")
    streaming_group.add_argument(
        "--dataset-text-column",
        type=str,
        default="text",
        help="Dataset text column name.",
    )
    streaming_group.add_argument(
        "--context-size",
        type=int,
        default=128,
        help="Context size for tokenization/inference.",
    )
    streaming_group.add_argument("--inference-batch-size", type=int, default=512, help="Inference batch size.")
    streaming_group.add_argument(
        "--exclude-special-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude special tokens.",
    )
    streaming_group.add_argument(
        "--prepend-bos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Prepend BOS token.",
    )


    args = parser.parse_args()

    # --- Validate conditional arguments ---
    if args.activation_source == "remote":
        if not args.server_url:
            parser.error("--server-url is required when --activation-source is 'remote'")
        if not args.dataset_id:
            parser.error("--dataset-id is required when --activation-source is 'remote'")
    elif args.activation_source == "local_manifest":
        if not args.activation_path:
            # Allow activation_path to be None if resuming, as it will be loaded from cli_args.json
            if not args.resume_from_checkpoint_dir:
                parser.error(
                    "--activation-path is required when --activation-source is 'local_manifest' and not resuming."
                )

    return args

def main():
    """Main function to configure and run the CLTTrainer."""
    args = parse_args()

    output_dir_for_trainer_str = args.output_dir
    actual_checkpoint_path_to_load: Optional[str] = None
    resuming_run = False

    if args.resume_from_checkpoint_dir:
        resuming_run = True
        resume_base_dir = Path(args.resume_from_checkpoint_dir)
        logger.info(f"Attempting to resume training from directory: {resume_base_dir}")

        # Override output_dir to be the resume directory
        output_dir_for_trainer_str = str(resume_base_dir.resolve())

        # Load original CLI args from the run being resumed
        original_cli_args_path = resume_base_dir / "cli_args.json"
        if original_cli_args_path.exists():
            logger.info(f"Loading original CLI arguments from {original_cli_args_path}")
            with open(original_cli_args_path, "r") as f:
                original_cli_vars = json.load(f)

            # Create a new argparse.Namespace from the loaded dict
            # Update this new namespace with the original args, then override with any current CLI args
            # that are relevant for resuming (like resume_step, or if user wants to change e.g. training_steps for the resumed run)

            # Start with current args (which include resume_from_checkpoint_dir, resume_step)
            # Then load original args, but current resume-specific args should take precedence if they were specified.
            # Also, things like output_dir might change if we allow resuming to a NEW directory (not supported yet, logs to original)

            current_args_dict = vars(args).copy()
            # args_from_file = argparse.Namespace(**original_cli_vars) # Unused / Can be removed

            # Update args_from_file with any overriding CLI args from current invocation
            # For most params, we want the original run's params. But some (e.g. training_steps) user might want to extend.
            # For now, let's prioritize original CLI args for most things, except for resume flags and potentially output_dir.

            # Convert original_cli_vars to Namespace and then update it with relevant current args.
            # The `args` variable will be rebuilt from original_cli_vars, with care for resume flags.
            temp_args_dict = original_cli_vars.copy()

            # Keep current resume flags and potentially new output_dir if we decide to support it
            # For now, output_dir is forced to be the resume_from_checkpoint_dir
            temp_args_dict["resume_from_checkpoint_dir"] = args.resume_from_checkpoint_dir
            temp_args_dict["resume_step"] = args.resume_step
            temp_args_dict["output_dir"] = (
                output_dir_for_trainer_str  # Ensure output_dir is the one we are resuming into
            )
            # If user wants to override training_steps for a resumed run, they can pass it.
            if current_args_dict.get("training_steps") != original_cli_vars.get("training_steps"):
                logger.info(
                    f"Overriding training_steps from {original_cli_vars.get('training_steps')} to {current_args_dict.get('training_steps')}"
                )
                temp_args_dict["training_steps"] = current_args_dict.get("training_steps")
            # Potentially other overridable args like learning_rate, wandb settings etc.

            args = argparse.Namespace(**temp_args_dict)  # Re-assign args with merged values
            logger.info(f"Effective arguments for resumed run: {vars(args)}")

        else:
            logger.warning(
                f"Original cli_args.json not found at {original_cli_args_path}. "
                f"Configuration will be based on the currently provided command-line arguments. "
                f"Ensure all necessary configuration parameters are supplied."
            )
            # In this case, `args` remains as parsed from the current command line, which is desired.

        # Determine the specific checkpoint path to load (model file or distributed dir)
        # This logic assumes CLTTrainer's load_checkpoint handles whether it's a file or dir based on distributed status
        if args.distributed:
            if args.resume_step is not None:
                actual_checkpoint_path_to_load = str(resume_base_dir / f"step_{args.resume_step}")
            else:
                actual_checkpoint_path_to_load = str(resume_base_dir / "latest")
        else:  # Non-distributed
            if args.resume_step is not None:
                actual_checkpoint_path_to_load = str(resume_base_dir / f"clt_checkpoint_{args.resume_step}.safetensors")
            else:
                actual_checkpoint_path_to_load = str(resume_base_dir / "clt_checkpoint_latest.safetensors")

        if not Path(actual_checkpoint_path_to_load).exists():
            logger.error(f"Checkpoint to load does not exist: {actual_checkpoint_path_to_load}")
            if args.distributed and (
                actual_checkpoint_path_to_load.endswith("latest")
                or actual_checkpoint_path_to_load.endswith(f"step_{args.resume_step}")
            ):
                logger.error("For distributed runs, ensure the directory exists.")
            elif not args.distributed and actual_checkpoint_path_to_load.endswith(".safetensors"):
                logger.error("For non-distributed runs, ensure the .safetensors file exists.")
            return

        logger.info(f"Will attempt to load checkpoint state from: {actual_checkpoint_path_to_load}")

    # --- Setup Output Directory (now based on output_dir_for_trainer_str) ---
    output_dir_path = Path(output_dir_for_trainer_str)
    output_dir_path.mkdir(exist_ok=True, parents=True)
    logger.info(f"Using output directory: {output_dir_path.resolve()}")

    # --- Determine Device ---
    if args.device:
        device_str = args.device
    else:
        if torch.cuda.is_available():
            device_str = "cuda"
        elif torch.backends.mps.is_available():  # For Apple Silicon
            device_str = "mps"
        else:
            device_str = "cpu"
    logger.info(f"Using device: {device_str}")
    # Trainer will handle torch.device object creation

    # --- Determine Base Model Dimensions ---
    base_model_name = args.model_name
    num_layers_auto, d_model_auto = get_model_dimensions(base_model_name)
    if num_layers_auto is None or d_model_auto is None:
        # This case implies get_model_dimensions failed or returned Nones.
        # CLTConfig requires num_layers and d_model.
        # If they couldn't be auto-detected, it's a fatal error.
        logger.error(
            f"Could not determine dimensions (num_layers, d_model) for model '{base_model_name}'. "
            "These are required for CLTConfig. Please ensure the model name is correct and visible "
            "to the Hugging Face AutoConfig, or that the CLT library can derive them."
        )
        return  # Exit if dimensions are critical and not found

    # --- Create CLT Configuration ---
    clt_config = CLTConfig(
        num_features=args.num_features,
        num_layers=num_layers_auto,  # d_model and num_layers are now from auto-detection
        d_model=d_model_auto,
        model_name=base_model_name,  # Store for reference
        activation_fn=args.activation_fn,
        jumprelu_threshold=args.jumprelu_threshold,
        batchtopk_k=args.batchtopk_k,
        batchtopk_straight_through=(not args.disable_batchtopk_straight_through),
        clt_dtype=args.clt_dtype,
        topk_k=args.topk_k,
        topk_straight_through=(not args.disable_topk_straight_through),
        decoder_tying=args.decoder_tying,
        enable_feature_offset=args.enable_feature_offset,
        enable_feature_scale=args.enable_feature_scale,
        skip_connection=args.skip_connection,
        topk_mode=args.topk_mode,
    )
    logger.info(f"CLT Config: {clt_config}")

    # --- Create Training Configuration ---
    lr_scheduler_arg: Optional[Literal["linear", "cosine", "linear_final20"]] = (
        args.lr_scheduler if args.lr_scheduler != "none" else None
    )

    activation_path_arg: Optional[str] = None
    remote_config_dict: Optional[Dict[str, Any]] = None

    if args.activation_source == "local_manifest":
        activation_path_arg = args.activation_path
        logger.info(f"Using local activation source: {activation_path_arg}")
    elif args.activation_source == "remote":
        remote_config_dict = {
            "server_url": args.server_url,
            "dataset_id": args.dataset_id,
            "timeout": args.remote_timeout,
            "max_retries": args.remote_max_retries,
            "prefetch_batches": args.remote_prefetch_batches,
        }
        logger.info(f"Using remote activation source: {args.server_url}, dataset: {args.dataset_id}")

    training_config = TrainingConfig(
        # Core Training
        learning_rate=args.learning_rate,
        training_steps=args.training_steps,
        seed=args.seed,
        gradient_clip_val=args.gradient_clip_val,
        train_batch_size_tokens=args.train_batch_size_tokens,
        # Activation Source
        activation_source=args.activation_source,  # Directly from args
        activation_path=activation_path_arg,  # Populated if local
        remote_config=remote_config_dict,  # Populated if remote
        activation_dtype=args.activation_dtype,
        # Normalization
        normalization_method=args.normalization_method,
        # Sampling Strategy
        sampling_strategy=args.sampling_strategy,
        # Loss Coeffs
        sparsity_lambda=args.sparsity_lambda,
        sparsity_lambda_schedule=args.sparsity_lambda_schedule,
        sparsity_lambda_delay_frac=args.sparsity_lambda_delay_frac,
        sparsity_c=args.sparsity_c,
        preactivation_coef=args.preactivation_coef,
        aux_loss_factor=args.aux_loss_factor,
        apply_sparsity_penalty_to_batchtopk=args.apply_sparsity_penalty_to_batchtopk,
        # Optimizer & Scheduler
        optimizer=args.optimizer,
        optimizer_beta1=args.optimizer_beta1,
        optimizer_beta2=args.optimizer_beta2,
        lr_scheduler=lr_scheduler_arg,
        # Logging & Checkpointing
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        keep_n_checkpoints=args.keep_n_checkpoints,
        # Dead Features & Diagnostics
        dead_feature_window=args.dead_feature_window,
        compute_sparsity_diagnostics=args.compute_sparsity_diagnostics,
        enable_profiling=args.enable_profiling,
        # WandB
        enable_wandb=args.enable_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=None,  # Use the decision from above
        wandb_tags=None,
        # Precision & Debugging
        precision=args.precision,
        debug_anomaly=args.debug_anomaly,
        fp16_convert_weights=args.fp16_convert_weights,
    )
    logger.info(f"Training Config: {training_config}")

    activation_cfg = None
    if args.activation_source == 'streaming':
        activation_cfg = ActivationConfig(
            model_name=args.model_name,
            mlp_input_module_path_template=args.mlp_input_template,
            mlp_output_module_path_template=args.mlp_output_template,
            model_dtype=args.model_dtype,
            exclude_special_tokens=args.exclude_special_tokens,
            dataset_split=args.dataset_split,
            dataset_text_column=args.dataset_text_column,
            activation_dtype=args.activation_dtype,
            dataset_path=args.dataset_path,
            context_size=args.context_size,
            inference_batch_size=args.inference_batch_size,
            prepend_bos=args.prepend_bos,
            target_total_tokens=1000000,
            activation_dir=None,
            compression=None,
            chunk_token_threshold=1,
            compute_norm_stats=True)


    # --- Initialize Evaluator from Trainer ---
    logger.info(f"Initializing Evaluator for {args.activation_source} training...")
    try:
        evaluator = Evaluator(
            clt_config=clt_config,
            training_config=training_config,
            log_dir=str(output_dir_path),  # Use the resolved output_dir_path
            device=device_str,
            distributed=args.distributed,
            activation_config=activation_cfg,
            resume_from_checkpoint_path=actual_checkpoint_path_to_load if resuming_run else None,
        )
    except Exception as e:
        logger.exception(f"Failed to initialize Evaluator: {e}")
        raise

    evaluator.evaluate()


if __name__ == "__main__":
    main()
