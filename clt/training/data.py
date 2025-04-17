import torch
from typing import Dict, List, Tuple, Optional, Union, Generator, Any
import logging
import time
from tqdm import tqdm
import sys
import datetime
import gc  # Import Python garbage collector
from abc import ABC, abstractmethod
import os
import json
import numpy as np
import h5py  # Requires pip install h5py
import requests  # Requires pip install requests
from threading import Thread, Event
from queue import Queue, Empty, Full
from urllib.parse import urljoin, quote  # Add quote and urljoin
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type hint for the generator output & batch format
ActivationBatchCLT = Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]


# Helper function to format elapsed time
def _format_elapsed_time(seconds: float) -> str:
    """Formats elapsed seconds into HH:MM:SS or MM:SS."""
    td = datetime.timedelta(seconds=int(seconds))
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if td.days > 0 or hours > 0:
        return f"{td.days * 24 + hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"


# --------------------
# Base Class
# --------------------


class BaseActivationStore(ABC):
    """Abstract base class for activation stores."""

    # Common attributes (to be set by subclasses)
    layer_indices: List[int]
    d_model: int
    dtype: torch.dtype
    device: torch.device
    train_batch_size_tokens: int
    total_tokens: int  # Total tokens available in the store (if known)

    @abstractmethod
    def get_batch(self) -> ActivationBatchCLT:
        """Yields the next batch of activations."""
        pass

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Return a dictionary containing the state for saving/resumption."""
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state from a dictionary."""
        pass

    def __iter__(self):
        """Make the store iterable."""
        return self

    def __next__(self):
        """Allows the store to be used as an iterator yielding batches."""
        return self.get_batch()

    def __len__(self):
        """Estimate the number of batches in the dataset."""
        if (
            not hasattr(self, "total_tokens")
            or self.total_tokens <= 0
            or self.train_batch_size_tokens <= 0
        ):
            return 0
        return (
            self.total_tokens + self.train_batch_size_tokens - 1
        ) // self.train_batch_size_tokens


# --------------------------
# Streaming Implementation
# --------------------------


class StreamingActivationStore(BaseActivationStore):
    """Manages model activations for CLT training using a live streaming generator.

    Buffers activations efficiently, yields batches for training, and handles
    optional normalization estimation on-the-fly.
    Inherits buffer management, batching, and normalization logic from the original ActivationStore.
    """

    # Type hints for instance variables specific to streaming
    activation_generator: Generator[ActivationBatchCLT, None, None]
    n_batches_in_buffer: int
    normalization_method: str  # 'none' or 'estimated_mean_std'
    normalization_estimation_batches: int

    # Buffers store activations per layer
    buffered_inputs: Dict[int, torch.Tensor]
    buffered_targets: Dict[int, torch.Tensor]

    # Read mask tracks yielded tokens across the unified buffer length
    read_indices: torch.Tensor

    # Normalization statistics (per layer) - estimated live
    input_means: Dict[int, torch.Tensor]
    input_stds: Dict[int, torch.Tensor]
    output_means: Dict[int, torch.Tensor]
    output_stds: Dict[int, torch.Tensor]

    # State tracking
    target_buffer_size_tokens: int
    total_tokens_yielded_by_generator: int = 0
    buffer_initialized: bool = False
    generator_exhausted: bool = False
    start_time: float

    def __init__(
        self,
        activation_generator: Generator[ActivationBatchCLT, None, None],
        train_batch_size_tokens: int = 4096,
        n_batches_in_buffer: int = 16,
        normalization_method: str = "none",
        normalization_estimation_batches: int = 50,
        device: Optional[Union[str, torch.device]] = None,
        start_time: Optional[float] = None,
    ):
        """Initialize the streaming activation store for CLT.

        Args:
            activation_generator: Generator yielding (inputs_dict, targets_dict).
            train_batch_size_tokens: Number of tokens per training batch.
            n_batches_in_buffer: Number of training batches worth of tokens to buffer.
            normalization_method: 'none' or 'estimated_mean_std'.
            normalization_estimation_batches: Batches used for estimating stats.
            device: Device to store activations on.
            start_time: Optional start time for logging.
        """
        self.activation_generator = activation_generator
        self.n_batches_in_buffer = n_batches_in_buffer
        self.train_batch_size_tokens = train_batch_size_tokens  # From Base
        self.normalization_method = normalization_method
        self.normalization_estimation_batches = normalization_estimation_batches
        self.start_time = start_time or time.time()

        # Set device (Common logic, could be in Base if needed)
        _device_input = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.device = (
            torch.device(_device_input)
            if isinstance(_device_input, str)
            else _device_input
        )

        # Calculate target buffer size
        self.target_buffer_size_tokens = (
            self.n_batches_in_buffer * self.train_batch_size_tokens
        )

        # Initialize state (buffers etc. will be initialized lazily)
        self.buffer_initialized = False
        self.generator_exhausted = False
        self.layer_indices = []  # From Base
        self.d_model = -1  # From Base
        self.dtype = torch.float32  # From Base

        self.buffered_inputs = {}
        self.buffered_targets = {}
        self.read_indices = torch.empty(0, dtype=torch.bool, device=self.device)

        # Initialize normalization stats placeholders
        self.input_means = {}
        self.input_stds = {}
        self.output_means = {}
        self.output_stds = {}

        # total_tokens is unknown in streaming mode until exhaustion
        self.total_tokens = -1  # Indicates unknown

        logger.info(f"StreamingActivationStore initialized on {self.device}.")
        logger.info(f"Target buffer size: {self.target_buffer_size_tokens} tokens.")
        logger.info(f"Normalization method: {self.normalization_method}")

        # Estimate normalization stats immediately if requested
        if self.normalization_method == "estimated_mean_std":
            self._estimate_normalization_stats()

    def _initialize_buffer_metadata(self, first_batch: ActivationBatchCLT):
        """Initialize buffer metadata based on the first batch from the generator."""
        inputs_dict, targets_dict = first_batch
        self.layer_indices = sorted(inputs_dict.keys())
        if not self.layer_indices:
            raise ValueError("Received empty dictionaries from activation generator.")

        # Infer d_model and dtype from the first layer's input tensor
        first_layer_idx = self.layer_indices[0]
        first_input_tensor = inputs_dict[first_layer_idx]
        self.d_model = first_input_tensor.shape[-1]
        self.dtype = first_input_tensor.dtype

        # Initialize empty buffers for each layer
        for layer_idx in self.layer_indices:
            self.buffered_inputs[layer_idx] = torch.empty(
                (0, self.d_model), device=self.device, dtype=self.dtype
            )
            self.buffered_targets[layer_idx] = torch.empty(
                (0, self.d_model), device=self.device, dtype=self.dtype
            )
            # Validate dimensions for all layers in the first batch
            if (
                inputs_dict[layer_idx].shape[-1] != self.d_model
                or targets_dict[layer_idx].shape[-1] != self.d_model
            ):
                raise ValueError(
                    f"Inconsistent d_model across layers in the first batch. "
                    f"Expected {self.d_model}, got {inputs_dict[layer_idx].shape[-1]} "
                    f"for input or {targets_dict[layer_idx].shape[-1]} for target "
                    f"at layer {layer_idx}."
                )
            if (
                inputs_dict[layer_idx].dtype != self.dtype
                or targets_dict[layer_idx].dtype != self.dtype
            ):
                logger.warning(
                    f"Inconsistent dtype across layers/tensors in first batch. Using {self.dtype}."
                )

        self.buffer_initialized = True
        logger.info(
            f"Buffer metadata initialized. Layers: {self.layer_indices}, d_model: {self.d_model}, dtype: {self.dtype}"
        )

    def _add_batch_to_buffer(self, batch: ActivationBatchCLT):
        """Adds a single batch from the generator to the internal buffers."""
        inputs_dict, targets_dict = batch

        # --- Apply Normalization ---
        if self.normalization_method == "estimated_mean_std" and self.input_means:
            inputs_dict, targets_dict = self._normalize_batch(inputs_dict, targets_dict)
        elif self.normalization_method != "none" and not self.input_means:
            logger.warning(
                "Normalization requested but statistics not computed yet. Skipping normalization."
            )

        # Validate layer indices match
        if (
            sorted(inputs_dict.keys()) != self.layer_indices
            or sorted(targets_dict.keys()) != self.layer_indices
        ):
            raise ValueError(
                f"Inconsistent layer indices received from generator. Expected {self.layer_indices}, "
                f"got inputs: {sorted(inputs_dict.keys())}, targets: {sorted(targets_dict.keys())}"
            )

        num_tokens_in_batch = 0
        for i, layer_idx in enumerate(self.layer_indices):
            inp_tensor = inputs_dict[layer_idx].to(self.device)
            tgt_tensor = targets_dict[layer_idx].to(self.device)

            # Validate shapes and types before concatenating
            if (
                inp_tensor.shape[-1] != self.d_model
                or tgt_tensor.shape[-1] != self.d_model
            ):
                raise ValueError(
                    f"Inconsistent d_model in batch for layer {layer_idx}. "
                    f"Expected {self.d_model}, got input: {inp_tensor.shape[-1]}, target: {tgt_tensor.shape[-1]}"
                )
            if inp_tensor.dtype != self.dtype or tgt_tensor.dtype != self.dtype:
                logger.warning(
                    f"Inconsistent dtype in batch for layer {layer_idx}. Expected {self.dtype}, "
                    f"got input: {inp_tensor.dtype}, target: {tgt_tensor.dtype}. Casting to {self.dtype}."
                )
                inp_tensor = inp_tensor.to(self.dtype)
                tgt_tensor = tgt_tensor.to(self.dtype)
            if inp_tensor.shape[0] != tgt_tensor.shape[0]:
                raise ValueError(
                    f"Mismatched number of tokens between input ({inp_tensor.shape[0]}) and target ({tgt_tensor.shape[0]}) "
                    f"for layer {layer_idx}."
                )

            # Use the number of tokens from the first layer of the batch
            if i == 0:
                num_tokens_in_batch = inp_tensor.shape[0]
                if num_tokens_in_batch == 0:
                    logger.warning(
                        f"Received an empty batch (0 tokens) for layer {layer_idx}. Skipping."
                    )
                    return 0  # Return 0 tokens added

            # Check consistency of token count across layers within the batch
            elif inp_tensor.shape[0] != num_tokens_in_batch:
                raise ValueError(
                    f"Inconsistent number of tokens across layers within the same batch. "
                    f"Layer {self.layer_indices[0]} had {num_tokens_in_batch}, layer {layer_idx} has {inp_tensor.shape[0]}."
                )

            # Concatenate to buffers
            self.buffered_inputs[layer_idx] = torch.cat(
                (self.buffered_inputs[layer_idx], inp_tensor), dim=0
            )
            self.buffered_targets[layer_idx] = torch.cat(
                (self.buffered_targets[layer_idx], tgt_tensor), dim=0
            )

        # Add corresponding read indices (initialized to False)
        if num_tokens_in_batch > 0:
            new_read_indices = torch.zeros(
                num_tokens_in_batch, dtype=torch.bool, device=self.device
            )
            self.read_indices = torch.cat((self.read_indices, new_read_indices), dim=0)
            self.total_tokens_yielded_by_generator += num_tokens_in_batch

        return num_tokens_in_batch

    def _fill_buffer(self):
        """Fills the buffer by pulling data from the generator until target size is reached."""
        if self.generator_exhausted:
            return  # Don't try to fill if we know the generator is done

        num_unread = (~self.read_indices).sum().item()
        tokens_needed = self.target_buffer_size_tokens - num_unread
        tokens_added_this_fill = 0
        start_time = time.time()

        while tokens_added_this_fill < tokens_needed:
            try:
                batch = next(self.activation_generator)

                # Initialize buffer metadata if this is the very first batch
                if not self.buffer_initialized:
                    self._initialize_buffer_metadata(batch)

                # Add the batch to the buffer
                tokens_added = self._add_batch_to_buffer(batch)
                tokens_added_this_fill += tokens_added

            except StopIteration:
                logger.info("Activation generator exhausted.")
                self.generator_exhausted = True
                # Update total_tokens if previously unknown
                if self.total_tokens == -1:
                    self.total_tokens = self.total_tokens_yielded_by_generator
                break  # Exit loop if generator is done
            except Exception as e:
                logger.error(
                    f"Error fetching or processing batch from generator: {e}",
                    exc_info=True,
                )
                raise e  # Re-raise by default

        end_time = time.time()
        current_buffer_size = self.read_indices.shape[0]
        final_unread = (~self.read_indices).sum().item()
        logger.debug(
            f"Buffer fill finished in {end_time - start_time:.2f}s. Added {tokens_added_this_fill} tokens. "
            f"Total buffer size: {current_buffer_size}. Unread tokens: {final_unread}."
        )

        if self.read_indices.shape[0] == 0 and self.generator_exhausted:
            logger.warning(
                "Buffer is empty and generator is exhausted. No data available."
            )

    def _prune_buffer(self):
        """Removes fully read tokens from the beginning of the buffer."""
        if not self.buffer_initialized or self.read_indices.shape[0] == 0:
            return

        # Find the first index that is False (not read)
        first_unread_idx = torch.argmin(
            self.read_indices.int()
        )  # argmin returns first 0 if available
        if self.read_indices[
            first_unread_idx
        ]:  # If the first unread is True, all are True
            first_unread_idx = self.read_indices.shape[0]

        if first_unread_idx > 0:
            # Prune the buffers and the read_indices tensor
            for layer_idx in self.layer_indices:
                self.buffered_inputs[layer_idx] = self.buffered_inputs[layer_idx][
                    first_unread_idx:
                ]
                self.buffered_targets[layer_idx] = self.buffered_targets[layer_idx][
                    first_unread_idx:
                ]
            self.read_indices = self.read_indices[first_unread_idx:]

            if torch.cuda.is_available() and self.device.type == "cuda":
                gc.collect()
                torch.cuda.empty_cache()

    def get_batch(self) -> ActivationBatchCLT:
        """Gets a randomly sampled batch of activations for training."""
        # Initialize and fill buffer on first call or if needed
        num_unread = (~self.read_indices).sum().item()
        # Refill needed if buffer not initialized OR less than half full
        # OR exactly full but not exhausted (avoids getting stuck if buffer = target size exactly)
        needs_refill = (
            (not self.buffer_initialized)
            or (num_unread < self.target_buffer_size_tokens // 2)
            or (
                num_unread == self.read_indices.shape[0]
                and not self.generator_exhausted
                and self.read_indices.shape[0] < self.target_buffer_size_tokens
            )
        )

        if needs_refill and not self.generator_exhausted:
            self._fill_buffer()
            # Re-check unread count after trying to fill
            num_unread = (~self.read_indices).sum().item()

        # If still no unread tokens after trying to fill
        if num_unread == 0:
            if self.generator_exhausted:
                logger.info(
                    "Generator exhausted and buffer empty. Signalling end of iteration."
                )
                raise StopIteration
            else:
                logger.error(
                    "Buffer has no unread tokens despite generator not being marked as exhausted."
                )
                raise RuntimeError("Failed to get unread tokens after buffer refill.")

        # --- Sample indices ---
        unread_token_indices = (~self.read_indices).nonzero().squeeze(-1)
        num_to_sample = min(self.train_batch_size_tokens, len(unread_token_indices))
        if num_to_sample == 0:
            raise RuntimeError(
                "No unread indices available for sampling, despite earlier checks."
            )

        perm = torch.randperm(len(unread_token_indices), device=self.device)[
            :num_to_sample
        ]
        sampled_buffer_indices = unread_token_indices[perm]

        # --- Create batch dictionaries ---
        batch_inputs: Dict[int, torch.Tensor] = {}
        batch_targets: Dict[int, torch.Tensor] = {}
        for layer_idx in self.layer_indices:
            batch_inputs[layer_idx] = self.buffered_inputs[layer_idx][
                sampled_buffer_indices
            ]
            batch_targets[layer_idx] = self.buffered_targets[layer_idx][
                sampled_buffer_indices
            ]

        # --- Mark indices as read ---
        self.read_indices[sampled_buffer_indices] = True

        # --- Prune buffer --- (Prune every time for simplicity here)
        self._prune_buffer()

        return batch_inputs, batch_targets

    def _estimate_normalization_stats(self):
        """Estimates normalization stats using the generator."""
        if self.normalization_method != "estimated_mean_std":
            return

        logger.info(
            f"Starting normalization statistics estimation using {self.normalization_estimation_batches} generator batches..."
        )
        self.input_means, self.input_stds = {}, {}
        self.output_means, self.output_stds = {}, {}
        all_inputs_for_norm: Dict[int, List[torch.Tensor]] = {}
        all_outputs_for_norm: Dict[int, List[torch.Tensor]] = {}
        first_batch_seen = False
        batches_processed = 0

        pbar_norm = tqdm(
            range(self.normalization_estimation_batches), desc="Estimating Norm Stats"
        )
        try:
            for _ in pbar_norm:
                batch_inputs, batch_targets = next(self.activation_generator)
                batches_processed += 1

                if not first_batch_seen:
                    self._initialize_buffer_metadata((batch_inputs, batch_targets))
                    for layer_idx in self.layer_indices:
                        all_inputs_for_norm[layer_idx] = []
                        all_outputs_for_norm[layer_idx] = []
                    first_batch_seen = True

                for layer_idx in self.layer_indices:
                    # Collect tensors (move to CPU to avoid GPU OOM during estimation)
                    all_inputs_for_norm[layer_idx].append(batch_inputs[layer_idx].cpu())
                    all_outputs_for_norm[layer_idx].append(
                        batch_targets[layer_idx].cpu()
                    )
        except StopIteration:
            self.generator_exhausted = True
            logger.warning(
                f"Generator exhausted after {batches_processed} batches during norm estimation."
            )
        finally:
            pbar_norm.close()

        if not first_batch_seen:
            logger.error(
                "No batches received from generator during normalization estimation. Cannot compute stats."
            )
            self.normalization_method = "none"  # Fallback
            return

        logger.info("Calculating mean and std from collected tensors...")
        for layer_idx in tqdm(self.layer_indices, desc="Calculating Stats"):
            if not all_inputs_for_norm[layer_idx]:
                logger.warning(
                    f"No data collected for layer {layer_idx} during norm estimation."
                )
                continue

            try:
                # Concatenate on CPU, compute stats, then move results to target device
                in_cat = torch.cat(
                    all_inputs_for_norm[layer_idx], dim=0
                ).float()  # Ensure float32 for stable stats
                out_cat = torch.cat(all_outputs_for_norm[layer_idx], dim=0).float()

                # Calculate stats and move to target device
                self.input_means[layer_idx] = in_cat.mean(dim=0, keepdim=True).to(
                    self.device, dtype=self.dtype
                )
                self.input_stds[layer_idx] = (
                    in_cat.std(dim=0, keepdim=True) + 1e-6
                ).to(self.device, dtype=self.dtype)
                self.output_means[layer_idx] = out_cat.mean(dim=0, keepdim=True).to(
                    self.device, dtype=self.dtype
                )
                self.output_stds[layer_idx] = (
                    out_cat.std(dim=0, keepdim=True) + 1e-6
                ).to(self.device, dtype=self.dtype)

                # --- Crucially, add the collected batches back to the buffer --- #
                # We iterate through the original CPU tensors we collected
                logger.debug(
                    f"Adding {len(all_inputs_for_norm[layer_idx])} norm batches back to buffer..."
                )
                temp_rebuild_batches = []
                num_batches = len(all_inputs_for_norm[layer_idx])
                for i in range(num_batches):
                    batch_input_dict = {
                        l: all_inputs_for_norm[l][i]
                        for l in self.layer_indices
                        if i < len(all_inputs_for_norm[l])
                    }
                    batch_output_dict = {
                        l: all_outputs_for_norm[l][i]
                        for l in self.layer_indices
                        if i < len(all_outputs_for_norm[l])
                    }
                    temp_rebuild_batches.append((batch_input_dict, batch_output_dict))

                for batch_in_dict, batch_out_dict in temp_rebuild_batches:
                    # Ensure metadata is initialized (should be already)
                    if not self.buffer_initialized:
                        self._initialize_buffer_metadata(
                            (batch_in_dict, batch_out_dict)
                        )
                    # Add batch (this handles moving to device and normalization)
                    self._add_batch_to_buffer((batch_in_dict, batch_out_dict))
                logger.debug("Finished adding norm batches back.")

            except Exception as e:
                logger.error(
                    f"Error calculating norm stats for layer {layer_idx}: {e}",
                    exc_info=True,
                )
                # Decide how to handle: skip layer? fallback? For now, just log.

        del all_inputs_for_norm, all_outputs_for_norm  # Free memory
        gc.collect()
        logger.info(
            f"Normalization estimation complete using {batches_processed} batches."
        )

    def _normalize_batch(
        self,
        inputs_dict: Dict[int, torch.Tensor],
        targets_dict: Dict[int, torch.Tensor],
    ) -> ActivationBatchCLT:
        """Applies the estimated mean/std to each layer's inputs and targets."""
        normalized_inputs = {}
        normalized_targets = {}
        for layer_idx in self.layer_indices:
            inp = inputs_dict[layer_idx].to(self.device, dtype=self.dtype)
            tgt = targets_dict[layer_idx].to(self.device, dtype=self.dtype)

            if layer_idx in self.input_means and layer_idx in self.input_stds:
                inp = (inp - self.input_means[layer_idx]) / self.input_stds[layer_idx]
            if layer_idx in self.output_means and layer_idx in self.output_stds:
                tgt = (tgt - self.output_means[layer_idx]) / self.output_stds[layer_idx]

            normalized_inputs[layer_idx] = inp
            normalized_targets[layer_idx] = tgt
        return normalized_inputs, normalized_targets

    def denormalize_outputs(
        self, outputs: Dict[int, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:
        """Denormalizes output activations to their original scale."""
        if self.normalization_method == "none" or not self.output_means:
            return outputs

        denormalized = {}
        for layer_idx, output in outputs.items():
            if layer_idx in self.output_means and layer_idx in self.output_stds:
                mean = self.output_means[layer_idx]
                std = self.output_stds[layer_idx]
                denormalized[layer_idx] = (output * std.to(output.device)) + mean.to(
                    output.device
                )
            else:
                logger.warning(
                    f"Attempting denormalize layer {layer_idx} but no stats found."
                )
                denormalized[layer_idx] = output
        return denormalized

    # Overrides BaseActivationStore.state_dict
    def state_dict(self) -> Dict:
        """Return state (incl. estimated normalization stats) for saving."""
        cpu_input_means = {k: v.cpu() for k, v in self.input_means.items()}
        cpu_input_stds = {k: v.cpu() for k, v in self.input_stds.items()}
        cpu_output_means = {k: v.cpu() for k, v in self.output_means.items()}
        cpu_output_stds = {k: v.cpu() for k, v in self.output_stds.items()}

        return {
            "store_type": "StreamingActivationStore",
            "layer_indices": self.layer_indices,
            "d_model": self.d_model,
            "dtype": str(self.dtype),
            "input_means": cpu_input_means,
            "input_stds": cpu_input_stds,
            "output_means": cpu_output_means,
            "output_stds": cpu_output_stds,
            "total_tokens_yielded_by_generator": self.total_tokens_yielded_by_generator,
            "target_buffer_size_tokens": self.target_buffer_size_tokens,
            "train_batch_size_tokens": self.train_batch_size_tokens,
            "normalization_method": self.normalization_method,
            "normalization_estimation_batches": self.normalization_estimation_batches,
            # Note: Buffer/generator state not saved.
        }

    # Overrides BaseActivationStore.load_state_dict
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state. Requires a new generator to be provided externally."""
        if state_dict.get("store_type") != "StreamingActivationStore":
            logger.warning("Attempting to load state from incompatible store type.")
        self.layer_indices = state_dict["layer_indices"]
        self.d_model = state_dict["d_model"]
        try:
            self.dtype = getattr(torch, state_dict["dtype"].split(".")[-1])
        except AttributeError:
            logger.warning(f"Could parse dtype '{state_dict['dtype']}', defaulting.")

        self.input_means = {
            k: v.to(self.device) for k, v in state_dict["input_means"].items()
        }
        self.input_stds = {
            k: v.to(self.device) for k, v in state_dict["input_stds"].items()
        }
        self.output_means = {
            k: v.to(self.device) for k, v in state_dict["output_means"].items()
        }
        self.output_stds = {
            k: v.to(self.device) for k, v in state_dict["output_stds"].items()
        }

        self.total_tokens_yielded_by_generator = state_dict[
            "total_tokens_yielded_by_generator"
        ]
        self.target_buffer_size_tokens = state_dict["target_buffer_size_tokens"]
        self.train_batch_size_tokens = state_dict["train_batch_size_tokens"]
        self.normalization_method = state_dict["normalization_method"]
        self.normalization_estimation_batches = state_dict[
            "normalization_estimation_batches"
        ]
        self.total_tokens = (
            -1
        )  # Reset total tokens, will update on generator exhaustion

        # Reset buffer state
        self.buffered_inputs, self.buffered_targets = {}, {}
        self.read_indices = torch.empty(0, dtype=torch.bool, device=self.device)
        self.buffer_initialized = False
        self.generator_exhausted = False
        logger.info("StreamingActivationStore state loaded. Requires a new generator.")


# --------------------------
# Mapped File Implementation
# --------------------------


class MappedActivationStore(BaseActivationStore):
    """Reads pre-generated activation chunks from local disk (HDF5 or NPZ)."""

    # Type hints for instance variables
    activation_path: str
    metadata: Dict[str, Any]
    num_chunks: int
    format: str
    apply_normalization: bool
    input_means: Dict[int, torch.Tensor]
    input_stds: Dict[int, torch.Tensor]
    output_means: Dict[int, torch.Tensor]
    output_stds: Dict[int, torch.Tensor]

    # Chunk management
    current_chunk_idx: int
    current_chunk_data: Optional[Dict[int, Dict[str, torch.Tensor]]]
    read_indices: Optional[torch.Tensor]  # Tracks read tokens within the current chunk
    chunk_tokens_remaining: int

    def __init__(
        self,
        activation_path: str,
        train_batch_size_tokens: int = 4096,
        normalization: Union[
            str, bool
        ] = "auto",  # 'auto', 'none', False, or path/to/stats.json
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Initialize store using pre-generated activations on disk.

        Args:
            activation_path: Path to the activation dataset directory (created by ActivationGenerator).
            train_batch_size_tokens: Number of tokens per training batch.
            normalization: How to handle normalization. 'auto': use norm_stats.json if exists,
                           'none'/False: disable, 'path/to/stats.json': use specific file.
            device: Device to place loaded tensors on.
        """
        self.activation_path = activation_path
        self.train_batch_size_tokens = train_batch_size_tokens  # From Base

        # Set device
        _device_input = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.device = (
            torch.device(_device_input)
            if isinstance(_device_input, str)
            else _device_input
        )  # From Base

        # --- Load metadata --- #
        metadata_path = os.path.join(activation_path, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        # --- Set attributes from metadata --- #
        try:
            dataset_stats = self.metadata["dataset_stats"]
            activation_config = self.metadata.get(
                "activation_config", {}
            )  # Get config if saved

            self.layer_indices = dataset_stats["layer_indices"]  # From Base
            self.num_chunks = dataset_stats["num_chunks"]
            # Use total_tokens_generated if available, else try old key
            self.total_tokens = dataset_stats.get(
                "total_tokens_generated", dataset_stats.get("total_tokens", 0)
            )  # From Base
            self.d_model = dataset_stats["d_model"]  # From Base
            # Get dtype string from dataset_stats (saved by new generator) or fallback
            dtype_str = dataset_stats.get(
                "dtype", activation_config.get("model_dtype", str(torch.float32))
            )
            self.dtype = getattr(
                torch, dtype_str.split(".")[-1], torch.float32
            )  # From Base
            # Get format from activation_config if present, else fallback to old storage_params
            storage_params = self.metadata.get(
                "storage_params", activation_config
            )  # Fallback to activation_config itself
            self.format = storage_params.get("output_format", "hdf5")

        except KeyError as e:
            raise ValueError(
                f"Metadata file {metadata_path} is missing required key: {e}"
            )

        # --- Format Validation ---
        if self.format not in ["hdf5", "npz"]:
            raise ValueError(
                f"Unsupported data format specified in metadata: {self.format}"
            )
        if self.format == "hdf5" and h5py is None:
            raise ImportError(
                "Metadata indicates HDF5 format, but h5py is not installed."
            )

        logger.info(
            f"MappedActivationStore initialized for path: {self.activation_path}"
        )
        logger.info(
            f"Format: {self.format}, Chunks: {self.num_chunks}, Total Tokens: {self.total_tokens:,}"
        )
        logger.info(
            f"Layers: {self.layer_indices}, d_model: {self.d_model}, dtype: {self.dtype}"
        )

        # --- Load normalization statistics if needed --- #
        self.input_means, self.input_stds = {}, {}
        self.output_means, self.output_stds = {}, {}
        self.apply_normalization = False
        self._load_normalization_stats(normalization, dataset_stats)

        # --- Initialize chunk tracking --- #
        self.current_chunk_idx = -1
        self.current_chunk_data = None
        self.read_indices = None
        self.chunk_tokens_remaining = 0
        # Load the first chunk lazily on first get_batch()

    def _load_normalization_stats(
        self, normalization: Union[str, bool], dataset_stats: Dict
    ):
        """Loads normalization stats based on the normalization argument and metadata."""
        norm_stats_path = None
        stats_were_computed = dataset_stats.get("computed_norm_stats", False)

        if normalization == "auto":
            potential_path = os.path.join(self.activation_path, "norm_stats.json")
            if os.path.exists(potential_path):
                if stats_were_computed:
                    norm_stats_path = potential_path
                    logger.info(
                        "'auto' normalization: Found norm_stats.json and metadata confirms computation."
                    )
                else:
                    logger.warning(
                        "'auto' normalization: Found norm_stats.json, but metadata indicates stats were NOT computed during generation. Disabling normalization."
                    )
            else:
                if stats_were_computed:
                    logger.warning(
                        "'auto' normalization: Metadata indicates stats computed, but norm_stats.json not found. Disabling normalization."
                    )
                else:
                    logger.info(
                        "'auto' normalization: No statistics file found and none computed. Disabling normalization."
                    )
        elif normalization == "none" or normalization is False:
            logger.info("Normalization explicitly disabled.")
        elif isinstance(normalization, str):
            if os.path.exists(normalization):
                norm_stats_path = normalization
                logger.info(f"Using specified normalization file: {norm_stats_path}")
            else:
                logger.warning(
                    f"Specified normalization file not found: {normalization}. Disabling normalization."
                )

        if norm_stats_path:
            try:
                with open(norm_stats_path, "r") as f:
                    norm_stats_data = json.load(f)

                # Ensure stats are loaded as tensors onto the correct device
                for layer_idx_str, stats in norm_stats_data.items():
                    layer_idx = int(layer_idx_str)
                    if layer_idx in self.layer_indices:
                        # Convert list back to tensor, add batch dim, move to device, set dtype
                        if (
                            stats.get("input_mean") is not None
                            and stats.get("input_std") is not None
                        ):
                            self.input_means[layer_idx] = torch.tensor(
                                stats["input_mean"],
                                device=self.device,
                                dtype=self.dtype,
                            ).unsqueeze(0)
                            self.input_stds[layer_idx] = (
                                torch.tensor(
                                    stats["input_std"],
                                    device=self.device,
                                    dtype=self.dtype,
                                ).unsqueeze(0)
                                + 1e-6
                            )
                        if (
                            stats.get("output_mean") is not None
                            and stats.get("output_std") is not None
                        ):
                            self.output_means[layer_idx] = torch.tensor(
                                stats["output_mean"],
                                device=self.device,
                                dtype=self.dtype,
                            ).unsqueeze(0)
                            self.output_stds[layer_idx] = (
                                torch.tensor(
                                    stats["output_std"],
                                    device=self.device,
                                    dtype=self.dtype,
                                ).unsqueeze(0)
                                + 1e-6
                            )
                self.apply_normalization = True
                logger.info("Normalization statistics loaded and ready.")
            except Exception as e:
                logger.error(
                    f"Error loading or processing normalization stats from {norm_stats_path}: {e}. Disabling normalization."
                )
                self.apply_normalization = False
        else:
            logger.info("Normalization is disabled.")
            self.apply_normalization = False

    def _load_chunk(self, chunk_idx: int):
        """Loads a specific chunk into memory."""
        if chunk_idx >= self.num_chunks or chunk_idx < 0:
            # This indicates we've cycled through all chunks
            raise StopIteration(
                f"Requested chunk {chunk_idx} is out of bounds (0 to {self.num_chunks - 1})."
            )

        self.current_chunk_idx = chunk_idx
        chunk_path = os.path.join(
            self.activation_path, f"chunk_{chunk_idx}.{self.format}"
        )
        logger.info(f"Loading chunk {chunk_idx} from {chunk_path}...")
        load_start_time = time.time()

        self.current_chunk_data = {}  # Reset buffer
        num_tokens_in_chunk = 0

        try:
            if self.format == "hdf5":
                with h5py.File(chunk_path, "r") as f:
                    # Get num_tokens from attribute if available, otherwise infer
                    num_tokens_in_chunk = f.attrs.get("num_tokens", -1)

                    for layer_idx in self.layer_indices:
                        layer_group = f[f"layer_{layer_idx}"]
                        # Load directly to the target device and dtype
                        inputs = torch.from_numpy(layer_group["inputs"][:]).to(
                            self.device, dtype=self.dtype
                        )
                        targets = torch.from_numpy(layer_group["targets"][:]).to(
                            self.device, dtype=self.dtype
                        )
                        self.current_chunk_data[layer_idx] = {
                            "inputs": inputs,
                            "targets": targets,
                        }
                        if (
                            layer_idx == self.layer_indices[0]
                            and num_tokens_in_chunk == -1
                        ):
                            num_tokens_in_chunk = inputs.shape[0]
            elif self.format == "npz":
                data = np.load(chunk_path)
                num_tokens_in_chunk_arr = data.get("num_tokens", np.array(-1))
                # Ensure we extract the scalar item if it's a 0-dim array
                num_tokens_in_chunk = (
                    num_tokens_in_chunk_arr.item()
                    if num_tokens_in_chunk_arr.size == 1
                    else -1
                )
                if num_tokens_in_chunk == -1 and len(self.layer_indices) > 0:
                    # Infer from first layer if 'num_tokens' not saved
                    first_layer_key = f"layer_{self.layer_indices[0]}_inputs"
                    if first_layer_key in data:
                        num_tokens_in_chunk = data[first_layer_key].shape[0]
                    else:
                        raise ValueError(
                            f"Cannot determine token count in NPZ chunk {chunk_idx}"
                        )

                for layer_idx in self.layer_indices:
                    inputs = torch.from_numpy(data[f"layer_{layer_idx}_inputs"]).to(
                        self.device, dtype=self.dtype
                    )
                    targets = torch.from_numpy(data[f"layer_{layer_idx}_targets"]).to(
                        self.device, dtype=self.dtype
                    )
                    self.current_chunk_data[layer_idx] = {
                        "inputs": inputs,
                        "targets": targets,
                    }
                del data  # Free numpy memory

            if num_tokens_in_chunk <= 0:
                logger.warning(
                    f"Chunk {chunk_idx} loaded but contains {num_tokens_in_chunk} tokens. Skipping."
                )
                self.chunk_tokens_remaining = 0
                self.read_indices = torch.empty(0, dtype=torch.bool, device=self.device)
                return  # Skip this chunk

            # Initialize read tracking for the newly loaded chunk
            self.read_indices = torch.zeros(
                num_tokens_in_chunk, dtype=torch.bool, device=self.device
            )
            self.chunk_tokens_remaining = num_tokens_in_chunk
            logger.info(
                f"Chunk {chunk_idx} ({num_tokens_in_chunk} tokens) loaded in {time.time() - load_start_time:.2f}s."
            )

        except FileNotFoundError:
            logger.error(f"Chunk file not found: {chunk_path}")
            raise  # Re-raise error
        except Exception as e:
            logger.error(
                f"Error loading chunk {chunk_idx} from {chunk_path}: {e}", exc_info=True
            )
            raise  # Re-raise error

    def get_batch(self):
        """Return a random batch of unread tokens from the current or next chunk."""
        if self.num_chunks == 0:
            logger.error("No chunks found in the activation directory.")
            raise StopIteration("No activation data chunks available.")

        # Loop to find the next available chunk with unread tokens
        while self.chunk_tokens_remaining <= 0 or (
            self.read_indices is not None and (~self.read_indices).sum() == 0
        ):
            # If read_indices exist and all are read, mark chunk as depleted
            if self.read_indices is not None and (~self.read_indices).sum() == 0:
                logger.debug(f"Chunk {self.current_chunk_idx} fully read, advancing...")
                self.chunk_tokens_remaining = 0  # Ensure we load the next one

            # Try loading next chunk (cycling back to 0 if needed)
            next_chunk_idx = (self.current_chunk_idx + 1) % self.num_chunks
            is_new_epoch = next_chunk_idx == 0 and self.current_chunk_idx != -1

            if is_new_epoch:
                logger.info("Completed full pass through activation chunks.")
                # Optional: Uncomment below to stop after one epoch
                # raise StopIteration("Completed one epoch through chunks.")

            try:
                self._load_chunk(next_chunk_idx)
                # If _load_chunk returned because the chunk was empty,
                # self.chunk_tokens_remaining will still be 0, and the loop continues.
            except (
                StopIteration
            ):  # Propagate if _load_chunk signals end of available chunks
                logger.info("StopIteration received from _load_chunk, signaling end.")
                raise
            except (IndexError, FileNotFoundError, ValueError) as e:
                logger.error(f"Failed to load chunk {next_chunk_idx}: {e}")
                raise StopIteration(
                    f"Could not load chunk {next_chunk_idx} or no chunks available."
                )

            # Sanity check after load attempt: if still no tokens, something is wrong or dataset is empty
            if (
                self.chunk_tokens_remaining <= 0
                and next_chunk_idx == self.current_chunk_idx
            ):
                # This could happen if the only chunk is empty or fails to load repeatedly
                logger.error(
                    f"Failed to find a non-empty chunk after attempting index {next_chunk_idx}."
                )
                raise StopIteration("Could not find any valid data chunks.")

        # --- At this point, we have a chunk loaded with chunk_tokens_remaining > 0 ---

        if self.read_indices is None or self.current_chunk_data is None:
            # This should ideally not happen if _load_chunk succeeded
            raise RuntimeError(
                "Chunk data or read_indices are unexpectedly None after load loop."
            )

        unread_indices_in_chunk = (~self.read_indices).nonzero().squeeze(-1)

        # If, despite checks, there are no unread indices (shouldn't happen here)
        if unread_indices_in_chunk.numel() == 0:
            logger.error(
                f"Logic error: Chunk {self.current_chunk_idx} has {self.chunk_tokens_remaining} remaining tokens but no unread indices found."
            )
            # Attempt to recover by forcing reload on next call
            self.chunk_tokens_remaining = 0
            raise StopIteration(
                "Internal error: Failed to find unread indices in loaded chunk."
            )

        batch_size = min(self.train_batch_size_tokens, unread_indices_in_chunk.numel())

        perm = torch.randperm(unread_indices_in_chunk.numel(), device=self.device)[
            :batch_size
        ]
        sampled_chunk_indices = unread_indices_in_chunk[perm]

        # --- Prepare batch dictionaries --- #
        batch_inputs = {}
        batch_targets = {}

        for layer_idx in self.layer_indices:
            # Ensure layer exists in the current chunk data
            if layer_idx not in self.current_chunk_data:
                logger.warning(
                    f"Layer {layer_idx} not found in loaded chunk {self.current_chunk_idx}. Skipping layer for this batch."
                )
                continue

            inputs = self.current_chunk_data[layer_idx]["inputs"][sampled_chunk_indices]
            targets = self.current_chunk_data[layer_idx]["targets"][
                sampled_chunk_indices
            ]

            # Apply normalization if enabled
            if self.apply_normalization:
                if layer_idx in self.input_means and layer_idx in self.input_stds:
                    inputs = (inputs - self.input_means[layer_idx]) / self.input_stds[
                        layer_idx
                    ]
                if layer_idx in self.output_means and layer_idx in self.output_stds:
                    targets = (
                        targets - self.output_means[layer_idx]
                    ) / self.output_stds[layer_idx]

            batch_inputs[layer_idx] = inputs
            batch_targets[layer_idx] = targets

        # Mark tokens as read within the current chunk
        self.read_indices[sampled_chunk_indices] = True
        self.chunk_tokens_remaining -= batch_size

        return batch_inputs, batch_targets

    def state_dict(self) -> Dict[str, Any]:
        """Return state for saving/resumption."""
        # Convert stats to CPU for saving
        cpu_input_means = {k: v.cpu().tolist() for k, v in self.input_means.items()}
        cpu_input_stds = {k: v.cpu().tolist() for k, v in self.input_stds.items()}
        cpu_output_means = {k: v.cpu().tolist() for k, v in self.output_means.items()}
        cpu_output_stds = {k: v.cpu().tolist() for k, v in self.output_stds.items()}

        return {
            "store_type": "MappedActivationStore",
            "activation_path": self.activation_path,
            "train_batch_size_tokens": self.train_batch_size_tokens,
            "normalization_applied": self.apply_normalization,  # Record if norm was used
            "input_means": cpu_input_means,
            "input_stds": cpu_input_stds,
            "output_means": cpu_output_means,
            "output_stds": cpu_output_stds,
            "current_chunk_idx": self.current_chunk_idx,
            # We don't save the actual chunk data or read_indices, just the position
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state from a dictionary."""
        if state_dict.get("store_type") != "MappedActivationStore":
            logger.warning("Attempting to load state from incompatible store type.")

        # Configuration params should match if loading into the same object instance
        # self.activation_path = state_dict["activation_path"]
        # self.train_batch_size_tokens = state_dict["train_batch_size_tokens"]

        # Load normalization stats and move to the correct device
        self.apply_normalization = state_dict.get("normalization_applied", False)
        # Convert lists back to tensors
        self.input_means = {
            k: torch.tensor(v, device=self.device, dtype=self.dtype)
            for k, v in state_dict["input_means"].items()
        }
        self.input_stds = {
            k: torch.tensor(v, device=self.device, dtype=self.dtype)
            for k, v in state_dict["input_stds"].items()
        }
        self.output_means = {
            k: torch.tensor(v, device=self.device, dtype=self.dtype)
            for k, v in state_dict["output_means"].items()
        }
        self.output_stds = {
            k: torch.tensor(v, device=self.device, dtype=self.dtype)
            for k, v in state_dict["output_stds"].items()
        }

        # Restore position, but don't load data yet
        self.current_chunk_idx = state_dict.get("current_chunk_idx", -1)
        self.current_chunk_data = None
        self.read_indices = None
        self.chunk_tokens_remaining = 0

        logger.info(
            f"MappedActivationStore state loaded. Resuming from chunk {self.current_chunk_idx}."
        )
        # The next call to get_batch() will load the required chunk


# --------------------------
# Remote Implementation (STUB)
# --------------------------


class RemoteActivationStore(BaseActivationStore):
    """Client for retrieving pre-generated activations from a remote server."""

    # Type hints for instance variables
    server_url: str
    dataset_id: str
    prefetch_batches: int
    requested_layer_indices: Optional[List[int]]
    normalization_mode: Union[str, bool]
    timeout: int
    apply_normalization: bool
    input_means: Dict[int, torch.Tensor]
    input_stds: Dict[int, torch.Tensor]
    output_means: Dict[int, torch.Tensor]
    output_stds: Dict[int, torch.Tensor]
    metadata: Optional[Dict[str, Any]]  # Store fetched metadata
    prefetch_queue: Queue
    prefetch_thread: Optional[Thread]
    stop_event: Event
    # Flag to indicate if the server has signaled end of data
    _server_exhausted: bool = False

    def __init__(
        self,
        server_url: str,
        dataset_id: str,  # e.g., "gpt2/openwebtext_train"
        train_batch_size_tokens: int = 4096,
        prefetch_batches: int = 4,  # Add prefetch param back, default to 4
        layer_indices_to_fetch: Optional[List[int]] = None,
        normalization: Union[
            str, bool
        ] = "auto",  # 'auto', 'none', False, or path to local JSON stats
        device: Optional[Union[str, torch.device]] = None,
        timeout: int = 60,  # Increase default timeout for potentially large batch requests
    ):
        """Initialize the remote activation store client."""
        self.server_url = server_url.rstrip("/")
        # URL-encode dataset_id for safe use in URLs
        self.dataset_id_encoded = quote(dataset_id, safe="")
        self.dataset_id_original = dataset_id  # Keep original for logging

        self.train_batch_size_tokens = train_batch_size_tokens  # From Base
        self.prefetch_batches = max(1, prefetch_batches)  # Ensure at least 1
        self.requested_layer_indices = layer_indices_to_fetch
        self.normalization_mode = normalization
        self.timeout = timeout

        # Set device
        _device_input = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.device = (
            torch.device(_device_input)
            if isinstance(_device_input, str)
            else _device_input
        )  # From Base

        logger.info(
            f"Initializing RemoteActivationStore for {self.dataset_id_original} at {self.server_url}"
        )

        # --- Set Base attributes (will be updated by metadata fetch) ---
        self.layer_indices = []
        self.d_model = -1
        self.dtype = torch.float32
        self.total_tokens = 0
        self.metadata = None  # Initialize metadata

        # --- Fetch metadata --- #
        try:
            self.metadata = self._fetch_metadata()
            # --- Update Base attributes from fetched metadata --- #
            if self.metadata:
                dataset_stats = self.metadata.get("dataset_stats", {})
                activation_config = self.metadata.get("activation_config", {})

                self.layer_indices = dataset_stats.get("layer_indices", [])
                self.d_model = dataset_stats.get("d_model", -1)
                # Use total_tokens_generated if available, else try old key
                self.total_tokens = dataset_stats.get(
                    "total_tokens_generated", dataset_stats.get("total_tokens", 0)
                )
                # Get dtype string from dataset_stats or activation_config
                dtype_str = dataset_stats.get(
                    "dtype", activation_config.get("model_dtype", str(torch.float32))
                )
                try:
                    self.dtype = getattr(torch, dtype_str.split(".")[-1])
                except AttributeError:
                    logger.warning(
                        f"Could not parse dtype '{dtype_str}' from metadata, defaulting to float32."
                    )
                    self.dtype = torch.float32

                logger.info(
                    f"Successfully fetched metadata: Layers={self.layer_indices}, "
                    f"d_model={self.d_model}, dtype={self.dtype}, TotalTokens={self.total_tokens:,}"
                )
            else:
                logger.error(
                    "Metadata fetch returned None. Store initialization incomplete."
                )
                # Optionally raise an error here

        except (requests.exceptions.RequestException, RuntimeError, ValueError) as e:
            logger.error(f"Failed to fetch or parse metadata during init: {e}")
            # Decide how to handle: raise error? operate without metadata?
            # For now, log error and continue with defaults
            self.metadata = None
        except Exception as e:
            logger.error(f"Unexpected error during metadata fetch: {e}", exc_info=True)
            self.metadata = None

        # If layer indices were not set by metadata, log warning
        if not self.layer_indices:
            logger.warning("Could not determine layer indices from server metadata.")
        # If specific layers requested, validate against available layers
        if self.requested_layer_indices is not None and self.layer_indices:
            invalid_layers = set(self.requested_layer_indices) - set(self.layer_indices)
            if invalid_layers:
                logger.warning(
                    f"Requested layers {invalid_layers} not available in dataset layers {self.layer_indices}."
                )
                # Filter requested layers to only those available
                self.requested_layer_indices = [
                    l for l in self.requested_layer_indices if l in self.layer_indices
                ]
                logger.info(f"Fetching layers: {self.requested_layer_indices}")

        # --- Load normalization stats --- #
        self.input_means, self.input_stds = {}, {}
        self.output_means, self.output_stds = {}, {}
        self.apply_normalization = (
            self._load_normalization_stats()
        )  # Call implementation

        # --- Prefetching --- #
        self.prefetch_queue = Queue(maxsize=self.prefetch_batches)
        self.stop_event = Event()
        self._server_exhausted = False  # Initialize flag
        self.prefetch_thread = Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()

        logger.info(f"RemoteActivationStore initialization finished.")

    def _fetch_metadata(self) -> Optional[Dict[str, Any]]:
        """Fetch dataset metadata from the server."""
        # Construct URL using the encoded dataset ID
        url = urljoin(
            f"{self.server_url}/api/v1/datasets/", f"{self.dataset_id_encoded}/info"
        )
        logger.debug(f"Fetching metadata from: {url}")
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch metadata from {url}: {e}")
            # Optionally return None or raise a custom exception
            raise RuntimeError(f"Failed to fetch metadata from server: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode metadata JSON from {url}: {e}")
            raise RuntimeError(f"Invalid metadata format received from server: {e}")

    def _fetch_norm_stats_remote(self) -> Optional[Dict[str, Any]]:
        """Fetch normalization stats from the server."""
        url = urljoin(
            f"{self.server_url}/api/v1/datasets/",
            f"{self.dataset_id_encoded}/norm_stats",
        )
        logger.debug(f"Fetching normalization stats from: {url}")
        try:
            response = requests.get(url, timeout=self.timeout)
            if response.status_code == 404:
                logger.info(
                    f"Normalization stats not found on server for {self.dataset_id_original}."
                )
                return None
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch normalization stats from {url}: {e}")
            return None  # Don't prevent startup if stats fetch fails
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode norm_stats JSON from {url}: {e}")
            return None

    def _load_normalization_stats(self) -> bool:
        """Load normalization stats based on self.normalization_mode."""
        norm_stats_data = None
        source_description = ""

        if self.normalization_mode == "auto":
            logger.info(
                "Normalization set to 'auto', attempting to fetch from server..."
            )
            norm_stats_data = self._fetch_norm_stats_remote()
            source_description = "remote server"
            if norm_stats_data is None:
                logger.warning("Could not fetch normalization stats from server.")

        elif isinstance(self.normalization_mode, str) and os.path.exists(
            self.normalization_mode
        ):
            logger.info(
                f"Loading normalization stats from local file: {self.normalization_mode}"
            )
            source_description = f"local file ({self.normalization_mode})"
            try:
                with open(self.normalization_mode, "r") as f:
                    norm_stats_data = json.load(f)
            except Exception as e:
                logger.error(
                    f"Error loading normalization stats from {self.normalization_mode}: {e}"
                )
                norm_stats_data = None

        elif self.normalization_mode == "none" or self.normalization_mode is False:
            logger.info("Normalization explicitly disabled.")
            return False
        else:
            logger.warning(
                f"Invalid normalization mode or file not found: '{self.normalization_mode}'. Disabling normalization."
            )
            return False

        # Process loaded/fetched stats
        if norm_stats_data:
            logger.info(
                f"Processing normalization statistics from {source_description}..."
            )
            stats_loaded_count = 0
            try:
                # Ensure layer_indices is populated before processing stats
                if not self.layer_indices:
                    logger.warning(
                        "Layer indices not available from metadata, cannot load normalization stats."
                    )
                    return False

                for layer_idx_str, stats in norm_stats_data.items():
                    try:
                        layer_idx = int(layer_idx_str)
                    except ValueError:
                        logger.warning(
                            f"Skipping invalid layer index '{layer_idx_str}' in normalization stats."
                        )
                        continue

                    if layer_idx in self.layer_indices:
                        # Convert list back to tensor, add batch dim, move to device, set dtype
                        if (
                            stats.get("input_mean") is not None
                            and stats.get("input_std") is not None
                        ):
                            self.input_means[layer_idx] = torch.tensor(
                                stats["input_mean"],
                                device=self.device,
                                dtype=self.dtype,  # Use dtype determined from metadata
                            ).unsqueeze(0)
                            self.input_stds[layer_idx] = (
                                torch.tensor(
                                    stats["input_std"],
                                    device=self.device,
                                    dtype=self.dtype,
                                ).unsqueeze(0)
                                + 1e-6
                            )
                        if (
                            stats.get("output_mean") is not None
                            and stats.get("output_std") is not None
                        ):
                            self.output_means[layer_idx] = torch.tensor(
                                stats["output_mean"],
                                device=self.device,
                                dtype=self.dtype,
                            ).unsqueeze(0)
                            self.output_stds[layer_idx] = (
                                torch.tensor(
                                    stats["output_std"],
                                    device=self.device,
                                    dtype=self.dtype,
                                ).unsqueeze(0)
                                + 1e-6
                            )
                        stats_loaded_count += 1
                    else:
                        logger.warning(
                            f"Layer index {layer_idx} from norm stats not found in dataset layers {self.layer_indices}. Skipping."
                        )

                if stats_loaded_count > 0:
                    logger.info(
                        f"Normalization statistics loaded for {stats_loaded_count} layers. Normalization enabled."
                    )
                    return True
                else:
                    logger.warning(
                        "Normalization stats found but contained no matching layer data. Normalization disabled."
                    )
                    return False

            except Exception as e:
                logger.error(
                    f"Error processing normalization stats: {e}. Disabling normalization.",
                    exc_info=True,
                )
                return False
        else:
            logger.info("No normalization statistics loaded. Normalization disabled.")
            return False

    def _apply_normalization(self, batch_inputs, batch_targets):
        """Applies normalization if enabled and stats are available."""
        if not self.apply_normalization:
            return batch_inputs, batch_targets

        norm_inputs = {}
        norm_targets = {}
        for (
            layer_idx
        ) in batch_inputs.keys():  # Iterate over layers present in the batch
            inp = batch_inputs[layer_idx]
            tgt = batch_targets.get(
                layer_idx
            )  # Targets might be missing if only inputs requested?

            # Ensure tensors are on the correct device before normalization
            inp = inp.to(self.device)

            if layer_idx in self.input_means and layer_idx in self.input_stds:
                inp = (inp - self.input_means[layer_idx]) / self.input_stds[layer_idx]
            norm_inputs[layer_idx] = inp

            if tgt is not None:
                tgt = tgt.to(self.device)
                if layer_idx in self.output_means and layer_idx in self.output_stds:
                    tgt = (tgt - self.output_means[layer_idx]) / self.output_stds[
                        layer_idx
                    ]
                norm_targets[layer_idx] = tgt
            # If tgt was None, norm_targets won't have this layer_idx

        return norm_inputs, norm_targets

    def _fetch_batch_from_server(self) -> bytes:
        """Fetches a single serialized batch from the server. Raises errors or StopIteration."""
        # Construct request URL
        params = {"num_tokens": self.train_batch_size_tokens}
        if self.requested_layer_indices:
            params["layers"] = ",".join(map(str, self.requested_layer_indices))

        # Use the encoded dataset ID in the URL path
        url = urljoin(
            f"{self.server_url}/api/v1/datasets/", f"{self.dataset_id_encoded}/batch"
        )

        logger.debug(
            f"Prefetch worker requesting batch from {url} with params: {params}"
        )
        try:
            response = requests.get(url, params=params, timeout=self.timeout)

            # Check for 404 specifically to signal exhaustion cleanly
            if response.status_code == 404:
                logger.info(
                    f"Server returned 404 for {self.dataset_id_original}, assuming dataset exhausted."
                )
                raise StopIteration(
                    f"Dataset {self.dataset_id_original} exhausted or not found on server."
                )

            response.raise_for_status()  # Raise for other bad statuses (5xx, etc.)

            # Check content type
            content_type = response.headers.get("content-type", "").lower()
            if "application/octet-stream" not in content_type:
                logger.warning(
                    f"Received unexpected content type '{content_type}' from batch endpoint. Expected 'application/octet-stream'."
                )
            return response.content

        except requests.exceptions.Timeout:
            logger.error(f"Timeout requesting batch from {url}")
            raise StopIteration(
                f"Timeout connecting to activation server: {self.server_url}"
            )
        except requests.exceptions.RequestException as e:
            # Includes HTTPError for non-404 bad status codes
            logger.error(f"Error requesting batch from {url}: {e}")
            if hasattr(e, "response") and e.response is not None:
                # Don't raise StopIteration for server errors, maybe retry later?
                # For now, re-raise as RuntimeError to signal worker issue
                raise RuntimeError(
                    f"Server error fetching batch ({e.response.status_code}): {e.response.text}"
                )
            else:
                # Network error likely
                raise StopIteration(
                    f"Network error connecting to activation server: {e}"
                )
        # Catch other potential errors during request
        except Exception as e:
            logger.error(f"Unexpected error fetching batch: {e}", exc_info=True)
            raise StopIteration(f"Unexpected error fetching batch: {e}")

    def _prefetch_worker(self):
        """Background worker to fetch batches and put them in the queue."""
        logger.info("Prefetch worker started.")
        while not self.stop_event.is_set():
            try:
                # Only fetch if queue has space
                if not self.prefetch_queue.full():
                    # Check if server already signaled exhaustion
                    if self._server_exhausted:
                        logger.debug(
                            "Prefetch worker: Server exhausted, not fetching more."
                        )
                        time.sleep(0.5)  # Sleep longer if exhausted
                        continue

                    batch_bytes = self._fetch_batch_from_server()
                    # --- Log Queue Size Before Put --- #
                    qsize_before = self.prefetch_queue.qsize()
                    self.prefetch_queue.put(batch_bytes, timeout=5.0)
                    # --- Log Queue Size After Put --- #
                    qsize_after = self.prefetch_queue.qsize()
                    logger.debug(
                        f"Prefetch worker put batch in queue (size before: {qsize_before}, after: {qsize_after})"
                    )
                    # Optional: small sleep even after success to prevent hammering server
                    time.sleep(0.01)
                else:
                    # --- Log When Queue is Full --- #
                    logger.warning(
                        f"Prefetch queue full (size: {self.prefetch_queue.qsize()}). Worker waiting..."
                    )
                    time.sleep(0.1)

            except StopIteration as e:
                # Server signaled end of data (404) or critical network error
                logger.info(f"Prefetch worker stopping: {e}")
                self._server_exhausted = True
                break  # Exit the worker loop
            except RuntimeError as e:
                # Non-critical error during fetch (e.g., server 500 error)
                logger.error(
                    f"Prefetch worker encountered runtime error: {e}. Retrying after delay..."
                )
                time.sleep(5)  # Wait before retrying
            except Full:
                # Should ideally be caught by the `else` block above, but kept here as a failsafe.
                logger.warning(
                    f"Prefetch queue full (exception caught, size: {self.prefetch_queue.qsize()}). Worker waiting..."
                )
                time.sleep(0.1)
            except Exception as e:
                # Catch unexpected errors in the worker loop
                logger.error(f"Unexpected error in prefetch worker: {e}", exc_info=True)
                time.sleep(5)

        logger.info(
            f"Prefetch worker finished. Final queue size: {self.prefetch_queue.qsize()}"
        )

    # Overrides BaseActivationStore.get_batch
    def get_batch(self):
        """Get a batch of activations from the prefetch queue."""
        try:
            # --- Log Queue Size Before Get --- #
            qsize_before_get = self.prefetch_queue.qsize()
            logger.debug(f"get_batch requesting item. Queue size: {qsize_before_get}")
            batch_bytes = self.prefetch_queue.get(timeout=self.timeout)

            if not batch_bytes:
                raise ValueError("Received empty bytes from prefetch queue.")

            # Deserialize the binary response using torch.load
            buffer = io.BytesIO(batch_bytes)
            batch_data = torch.load(
                buffer, map_location=torch.device("cpu")
            )  # Load to CPU first
            del buffer

            # Extract, move to device, and apply normalization
            raw_inputs = batch_data.get("inputs", {})
            raw_targets = batch_data.get("targets", {})

            # Apply normalization (which also moves tensors to self.device)
            batch_inputs, batch_targets = self._apply_normalization(
                raw_inputs, raw_targets
            )

            # Validate batch content (optional)
            if not batch_inputs:
                logger.warning("Received batch with no input data after processing.")
                raise RuntimeError("Processed empty batch data from queue.")

            return batch_inputs, batch_targets

        except Empty:
            # --- Log Queue Size on Empty --- #
            qsize_on_empty = self.prefetch_queue.qsize()
            logger.warning(
                f"Prefetch queue empty (size: {qsize_on_empty}) during get(). Checking worker/server status."
            )
            # Check if the server is marked as exhausted
            if self._server_exhausted and self.prefetch_queue.empty():
                logger.info(
                    "Prefetch queue empty and server is exhausted. Signaling end of iteration."
                )
                raise StopIteration("Activation data exhausted.")
            elif self.prefetch_thread and not self.prefetch_thread.is_alive():
                logger.error("Prefetch queue empty and worker thread is not alive!")
                raise RuntimeError("Prefetch worker thread died unexpectedly.")
            else:
                logger.error(
                    f"Timeout waiting for batch from prefetch queue (timeout: {self.timeout}s). Worker might be slow or stuck."
                )
                raise RuntimeError(f"Timeout waiting for batch from prefetch queue.")
        except (EOFError, RuntimeError, ValueError, KeyError, TypeError) as e:
            # Catch potential errors during torch.load or processing
            logger.error(
                f"Error processing batch data received from queue: {e}", exc_info=True
            )
            raise RuntimeError(f"Failed to process batch data from queue: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in get_batch: {e}", exc_info=True)
            raise StopIteration(f"Unexpected error getting batch: {e}")

    def close(self):
        """Signals the prefetch worker to stop and waits for it to join."""
        logger.info("Stopping RemoteActivationStore prefetch worker...")
        self.stop_event.set()
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            # Don't wait indefinitely, use a timeout
            join_timeout = 5.0
            self.prefetch_thread.join(timeout=join_timeout)
            if self.prefetch_thread.is_alive():
                logger.warning(
                    f"Prefetch worker did not exit within {join_timeout} seconds."
                )
            else:
                logger.info("Prefetch worker joined.")
        self.prefetch_thread = None

    # Ensure close is called when the object is deleted
    def __del__(self):
        self.close()

    # Overrides BaseActivationStore.state_dict
    def state_dict(self) -> Dict[str, Any]:
        """Return state for saving/resumption."""
        # Convert stats to lists for JSON if saving them
        cpu_input_means = {k: v.cpu().tolist() for k, v in self.input_means.items()}
        cpu_input_stds = {k: v.cpu().tolist() for k, v in self.input_stds.items()}
        cpu_output_means = {k: v.cpu().tolist() for k, v in self.output_means.items()}
        cpu_output_stds = {k: v.cpu().tolist() for k, v in self.output_stds.items()}
        return {
            "store_type": "RemoteActivationStore",
            "server_url": self.server_url,
            "dataset_id": self.dataset_id_original,
            "train_batch_size_tokens": self.train_batch_size_tokens,
            "normalization_applied": self.apply_normalization,
            "input_means": cpu_input_means,
            "input_stds": cpu_input_stds,
            "output_means": cpu_output_means,
            "output_stds": cpu_output_stds,
            "requested_layer_indices": self.requested_layer_indices,
            "normalization_mode": self.normalization_mode,
            "timeout": self.timeout,
            # Add other relevant config needed for resumption
        }

    # Overrides BaseActivationStore.load_state_dict
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state from a dictionary."""
        # Note: Does not restart the prefetch thread automatically.
        # Assumes a new instance is created or thread is managed externally if resuming.
        if state_dict.get("store_type") != "RemoteActivationStore":
            logger.warning("Attempting to load state from incompatible store type.")
