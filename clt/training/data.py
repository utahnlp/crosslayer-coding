import torch
from typing import Dict, List, Tuple, Optional, Union, Generator, Any
import logging
import time
from tqdm import tqdm
import datetime
import gc  # Import Python garbage collector
from abc import ABC, abstractmethod

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

    @abstractmethod
    def close(self):
        """Close the store and clean up resources (e.g., threads). Stub for base class."""
        pass

    def __iter__(self):
        """Make the store iterable."""
        return self

    def __next__(self):
        """Allows the store to be used as an iterator yielding batches."""
        try:
            return self.get_batch()
        except StopIteration:
            # Re-raise StopIteration if get_batch signals exhaustion
            raise
        except Exception as e:
            logger.error(f"Error during iteration: {e}", exc_info=True)
            raise  # Propagate other errors

    def __len__(self):
        """Estimate the number of batches in the dataset."""
        if (
            not hasattr(self, "total_tokens")
            or self.total_tokens <= 0
            or not hasattr(self, "train_batch_size_tokens")  # Check attribute exists
            or self.train_batch_size_tokens <= 0
        ):
            return 0
        # Ensure division is safe
        return (self.total_tokens + self.train_batch_size_tokens - 1) // self.train_batch_size_tokens


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
        _device_input = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.device = torch.device(_device_input) if isinstance(_device_input, str) else _device_input

        # Calculate target buffer size
        self.target_buffer_size_tokens = self.n_batches_in_buffer * self.train_batch_size_tokens

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
            self.buffered_inputs[layer_idx] = torch.empty((0, self.d_model), device=self.device, dtype=self.dtype)
            self.buffered_targets[layer_idx] = torch.empty((0, self.d_model), device=self.device, dtype=self.dtype)
            # Validate dimensions for all layers in the first batch
            if inputs_dict[layer_idx].shape[-1] != self.d_model or targets_dict[layer_idx].shape[-1] != self.d_model:
                raise ValueError(
                    f"Inconsistent d_model across layers in the first batch. "
                    f"Expected {self.d_model}, got {inputs_dict[layer_idx].shape[-1]} "
                    f"for input or {targets_dict[layer_idx].shape[-1]} for target "
                    f"at layer {layer_idx}."
                )
            if inputs_dict[layer_idx].dtype != self.dtype or targets_dict[layer_idx].dtype != self.dtype:
                logger.warning(f"Inconsistent dtype across layers/tensors in first batch. Using {self.dtype}.")

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
            logger.warning("Normalization requested but statistics not computed yet. Skipping normalization.")

        # Validate layer indices match
        if sorted(inputs_dict.keys()) != self.layer_indices or sorted(targets_dict.keys()) != self.layer_indices:
            raise ValueError(
                f"Inconsistent layer indices received from generator. Expected {self.layer_indices}, "
                f"got inputs: {sorted(inputs_dict.keys())}, targets: {sorted(targets_dict.keys())}"
            )

        num_tokens_in_batch = 0
        for i, layer_idx in enumerate(self.layer_indices):
            inp_tensor = inputs_dict[layer_idx].to(self.device)
            tgt_tensor = targets_dict[layer_idx].to(self.device)

            # Validate shapes and types before concatenating
            if inp_tensor.shape[-1] != self.d_model or tgt_tensor.shape[-1] != self.d_model:
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
                    logger.warning(f"Received an empty batch (0 tokens) for layer {layer_idx}. Skipping.")
                    return 0  # Return 0 tokens added

            # Check consistency of token count across layers within the batch
            elif inp_tensor.shape[0] != num_tokens_in_batch:
                raise ValueError(
                    f"Inconsistent number of tokens across layers within the same batch. "
                    f"Layer {self.layer_indices[0]} had {num_tokens_in_batch}, layer {layer_idx} has {inp_tensor.shape[0]}."
                )

            # Concatenate to buffers
            self.buffered_inputs[layer_idx] = torch.cat((self.buffered_inputs[layer_idx], inp_tensor), dim=0)
            self.buffered_targets[layer_idx] = torch.cat((self.buffered_targets[layer_idx], tgt_tensor), dim=0)

        # Add corresponding read indices (initialized to False)
        if num_tokens_in_batch > 0:
            new_read_indices = torch.zeros(num_tokens_in_batch, dtype=torch.bool, device=self.device)
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
            logger.warning("Buffer is empty and generator is exhausted. No data available.")

    def _prune_buffer(self):
        """Removes fully read tokens from the beginning of the buffer."""
        if not self.buffer_initialized or self.read_indices.shape[0] == 0:
            return

        # Find the first index that is False (not read)
        first_unread_idx = torch.argmin(self.read_indices.int())  # argmin returns first 0 if available
        if self.read_indices[first_unread_idx]:  # If the first unread is True, all are True
            first_unread_idx = self.read_indices.shape[0]

        if first_unread_idx > 0:
            # Prune the buffers and the read_indices tensor
            for layer_idx in self.layer_indices:
                self.buffered_inputs[layer_idx] = self.buffered_inputs[layer_idx][first_unread_idx:]
                self.buffered_targets[layer_idx] = self.buffered_targets[layer_idx][first_unread_idx:]
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
                logger.info("Generator exhausted and buffer empty. Signalling end of iteration.")
                raise StopIteration
            else:
                logger.error("Buffer has no unread tokens despite generator not being marked as exhausted.")
                raise RuntimeError("Failed to get unread tokens after buffer refill.")

        # --- Sample indices ---
        unread_token_indices = (~self.read_indices).nonzero().squeeze(-1)
        num_to_sample = min(self.train_batch_size_tokens, len(unread_token_indices))
        if num_to_sample == 0:
            raise RuntimeError("No unread indices available for sampling, despite earlier checks.")

        perm = torch.randperm(len(unread_token_indices), device=self.device)[:num_to_sample]
        sampled_buffer_indices = unread_token_indices[perm]

        # --- Create batch dictionaries ---
        batch_inputs: Dict[int, torch.Tensor] = {}
        batch_targets: Dict[int, torch.Tensor] = {}
        for layer_idx in self.layer_indices:
            batch_inputs[layer_idx] = self.buffered_inputs[layer_idx][sampled_buffer_indices]
            batch_targets[layer_idx] = self.buffered_targets[layer_idx][sampled_buffer_indices]

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

        pbar_norm = tqdm(range(self.normalization_estimation_batches), desc="Estimating Norm Stats")
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
                    all_outputs_for_norm[layer_idx].append(batch_targets[layer_idx].cpu())
        except StopIteration:
            self.generator_exhausted = True
            logger.warning(f"Generator exhausted after {batches_processed} batches during norm estimation.")
        finally:
            pbar_norm.close()

        if not first_batch_seen:
            logger.error("No batches received from generator during normalization estimation. Cannot compute stats.")
            self.normalization_method = "none"  # Fallback
            return

        logger.info("Calculating mean and std from collected tensors...")
        for layer_idx in tqdm(self.layer_indices, desc="Calculating Stats"):
            if not all_inputs_for_norm[layer_idx]:
                logger.warning(f"No data collected for layer {layer_idx} during norm estimation.")
                continue

            try:
                # Concatenate on CPU, compute stats, then move results to target device
                in_cat = torch.cat(all_inputs_for_norm[layer_idx], dim=0).float()  # Ensure float32 for stable stats
                out_cat = torch.cat(all_outputs_for_norm[layer_idx], dim=0).float()

                # Calculate stats and move to target device
                self.input_means[layer_idx] = in_cat.mean(dim=0, keepdim=True).to(self.device, dtype=self.dtype)
                self.input_stds[layer_idx] = (in_cat.std(dim=0, keepdim=True) + 1e-6).to(self.device, dtype=self.dtype)
                self.output_means[layer_idx] = out_cat.mean(dim=0, keepdim=True).to(self.device, dtype=self.dtype)
                self.output_stds[layer_idx] = (out_cat.std(dim=0, keepdim=True) + 1e-6).to(
                    self.device, dtype=self.dtype
                )

                # --- Crucially, add the collected batches back to the buffer --- #
                # We iterate through the original CPU tensors we collected
                logger.debug(f"Adding {len(all_inputs_for_norm[layer_idx])} norm batches back to buffer...")
                temp_rebuild_batches = []
                num_batches = len(all_inputs_for_norm[layer_idx])
                for i in range(num_batches):
                    batch_input_dict = {
                        layer_idx_inner: all_inputs_for_norm[layer_idx_inner][i]
                        for layer_idx_inner in self.layer_indices
                        if i < len(all_inputs_for_norm[layer_idx_inner])
                    }
                    batch_output_dict = {
                        layer_idx_inner: all_outputs_for_norm[layer_idx_inner][i]
                        for layer_idx_inner in self.layer_indices
                        if i < len(all_outputs_for_norm[layer_idx_inner])
                    }
                    temp_rebuild_batches.append((batch_input_dict, batch_output_dict))

                for batch_in_dict, batch_out_dict in temp_rebuild_batches:
                    # Ensure metadata is initialized (should be already)
                    if not self.buffer_initialized:
                        self._initialize_buffer_metadata((batch_in_dict, batch_out_dict))
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
        logger.info(f"Normalization estimation complete using {batches_processed} batches.")

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

    def denormalize_outputs(self, outputs: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """Denormalizes output activations to their original scale."""
        if self.normalization_method == "none" or not self.output_means:
            return outputs

        denormalized = {}
        for layer_idx, output in outputs.items():
            if layer_idx in self.output_means and layer_idx in self.output_stds:
                mean = self.output_means[layer_idx]
                std = self.output_stds[layer_idx]
                denormalized[layer_idx] = (output * std.to(output.device)) + mean.to(output.device)
            else:
                logger.warning(f"Attempting denormalize layer {layer_idx} but no stats found.")
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

        self.input_means = {k: v.to(self.device) for k, v in state_dict["input_means"].items()}
        self.input_stds = {k: v.to(self.device) for k, v in state_dict["input_stds"].items()}
        self.output_means = {k: v.to(self.device) for k, v in state_dict["output_means"].items()}
        self.output_stds = {k: v.to(self.device) for k, v in state_dict["output_stds"].items()}

        self.total_tokens_yielded_by_generator = state_dict["total_tokens_yielded_by_generator"]
        self.target_buffer_size_tokens = state_dict["target_buffer_size_tokens"]
        self.train_batch_size_tokens = state_dict["train_batch_size_tokens"]
        self.normalization_method = state_dict["normalization_method"]
        self.normalization_estimation_batches = state_dict["normalization_estimation_batches"]
        self.total_tokens = -1  # Reset total tokens, will update on generator exhaustion

        # Reset buffer state
        self.buffered_inputs, self.buffered_targets = {}, {}
        self.read_indices = torch.empty(0, dtype=torch.bool, device=self.device)
        self.buffer_initialized = False
        self.generator_exhausted = False
        logger.info("StreamingActivationStore state loaded. Requires a new generator.")

    # Add close method stub for compatibility
    def close(self):
        """Close method stub for StreamingActivationStore."""
        pass


# --------------------------
# Mapped File Implementation (Original - Kept for legacy datasets without manifest)
# DELETING THIS ENTIRE CLASS
# --------------------------

# class MappedActivationStore(BaseActivationStore):
#    ... (all code for MappedActivationStore deleted) ...

# Ensure StreamingActivationStore definition is complete if it was above the deleted section.
# (No changes needed here if StreamingActivationStore was already fully defined)
