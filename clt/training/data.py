import torch
from typing import Dict, List, Tuple, Optional, Union, Generator
import logging
import time
from tqdm import tqdm
import sys
import datetime
import gc  # Import Python garbage collector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type hint for the generator output
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


class ActivationStore:
    """Manages model activations for CLT training using a streaming generator.

    Buffers activations efficiently, yields batches for training, and handles
    optional normalization.
    """

    # Type hints for instance variables
    activation_generator: Generator[ActivationBatchCLT, None, None]
    n_batches_in_buffer: int
    train_batch_size_tokens: int
    normalization_method: str
    normalization_estimation_batches: int
    device: torch.device

    layer_indices: List[int]
    d_model: int
    dtype: torch.dtype

    # Buffers store activations per layer
    buffered_inputs: Dict[int, torch.Tensor]
    buffered_targets: Dict[int, torch.Tensor]

    # Read mask tracks yielded tokens across the unified buffer length
    read_indices: torch.Tensor

    # Normalization statistics (per layer)
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
        n_batches_in_buffer: int = 16,
        train_batch_size_tokens: int = 4096,
        normalization_method: str = "none",  # Default to no normalization
        normalization_estimation_batches: int = 50,
        device: Optional[Union[str, torch.device]] = None,
        start_time: Optional[float] = None,
    ):
        """Initialize the streaming activation store for CLT.

        Args:
            activation_generator: Generator yielding tuples of
                                  (inputs_dict, targets_dict). Dictionaries map layer_idx
                                  to tensors of shape [n_valid_tokens, d_model].
            n_batches_in_buffer: Number of training batches worth of tokens to aim for
                                 in the buffer.
            train_batch_size_tokens: Number of tokens per training batch yielded by get_batch().
            normalization_method: Normalization method ('none', 'estimated_mean_std').
            normalization_estimation_batches: Number of generator batches to use for estimating
                                             normalization statistics if method is 'estimated_mean_std'.
            device: Device to store activations on ('cuda', 'cpu', etc.). Auto-detects if None.
            start_time: The initial time.time() from the trainer for elapsed time logging.
        """
        self.activation_generator = activation_generator
        self.n_batches_in_buffer = n_batches_in_buffer
        self.train_batch_size_tokens = train_batch_size_tokens
        self.normalization_method = normalization_method
        self.normalization_estimation_batches = normalization_estimation_batches
        self.start_time = start_time or time.time()

        # Set device
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
        self.layer_indices = []
        self.d_model = -1
        self.dtype = torch.float32  # Default, will be updated

        self.buffered_inputs = {}
        self.buffered_targets = {}
        self.read_indices = torch.empty(0, dtype=torch.bool, device=self.device)

        # Initialize normalization stats placeholders
        self.input_means = {}
        self.input_stds = {}
        self.output_means = {}
        self.output_stds = {}

        logger.info(f"ActivationStore initialized on {self.device}.")
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

        # --- Commented out memory logging --- #
        # if torch.cuda.is_available() and self.device.type == "cuda":
        #     mem_before = torch.cuda.memory_allocated(self.device) / (1024**2)  # MB
        #     elapsed_str = _format_elapsed_time(time.time() - self.start_time)
        #     logger.debug(
        #         f"Fill Buffer - Start [{elapsed_str}]. Mem: {mem_before:.2f} MB. Unread: {num_unread}, Needed: {max(0, tokens_needed)}"
        #     )
        # else:
        #     elapsed_str = _format_elapsed_time(time.time() - self.start_time)
        #     logger.debug(
        #         f"Fill Buffer - Start [{elapsed_str}]. Unread: {num_unread}, Needed: {max(0, tokens_needed)}"
        #     )

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
                break  # Exit loop if generator is done
            except Exception as e:
                logger.error(
                    f"Error fetching or processing batch from generator: {e}",
                    exc_info=True,
                )
                # Decide whether to re-raise or try to continue
                raise e  # Re-raise by default

        end_time = time.time()
        current_buffer_size = self.read_indices.shape[0]
        final_unread = (~self.read_indices).sum().item()
        logger.debug(
            f"Buffer fill finished in {end_time - start_time:.2f}s. Added {tokens_added_this_fill} tokens. "
            f"Total buffer size: {current_buffer_size}. Unread tokens: {final_unread}."
        )

        # --- Commented out memory logging --- #
        # if torch.cuda.is_available() and self.device.type == "cuda":
        #     mem_after = torch.cuda.memory_allocated(self.device) / (1024**2)  # MB
        #     elapsed_str = _format_elapsed_time(time.time() - self.start_time)
        #     logger.debug(
        #         f"Fill Buffer - End [{elapsed_str}]: Mem: {mem_after:.2f} MB (+{mem_after - mem_before:.2f} MB). Added: {tokens_added_this_fill}"
        #     )
        # else:
        #     elapsed_str = _format_elapsed_time(time.time() - self.start_time)
        #     logger.debug(
        #         f"Fill Buffer - End [{elapsed_str}]. Added: {tokens_added_this_fill}"
        #     )

        # Check if buffer is still empty after trying to fill
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
            num_to_prune = first_unread_idx
            # --- Commented out memory logging --- #
            # mem_before = 0
            # if torch.cuda.is_available() and self.device.type == "cuda":
            #     mem_before = torch.cuda.memory_allocated(self.device) / (1024**2)  # MB

            for layer_idx in self.layer_indices:
                self.buffered_inputs[layer_idx] = self.buffered_inputs[layer_idx][
                    num_to_prune:
                ]
                self.buffered_targets[layer_idx] = self.buffered_targets[layer_idx][
                    num_to_prune:
                ]
            self.read_indices = self.read_indices[num_to_prune:]
            # logger.debug(f"Pruned {num_to_prune} read tokens from buffer. New size: {self.read_indices.shape[0]}")

            if torch.cuda.is_available() and self.device.type == "cuda":
                # --- Force Python GC and CUDA cache clearing AFTER pruning tensors --- #
                gc.collect()  # Force Python GC
                torch.cuda.empty_cache()
                # --- Log memory AFTER cache clearing --- #
                # --- Commented out memory logging --- #
                # mem_after = torch.cuda.memory_allocated(self.device) / (1024**2)  # MB
                # elapsed_str = _format_elapsed_time(time.time() - self.start_time)
                # logger.debug(
                #     f"Prune Buffer [{elapsed_str}]: Pruned {num_to_prune}. Mem Before: {mem_before:.2f} MB, After: {mem_after:.2f} MB, Diff: {mem_after - mem_before:.2f} MB"
                # )

    def get_batch(self) -> ActivationBatchCLT:
        """Gets a randomly sampled batch of activations for training.

        Refills the buffer from the generator if it falls below half capacity.
        Prunes read tokens from the start of the buffer periodically.

        Returns:
            Tuple of (inputs_dict, targets_dict) dictionaries, where each maps
            layer_idx to a tensor of shape [train_batch_size_tokens, d_model].

        Raises:
            StopIteration: If the generator is exhausted and the buffer becomes empty.
            RuntimeError: If unable to provide a batch after attempting to refill.
        """
        # --- Commented out per-batch memory logging --- #
        # mem_start_get_batch = 0.0 # Type hint was fixed previously
        # if torch.cuda.is_available() and self.device.type == "cuda":
        #     mem_start_get_batch = torch.cuda.memory_allocated(self.device) / (
        #         1024**2
        #     )  # MB
        #     elapsed_str = _format_elapsed_time(time.time() - self.start_time)
        #     logger.debug(
        #         f"Get Batch - Start [{elapsed_str}]. Mem: {mem_start_get_batch:.2f} MB"
        #     )

        # Initialize and fill buffer on first call or if needed
        num_unread = (~self.read_indices).sum().item()
        if (
            not self.buffer_initialized
            or num_unread < self.target_buffer_size_tokens // 2
        ):
            if not self.generator_exhausted:
                self._fill_buffer()
            # Re-check unread count after trying to fill
            num_unread = (~self.read_indices).sum().item()

        # If still no unread tokens after trying to fill (generator might be exhausted)
        if num_unread == 0:
            if self.generator_exhausted:
                logger.info(
                    "Generator exhausted and buffer empty. Signalling end of iteration."
                )
                raise StopIteration
            else:
                # This case should ideally be rare if _fill_buffer works correctly
                logger.error(
                    "Buffer has no unread tokens despite generator not being marked as exhausted."
                )
                raise RuntimeError("Failed to get unread tokens after buffer refill.")

        # --- Sample indices ---
        unread_token_indices = (~self.read_indices).nonzero().squeeze(-1)

        # Determine how many tokens to sample for the batch
        num_to_sample = min(self.train_batch_size_tokens, len(unread_token_indices))
        if (
            num_to_sample == 0
        ):  # Should not happen if checks above passed, but safeguard
            raise RuntimeError(
                "No unread indices available for sampling, despite earlier checks."
            )

        # Randomly sample from the unread indices
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

        # --- Prune buffer ---
        # Pruning can be done less frequently for efficiency, e.g., every N batches
        # Or based on the number of read tokens at the start.
        self._prune_buffer()  # Prune every time for simplicity here

        # --- Commented out per-batch memory logging --- #
        # if torch.cuda.is_available() and self.device.type == "cuda":
        #     mem_end_get_batch = torch.cuda.memory_allocated(self.device) / (
        #         1024**2
        #     )  # MB
        #     elapsed_str = _format_elapsed_time(time.time() - self.start_time)
        #     logger.debug(
        #         f"Get Batch - End [{elapsed_str}]. Mem: {mem_end_get_batch:.2f} MB (+{mem_end_get_batch - mem_start_get_batch:.2f} MB)"
        #     )

        return batch_inputs, batch_targets

    def _estimate_normalization_stats(self):
        """
        Simplified estimation of normalization stats by collecting up to
        'normalization_estimation_batches' from the generator and computing
        mean/std for each layer in a single pass.
        """
        if self.normalization_method != "estimated_mean_std":
            return

        # Reset or initialize stats containers
        self.input_means = {}
        self.input_stds = {}
        self.output_means = {}
        self.output_stds = {}

        # Prepare placeholders to accumulate activations
        all_inputs = {}
        all_outputs = {}

        # We'll initialize buffer metadata after the first actual batch
        # so that layer_indices/dtype/device are set properly.
        first_batch_seen = False

        # Print status and force display
        print("\nComputing normalization statistics...")
        sys.stdout.flush()  # Force output to display

        # Create progress bar
        norm_progress = tqdm(
            range(self.normalization_estimation_batches),
            desc="Normalization Progress",
            leave=True,  # Keep the progress bar after completion
        )

        for batch_idx in norm_progress:
            try:
                batch_inputs, batch_targets = next(self.activation_generator)

                # Update progress description
                norm_progress.set_description(
                    f"Processed batch {batch_idx+1}/{self.normalization_estimation_batches}"
                )

                if not first_batch_seen:
                    self._initialize_buffer_metadata((batch_inputs, batch_targets))
                    first_batch_seen = True
                    # Build empty lists for each layer
                    for layer_idx in self.layer_indices:
                        all_inputs[layer_idx] = []
                        all_outputs[layer_idx] = []

                for layer_idx in self.layer_indices:
                    in_tensor = batch_inputs[layer_idx].to(self.device).to(self.dtype)
                    out_tensor = batch_targets[layer_idx].to(self.device).to(self.dtype)
                    all_inputs[layer_idx].append(in_tensor)
                    all_outputs[layer_idx].append(out_tensor)

            except StopIteration:
                self.generator_exhausted = True
                break

        # Compute mean and std for each layer
        if first_batch_seen:
            for layer_idx in self.layer_indices:
                # Concatenate all data for layer_idx
                in_cat = (
                    torch.cat(all_inputs[layer_idx], dim=0)
                    if len(all_inputs[layer_idx]) > 0
                    else None
                )
                out_cat = (
                    torch.cat(all_outputs[layer_idx], dim=0)
                    if len(all_outputs[layer_idx]) > 0
                    else None
                )

                if in_cat is None or out_cat is None:
                    continue

                self.input_means[layer_idx] = in_cat.mean(dim=0, keepdim=True)
                self.input_stds[layer_idx] = in_cat.std(dim=0, keepdim=True) + 1e-6

                self.output_means[layer_idx] = out_cat.mean(dim=0, keepdim=True)
                self.output_stds[layer_idx] = out_cat.std(dim=0, keepdim=True) + 1e-6

            # Store initial batches in buffer for training
            for in_dict, out_dict in zip(all_inputs.values(), all_outputs.values()):
                for batch_in, batch_out in zip(in_dict, out_dict):
                    if not self.buffer_initialized:
                        self._initialize_buffer_metadata((batch_in, batch_out))
                    self._add_batch_to_buffer((batch_in, batch_out))

            total_batches = sum(len(tensors) for tensors in all_inputs.values())
            logger.info(f"Normalization complete using {total_batches} batches")

            # Signal completion
            print("\n>>> NORMALIZATION PHASE COMPLETE - STARTING TRAINING <<<\n")
            sys.stdout.flush()  # Force output to display
        else:
            logger.warning(
                "No data processed for normalization. Using identity normalization."
            )
            self.normalization_method = "none"  # Fall back to no normalization

    def _normalize_batch(
        self,
        inputs_dict: Dict[int, torch.Tensor],
        targets_dict: Dict[int, torch.Tensor],
    ) -> ActivationBatchCLT:
        """
        Applies the estimated mean/std to each layer's inputs and targets.
        If no stats exist for a layer, returns the tensors unchanged.
        """
        normalized_inputs = {}
        normalized_targets = {}

        for layer_idx in self.layer_indices:
            inp = inputs_dict[layer_idx].to(self.device)
            tgt = targets_dict[layer_idx].to(self.device)

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
            return outputs  # No-op if not normalized or no stats

        denormalized = {}
        for layer_idx, output in outputs.items():
            if layer_idx in self.output_means:
                mean = self.output_means[layer_idx]
                std = self.output_stds[layer_idx]
                # Ensure mean/std are on the same device as the output tensor
                denormalized[layer_idx] = (output * std.to(output.device)) + mean.to(
                    output.device
                )
            else:
                logger.warning(
                    f"Attempting to denormalize layer {layer_idx} but no statistics found."
                )
                denormalized[layer_idx] = output  # Return as is if no stats
        return denormalized

    def state_dict(self) -> Dict:
        """Return a dictionary containing the state for saving/resumption."""
        # Convert tensors to CPU before saving to ensure compatibility
        cpu_input_means = {k: v.cpu() for k, v in self.input_means.items()}
        cpu_input_stds = {k: v.cpu() for k, v in self.input_stds.items()}
        cpu_output_means = {k: v.cpu() for k, v in self.output_means.items()}
        cpu_output_stds = {k: v.cpu() for k, v in self.output_stds.items()}

        return {
            "layer_indices": self.layer_indices,
            "d_model": self.d_model,
            "dtype": str(self.dtype),  # Save dtype as string
            "input_means": cpu_input_means,
            "input_stds": cpu_input_stds,
            "output_means": cpu_output_means,
            "output_stds": cpu_output_stds,
            "total_tokens_yielded_by_generator": self.total_tokens_yielded_by_generator,
            "target_buffer_size_tokens": self.target_buffer_size_tokens,
            "normalization_method": self.normalization_method,
            # Note: We don't save buffer data, read indices, or generator state.
            # Resumption requires a fresh generator.
        }

    def load_state_dict(self, state_dict: Dict):
        """Load state from a dictionary."""
        self.layer_indices = state_dict["layer_indices"]
        self.d_model = state_dict["d_model"]
        # Convert dtype string back to torch.dtype
        try:
            self.dtype = getattr(torch, state_dict["dtype"].split(".")[-1])
        except AttributeError:
            logger.warning(
                f"Could not parse dtype string '{state_dict['dtype']}'. Defaulting to {self.dtype}."
            )

        # Load normalization stats and move to the correct device
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
        self.normalization_method = state_dict["normalization_method"]

        # Reset buffer state as it's not saved
        self.buffered_inputs = {}
        self.buffered_targets = {}
        self.read_indices = torch.empty(0, dtype=torch.bool, device=self.device)
        self.buffer_initialized = False  # Re-initialize on first get_batch
        self.generator_exhausted = False  # Assume new generator is not exhausted

        logger.info(
            f"ActivationStore state loaded. Resuming from {self.total_tokens_yielded_by_generator} generator tokens."
        )
        # It's crucial that the provided activation_generator on __init__
        # can correctly resume or start fresh as needed.

    def __iter__(self):
        return self

    def __next__(self):
        """Allows the store to be used as an iterator yielding batches."""
        return self.get_batch()
