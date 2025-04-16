import os
import json
import time
import logging
import numpy as np
import torch
import h5py  # Requires: pip install h5py
from tqdm import tqdm
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import asdict

from clt.nnsight.extractor import ActivationExtractorCLT
from clt.config.data_config import ActivationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Helper function for Welford's online algorithm
def _update_welford(
    existing_count: int,
    existing_mean: torch.Tensor,
    existing_m2: torch.Tensor,
    new_batch: torch.Tensor,
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """
    Updates Welford's algorithm accumulators for a new batch of data.

    Args:
        existing_count: Current count of samples.
        existing_mean: Current mean tensor (shape [d_model]).
        existing_m2: Current sum of squares of differences from the mean (shape [d_model]).
        new_batch: New batch of data (shape [batch_size, d_model]), must be on CPU.

    Returns:
        Tuple containing (new_count, new_mean, new_m2).
    """
    batch_count = new_batch.shape[0]
    if batch_count == 0:
        return existing_count, existing_mean, existing_m2

    new_count = existing_count + batch_count
    # delta_batch shape: [batch_size, d_model]
    delta_batch = new_batch - existing_mean.unsqueeze(0)
    new_mean = existing_mean + delta_batch.sum(dim=0) / new_count

    delta_batch2 = new_batch - new_mean.unsqueeze(0)  # Delta from new mean
    new_m2 = existing_m2 + (delta_batch * delta_batch2).sum(dim=0)

    return new_count, new_mean, new_m2


class ActivationGenerator:
    """Generates and saves model activations based on ActivationConfig."""

    config: ActivationConfig
    extractor: ActivationExtractorCLT

    def __init__(
        self,
        activation_config: ActivationConfig,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initializes the ActivationGenerator.

        Args:
            activation_config: Configuration object detailing generation parameters.
            device: Device for model inference ('cuda', 'cpu', etc.). Overrides device in config if provided.
        """
        self.config = activation_config
        self.storage_type = "local"  # Default, can be updated by driver script

        # --- Basic validation (moved sebagian besar to config.__post_init__) ---
        if self.config.output_format not in ["hdf5", "npz"]:
            raise ValueError(f"Unsupported output_format: {self.config.output_format}")
        if self.config.output_format == "hdf5" and h5py is None:
            raise ImportError(
                "h5py library is required for HDF5 format. Please install it: pip install h5py"
            )
        # Compression validation is handled in config

        # --- Initialize the extractor ---
        # Determine device: use provided arg, else auto-detect
        _device_input = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.device = (
            torch.device(_device_input)
            if isinstance(_device_input, str)
            else _device_input
        )

        self.extractor = ActivationExtractorCLT(
            model_name=self.config.model_name,
            mlp_input_module_path_template=self.config.mlp_input_module_path_template,
            mlp_output_module_path_template=self.config.mlp_output_module_path_template,
            device=self.device,
            model_dtype=self.config.model_dtype,
            context_size=self.config.context_size,
            inference_batch_size=self.config.inference_batch_size,
            exclude_special_tokens=self.config.exclude_special_tokens,
            prepend_bos=self.config.prepend_bos,
            nnsight_tracer_kwargs=self.config.nnsight_tracer_kwargs,
            nnsight_invoker_args=self.config.nnsight_invoker_args,
        )
        logger.info(
            f"ActivationGenerator initialized for model '{self.config.model_name}' on device '{self.extractor.device}'."
        )
        logger.info(
            f"Saving activations to '{self.config.activation_dir}' in '{self.config.output_format}' format."
        )

    def set_storage_type(self, storage_type: str):
        """Sets the storage type ('local' or 'remote'). Called by the driver script."""
        if storage_type.lower() not in ["local", "remote"]:
            raise ValueError(
                f"Invalid storage_type: {storage_type}. Must be 'local' or 'remote'."
            )
        self.storage_type = storage_type.lower()
        logger.info(f"Storage type set to: {self.storage_type}")

    def generate_and_save(self):
        """
        Generates activations for the dataset specified in self.config and saves them.
        Uses parameters directly from self.config.
        """
        # Use parameters from self.config
        dataset_path = self.config.dataset_path
        dataset_split = self.config.dataset_split
        dataset_text_column = self.config.dataset_text_column
        streaming = self.config.streaming
        dataset_trust_remote_code = self.config.dataset_trust_remote_code
        cache_path = self.config.cache_path
        target_total_tokens = self.config.target_total_tokens
        compute_norm_stats = self.config.compute_norm_stats

        logger.info(
            f"Starting activation generation for dataset: '{dataset_path}' [{dataset_split}] column:'{dataset_text_column}'"
        )
        if target_total_tokens:
            logger.info(f"Targeting approximately {target_total_tokens:,} tokens.")
        if compute_norm_stats:
            logger.info("Normalization statistics computation enabled.")

        # --- Create output directory structure ---
        # Example: ./activations/gpt2/openwebtext_train/
        dataset_name = os.path.basename(dataset_path)
        dataset_dir = os.path.join(
            self.config.activation_dir,
            self.config.model_name,
            f"{dataset_name}_{dataset_split}",
        )
        os.makedirs(dataset_dir, exist_ok=True)
        logger.info(f"Output directory: {dataset_dir}")

        # --- Set up activation stream ---
        activation_stream = self.extractor.stream_activations(
            dataset_path=dataset_path,
            dataset_split=dataset_split,
            dataset_text_column=dataset_text_column,
            streaming=streaming,
            dataset_trust_remote_code=dataset_trust_remote_code,
            cache_path=cache_path,
        )

        # --- Process dataset in chunks ---
        chunk_idx = 0
        total_token_count = 0
        tokens_in_current_chunk = 0
        start_time = time.time()

        current_chunk_inputs: Dict[int, List[torch.Tensor]] = {}
        current_chunk_targets: Dict[int, List[torch.Tensor]] = {}

        # Welford's algorithm variables (if enabled)
        stats_counts: Dict[int, Dict[str, int]] = {}
        stats_means: Dict[int, Dict[str, torch.Tensor]] = {}
        stats_m2s: Dict[int, Dict[str, torch.Tensor]] = {}
        final_norm_stats: Dict[int, Dict[str, List[float]]] = {}  # Final results

        layer_indices: Optional[List[int]] = None
        d_model = -1
        dtype_str = "unknown"  # Store dtype as string for metadata
        generator_stopped_early = False

        pbar = tqdm(activation_stream, desc="Generating activations")
        try:
            for batch_inputs, batch_targets in pbar:
                if not batch_inputs:  # Skip empty batches
                    continue

                # --- Initialize layer info on first valid batch ---
                if layer_indices is None:
                    layer_indices = sorted(batch_inputs.keys())
                    if not layer_indices:
                        logger.warning(
                            "Received empty dicts from extractor, skipping batch."
                        )
                        continue
                    first_layer_idx = layer_indices[0]
                    # Use float32 for stats calculation stability, regardless of model dtype
                    stats_dtype = torch.float32
                    # Get d_model and dtype from first layer tensor
                    first_tensor = batch_inputs[first_layer_idx]
                    d_model = first_tensor.shape[-1]
                    tensor_dtype = first_tensor.dtype
                    dtype_str = str(tensor_dtype)  # e.g., "torch.float32"

                    logger.info(
                        f"Detected {len(layer_indices)} layers, d_model={d_model}, dtype={dtype_str}"
                    )
                    # Initialize accumulators
                    for l_idx in layer_indices:
                        current_chunk_inputs[l_idx] = []
                        current_chunk_targets[l_idx] = []
                        if compute_norm_stats:
                            # Initialize Welford stats on CPU with float32
                            stats_counts[l_idx] = {"inputs": 0, "targets": 0}
                            stats_means[l_idx] = {
                                "inputs": torch.zeros(d_model, dtype=stats_dtype),
                                "targets": torch.zeros(d_model, dtype=stats_dtype),
                            }
                            stats_m2s[l_idx] = {
                                "inputs": torch.zeros(d_model, dtype=stats_dtype),
                                "targets": torch.zeros(d_model, dtype=stats_dtype),
                            }

                # --- Accumulate batch data ---
                batch_token_count = 0
                if layer_indices:  # Check if initialized
                    for i, layer_idx in enumerate(layer_indices):
                        # Ensure layer exists in current batch (might not if extractor fails?)
                        if (
                            layer_idx not in batch_inputs
                            or layer_idx not in batch_targets
                        ):
                            logger.warning(
                                f"Layer {layer_idx} missing from current batch. Skipping."
                            )
                            continue

                        inp_tensor = batch_inputs[layer_idx].cpu()  # Move to CPU
                        tgt_tensor = batch_targets[layer_idx].cpu()  # Move to CPU

                        if i == 0:
                            batch_token_count = inp_tensor.shape[0]

                        if batch_token_count > 0:
                            current_chunk_inputs[layer_idx].append(inp_tensor)
                            current_chunk_targets[layer_idx].append(tgt_tensor)

                            # Update normalization stats here (Welford's) if enabled
                            if compute_norm_stats:
                                try:
                                    # Use float32 for update stability
                                    inp_batch_stats = inp_tensor.to(stats_dtype)
                                    (
                                        stats_counts[layer_idx]["inputs"],
                                        stats_means[layer_idx]["inputs"],
                                        stats_m2s[layer_idx]["inputs"],
                                    ) = _update_welford(
                                        stats_counts[layer_idx]["inputs"],
                                        stats_means[layer_idx]["inputs"],
                                        stats_m2s[layer_idx]["inputs"],
                                        inp_batch_stats,
                                    )

                                    tgt_batch_stats = tgt_tensor.to(stats_dtype)
                                    (
                                        stats_counts[layer_idx]["targets"],
                                        stats_means[layer_idx]["targets"],
                                        stats_m2s[layer_idx]["targets"],
                                    ) = _update_welford(
                                        stats_counts[layer_idx]["targets"],
                                        stats_means[layer_idx]["targets"],
                                        stats_m2s[layer_idx]["targets"],
                                        tgt_batch_stats,
                                    )
                                except KeyError as e:
                                    logger.error(
                                        f"KeyError during Welford update for layer {layer_idx}: {e}. "
                                        f"Stat dicts might not be initialized correctly.",
                                        exc_info=True,
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"Error during Welford update for layer {layer_idx}: {e}",
                                        exc_info=True,
                                    )

                # --- Update counts and progress bar ---
                if batch_token_count > 0:
                    tokens_in_current_chunk += batch_token_count
                    total_token_count += batch_token_count
                    pbar.set_postfix(
                        {
                            "Total Tokens": f"{total_token_count:,}",
                            "Chunk Tokens": f"{tokens_in_current_chunk:,}",
                        }
                    )

                    # --- Save chunk if needed ---
                    if tokens_in_current_chunk >= self.config.chunk_token_threshold:
                        logger.info(
                            f"\nChunk {chunk_idx} full ({tokens_in_current_chunk} tokens). Processing..."
                        )
                        self._process_and_save_chunk(
                            current_chunk_inputs,
                            current_chunk_targets,
                            dataset_dir,
                            chunk_idx,
                        )

                        # Reset chunk accumulators
                        tokens_in_current_chunk = 0
                        chunk_idx += 1
                        if layer_indices:  # Check if layer_indices is initialized
                            for l_idx in layer_indices:
                                current_chunk_inputs[l_idx] = []
                                current_chunk_targets[l_idx] = []

                    # --- Check target token count ---
                    if (
                        target_total_tokens is not None
                        and total_token_count >= target_total_tokens
                    ):
                        logger.info(
                            f"\nTarget tokens ({target_total_tokens:,}) reached."
                        )
                        generator_stopped_early = True
                        break  # Stop processing stream

        except StopIteration:
            logger.info("\nActivation stream finished.")
        except Exception as e:
            logger.error(f"Error during activation generation: {e}", exc_info=True)
        finally:
            pbar.close()
            self.extractor.close()  # Clean up nnsight resources

        # --- Save final chunk if any data remains ---
        if (
            tokens_in_current_chunk > 0 and layer_indices
        ):  # Ensure initialization happened
            logger.info(
                f"\nProcessing final chunk {chunk_idx} ({tokens_in_current_chunk} tokens)..."
            )
            self._process_and_save_chunk(
                current_chunk_inputs, current_chunk_targets, dataset_dir, chunk_idx
            )
            chunk_idx += 1

        # --- Finalize and save norm stats (if computed) ---
        stats_were_computed = False
        if compute_norm_stats and layer_indices and stats_counts:
            logger.info("\nComputing final normalization statistics...")
            final_norm_stats = self._finalize_welford(
                stats_counts, stats_means, stats_m2s
            )
            norm_stats_path = os.path.join(dataset_dir, "norm_stats.json")
            if final_norm_stats:
                try:
                    with open(norm_stats_path, "w") as f:
                        json.dump(final_norm_stats, f, indent=2)
                    logger.info(f"Normalization statistics saved to {norm_stats_path}")
                    stats_were_computed = True
                except TypeError as e:
                    logger.error(
                        f"Error serializing normalization stats to JSON: {e}. "
                        f"Stats may contain non-serializable types.",
                        exc_info=True,
                    )
                except Exception as e:
                    logger.error(
                        f"Error saving normalization stats: {e}", exc_info=True
                    )
            else:
                logger.warning(
                    "Normalization computation enabled, but final stats are empty. "
                    "Skipping save."
                )

        # --- Save metadata ---
        logger.info("Saving metadata...")
        end_time = time.time()
        # Prepare config dict for saving (excluding potentially non-serializable fields if any)
        config_dict = asdict(self.config)
        # Remove fields that might be problematic or redundant if needed
        # config_dict.pop('nnsight_tracer_kwargs', None)
        # config_dict.pop('nnsight_invoker_args', None)

        metadata = {
            "activation_config": config_dict,  # Store the config used
            # Add specific dataset stats that might differ from config (e.g., actual tokens)
            "dataset_stats": {
                "num_chunks": chunk_idx,
                "total_tokens_generated": total_token_count,
                "layer_indices": layer_indices,
                "d_model": d_model,
                "dtype": dtype_str,  # Save the detected dtype string
                "computed_norm_stats": compute_norm_stats
                and stats_were_computed,  # Reflect if stats were actually saved
                "generation_duration_seconds": end_time - start_time,
                "generator_stopped_early": generator_stopped_early,
            },
            "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        metadata_path = os.path.join(dataset_dir, "metadata.json")
        try:
            with open(metadata_path, "w") as f:
                # Use default=str to handle potential non-serializable types gracefully
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Metadata saved to {metadata_path}")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}", exc_info=True)

        logger.info(
            f"Activation generation completed in {end_time - start_time:.2f} seconds."
        )
        logger.info(f"Total tokens processed: {total_token_count:,}")
        logger.info(f"Total chunks saved: {chunk_idx}")

    def _process_and_save_chunk(
        self, chunk_inputs, chunk_targets, dataset_dir, chunk_idx
    ):
        """Concatenates lists of tensors for a chunk and saves/sends it."""
        processed_inputs = {}
        processed_targets = {}
        num_tokens = 0

        if not chunk_inputs:
            logger.warning(f"Attempting to save empty chunk {chunk_idx}. Skipping.")
            return

        layer_indices = sorted(chunk_inputs.keys())
        if not layer_indices:
            logger.warning(
                f"Chunk {chunk_idx} input dict has no layers. Skipping save."
            )
            return

        first_layer_data = chunk_inputs.get(layer_indices[0])
        if not first_layer_data:
            logger.warning(
                f"Chunk {chunk_idx} has no data for first layer {layer_indices[0]}. Skipping save."
            )
            return

        try:
            # Determine num_tokens from the first layer's first tensor to avoid iterating if possible
            num_tokens = sum(t.shape[0] for t in first_layer_data)
            if num_tokens == 0:
                logger.warning(
                    f"Chunk {chunk_idx} resulted in 0 tokens before concatenation. Skipping save."
                )
                return

            logger.debug(f"Concatenating tensors for chunk {chunk_idx}...")
            for i, layer_idx in enumerate(layer_indices):
                if not chunk_inputs.get(
                    layer_idx
                ):  # Check if list is empty for this layer
                    logger.warning(
                        f"No input data collected for layer {layer_idx} in chunk {chunk_idx}. "
                        f"Skipping layer."
                    )
                    continue
                if not chunk_targets.get(layer_idx):
                    logger.warning(
                        f"No target data collected for layer {layer_idx} in chunk {chunk_idx}. "
                        f"Skipping layer."
                    )
                    continue

                inputs_cat = torch.cat(chunk_inputs[layer_idx], dim=0)
                targets_cat = torch.cat(chunk_targets[layer_idx], dim=0)
                # Convert back to original model dtype for saving space
                if self.config.model_dtype:
                    try:
                        save_dtype = getattr(
                            torch, self.config.model_dtype.split(".")[-1]
                        )
                        inputs_cat = inputs_cat.to(save_dtype)
                        targets_cat = targets_cat.to(save_dtype)
                    except AttributeError:
                        logger.warning(
                            f"Could not parse model_dtype '{self.config.model_dtype}' "
                            f"for saving chunk. Using original dtype."
                        )
                    except Exception as e:
                        logger.error(
                            f"Error converting chunk to {self.config.model_dtype}: {e}. "
                            f"Using original dtype.",
                            exc_info=True,
                        )

                processed_inputs[layer_idx] = inputs_cat
                processed_targets[layer_idx] = targets_cat
                # Verify token count consistency across layers after concatenation
                if inputs_cat.shape[0] != num_tokens:
                    logger.error(
                        f"Token count mismatch in chunk {chunk_idx}, layer {layer_idx}. "
                        f"Expected {num_tokens}, got {inputs_cat.shape[0]}. Aborting chunk save."
                    )
                    return

        except Exception as e:
            logger.error(
                f"Error concatenating tensors for chunk {chunk_idx}: {e}",
                exc_info=True,
            )
            return  # Skip saving this chunk if concatenation fails

        # Check again if processed dictionaries are empty
        if not processed_inputs or not processed_targets:
            logger.warning(
                f"Chunk {chunk_idx} resulted in empty processed data. Skipping save."
            )
            return

        logger.info(
            f"Saving chunk {chunk_idx} ({num_tokens} tokens) to {self.storage_type}..."
        )
        if self.storage_type == "local":
            self._save_chunk_local(
                processed_inputs, processed_targets, dataset_dir, chunk_idx, num_tokens
            )
        elif self.storage_type == "remote":
            # Generate locally first, then send (stubbed)
            self._save_chunk_local(
                processed_inputs, processed_targets, dataset_dir, chunk_idx, num_tokens
            )
            self._send_chunk_to_server(processed_inputs, processed_targets, chunk_idx)

    def _save_chunk_local(
        self, inputs_dict, targets_dict, dataset_dir, chunk_idx, num_tokens
    ):
        """Saves a processed chunk to disk (HDF5 or NPZ)."""
        chunk_file_base = os.path.join(dataset_dir, f"chunk_{chunk_idx}")
        save_path = f"{chunk_file_base}.{self.config.output_format}"
        start_save = time.time()

        try:
            if self.config.output_format == "hdf5":
                with h5py.File(save_path, "w") as f:
                    f.attrs["num_tokens"] = num_tokens
                    # Optionally store the saved dtype as an attribute
                    if inputs_dict:
                        first_layer = next(iter(inputs_dict))
                        f.attrs["saved_dtype"] = str(inputs_dict[first_layer].dtype)

                    for layer_idx, tensor in inputs_dict.items():
                        # Check if targets exist for this layer before saving
                        if layer_idx not in targets_dict:
                            logger.warning(
                                f"Skipping HDF5 save for layer {layer_idx} in chunk {chunk_idx}: "
                                f"missing target tensor."
                            )
                            continue
                        group = f.create_group(f"layer_{layer_idx}")
                        group.create_dataset(
                            "inputs",
                            data=tensor.numpy(),
                            compression=self.config.compression,
                        )
                        group.create_dataset(
                            "targets",
                            data=targets_dict[layer_idx].numpy(),
                            compression=self.config.compression,
                        )

            elif self.config.output_format == "npz":
                save_dict = {"num_tokens": np.array(num_tokens)}
                # Optionally store saved dtype
                if inputs_dict:
                    first_layer = next(iter(inputs_dict))
                    save_dict["saved_dtype"] = str(inputs_dict[first_layer].dtype)

                for layer_idx, tensor in inputs_dict.items():
                    if layer_idx not in targets_dict:
                        logger.warning(
                            f"Skipping NPZ save for layer {layer_idx} in chunk {chunk_idx}: "
                            f"missing target tensor."
                        )
                        continue
                    save_dict[f"layer_{layer_idx}_inputs"] = tensor.numpy()
                    save_dict[f"layer_{layer_idx}_targets"] = targets_dict[
                        layer_idx
                    ].numpy()

                if not save_dict:  # Don't save empty npz
                    logger.warning(
                        f"NPZ save dictionary for chunk {chunk_idx} is empty. "
                        f"Skipping save."
                    )
                    return

                if self.config.compression:
                    # Note: np.savez_compressed uses zipfile with DEFLATE (like gzip)
                    # It doesn't directly support lz4.
                    if self.config.compression == "lz4":
                        logger.warning(
                            "LZ4 compression not directly supported for NPZ. Using default DEFLATE."
                        )
                    np.savez_compressed(save_path, **save_dict)
                else:
                    np.savez(save_path, **save_dict)

            logger.info(
                f"Chunk {chunk_idx} saved locally to {save_path} in {time.time() - start_save:.2f}s"
            )

        except Exception as e:
            logger.error(
                f"Error saving chunk {chunk_idx} locally to {save_path}: {e}",
                exc_info=True,
            )
            # Attempt to clean up partially written file
            if os.path.exists(save_path):
                try:
                    os.remove(save_path)
                    logger.info(f"Removed partially written file: {save_path}")
                except OSError as rm_err:
                    logger.error(
                        f"Error removing partially written file {save_path}: {rm_err}"
                    )

    def _send_chunk_to_server(self, inputs_dict, targets_dict, chunk_idx):
        """(STUB) Placeholder for sending a processed chunk to a remote server."""
        logger.info(
            f"[STUB] Preparing to send chunk {chunk_idx} to remote server... (Not implemented)"
        )
        # --- Future implementation ---
        # 1. Serialize the inputs_dict and targets_dict (e.g., using torch.save or custom format)
        # 2. Connect to the ActivationStorageServer endpoint
        # 3. Send the serialized data (potentially with metadata like chunk_idx, model_name, dataset_id)
        # 4. Handle response/confirmation from the server
        pass  # Do nothing for now

    def close(self):
        """Clean up resources, like the nnsight model."""
        if hasattr(self, "extractor") and self.extractor:
            self.extractor.close()
            logger.info("ActivationExtractorCLT resources closed.")

    def _finalize_welford(
        self,
        counts: Dict[int, Dict[str, int]],
        means: Dict[int, Dict[str, torch.Tensor]],
        m2s: Dict[int, Dict[str, torch.Tensor]],
    ) -> Dict[int, Dict[str, Dict[str, List[float]]]]:
        """
        Calculates final mean and std deviation from Welford accumulators.

        Args:
            counts: Dictionary {layer_idx: {'inputs': count, 'targets': count}}.
            means: Dictionary {layer_idx: {'inputs': mean_tensor, 'targets': mean_tensor}}.
            m2s: Dictionary {layer_idx: {'inputs': m2_tensor, 'targets': m2_tensor}}.

        Returns:
            Dictionary {layer_idx: {'inputs': {'mean': list, 'std': list},
                                     'targets': {'mean': list, 'std': list}}}
            Returns empty dict if no stats were computed.
        """
        final_stats = {}
        if not counts:  # No stats computed
            return final_stats

        layer_indices = sorted(counts.keys())
        for layer_idx in layer_indices:
            layer_stats = {}
            for key in ["inputs", "targets"]:  # Process inputs and targets
                count = counts[layer_idx][key]
                mean = means[layer_idx][key]
                m2 = m2s[layer_idx][key]

                if count < 2:  # Cannot compute variance with less than 2 samples
                    logger.warning(
                        f"Layer {layer_idx} {key}: Count ({count}) < 2, cannot compute std dev."
                    )
                    variance = torch.full_like(mean, float("nan"))  # Or zeros?
                else:
                    variance = m2 / count  # Population variance

                std = torch.sqrt(variance)
                # Handle potential NaNs from sqrt(negative) due to float precision
                std = torch.nan_to_num(std, nan=0.0)

                layer_stats[key] = {
                    "mean": mean.tolist(),  # Convert to list for JSON
                    "std": std.tolist(),  # Convert to list for JSON
                }
            if layer_stats:  # Only add if we got stats for this layer
                final_stats[layer_idx] = layer_stats

        return final_stats

    # TODO: Add helper to convert tensor stats to JSON serializable format
    # def _norm_stats_to_serializable(stats_dict):
    #     serializable = {}
    #     for layer, stats in stats_dict.items():
    #          serializable[layer] = {k: v.tolist() for k, v in stats.items()}
    #     return serializable
