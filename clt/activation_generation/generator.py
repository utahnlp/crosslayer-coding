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
import requests  # Add requests import
import io  # Add io import
from urllib.parse import urljoin, quote  # Add urljoin and quote import

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
        # --- Enforce HDF5 for remote for now --- #
        if self.storage_type == "remote" and self.config.output_format != "hdf5":
            logger.warning(
                f"Remote storage currently requires HDF5 format. Overriding output_format from '{self.config.output_format}' to 'hdf5'."
            )
            self.config.output_format = "hdf5"
        elif self.config.output_format == "hdf5" and h5py is None:
            raise ImportError(
                "h5py library is required for HDF5 format. Please install it: pip install h5py"
            )

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
        final_norm_stats: Dict[int, Dict[str, Dict[str, List[float]]]] = (
            {}
        )  # Final results

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
        norm_stats_path = None  # Initialize path
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
                # Add format info for clarity
                "output_format": self.config.output_format,
            },
            "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        metadata_path = os.path.join(dataset_dir, "metadata.json")
        try:
            with open(metadata_path, "w") as f:
                # Use default=str to handle potential non-serializable types gracefully
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Metadata saved to {metadata_path}")
            # Upload metadata if remote storage is enabled
            if self.storage_type == "remote":
                self._upload_json_file(metadata_path, "metadata", dataset_dir)
        except Exception as e:
            logger.error(f"Error saving or uploading metadata: {e}", exc_info=True)

        # Upload norm stats after metadata (server might need metadata first)
        if self.storage_type == "remote" and stats_were_computed and norm_stats_path:
            self._upload_json_file(norm_stats_path, "norm_stats", dataset_dir)

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
            first_tensor_dtype_str = None  # Track dtype for HDF5 saving
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

                # Store first tensor dtype for HDF5 attributes
                if i == 0:
                    first_tensor_dtype_str = str(inputs_cat.dtype)

                # Convert back to original model dtype if specified, primarily for NPZ space saving
                # HDF5 handles different dtypes per dataset well
                if self.config.output_format == "npz" and self.config.model_dtype:
                    try:
                        save_dtype = getattr(
                            torch, self.config.model_dtype.split(".")[-1]
                        )
                        inputs_cat = inputs_cat.to(save_dtype)
                        targets_cat = targets_cat.to(save_dtype)
                        if i == 0:
                            first_tensor_dtype_str = str(
                                save_dtype
                            )  # Update if converted
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
        # Pass dtype info to savers
        saved_dtype_str = first_tensor_dtype_str or "unknown"

        if self.storage_type == "local":
            self._save_chunk_local(
                processed_inputs,
                processed_targets,
                dataset_dir,
                chunk_idx,
                num_tokens,
                saved_dtype_str,
            )
        elif self.storage_type == "remote":
            # For remote, we need to save as HDF5 temporarily to get bytes to send
            # unless we implement direct HDF5 byte streaming
            temp_h5_path = os.path.join(dataset_dir, f"_temp_chunk_{chunk_idx}.h5")
            try:
                self._save_chunk_hdf5(
                    temp_h5_path,
                    processed_inputs,
                    processed_targets,
                    num_tokens,
                    saved_dtype_str,
                )
                # Send the HDF5 file bytes
                self._send_chunk_file_to_server(
                    temp_h5_path, chunk_idx, num_tokens, dataset_dir, saved_dtype_str
                )
            except Exception as e:
                logger.error(
                    f"Error during remote processing/sending chunk {chunk_idx}: {e}",
                    exc_info=True,
                )
            finally:
                # Clean up temporary HDF5 file
                if os.path.exists(temp_h5_path):
                    try:
                        os.remove(temp_h5_path)
                        logger.debug(f"Removed temporary HDF5 file: {temp_h5_path}")
                    except OSError as rm_err:
                        logger.error(
                            f"Error removing temporary HDF5 file {temp_h5_path}: {rm_err}"
                        )

            # Optionally still save locally in the configured format if needed for backup/inspection
            # self._save_chunk_local(
            #     processed_inputs, processed_targets, dataset_dir, chunk_idx, num_tokens, saved_dtype_str
            # )

    def _save_chunk_local(
        self,
        inputs_dict,
        targets_dict,
        dataset_dir,
        chunk_idx,
        num_tokens,
        saved_dtype_str,
    ):
        """Saves a processed chunk to disk locally (HDF5 or NPZ)."""
        chunk_file_base = os.path.join(dataset_dir, f"chunk_{chunk_idx}")
        save_path = f"{chunk_file_base}.{self.config.output_format}"
        start_save = time.time()

        try:
            if self.config.output_format == "hdf5":
                self._save_chunk_hdf5(
                    save_path, inputs_dict, targets_dict, num_tokens, saved_dtype_str
                )
            elif self.config.output_format == "npz":
                self._save_chunk_npz(
                    save_path, inputs_dict, targets_dict, num_tokens, saved_dtype_str
                )

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

    def _save_chunk_hdf5(
        self, save_path: str, inputs_dict, targets_dict, num_tokens, saved_dtype_str
    ):
        """Saves a single chunk to an HDF5 file."""
        with h5py.File(save_path, "w") as f:
            f.attrs["num_tokens"] = num_tokens
            f.attrs["saved_dtype"] = saved_dtype_str
            for layer_idx, tensor in inputs_dict.items():
                if layer_idx not in targets_dict:
                    logger.warning(
                        f"Skipping HDF5 save for layer {layer_idx} in {os.path.basename(save_path)}: missing target tensor."
                    )
                    continue
                group = f.create_group(f"layer_{layer_idx}")
                # Convert PyTorch tensor to NumPy array for saving
                group.create_dataset(
                    "inputs",
                    data=tensor.numpy(),
                    compression=self.config.compression,
                    # Optional: Add chunking/shuffling for potentially better read perf
                    # chunks=(64, tensor.shape[1]) if tensor.ndim == 2 else True,
                    # shuffle=True
                )
                group.create_dataset(
                    "targets",
                    data=targets_dict[layer_idx].numpy(),
                    compression=self.config.compression,
                    # chunks=(64, targets_dict[layer_idx].shape[1]) if targets_dict[layer_idx].ndim == 2 else True,
                    # shuffle=True
                )

    def _save_chunk_npz(
        self, save_path: str, inputs_dict, targets_dict, num_tokens, saved_dtype_str
    ):
        """Saves a single chunk to an NPZ file."""
        save_dict = {"num_tokens": np.array(num_tokens), "saved_dtype": saved_dtype_str}
        for layer_idx, tensor in inputs_dict.items():
            if layer_idx not in targets_dict:
                logger.warning(
                    f"Skipping NPZ save for layer {layer_idx} in {os.path.basename(save_path)}: missing target tensor."
                )
                continue
            save_dict[f"layer_{layer_idx}_inputs"] = tensor.numpy()
            save_dict[f"layer_{layer_idx}_targets"] = targets_dict[layer_idx].numpy()

        if len(save_dict) <= 2:  # Only contains num_tokens and dtype
            logger.warning(
                f"NPZ save dictionary for {os.path.basename(save_path)} is effectively empty. Skipping save."
            )
            return

        if self.config.compression:
            if self.config.compression == "lz4":
                logger.warning(
                    "LZ4 compression not directly supported for NPZ. Using default DEFLATE."
                )
            np.savez_compressed(save_path, **save_dict)
        else:
            np.savez(save_path, **save_dict)

    def _send_chunk_file_to_server(
        self,
        file_path: str,
        chunk_idx: int,
        num_tokens: int,
        dataset_dir: str,
        saved_dtype_str: str,
    ):
        """Sends a pre-saved chunk file (e.g., HDF5) to the remote server."""
        if not self.config.remote_server_url:
            logger.error("Remote server URL not configured. Cannot send chunk file.")
            return
        if not os.path.exists(file_path):
            logger.error(f"Chunk file {file_path} not found for upload.")
            return

        try:
            # Construct dataset_id
            model_name = self.config.model_name
            dataset_name = os.path.basename(self.config.dataset_path)
            split = self.config.dataset_split
            dataset_id = quote(f"{model_name}/{dataset_name}_{split}", safe="")

            # Construct target URL
            base_url = self.config.remote_server_url.rstrip("/") + "/"
            # Prepend /api/v1/
            endpoint = f"api/v1/datasets/{dataset_id}/chunks/{chunk_idx}"
            target_url = urljoin(base_url, endpoint)

            logger.info(
                f"Sending chunk file {os.path.basename(file_path)} ({num_tokens} tokens) to {target_url}..."
            )
            start_send = time.time()

            # Prepare headers
            headers = {
                # Content-Type is set automatically by requests for multipart
                # "Content-Type": "application/x-hdf5",
                "X-Num-Tokens": str(num_tokens),
                "X-Saved-Dtype": saved_dtype_str,
            }

            # Read file bytes and send as multipart/form-data
            with open(file_path, "rb") as f:
                # Define the files dictionary for requests
                files = {
                    # Key matches the parameter name in the FastAPI endpoint
                    "chunk_file": (os.path.basename(file_path), f, "application/x-hdf5")
                }
                # Send using the 'files' argument, remove 'data'
                response = requests.post(
                    target_url, files=files, headers=headers, timeout=120
                )  # Increased timeout for large files

            # Check response
            if response.status_code == 201:
                logger.info(
                    f"Chunk file {os.path.basename(file_path)} successfully sent to server in {time.time() - start_send:.2f}s."
                )
            else:
                logger.error(
                    f"Failed to send chunk file {os.path.basename(file_path)}. Server responded with {response.status_code}: {response.text}"
                )
        except requests.exceptions.RequestException as e:
            logger.error(
                f"Network error sending chunk file {os.path.basename(file_path)}: {e}",
                exc_info=True,
            )
        except Exception as e:
            logger.error(
                f"Unexpected error sending chunk file {os.path.basename(file_path)}: {e}",
                exc_info=True,
            )

    def _send_chunk_to_server(
        self, inputs_dict, targets_dict, chunk_idx, num_tokens, dataset_dir
    ):
        """(DEPRECATED - use _send_chunk_file_to_server) Sends a processed chunk to the remote activation server using torch.save bytes."""
        logger.warning(
            "_send_chunk_to_server using torch.save is deprecated. Use HDF5 and _send_chunk_file_to_server instead."
        )
        # ... (previous implementation using torch.save remains here but marked deprecated) ...
        if not self.config.remote_server_url:
            logger.error("Remote server URL not configured. Cannot send chunk.")
            return

        try:
            # Construct dataset_id (e.g., model_name/dataset_name_split)
            model_name = self.config.model_name
            dataset_name = os.path.basename(self.config.dataset_path)
            split = self.config.dataset_split
            # URL-encode the dataset_id to handle characters like '/'
            dataset_id = quote(f"{model_name}/{dataset_name}_{split}", safe="")

            # Construct target URL
            # Ensure base URL ends with / and join paths
            base_url = self.config.remote_server_url.rstrip("/") + "/"
            endpoint = f"datasets/{dataset_id}/chunks/{chunk_idx}"
            target_url = urljoin(base_url, endpoint)

            logger.info(
                f"Sending chunk {chunk_idx} ({num_tokens} tokens) to {target_url} via torch.save..."
            )
            start_send = time.time()

            # Prepare payload (serialize dictionary of tensors)
            payload_dict = {"inputs": inputs_dict, "targets": targets_dict}
            buffer = io.BytesIO()
            torch.save(payload_dict, buffer)
            buffer.seek(0)
            payload_bytes = buffer.read()
            del buffer  # Free buffer memory

            # Prepare headers
            headers = {
                "Content-Type": "application/octet-stream",  # Indicate torch.save bytes
                "X-Num-Tokens": str(num_tokens),
            }

            # Send request
            response = requests.post(
                target_url, data=payload_bytes, headers=headers, timeout=60
            )  # 60s timeout

            # Check response
            if response.status_code == 201:
                logger.info(
                    f"Chunk {chunk_idx} successfully sent to server (torch.save format) in {time.time() - start_send:.2f}s."
                )
            else:
                logger.error(
                    f"Failed to send chunk {chunk_idx} (torch.save format). Server responded with {response.status_code}: {response.text}"
                )

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error sending chunk {chunk_idx}: {e}", exc_info=True)
        except Exception as e:
            logger.error(
                f"Unexpected error sending chunk {chunk_idx}: {e}", exc_info=True
            )

    def _upload_json_file(self, file_path: str, file_type: str, dataset_dir: str):
        """Reads a JSON file and uploads its content to the server."""
        if not self.config.remote_server_url:
            logger.error(
                f"Remote server URL not configured. Cannot upload {file_type}."
            )
            return
        if not os.path.exists(file_path):
            logger.error(
                f"{file_type.capitalize()} file not found at {file_path}. Cannot upload."
            )
            return

        try:
            # Construct dataset_id
            model_name = self.config.model_name
            dataset_name = os.path.basename(self.config.dataset_path)
            split = self.config.dataset_split
            dataset_id = quote(f"{model_name}/{dataset_name}_{split}", safe="")

            # Construct target URL
            base_url = self.config.remote_server_url.rstrip("/") + "/"
            # Prepend /api/v1/
            endpoint = f"api/v1/datasets/{dataset_id}/{file_type}"  # file_type is 'metadata' or 'norm_stats'
            target_url = urljoin(base_url, endpoint)

            logger.info(f"Uploading {file_type} from {file_path} to {target_url}...")

            # Read file content
            with open(file_path, "r") as f:
                json_data = json.load(f)

            # Prepare headers
            headers = {"Content-Type": "application/json"}

            # Send request
            response = requests.post(
                target_url, json=json_data, headers=headers, timeout=30
            )

            # Check response
            if response.status_code in [200, 201]:
                logger.info(
                    f"{file_type.capitalize()} successfully uploaded to server."
                )
            else:
                logger.error(
                    f"Failed to upload {file_type}. Server responded with {response.status_code}: {response.text}"
                )

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error uploading {file_type}: {e}", exc_info=True)
        except json.JSONDecodeError as e:
            logger.error(f"Error reading JSON from {file_path}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error uploading {file_type}: {e}", exc_info=True)

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
        final_stats: Dict[int, Dict[str, Dict[str, List[float]]]] = {}
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
