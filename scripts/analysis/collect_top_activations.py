import torch
import heapq
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

from tqdm import tqdm

from clt.config import CLTConfig
from clt.models.clt import CrossLayerTranscoder
from clt.nnsight.extractor import ActivationExtractorCLT

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define Example structure for typing
Example = Dict[
    str, Any
]  # {"text": List[str], "values": List[float], "raw_sequence_text": str, "token_indices_in_sequence": List[int]}
FeatureHeap = List[Tuple[float, int, Example]]  # Min-heap: (value, counter, example_data)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect top activating examples for CLT features.")
    parser.add_argument(
        "--clt-checkpoint", type=str, required=True, help="Path to the CLT model checkpoint (.pt or .safetensors)"
    )
    parser.add_argument(
        "--clt-config", type=str, required=True, help="Path to the CLT model configuration file (cfg.json)"
    )
    parser.add_argument(
        "--model-name", type=str, required=True, help="Name or path of the base Hugging Face model (e.g., gpt2)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name or path of the Hugging Face dataset (e.g., monology/pile-uncopyrighted)",
    )
    parser.add_argument(
        "--dataset-split", type=str, default="validation", help="Dataset split to use (default: validation)"
    )
    parser.add_argument(
        "--dataset-text-column", type=str, default="text", help="Name of the text column in the dataset (default: text)"
    )

    parser.add_argument("--layer", type=int, default=0, help="CLT layer index to analyze (default: 0)")
    parser.add_argument(
        "--topk-per-feature", type=int, default=10, help="Number of top examples to keep for each feature (default: 10)"
    )

    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for the base language model processing (default: 32)"
    )
    parser.add_argument(
        "--context-size", type=int, default=128, help="Maximum sequence length for the language model (default: 128)"
    )

    parser.add_argument(
        "--device", type=str, default=None, help="Device to run on (cuda, cpu, mps). Auto-detects if None."
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="./top_activating_examples.json",
        help="Path to save the output JSON file (default: ./top_activating_examples.json)",
    )

    parser.add_argument(
        "--max-dataset-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process from the dataset (for quick testing).",
    )
    parser.add_argument(
        "--mlp-input-template",
        type=str,
        default="transformer.h.{}.mlp.c_fc",
        help="NNsight path template for MLP input activations.",
    )
    parser.add_argument(
        "--mlp-output-template",
        type=str,
        default="transformer.h.{}.mlp.c_proj",
        help="NNsight path template for MLP output activations.",
    )

    parser.add_argument(
        "--features-per-file",
        type=int,
        default=1000,
        help="Number of features to include in each output JSON file (default: 1000). Set to 0 for a single file for all features.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device_str = args.device or (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    # 1. Load CLT Model
    logger.info(f"Loading CLT model from checkpoint: {args.clt_checkpoint}")
    clt_config = CLTConfig.from_json(args.clt_config)

    # Initialize model structure first
    clt_model = CrossLayerTranscoder(clt_config, process_group=None, device=device)

    loaded_object = torch.load(args.clt_checkpoint, map_location=device)
    if isinstance(loaded_object, dict):  # It's a state_dict
        # Remove 'module.' prefix if it exists (common from DDP)
        new_state_dict = {}
        for k, v in loaded_object.items():
            if k.startswith("module."):
                new_state_dict[k[len("module.") :]] = v
            else:
                new_state_dict[k] = v
        clt_model.load_state_dict(new_state_dict)
    elif isinstance(loaded_object, CrossLayerTranscoder):  # It's a full model object
        clt_model = loaded_object
        if str(clt_model.device) != str(device):  # Ensure it's on the correct device
            clt_model.to(device)
    else:
        raise ValueError(f"Unsupported checkpoint format. Loaded object type: {type(loaded_object)}")

    clt_model.eval()
    logger.info("CLT model loaded successfully.")

    # Determine MLP path templates: prioritize config, then CLI args
    mlp_input_template = clt_config.mlp_input_template or args.mlp_input_template
    mlp_output_template = clt_config.mlp_output_template or args.mlp_output_template

    if not mlp_input_template or not mlp_output_template:
        logger.error(
            "MLP input and output templates must be provided either in CLT config or via command line arguments."
        )
        exit(1)

    logger.info(f"Using MLP input template: {mlp_input_template}")
    logger.info(f"Using MLP output template: {mlp_output_template}")

    # Attempt to load normalization stats if method is 'auto'
    mean_tensor_for_norm = None
    std_tensor_for_norm = None
    apply_norm = False

    if clt_config.normalization_method == "auto":
        logger.info("CLT config specifies 'auto' normalization. Attempting to load norm_stats.json.")
        try:
            norm_stats_path = Path(args.clt_config).resolve().parent / "norm_stats.json"
            if norm_stats_path.exists():
                logger.info(f"Found potential norm_stats.json at: {norm_stats_path}")
                with open(norm_stats_path, "r") as f:
                    norm_data = json.load(f)

                layer_stats = norm_data.get(str(args.layer))
                if layer_stats:
                    input_stats = layer_stats.get("inputs")
                    if input_stats and "mean" in input_stats and "std" in input_stats:
                        mean_list = input_stats["mean"]
                        std_list = input_stats["std"]

                        # Ensure d_model consistency
                        if len(mean_list) == clt_config.d_model and len(std_list) == clt_config.d_model:
                            mean_tensor_for_norm = torch.tensor(
                                mean_list, device=device, dtype=torch.float32
                            ).unsqueeze(0)
                            std_tensor_for_norm = (
                                torch.tensor(std_list, device=device, dtype=torch.float32) + 1e-6
                            ).unsqueeze(0)
                            apply_norm = True
                            logger.info(f"Successfully loaded normalization stats for layer {args.layer} inputs.")
                        else:
                            logger.warning(
                                f"Mismatch in d_model for normalization stats. Expected {clt_config.d_model}, got mean={len(mean_list)}, std={len(std_list)} for layer {args.layer} inputs in {norm_stats_path}. Proceeding without normalization."
                            )
                    else:
                        logger.warning(
                            f"Normalization stats for layer {args.layer} inputs (mean/std) not found or incomplete in {norm_stats_path}. Proceeding without normalization."
                        )
                else:
                    logger.warning(
                        f"Normalization stats for layer {args.layer} not found in {norm_stats_path}. Proceeding without normalization."
                    )
            else:
                logger.warning(
                    f"CLT config normalization is 'auto', but norm_stats.json not found at expected path: {norm_stats_path}. Proceeding without normalization."
                )
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding norm_stats.json: {e}. Proceeding without normalization.")
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while trying to load norm_stats.json: {e}. Proceeding without normalization.",
                exc_info=True,
            )
    elif clt_config.normalization_method != "none":
        logger.info(
            f"CLT config normalization_method is '{clt_config.normalization_method}' (not 'auto' or 'none'). This script will not apply external normalization based on norm_stats.json."
        )
    else:
        logger.info("CLT config normalization_method is 'none'. No normalization will be applied by this script.")

    # 2. Initialize ActivationExtractorCLT for the base model
    logger.info(f"Initializing ActivationExtractorCLT for base model: {args.model_name}")
    model_dtype_for_extractor = clt_config.expected_input_dtype or "bfloat16"
    if clt_config.expected_input_dtype is None:
        logger.warning(
            "CLTConfig.expected_input_dtype is not set. Defaulting to 'bfloat16' for ActivationExtractorCLT's model_dtype."
        )

    extractor = ActivationExtractorCLT(
        model_name=args.model_name,
        mlp_input_module_path_template=mlp_input_template,
        mlp_output_module_path_template=mlp_output_template,
        device=device,
        model_dtype=model_dtype_for_extractor,
        context_size=args.context_size,
        inference_batch_size=args.batch_size,
    )
    tokenizer = extractor.tokenizer
    logger.info("ActivationExtractorCLT initialized.")

    # 3. Prepare Heaps
    num_features_in_layer = clt_config.num_features
    top_k_examples = args.topk_per_feature
    feature_heaps: List[FeatureHeap] = [[] for _ in range(num_features_in_layer)]
    logger.info(
        f"Initialized {num_features_in_layer} heaps, each to store top {top_k_examples} examples for layer {args.layer}."
    )

    # Unique counter for heap tie-breaking
    example_counter = 0

    # 4. Stream Dataset -> Activations -> Feature Activations -> Update Heaps
    processed_samples_count = 0
    # The extractor.stream_activations yields batches. Each batch contains activations for multiple sequences.
    # Each sequence in that batch, when tokenized, produces a set of token activations.
    # We need to link the text of each *original sequence* to its corresponding *token activations*.

    logger.info("Starting to process dataset samples...")
    # The `stream_activations` method in `ActivationExtractorCLT` processes texts in batches
    # and yields (inputs_dict, targets_dict) for *valid tokens* within those batches.
    # The `inputs_dict` will have keys for layer_idx and values are tensors of shape (n_valid_tokens_in_batch, d_model).
    # We need to reconstruct which tokens belong to which original text sequence.
    # `ActivationExtractorCLT` internally handles batching of texts for the underlying model.
    # The crucial part is that the *output* of `stream_activations` is already token-level activations.

    # We need to get the original texts that produced these batches of token activations.
    # The `ActivationExtractorCLT.stream_activations` takes `dataset_path` and `dataset_split`.
    # It internally loads the dataset and processes it text by text, managing `inference_batch_size`.

    # To link activations back to text, we need to tokenize texts in the same way
    # the extractor does for its internal processing, then align.
    # However, the extractor already yields token-level activations from *valid* (non-padding) tokens.
    # A simpler approach for this script, given its purpose, is to process one text at a time
    # or rely on the fact that the order of activations within a yielded batch from the extractor
    # corresponds to the tokenization of the texts fed into nnsight for that specific model.trace() call.

    # Let's get the text sequences and their tokenized versions to align with activations.
    # The `ActivationExtractorCLT` already processes texts in `inference_batch_size`.
    # We will iterate through the dataset, form batches of text, tokenize them,
    # then pass these *tokenized inputs* to a modified call that gives us activations *and* helps align.

    # Simpler approach: Iterate through the dataset and pass individual texts or small batches
    # directly to the CLT's encode method after getting their MLP activations from the extractor.
    # The current `extractor.stream_activations` is good for generating bulk data, but for finding
    # top activating examples, we need more direct control over text-to-activation mapping.

    # Re-thinking: `stream_activations` yields `batch_input_acts` and `batch_target_acts`
    # where `batch_input_acts[layer_idx]` is a tensor of shape [N_valid_tokens_in_batch, d_model].
    # `N_valid_tokens_in_batch` is the sum of valid tokens from all sequences in that `inference_batch_size`
    # processed by nnsight.
    # We need to know which tokens in this flat list belong to which original sequence.

    # The `ActivationExtractorCLT` processes `args.batch_size` text sequences at a time.
    # For each batch of texts, it tokenizes them, gets activations.
    # We need to capture the original texts for each batch processed by the extractor.

    # Let's refine the loop to work with how `ActivationExtractorCLT` provides data.
    # `ActivationExtractorCLT.stream_activations` yields `(inputs_dict, targets_dict)`
    # `inputs_dict[layer_idx]` has shape `(total_valid_tokens_in_this_yield, d_model)`.
    # This `total_valid_tokens_in_this_yield` comes from `args.batch_size` (extractor's `inference_batch_size`)
    # source texts. We need to segment `feats` according to these original source texts.

    # The `ActivationExtractorCLT` would need to be modified to yield text alongside activations,
    # or we replicate its batching and tokenization logic here carefully.
    # Given the "simple as possible" goal, let's assume we can approximate by
    # processing texts one by one or in small, controlled batches if `stream_activations` is too coarse.

    # Let's adapt the design slightly:
    # We'll use a helper to get activations *and* tokenized text for a single text sequence.

    # For the script's goal, we need text context for each activation.
    # `stream_activations` gives a flat list of valid tokens from a batch of texts.
    # This is hard to map back without more info from the extractor.

    # Alternative: Process dataset, for each text:
    # 1. Tokenize it.
    # 2. Get its MLP activations using a "direct" call to the extractor for that single text.
    #    (Need to add a method to ActivationExtractorCLT for this or replicate its core logic)
    # 3. Then pass to CLT.

    # For simplicity, let's assume `stream_activations` is the main source and try to work with its output.
    # The key challenge is mapping the flat `(N_tokens, d_model)` back to original texts.
    # `ActivationExtractorCLT` has `_preprocess_text` and then tokenizes `current_batch` of texts.
    # The `input_ids` and `attention_mask` from `tokenizer(current_batch, ...)` correspond to these texts.
    # The `valid_indices` are derived from `attention_mask`.
    # If we can reconstruct `current_batch` of texts inside this loop, we can map.

    # Let's load the dataset manually and process it text by text or in small batches here.
    from datasets import load_dataset, IterableDataset

    dataset = load_dataset(
        args.dataset,
        split=args.dataset_split,
        streaming=True,  # Stream to avoid downloading everything if not needed
    )
    if not isinstance(dataset, IterableDataset):
        logger.warning("Dataset is not an IterableDataset. This might be slow if not streaming.")

    pbar = tqdm(total=args.max_dataset_samples, desc="Processing Samples")
    texts_buffer = []

    for sample_idx, item in enumerate(dataset):
        if args.max_dataset_samples is not None and processed_samples_count >= args.max_dataset_samples:
            break

        text = item[args.dataset_text_column]
        if not isinstance(text, str) or not text.strip():
            continue

        texts_buffer.append(text)

        if len(texts_buffer) >= args.batch_size or (
            args.max_dataset_samples is not None
            and processed_samples_count + len(texts_buffer) >= args.max_dataset_samples
        ):

            current_processing_batch = texts_buffer
            texts_buffer = []

            try:
                # Tokenize the current batch of texts
                tokenized_batch = tokenizer(
                    current_processing_batch,
                    truncation=True,
                    max_length=args.context_size,
                    padding="longest",
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                input_ids_batch = tokenized_batch["input_ids"].to(device)
                attention_mask_batch = tokenized_batch["attention_mask"].to(device)

                # Get activations using nnsight trace directly (simplified from ActivationExtractorCLT)
                # This gives us per-sequence activations before flattening, which is easier to map.
                with torch.no_grad(), extractor.model.trace(input_ids_batch):
                    # Get MLP input for the target layer
                    # The proxy returns activations of shape (batch_size, seq_len, d_model)
                    mlp_input_proxy = extractor._get_module_proxy(args.layer, "input")
                    saved_mlp_input_acts = mlp_input_proxy.save()
                    # Ensure trace executes
                    _ = extractor.model.output.logits.shape

                # Now, saved_mlp_input_acts.value is a tensor for the whole batch
                # of shape (batch_size_actual, seq_len_padded, d_model)
                batch_mlp_input_activations_raw = saved_mlp_input_acts
                if isinstance(batch_mlp_input_activations_raw, tuple):  # nnsight sometimes wraps in tuple
                    batch_mlp_input_activations_raw = batch_mlp_input_activations_raw[0]

                # Iterate through each sequence in the batch
                for seq_idx in range(batch_mlp_input_activations_raw.shape[0]):
                    # Get activations for this specific sequence, remove padding
                    seq_len_unpadded = attention_mask_batch[seq_idx].sum().item()
                    mlp_acts_one_sequence = batch_mlp_input_activations_raw[seq_idx, :seq_len_unpadded, :]  # (S, D)

                    if mlp_acts_one_sequence.nelement() == 0:
                        continue

                    activations_to_encode = mlp_acts_one_sequence
                    if apply_norm and mean_tensor_for_norm is not None and std_tensor_for_norm is not None:
                        # logger.debug(f"Applying normalization to activations for layer {args.layer} before encoding.")
                        activations_to_encode = (
                            activations_to_encode.float() - mean_tensor_for_norm
                        ) / std_tensor_for_norm
                        # CLT's encode method will handle casting to its operational dtype

                    # Encode with CLT: mlp_acts_one_sequence is (S, D)
                    # clt.encode expects (B, S, D) or (B*S, D). For single sequence, (S,D) is fine.
                    feature_activations_one_sequence = clt_model.encode(
                        activations_to_encode, args.layer
                    )  # (S, num_features)

                    # Get decoded tokens for this sequence for "text" field
                    decoded_tokens_for_sequence = [
                        tokenizer.decode(token_id) for token_id in input_ids_batch[seq_idx, :seq_len_unpadded]
                    ]

                    # Iterate over features
                    for feat_idx in range(num_features_in_layer):
                        # Get all activation values for this feature across all tokens in the sequence
                        current_feature_values_for_seq = feature_activations_one_sequence[:, feat_idx]  # (S,)

                        if current_feature_values_for_seq.nelement() == 0:
                            continue

                        # Determine the "ranking" value for this example for this feature
                        # Here, we use the max activation value for this feature in the sequence
                        max_activation_for_feature_in_seq = torch.max(current_feature_values_for_seq).item()

                        if (
                            len(feature_heaps[feat_idx]) < top_k_examples
                            or max_activation_for_feature_in_seq > feature_heaps[feat_idx][0][0]
                        ):

                            example_data: Example = {
                                "text": decoded_tokens_for_sequence,  # List of decoded tokens
                                "values": current_feature_values_for_seq.tolist(),  # List of float act values for this feature
                                # For context, maybe store the original full text too
                                # "raw_sequence_text": original_text_for_seq
                            }

                            heapq.heappush(
                                feature_heaps[feat_idx],
                                (max_activation_for_feature_in_seq, example_counter, example_data),
                            )
                            example_counter += 1  # Increment unique counter
                            if len(feature_heaps[feat_idx]) > top_k_examples:
                                heapq.heappop(feature_heaps[feat_idx])

                    processed_samples_count += 1
                    pbar.update(1)
                    if args.max_dataset_samples is not None and processed_samples_count >= args.max_dataset_samples:
                        break
            except Exception as e:
                logger.error(f"Error processing batch: {e}. Original texts: {current_processing_batch}", exc_info=True)

            if args.max_dataset_samples is not None and processed_samples_count >= args.max_dataset_samples:
                break

    # Process any remaining texts in the buffer if the loop ended due to dataset exhaustion
    if texts_buffer and (args.max_dataset_samples is None or processed_samples_count < args.max_dataset_samples):
        current_processing_batch = texts_buffer
        # (Repeat the try-except block from above for the final batch)
        try:
            tokenized_batch = tokenizer(
                current_processing_batch,
                truncation=True,
                max_length=args.context_size,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_ids_batch = tokenized_batch["input_ids"].to(device)
            attention_mask_batch = tokenized_batch["attention_mask"].to(device)

            with torch.no_grad(), extractor.model.trace(input_ids_batch):
                mlp_input_proxy = extractor._get_module_proxy(args.layer, "input")
                saved_mlp_input_acts = mlp_input_proxy.save()
                _ = extractor.model.output.logits.shape

            # Now, saved_mlp_input_acts.value is a tensor for the whole batch
            # of shape (batch_size_actual, seq_len_padded, d_model)
            batch_mlp_input_activations_raw = saved_mlp_input_acts
            if isinstance(batch_mlp_input_activations_raw, tuple):
                batch_mlp_input_activations_raw = batch_mlp_input_activations_raw[0]

            for seq_idx in range(batch_mlp_input_activations_raw.shape[0]):
                if args.max_dataset_samples is not None and processed_samples_count >= args.max_dataset_samples:
                    break
                seq_len_unpadded = attention_mask_batch[seq_idx].sum().item()
                mlp_acts_one_sequence = batch_mlp_input_activations_raw[seq_idx, :seq_len_unpadded, :]

                if mlp_acts_one_sequence.nelement() == 0:
                    continue

                activations_to_encode = mlp_acts_one_sequence
                if apply_norm and mean_tensor_for_norm is not None and std_tensor_for_norm is not None:
                    # logger.debug(f"Applying normalization to activations for layer {args.layer} before encoding (final batch).")
                    activations_to_encode = (activations_to_encode.float() - mean_tensor_for_norm) / std_tensor_for_norm

                feature_activations_one_sequence = clt_model.encode(activations_to_encode, args.layer)
                decoded_tokens_for_sequence = [
                    tokenizer.decode(token_id) for token_id in input_ids_batch[seq_idx, :seq_len_unpadded]
                ]

                for feat_idx in range(num_features_in_layer):
                    current_feature_values_for_seq = feature_activations_one_sequence[:, feat_idx]
                    if current_feature_values_for_seq.nelement() == 0:
                        continue
                    max_activation_for_feature_in_seq = torch.max(current_feature_values_for_seq).item()

                    if (
                        len(feature_heaps[feat_idx]) < top_k_examples
                        or max_activation_for_feature_in_seq > feature_heaps[feat_idx][0][0]
                    ):
                        example_data: Example = {
                            "text": decoded_tokens_for_sequence,
                            "values": current_feature_values_for_seq.tolist(),
                        }
                        heapq.heappush(
                            feature_heaps[feat_idx], (max_activation_for_feature_in_seq, example_counter, example_data)
                        )
                        example_counter += 1  # Increment unique counter
                        if len(feature_heaps[feat_idx]) > top_k_examples:
                            heapq.heappop(feature_heaps[feat_idx])
                processed_samples_count += 1
                pbar.update(1)
        except Exception as e:
            logger.error(
                f"Error processing final batch: {e}. Original texts: {current_processing_batch}", exc_info=True
            )

    pbar.close()
    logger.info("Finished processing dataset samples.")

    # 5. Serialize heaps to JSON
    logger.info(f"Serializing results...")

    num_total_features = len(feature_heaps)
    features_per_file = args.features_per_file
    if features_per_file <= 0:  # Handle 0 or negative as single file for all
        features_per_file = num_total_features

    output_file_base = Path(args.output_json)
    output_dir = output_file_base.parent
    output_stem = output_file_base.stem
    output_suffix = output_file_base.suffix
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    current_feature_batch_data = {}
    file_counter = 0

    for feature_idx, heap in enumerate(feature_heaps):
        # Add current feature data to the current batch
        sorted_examples = sorted(heap, key=lambda x: x[0], reverse=True)
        current_feature_batch_data[str(feature_idx)] = {"activations": {}}
        for rank, (activation_value, _, example_data) in enumerate(sorted_examples):
            current_feature_batch_data[str(feature_idx)]["activations"][str(rank)] = {
                "text": example_data["text"],
                "values": example_data["values"],
                "score": activation_value,
            }

        # Check if the current batch is full or if it's the last feature
        if (feature_idx + 1) % features_per_file == 0 or (feature_idx + 1) == num_total_features:
            start_feature_in_file = file_counter * features_per_file
            end_feature_in_file = feature_idx  # Inclusive end index for this file

            if num_total_features <= features_per_file:  # Single file case
                current_output_filename = output_file_base
            else:
                current_output_filename = (
                    output_dir
                    / f"{output_stem}_features_{start_feature_in_file}_to_{end_feature_in_file}{output_suffix}"
                )

            logger.info(
                f"Writing feature batch to {current_output_filename} (features {start_feature_in_file}-{end_feature_in_file})"
            )
            try:
                with open(current_output_filename, "w") as f:
                    json.dump(current_feature_batch_data, f, indent=2)
                logger.info(f"Successfully saved batch to {current_output_filename}")
            except IOError as e:
                logger.error(f"Error writing JSON to {current_output_filename}: {e}")
            except TypeError as e:
                logger.error(
                    f"Error serializing data to JSON for {current_output_filename} (likely non-serializable type): {e}"
                )

            current_feature_batch_data = {}  # Reset for the next batch
            file_counter += 1

    logger.info("All features processed and results serialized.")


if __name__ == "__main__":
    main()
