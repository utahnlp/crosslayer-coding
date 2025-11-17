import torch
from datasets import load_dataset, Dataset, IterableDataset, Features, Value
from nnsight import LanguageModel
from tqdm import tqdm
from typing import Generator, Dict, Tuple, Optional, Union, List
import logging
import os

# manually specified features for olmo_mix since the hf dataset is not setup properly
from clt.nnsight.olmomix_features import wiki_features, pes2o_features, algebraicstack_features, arxiv_features, dclm_features, openwebmath_features, starcoder_features, olmomix_data_files, dolmino_data_files

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActivationExtractorCLT:
    """
    Extracts paired activations (MLP input and output) from specified layers
    of a transformer model using nnsight. Designed for CLT training.
    """

    def __init__(
        self,
        model_name: str,
        mlp_input_module_path_template: str,
        mlp_output_module_path_template: str,
        device: Optional[Union[str, torch.device]] = None,
        model_dtype: Optional[Union[str, torch.dtype]] = None,
        context_size: int = 128,
        inference_batch_size: int = 512,
        exclude_special_tokens: bool = True,
        prepend_bos: bool = False,
        nnsight_tracer_kwargs: Optional[Dict] = None,
        nnsight_invoker_args: Optional[Dict] = None,
        batchtopk_k: Optional[int] = None,
    ):
        """
        Initializes the ActivationExtractorCLT.

        Args:
            model_name: Name or path of the Hugging Face transformer model.
            mlp_input_module_path_template: String template for the NNsight path
                                             to MLP input modules. Must contain '{}'
                                             for layer index.
            mlp_output_module_path_template: String template for the NNsight path
                                              to MLP output modules. Must contain '{}'
                                              for layer index.
            device: Device to run the model on ('cuda', 'cpu', etc.).
                    Auto-detects if None.
            model_dtype: Optional data type for model weights
                         (e.g., torch.float16, "bfloat16").
            context_size: Maximum sequence length for tokenization.
            inference_batch_size: Number of text prompts to process in each
                                      model forward pass.
            exclude_special_tokens: Whether to exclude activations corresponding
                                    to special tokens.
            prepend_bos: Whether to prepend the BOS token (required by some models).
            nnsight_tracer_kwargs: Additional kwargs for nnsight model.trace().
            nnsight_invoker_args: Additional invoker_args for nnsight model.trace().
            batchtopk_k: Optional k parameter for BatchTopK.
        """
        self.model_name = model_name
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model_dtype = self._resolve_dtype(model_dtype)
        self.context_size = context_size
        self.inference_batch_size = inference_batch_size
        self.mlp_input_module_path_template = mlp_input_module_path_template
        self.mlp_output_module_path_template = mlp_output_module_path_template
        self.exclude_special_tokens = exclude_special_tokens
        self.prepend_bos = prepend_bos

        # Store BatchTopK params if provided, though not directly used by current extractor logic
        # This is for potential future use or if downstream components expect them via this config path
        self.batchtopk_k = batchtopk_k

        # Tokenizer arguments
        self.tokenizer_args = {
            "truncation": True,
            "max_length": self.context_size,
            "padding": "longest",  # Ensure all sequences in the batch have the same length
            "return_tensors": "pt",  # Ensure PyTorch tensors are returned
            "add_special_tokens": True
        }

        self._default_tracer_kwargs = {"scan": False, "validate": False}
        self.nnsight_tracer_kwargs = nnsight_tracer_kwargs or {}
        self._default_invoker_args = {
            "truncation": True,
            "max_length": self.context_size,
        }
        self.nnsight_invoker_args = nnsight_invoker_args or {}

        self.model, self.tokenizer = self._load_model_and_tokenizer()
        self.num_layers = self._get_num_layers()

    def _resolve_dtype(self, dtype_input: Optional[Union[str, torch.dtype]]) -> Optional[torch.dtype]:
        """Converts string dtype names to torch.dtype objects."""
        if isinstance(dtype_input, torch.dtype):
            return dtype_input
        if isinstance(dtype_input, str):
            try:
                return getattr(torch, dtype_input)
            except AttributeError:
                logger.warning(f"Unsupported dtype string: '{dtype_input}'. Ignoring.")
                return None
        return None

    def _load_model_and_tokenizer(self):
        """Loads the LanguageModel and its tokenizer, passing dtype."""
        model = LanguageModel(
            self.model_name,
            device_map=self.device,
            torch_dtype=self.model_dtype,
            dispatch=True,
        )
        tokenizer = model.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def _get_num_layers(self) -> int:
        """Detects the number of layers in the transformer model."""
        if hasattr(self.model.config, "num_hidden_layers"):
            return self.model.config.num_hidden_layers
        elif hasattr(self.model.config, "n_layer"):
            return self.model.config.n_layer
        else:
            # Try to infer by checking model structure directly
            try:
                layer_idx = 0
                while hasattr(self.model.transformer.h, str(layer_idx)):
                    layer_idx += 1
                return layer_idx
            except (AttributeError, IndexError):
                # Fallback: inspect module paths directly
                try:
                    layer_idx = 0
                    while True:
                        try:
                            # Try to access a layer to see if it exists
                            _ = self._get_module_proxy(layer_idx, "input")
                            layer_idx += 1
                        except Exception:
                            break
                    if layer_idx == 0:
                        raise ValueError("Could not determine number of layers")
                    return layer_idx
                except Exception as e:
                    raise ValueError(f"Failed to detect number of layers: {e}")

    def _get_module_proxy(self, layer_idx: int, module_type: str):
        """
        Gets the nnsight module proxy for a given layer and type.
        Uses the path templates defined in the instance.
        Navigates the model structure using getattr and indexing.
        """
        if module_type == "input":
            path_str = self.mlp_input_module_path_template.format(layer_idx)
        elif module_type == "output":
            path_str = self.mlp_output_module_path_template.format(layer_idx)
        else:
            raise ValueError(f"Invalid module_type: {module_type}")

        # Navigate the model structure using the path string
        proxy = self.model  # Start with the root model proxy
        try:
            parts = path_str.split(".")
            for part in parts:
                if part.isdigit():  # Handle numerical indices like in transformer.h[0]
                    proxy = proxy[int(part)]
                else:  # Handle attribute access
                    proxy = getattr(proxy, part)
            return proxy
        except (AttributeError, KeyError, IndexError, TypeError) as e:
            # Catch potential errors during navigation
            raise AttributeError(f"Could not find module at path '{path_str}': {e}")

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess and chunk long text into manageable segments."""
        if not isinstance(text, str) or not text.strip():
            return []

        # If text is already short, return it as is
        # Rough check based on characters; tokenization might vary
        # A more robust check would tokenize first, but adds overhead
        if len(text) < self.context_size * 3:
            return [text]

        # For long texts, tokenize and chunk to avoid sequence length issues
        # Ensure add_special_tokens is False for chunking logic
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        # If token length fits within context size (allowing for potential BOS/EOS later)
        if len(tokens) <= self.context_size - (1 if self.prepend_bos else 0) - 1:  # -1 for potential EOS
            return [text]

        # Split into chunks, leaving room for special tokens if needed
        # The exact room needed depends on tokenizer and prepend_bos
        # Conservatively leave space for BOS and EOS if necessary
        room_for_specials = (1 if self.prepend_bos else 0) + 1  # BOS + EOS
        chunk_size = self.context_size - room_for_specials
        if chunk_size <= 0:
            logger.warning(
                f"Context size {self.context_size} too small for chunking with special tokens. Skipping text."
            )
            return []

        token_chunks = [tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size)]

        # Convert token chunks back to strings
        text_chunks = [self.tokenizer.decode(chunk) for chunk in token_chunks]
        return text_chunks

    def stream_activations(
        self,
        dataset_path: str,
        dataset_split: str = "train",
        dataset_text_column: str = "text",
        dataset_skip: int = None,
        streaming: bool = True,
        dataset_trust_remote_code: Optional[bool] = False,
        cache_path: Optional[str] = None,
        pbar: bool = True
    ) -> Generator[Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]], None, None]:
        """
        Streams paired MLP input and output activations from the model for a given dataset.

        Args:
            dataset_path: Path or name of the Hugging Face dataset.
            dataset_split: Dataset split to use (e.g., 'train', 'validation').
            dataset_text_column: Name of the column containing text data.
            dataset_skip: Number of dataset examples to skip.
            streaming: Whether to use dataset streaming.
            dataset_trust_remote_code: Whether to trust remote code for the dataset.
            cache_path: Optional path to cache downloaded data (relevant if not streaming).

        Yields:
            Tuples of (inputs_dict, targets_dict), where each dict maps layer_idx
            to a tensor of activations for that layer, filtered for valid tokens.
            Tensor shape: (n_valid_tokens, d_model)
        """
        # Handle the case where dataset_trust_remote_code is None
        trust_remote_code = False if dataset_trust_remote_code is None else dataset_trust_remote_code

        # Normalize dataset_path to a list
        if isinstance(dataset_path, str):
            dataset_path = [dataset_path]

        def add_source_columns(subset, ds_name, ss=""):
            return subset.map(lambda x: {"data_source": ds_name, "subset_source": ss})
              
        def extract_text(batch, col_names="messages"):
            out_texts = []
            # col_names is a list → preference dataset
            if isinstance(col_names, list):
                chosen_list = batch[col_names[0]]
                rejected_list = batch[col_names[1]]

                for chosen, rejected in zip(chosen_list, rejected_list):
                    # chosen
                    t1 = self.tokenizer.apply_chat_template(chosen, tokenize=False)
                    # rejected
                    t2 = self.tokenizer.apply_chat_template(rejected, tokenize=False)
                    # query only (first message from chosen)
                    t3 = self.tokenizer.apply_chat_template([chosen[0]], tokenize=False)

                    out_texts.extend([t1, t2, t3])
                return {"text": out_texts}

            # single-column SFT dataset
            else:
                text = self.tokenizer.apply_chat_template(batch[col_names], tokenize=False)
                return {"text": text}

        
        PRETRAINING = ["allenai/dolmino-mix-1124", "allenai/olmo-mix-1124"]
        SFT = ["allenai/tulu-3-sft-olmo-2-mixture-0225"]
        RL = ["allenai/olmo-2-0425-1b-preference-mix"] + ["allenai/RLVR-GSM-MATH-IF-Mixed-Constraints", "allenai/RLVR-MATH"]

        streamed_datasets = []
        val_set_names = set()
        print('dataset_path=', dataset_path)
        for dataset in dataset_path:
            logger.info(f'Loading dataset {dataset}')
            ds_name = "/".join(dataset.split('/'))
            # Olmo-mix and Dolmino isn't setup properly on huggingface, so we need to load each subset individually and then interleave the subsets.
            if dataset in PRETRAINING:
                val_set_names.add('pretraining')
                # We have specified the fixed datapaths in a separate file.
                datafiles = olmomix_data_files if dataset == "allenai/olmo-mix-1124" else dolmino_data_files
                # get each separately loaded subset
                ds = []
                # for sub, sub_datafiles in zip(subsets, datafiles):
                for sub_datafiles in datafiles:
                    # print("sub=", sub_datafiles)
                    subset = load_dataset(
                        dataset,
                        split=dataset_split,
                        streaming=streaming,
                        trust_remote_code=trust_remote_code,
                        cache_dir=cache_path,
                        # features=sub,
                        data_files=sub_datafiles
                    )
                    # track which dataset and subset the example is coming from
                    ss = sub_datafiles.split('/')[1]
                    # Keep only text column to unify schema between datasets
                    subset = subset.select_columns(["text"]) 
                    # Add a constant columns
                    subset = add_source_columns(subset, ds_name, ss)
                    # shuffle the subset
                    subset = subset.shuffle(seed=1, buffer_size=10000)
                    ds.append(subset)
                
                logger.info(f'Pretraining subsets to be interleaved: {ds}')
                ds = interleave_datasets(ds)
                
            elif dataset in SFT + RL:
                val_set_names.add('SFT') if dataset in SFT else val_set_names.add('RL')
                ds = load_dataset(
                        dataset,
                        split=dataset_split,
                        streaming=streaming,
                        trust_remote_code=trust_remote_code,
                        cache_dir=cache_path,
                    )
                # Preference data has accepted and rejected responses
                if dataset in ["allenai/olmo-2-0425-1b-preference-mix"]:
                    ds = ds.map(extract_text, fn_kwargs={"col_names":["chosen", "rejected"]}, batched=True)         
                else:
                    ds = ds.map(extract_text, fn_kwargs={"col_names":"messages"})
                    
                ds = ds.select_columns(["text"])
                ds = ds = add_source_columns(ds, ds_name)
                # ds = ds.shuffle(seed=1, buffer_size=10000)

            else:
                raise NotImplementedError("Dataset not implemented")
            streamed_datasets.append(ds)
            logger.info(f'Datasets to be interleaved: {streamed_datasets}')
            
        # multiple datasets in streaming mode. This alternates an example from each dataset
        dataset = interleave_datasets(
            streamed_datasets,
            stopping_strategy="all_exhausted"
        )
        print('Final Dataset=', dataset)
        N_VAL = 1000
        VAL_SET_NAME = f"validation_set_{N_VAL}_{"_".join(val_set_names)}"

        # Only create validation set since training from start of the dataset
        if not os.path.exists(VAL_SET_NAME):
            logger.info(f'Creating validation set with {N_VAL} examples at {VAL_SET_NAME}')
            # Take fixed validation examples
            val_examples = list(dataset.take(N_VAL))
            val_dataset = Dataset.from_list(val_examples)
            val_dataset.save_to_disk(VAL_SET_NAME)
            logger.info(f'Created validation set with {N_VAL} examples at {VAL_SET_NAME}')

            # Sanity check: load val set and save to JSON for inspection
            val_ds = load_from_disk(VAL_SET_NAME)
            val_ds.to_json(f"{VAL_SET_NAME}/val.json")
        else:
            logger.info(f"Validation set already exists at {VAL_SET_NAME}, skipping creation.")

        
        if dataset_skip is None:
            dataset_skip = 0
        # skip val set examples + number already trained on
        logger.info(f'skipping {N_VAL:,d} validation examples and {dataset_skip:,d} examples already trained on')
        dataset_skip += N_VAL
        logger.info(f'Together, skipping first {dataset_skip:,d} lines of dataset')
        dataset = dataset.skip(dataset_skip)

        if not isinstance(dataset, (Dataset, IterableDataset)):
            raise TypeError("Loaded dataset is not a Hugging Face Dataset or IterableDataset.")

        batch_texts: List[str] = []

        if pbar:
            pbar = tqdm(desc="Processing dataset")

        for item in dataset:
            text = item[dataset_text_column]
            chat_template_applied = item["data_source"] in RL + SFT

            # Process potentially long texts into manageable chunks
            text_chunks = self._preprocess_text(text)
            # Add each chunk to batch_texts
            batch_texts.extend(text_chunks)

            # Process complete batches
            while len(batch_texts) >= self.inference_batch_size:
                current_batch = batch_texts[: self.inference_batch_size]
                batch_texts = batch_texts[self.inference_batch_size :]

                try:
                    # Pre-tokenize the batch of text (key change from notebook)
                    self.tokenizer_args["add_special_tokens"] = False if chat_template_applied else True
                    # print(f"self.tokenizer_args", self.tokenizer_args)
                    # print(self.tokenizer.decode([100277]))
                    tokenized_inputs = self.tokenizer(current_batch, **self.tokenizer_args)

                    # Move to device
                    input_ids = tokenized_inputs["input_ids"].to(self.device)
                    attention_mask = tokenized_inputs["attention_mask"].to(self.device)

                    # Dictionaries to hold saved activations
                    saved_mlp_inputs = {}
                    saved_mlp_outputs = {}

                    # Using simpler tracing approach from notebook
                    with torch.no_grad():
                        with self.model.trace(input_ids):
                            for layer_idx in range(self.num_layers):
                                saved_mlp_inputs[layer_idx] = self._get_module_proxy(layer_idx, "input").save()
                                saved_mlp_outputs[layer_idx] = self._get_module_proxy(layer_idx, "output").save()

                            # Ensure trace executes
                            _ = self.model.output.logits.shape

                    # --- Post-processing happens after trace context exits ---
                    # Initialize dictionaries for this batch's valid activations
                    batch_inputs_dict: Dict[int, torch.Tensor] = {}
                    batch_targets_dict: Dict[int, torch.Tensor] = {}
                    d_model = -1

                    # Process each layer's activations
                    for layer_idx in range(self.num_layers):
                        mlp_input_proxy = saved_mlp_inputs.get(layer_idx)
                        mlp_output_proxy = saved_mlp_outputs.get(layer_idx)

                        if mlp_input_proxy is None or mlp_output_proxy is None:
                            logger.warning(f"Missing input/output proxy for layer {layer_idx}. Skipping.")
                            continue

                        # Get actual tensor values
                        mlp_input_acts = mlp_input_proxy.value
                        mlp_output_acts = mlp_output_proxy.value

                        # Handle tuple outputs
                        if isinstance(mlp_input_acts, tuple):
                            mlp_input_acts = mlp_input_acts[0] if mlp_input_acts else None
                        if isinstance(mlp_output_acts, tuple):
                            mlp_output_acts = mlp_output_acts[0] if mlp_output_acts else None

                        if mlp_input_acts is None or mlp_output_acts is None:
                            logger.warning(f"Activation value is None for layer {layer_idx}. Skipping.")
                            continue

                        # Ensure tensors are on correct device
                        mlp_input_acts = mlp_input_acts.to(self.device)
                        mlp_output_acts = mlp_output_acts.to(self.device)

                        # Infer d_model (hidden dimension size)
                        if d_model == -1 and hasattr(mlp_input_acts, "shape") and mlp_input_acts.ndim == 3:
                            d_model = mlp_input_acts.shape[-1]

                        # Expected shape: [batch_size, sequence_length, d_model]
                        if mlp_input_acts.ndim == 3 and mlp_output_acts.ndim == 3:
                            batch_size, seq_len, _ = mlp_input_acts.shape

                            # Flatten activations and mask
                            flat_input_acts = mlp_input_acts.reshape(-1, d_model)
                            flat_output_acts = mlp_output_acts.reshape(-1, d_model)
                            flat_mask = attention_mask.reshape(-1)

                            # Select non-padding tokens
                            valid_indices = flat_mask != 0
                            valid_input_acts = flat_input_acts[valid_indices]
                            valid_output_acts = flat_output_acts[valid_indices]

                            # Store valid activations
                            batch_inputs_dict[layer_idx] = valid_input_acts
                            batch_targets_dict[layer_idx] = valid_output_acts

                            # Count tokens only for the first layer to avoid duplication
                            if layer_idx == 0:
                                pass
                        else:
                            logger.warning(f"Unexpected activation shape at layer {layer_idx}. Skipping.")

                    # Yield the processed activations
                    if batch_inputs_dict and any(t.numel() > 0 for t in batch_inputs_dict.values()):
                        yield batch_inputs_dict, batch_targets_dict

                except Exception as e:
                    logger.warning(
                        f"Error processing batch: {e}. Skipping this batch.",
                        exc_info=True,
                    )
            if pbar:
                pbar.update()

        # Process any remaining texts
        if batch_texts:
            try:
                # Pre-tokenize the remaining batch
                tokenized_inputs = self.tokenizer(batch_texts, **self.tokenizer_args)
                input_ids = tokenized_inputs["input_ids"].to(self.device)
                attention_mask = tokenized_inputs["attention_mask"].to(self.device)

                # Dictionaries to hold saved activations
                saved_mlp_inputs = {}
                saved_mlp_outputs = {}

                with torch.no_grad():
                    with self.model.trace(input_ids):
                        for layer_idx in range(self.num_layers):
                            saved_mlp_inputs[layer_idx] = self._get_module_proxy(layer_idx, "input").save()
                            saved_mlp_outputs[layer_idx] = self._get_module_proxy(layer_idx, "output").save()
                        # Ensure trace executes
                        _ = self.model.output.logits.shape

                # Process final batch activations
                batch_inputs_dict = {}
                batch_targets_dict = {}
                d_model = -1

                for layer_idx in range(self.num_layers):
                    mlp_input_proxy = saved_mlp_inputs.get(layer_idx)
                    mlp_output_proxy = saved_mlp_outputs.get(layer_idx)

                    if mlp_input_proxy is None or mlp_output_proxy is None:
                        logger.warning(f"(Final Batch) Missing proxy layer {layer_idx}. Skipping.")
                        continue

                    # Get tensor values
                    mlp_input_acts = mlp_input_proxy.value
                    mlp_output_acts = mlp_output_proxy.value

                    # Handle tuples
                    if isinstance(mlp_input_acts, tuple):
                        mlp_input_acts = mlp_input_acts[0] if mlp_input_acts else None
                    if isinstance(mlp_output_acts, tuple):
                        mlp_output_acts = mlp_output_acts[0] if mlp_output_acts else None

                    if mlp_input_acts is None or mlp_output_acts is None:
                        logger.warning(f"(Final Batch) Activation value is None for layer {layer_idx}. Skipping.")
                        continue

                    # Ensure tensors are on correct device
                    mlp_input_acts = mlp_input_acts.to(self.device)
                    mlp_output_acts = mlp_output_acts.to(self.device)

                    # Infer d_model
                    if d_model == -1 and hasattr(mlp_input_acts, "shape") and mlp_input_acts.ndim == 3:
                        d_model = mlp_input_acts.shape[-1]

                    # Process activations
                    if mlp_input_acts.ndim == 3 and mlp_output_acts.ndim == 3:
                        batch_size, seq_len, _ = mlp_input_acts.shape

                        flat_input_acts = mlp_input_acts.reshape(-1, d_model)
                        flat_output_acts = mlp_output_acts.reshape(-1, d_model)
                        flat_mask = attention_mask.reshape(-1)

                        valid_indices = flat_mask != 0
                        valid_input_acts = flat_input_acts[valid_indices]
                        valid_output_acts = flat_output_acts[valid_indices]

                        batch_inputs_dict[layer_idx] = valid_input_acts
                        batch_targets_dict[layer_idx] = valid_output_acts

                        # Count tokens only for the first layer to avoid duplication
                        if layer_idx == 0:
                            pass
                    else:
                        logger.warning(f"(Final Batch) Unexpected activation shape layer {layer_idx}. Skipping.")

                # Yield the final batch
                if batch_inputs_dict and any(t.numel() > 0 for t in batch_inputs_dict.values()):
                    yield batch_inputs_dict, batch_targets_dict

            except Exception as e:
                logger.warning(f"Error processing final batch: {e}. Skipping.", exc_info=True)

    def close(self):
        """Clean up resources (if any)."""
        pass
