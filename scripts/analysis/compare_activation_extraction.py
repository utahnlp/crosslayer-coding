# %% [markdown]
# Activation Comparison: nnsight vs. TransformerLens
# This script compares MLP input activations extracted by nnsight and TransformerLens
# for a given sentence and layer.

# %%
# Cell 1: Imports and Setup
import torch
from nnsight import LanguageModel
from transformer_lens import HookedTransformer
import logging
from transformers import AutoTokenizer

# Configure logging to reduce verbosity from libraries
logging.basicConfig(level=logging.WARNING)
logging.getLogger("nnsight").setLevel(logging.WARNING)
logging.getLogger("HookedTransformer").setLevel(logging.WARNING)


# --- Configuration ---
MODEL_NAME = "EleutherAI/pythia-70m"  # Hugging Face model name (e.g., "gpt2", "gpt2-medium")
# Example sentence. For GPT-2, it typically expects a BOS token to be prepended.
# TransformerLens' to_tokens() usually handles this. NNsight's tokenizer also.
SENTENCE = "Hello world, this is a test sentence."
LAYER_NUM = 2  # Layer number to extract activations from (0-indexed)

# NNsight path template for MLP input. For GPT-2, this is typically the input to the first linear layer of the MLP.
# For gpt2: model.transformer.h[layer_idx].mlp.c_fc
# You might need to adjust this based on the specific model architecture.
NNSIGHT_MLP_MODULE_PATH_TEMPLATE = "gpt_neox.layers.{}.mlp.input"

# TransformerLens hook point for MLP input.
# For gpt2: blocks.{layer_num}.hook_mlp_in
TL_MLP_HOOK_NAME_TEMPLATE = "blocks.{}.hook_mlp_in"
# --- End Configuration ---

# Determine device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")
print(f"Model: {MODEL_NAME}")
print(f"Sentence: '{SENTENCE}'")
print(f"Layer: {LAYER_NUM}")


# %%
# Cell 2: Tokenization
print("\n--- Tokenization ---")

# NNsight (Hugging Face Tokenizer)
# NNsight's LanguageModel loads the tokenizer internally.
# We load it separately here just to inspect tokens, but nnsight will use its own instance.
hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# NNsight/HF tokenization - by default, for models like GPT-2, add_special_tokens=True adds BOS
nnsight_tokens_dict = hf_tokenizer(SENTENCE, return_tensors="pt")
nnsight_token_ids = nnsight_tokens_dict["input_ids"].to(DEVICE)
nnsight_decoded_tokens = [hf_tokenizer.decode(token_id) for token_id in nnsight_token_ids[0]]

print(f"NNsight/HF tokenized IDs: {nnsight_token_ids}")
print(f"NNsight/HF decoded tokens: {nnsight_decoded_tokens}")


# TransformerLens Tokenization
# TransformerLens models also have a tokenizer.
# For GPT-2, prepend_bos=True is default and generally desired.
tl_model_for_tokenizer = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
tl_token_ids = tl_model_for_tokenizer.to_tokens(SENTENCE, prepend_bos=False)  # Explicitly ensure BOS for consistency
tl_decoded_tokens = tl_model_for_tokenizer.to_str_tokens(SENTENCE, prepend_bos=False)

print(f"TransformerLens tokenized IDs: {tl_token_ids}")
print(f"TransformerLens decoded tokens: {tl_decoded_tokens}")

# Basic check for consistency
if torch.equal(nnsight_token_ids.cpu(), tl_token_ids.cpu()):
    print("\nTokenization is consistent between NNsight/HF and TransformerLens.")
else:
    print("\nWARNING: Tokenization differs. This will likely lead to different activations.")
    print("This can happen due to different tokenizer settings (e.g., BOS handling, specific tokenizer versions).")
    # Forcing nnsight_token_ids to be what TransformerLens uses if they differ, for a more direct comparison of model internals.
    # This assumes TransformerLens tokenization is the reference for this comparison.
    # nnsight_token_ids = tl_token_ids.to(DEVICE)
    # print(f"Forcing NNsight input_ids to TransformerLens's for activation comparison: {nnsight_token_ids}")


# %%
# Cell 3: Activation Extraction with nnsight
print("\n--- NNsight Activation Extraction ---")

# Load nnsight model
# Using dispatch=True is often helpful for nnsight to manage memory and computation efficiently.
nnsight_model = LanguageModel(MODEL_NAME, device_map=DEVICE, dispatch=True)

# Construct the module path for the specified layer
# Example: model.transformer.h[0].mlp.c_fc
mlp_input_path_str = NNSIGHT_MLP_MODULE_PATH_TEMPLATE.format(LAYER_NUM)
print(f"NNsight module path: {mlp_input_path_str}")

activations_nnsight = None

with torch.no_grad():
    with nnsight_model.trace(nnsight_token_ids):  # Use the (potentially overridden) nnsight_token_ids
        # Navigate to the target module using the path string
        module_proxy = nnsight_model
        try:
            parts = mlp_input_path_str.split(".")
            for part in parts:
                if part.isdigit():
                    module_proxy = module_proxy[int(part)]
                else:
                    module_proxy = getattr(module_proxy, part)
        except Exception as e:
            raise AttributeError(f"Could not find module at path '{mlp_input_path_str}': {e}")

        # Save the module_proxy itself if the path already points to the desired tensor proxy.
        # This is the key change to address the error.
        saved_mlp_input_nnsight = module_proxy.save()

        # Ensure the trace executes by invoking something at the end of the model
        _ = nnsight_model.output.logits.shape


# Retrieve the activation tensor (robustly handle whether .value exists)
if hasattr(saved_mlp_input_nnsight, "value"):
    activations_nnsight = saved_mlp_input_nnsight.value
else:
    # In some nnsight versions, .save() may return the tensor directly
    activations_nnsight = saved_mlp_input_nnsight

if activations_nnsight is not None:
    activations_nnsight = activations_nnsight.squeeze(0)  # Remove batch dimension if it's 1
    print(f"NNsight Activations Shape: {activations_nnsight.shape}")
else:
    print("Failed to retrieve NNsight activations.")

# Clean up nnsight model explicitly if possible, though Python's GC should handle it.
del nnsight_model
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()


# %%
# Cell 4: Activation Extraction with TransformerLens
print("\n--- TransformerLens Activation Extraction ---")

# Load TransformerLens model (it might have been loaded for tokenizer already)
# If tl_model_for_tokenizer is already on the correct device, we can reuse it.
if "tl_model_for_tokenizer" in locals() and str(tl_model_for_tokenizer.cfg.device) == str(DEVICE):
    tl_model = tl_model_for_tokenizer
else:
    tl_model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, logging_level=logging.WARNING)


# Hook name for MLP input
tl_hook_name = TL_MLP_HOOK_NAME_TEMPLATE.format(LAYER_NUM)
print(f"TransformerLens hook name: {tl_hook_name}")

with torch.no_grad():
    # Run the model and cache activations
    # prepend_bos=True matches typical GPT-2 behavior and our nnsight tokenization
    _, cache = tl_model.run_with_cache(tl_token_ids, prepend_bos=False)  # BOS is already in tl_token_ids

# Extract activations
# Shape is typically [batch_size, sequence_length, d_model]
activations_tl = cache[tl_hook_name]
activations_tl = activations_tl.squeeze(0)  # Remove batch dimension

print(f"TransformerLens Activations Shape: {activations_tl.shape}")
# print(f"TransformerLens Activations (first 5 values of first token): {activations_tl[0, :5]}")

# Clean up TL model
del tl_model
del tl_model_for_tokenizer
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()


# %%
# Cell 5: Comparison
print("\n--- Comparison ---")

if activations_nnsight is not None and activations_tl is not None:
    # Ensure both are on CPU for comparison, and are float32
    act_nnsight_cpu = activations_nnsight.detach().cpu().float()
    act_tl_cpu = activations_tl.detach().cpu().float()

    print(f"NNsight final shape for comparison: {act_nnsight_cpu.shape}")
    print(f"TransformerLens final shape for comparison: {act_tl_cpu.shape}")

    if act_nnsight_cpu.shape != act_tl_cpu.shape:
        print("\nWARNING: Shapes differ, direct comparison might be misleading.")
        # Try to align sequence lengths if one is longer due to BOS/EOS differences not caught earlier
        min_seq_len = min(act_nnsight_cpu.shape[0], act_tl_cpu.shape[0])
        print(f"Aligning to min_seq_len: {min_seq_len}")
        act_nnsight_cpu = act_nnsight_cpu[:min_seq_len, :]
        act_tl_cpu = act_tl_cpu[:min_seq_len, :]
        print(f"Adjusted NNsight shape: {act_nnsight_cpu.shape}")
        print(f"Adjusted TL shape: {act_tl_cpu.shape}")

    if act_nnsight_cpu.shape == act_tl_cpu.shape:
        abs_diff = torch.abs(act_nnsight_cpu - act_tl_cpu)
        mean_abs_diff = torch.mean(abs_diff).item()
        max_abs_diff = torch.max(abs_diff).item()
        mse = torch.mean((act_nnsight_cpu - act_tl_cpu) ** 2).item()

        print(f"\nMean Absolute Difference: {mean_abs_diff:.6e}")
        print(f"Max Absolute Difference: {max_abs_diff:.6e}")
        print(f"Mean Squared Error: {mse:.6e}")

        print("\nSample values (first token, first 5 features):")
        print(f"  NNsight:          {act_nnsight_cpu[0, :5].tolist()}")
        print(f"  TransformerLens:  {act_tl_cpu[0, :5].tolist()}")

        # Check more tokens if available
        if act_nnsight_cpu.shape[0] > 1:
            print("\nSample values (second token if exists, first 5 features):")
            print(f"  NNsight:          {act_nnsight_cpu[1, :5].tolist()}")
            print(f"  TransformerLens:  {act_tl_cpu[1, :5].tolist()}")

        if mean_abs_diff < 1e-5:  # Threshold for "close enough"
            print("\nActivations are very similar.")
        else:
            print("\nActivations differ significantly. Check model versions, tokenization, and hook points.")
    else:
        print("\nCould not compare activations due to final shape mismatch after attempting alignment.")

else:
    print("Comparison skipped because one or both activation sets could not be retrieved.")

print("\nScript finished.")
