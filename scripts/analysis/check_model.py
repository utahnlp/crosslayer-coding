# %%
import torch
import os
import json
from safetensors.torch import load_file

# Load model from safetensors file
model_path = "/Users/curttigges/Projects/crosslayer-coding/conversion_test/gpt2_32k/full_model.safetensors"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists(model_path):
    state_dict = load_file(model_path, device=device.type)
    print(f"Loaded model from {model_path}")
else:
    print(f"Model file not found at {model_path}")

# %%
state_dict.keys()
# %%
state_dict["decoder_module.decoders.0->1.weight"].shape
# %%
