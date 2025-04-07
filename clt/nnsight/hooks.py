from typing import Dict, Any


def get_mlp_paths(model_name: str) -> Dict[str, Any]:
    """Get model-specific MLP paths for a given model.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary with paths to access MLP inputs and outputs
    """
    print(f"[DEBUG] Getting MLP paths for model: {model_name}")

    if "llama" in model_name.lower() or "meta" in model_name.lower():
        # LLaMA-style paths
        print("[DEBUG] Using LLaMA-style paths")
        return {
            "input_path": lambda model, layer_idx: model.model.layers[
                layer_idx
            ].post_attention_layernorm.output,
            "output_path": lambda model, layer_idx: model.model.layers[
                layer_idx
            ].mlp.output,
        }
    else:
        # Default to GPT-2 style paths
        print("[DEBUG] Using GPT-2 style paths")

        # Define path functions with try/except for debugging
        def input_path(model, layer_idx):
            try:
                # Only print debug for first layer
                if layer_idx == 0:
                    print(f"[DEBUG] Accessing ln_2.output for layer 0")
                return model.transformer.h[layer_idx].ln_2.output
            except AttributeError as e:
                print(
                    f"[ERROR] Failed to access ln_2.output for layer {layer_idx}: {e}"
                )
                # Only print full details for first layer error
                if layer_idx == 0:
                    attrs = dir(model.transformer.h[layer_idx])
                    print(f"[DEBUG] First layer attrs (sample): {attrs[:5]}")
                raise

        def output_path(model, layer_idx):
            try:
                # Only print debug for first layer
                if layer_idx == 0:
                    print(f"[DEBUG] Accessing mlp.output for layer 0")
                return model.transformer.h[layer_idx].mlp.output
            except AttributeError as e:
                print(f"[ERROR] Failed to access mlp.output for layer {layer_idx}: {e}")
                # Only print full details for first layer error
                if layer_idx == 0:
                    attrs = dir(model.transformer.h[layer_idx])
                    print(f"[DEBUG] First layer attrs (sample): {attrs[:5]}")
                raise

        return {
            "input_path": input_path,
            "output_path": output_path,
        }
