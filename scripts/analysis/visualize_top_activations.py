# %%
"""
Interactive visualization of top activating examples produced by `collect_top_activations.py`.

How to use:
1. Set `OUTPUT_JSON_PATH` in the second cell to point at the JSON file you saved with
   `collect_top_activations.py` (default: ./top_activating_examples.json).
2. Run the cells sequentially in a Jupyter / VS Code interactive session.
3. Use the provided widgets (if `ipywidgets` is installed) to browse features and examples,
   or call the helper function `show_example(feature_idx, example_rank)` manually.

The script colours tokens by activation strength using a matplotlib colormap, giving an
intuitive heat-map style view of which tokens a CLT feature responds to most strongly.
"""

# %%
# Standard library imports
import json
from pathlib import Path
from typing import Dict, Any, List

# Third-party imports – most are optional but recommended
import plotly.colors as pcolors

# Richer, interactive display (optional)
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output, HTML

    HAS_WIDGETS = True
except ImportError:  # Fallback to non-interactive mode if ipywidgets is missing
    HAS_WIDGETS = False
    print(
        "ipywidgets not found – interactive sliders disabled. You can still call\n"
        "show_example(feature_idx, example_rank) manually."
    )

# %%
# === CONFIGURATION ===
# Path to the JSON file produced by `collect_top_activations.py`
OUTPUT_JSON_PATH = Path(
    "/Users/curttigges/Projects/crosslayer-coding/vis/pythia_70m_layer3_top_examples_features_5000_to_5999.json"
)  # <- update if different

# Matplotlib colormap to use for token colouring
COLORMAP_NAME = "Greens"  # UPDATED: Changed to Greens colorscale

assert OUTPUT_JSON_PATH.exists(), f"JSON file not found: {OUTPUT_JSON_PATH}"  # noqa: S101

# %%
# --- Load data ---
with OUTPUT_JSON_PATH.open() as f:
    data: Dict[str, Any] = json.load(f)

num_features = len(data)
print(f"Loaded activations for {num_features} feature(s) from {OUTPUT_JSON_PATH}")

# Determine max examples per feature (assumed uniform, but we compute just in case)
max_examples_per_feature = max(len(feat_data["activations"]) for feat_data in data.values())
print(f"Up to {max_examples_per_feature} top example(s) stored per feature.")

# %%
# --- Helper functions ---


def _html_colour_tokens(tokens: List[str], values: List[float], cmap_name: str = COLORMAP_NAME) -> str:
    """Return HTML string that colours each token according to its activation value."""
    assert len(tokens) == len(values), "tokens and values length mismatch"

    positive_values = [v for v in values if v > 0]
    v_max_positive = max(positive_values) if positive_values else 0

    coloured = []
    for tok, val in zip(tokens, values):
        style = ""
        # Use a very small epsilon to handle potential floating point inaccuracies if needed for val == 0
        if val > 1e-9 and v_max_positive > 1e-9:  # Treat near-zero as zero for highlighting
            normalized_val = min(val / v_max_positive, 1.0)
            try:
                # sample_colorscale expects colorscale name or the scale itself
                # Scale normalized_val to avoid very dark colors, e.g., use up to 70% of the scale
                adjusted_normalized_val = normalized_val * 0.7
                color_str = pcolors.sample_colorscale(cmap_name, adjusted_normalized_val)[0]  # returns 'rgb(r,g,b)'
                style = f"background-color:{color_str}; padding:2px;"
            except Exception:
                style = (
                    f"background-color:rgba(0,255,0,{normalized_val * 0.7}); padding:2px;"  # Fallback green with alpha
                )

        safe_tok = tok.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "↵")
        coloured.append(f"<span style='{style}' title='{val:.4f}'>{safe_tok}</span>")
    return " ".join(coloured)


def display_top_examples_for_feature(feature_idx: int, output_widget: widgets.Output):
    """Display top 5 examples (HTML text + Plotly bar chart) for a given feature."""
    with output_widget:
        clear_output(wait=True)

        feature_key = str(feature_idx)
        if feature_key not in data:
            print(f"Feature {feature_idx} not found.")
            return

        activations_dict = data[feature_key]["activations"]

        num_examples_to_show = min(5, len(activations_dict))
        if num_examples_to_show == 0:
            print(f"No examples found for feature {feature_idx}.")
            return

        print(f"Displaying top {num_examples_to_show} examples for Feature {feature_idx}:\n")

        for rank in range(num_examples_to_show):
            rank_key = str(rank)
            if rank_key not in activations_dict:
                continue

            entry = activations_dict[rank_key]
            tokens: List[str] = entry["text"]
            values: List[float] = entry["values"]
            score: float = entry["score"]

            # 1. Display HTML colored text
            html_text = _html_colour_tokens(tokens, values)
            display(HTML(f"<b>Rank {rank} (Score: {score:.4f})</b><br>{html_text}<br><br>"))

            # Add a small separator - still useful between HTML text blocks
            if rank < num_examples_to_show - 1:
                display(
                    HTML("<hr style='margin-top:10px; margin-bottom:10px; border: 0; border-top: 1px solid #eee;'>")
                )


# %%
# --- Interactive widget UI (optional) ---
if HAS_WIDGETS:
    actual_feature_indices = []
    error_in_keys = False
    if data:  # Check if data was loaded successfully
        try:
            # Extract feature indices from data keys, convert to int, and sort them
            actual_feature_indices = sorted([int(k) for k in data.keys()])
            if not actual_feature_indices and data:  # data is not empty, but keys didn't yield integers
                print("Warning: Data loaded, but no valid integer feature keys found after processing.")
                # This scenario might indicate an issue with JSON structure or key naming
        except ValueError:
            print("Error: Feature keys in JSON are not all valid integers. Cannot populate dropdown correctly.")
            error_in_keys = True  # Mark that there was an issue processing keys

    # Determine dropdown options and initial value based on processed feature indices
    if not actual_feature_indices or error_in_keys:
        if not error_in_keys:  # Only print this if not already covered by key parsing error
            print("No valid features found in the data file to display.")
        dropdown_widget_options = [("N/A", -1)]
        initial_widget_value = -1
        is_widget_disabled = True
    else:
        dropdown_widget_options = [(f"Feature {idx}", idx) for idx in actual_feature_indices]
        initial_widget_value = actual_feature_indices[0]  # Default to the first actual feature
        is_widget_disabled = False

    feature_dropdown = widgets.Dropdown(
        options=dropdown_widget_options, value=initial_widget_value, description="Feature:", disabled=is_widget_disabled
    )

    global_output_widget = widgets.Output()

    def _on_feature_change(_change=None):
        selected_feature_idx = feature_dropdown.value
        # Ensure selected_feature_idx is an int and not the N/A value (-1)
        if isinstance(selected_feature_idx, int) and selected_feature_idx != -1:
            display_top_examples_for_feature(selected_feature_idx, global_output_widget)

    feature_dropdown.observe(_on_feature_change, names="value")

    display(widgets.VBox([feature_dropdown, global_output_widget]))

    # Initial display call only if the dropdown is enabled (meaning valid features were found)
    if not feature_dropdown.disabled:
        # initial_widget_value is guaranteed to be a valid feature index here
        display_top_examples_for_feature(initial_widget_value, global_output_widget)


# %%
# If widgets are unavailable, provide quick text-based summary and helpers
if not HAS_WIDGETS:
    print("\nNon-interactive mode. The interactive Plotly display is not available.")
    print("You would need to adapt 'display_top_examples_for_feature' for non-widget use if needed.")
    # Keep a way to call a simplified version if desired, or guide user
    # For now, the script is primarily interactive with widgets.
    # Example: show_example(0, 0) function is removed, main interaction is through widgets.
