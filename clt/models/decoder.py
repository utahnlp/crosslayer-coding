import torch
import torch.nn as nn
from typing import Dict, Optional
import logging

from clt.config import CLTConfig
from clt.models.parallel import RowParallelLinear
from clt.parallel import ops as dist_ops
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)


class Decoder(nn.Module):
    """
    Encapsulates the decoder functionality of the CrossLayerTranscoder.
    It holds the stack of decoder layers and provides methods to decode
    feature activations and compute decoder norms.
    """

    def __init__(
        self,
        config: CLTConfig,
        process_group: Optional[ProcessGroup],
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.config = config
        self.process_group = process_group
        self.device = device
        self.dtype = dtype

        if process_group is None or not dist_ops.is_dist_initialized_and_available():
            self.world_size = 1
            self.rank = 0
        else:
            self.world_size = dist_ops.get_world_size(process_group)
            self.rank = dist_ops.get_rank(process_group)

        # Initialize decoders based on tying configuration
        if config.decoder_tying == "per_source":
            # Tied decoders: one decoder per source layer
            self.decoders = nn.ModuleList([
                RowParallelLinear(
                    in_features=self.config.num_features,
                    out_features=self.config.d_model,
                    bias=True,
                    process_group=self.process_group,
                    input_is_parallel=False,
                    d_model_for_init=self.config.d_model,
                    num_layers_for_init=self.config.num_layers,
                    device=self.device,
                    dtype=self.dtype,
                )
                for _ in range(self.config.num_layers)
            ])
            
            # Initialize decoder weights to zeros for tied transcoders
            # This matches the reference implementation
            for decoder in self.decoders:
                nn.init.zeros_(decoder.weight)
                if decoder.bias_param is not None:
                    nn.init.zeros_(decoder.bias_param)
            
            # Initialize per-target scale and bias if enabled
            if config.per_target_scale:
                self.per_target_scale = nn.Parameter(
                    torch.ones(self.config.num_layers, self.config.num_layers, self.config.d_model, 
                              device=self.device, dtype=self.dtype)
                )
            else:
                self.per_target_scale = None
                
            if config.per_target_bias:
                self.per_target_bias = nn.Parameter(
                    torch.zeros(self.config.num_layers, self.config.num_layers, self.config.d_model,
                               device=self.device, dtype=self.dtype)
                )
            else:
                self.per_target_bias = None
        else:
            # Original untied decoders: one decoder per (src, tgt) pair
            self.decoders = nn.ModuleDict(
                {
                    f"{src_layer}->{tgt_layer}": RowParallelLinear(
                        in_features=self.config.num_features,
                        out_features=self.config.d_model,
                        bias=True,
                        process_group=self.process_group,
                        input_is_parallel=False,
                        d_model_for_init=self.config.d_model,
                        num_layers_for_init=self.config.num_layers,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    for src_layer in range(self.config.num_layers)
                    for tgt_layer in range(src_layer, self.config.num_layers)
                }
            )
            self.per_target_scale = None
            self.per_target_bias = None
        
        # Initialize skip connection weights if enabled
        if config.skip_connection:
            if config.decoder_tying == "per_source":
                # For tied decoders, one skip connection per target layer
                self.skip_weights = nn.ParameterList([
                    nn.Parameter(torch.zeros(self.config.d_model, self.config.d_model, 
                                           device=self.device, dtype=self.dtype))
                    for _ in range(self.config.num_layers)
                ])
            else:
                # For untied decoders, one skip connection per src->tgt pair
                self.skip_weights = nn.ParameterDict({
                    f"{src_layer}->{tgt_layer}": nn.Parameter(
                        torch.zeros(self.config.d_model, self.config.d_model, 
                                  device=self.device, dtype=self.dtype)
                    )
                    for src_layer in range(self.config.num_layers)
                    for tgt_layer in range(src_layer, self.config.num_layers)
                })
        else:
            self.skip_weights = None
            
        self.register_buffer("_cached_decoder_norms", None, persistent=False)

    def decode(self, a: Dict[int, torch.Tensor], layer_idx: int) -> torch.Tensor:
        """Decode the feature activations to reconstruct outputs at the specified layer.

        Input activations `a` are expected to be the *full* tensors.
        The RowParallelLinear decoder splits them internally.

        Args:
            a: Dictionary mapping layer indices to *full* feature activations [..., num_features]
            layer_idx: Index of the layer to reconstruct outputs for

        Returns:
            Reconstructed outputs [..., d_model]
        """
        available_keys = sorted(a.keys())
        if not available_keys:
            logger.warning(f"Rank {self.rank}: No activation keys available in decode method for layer {layer_idx}")
            return torch.zeros((0, self.config.d_model), device=self.device, dtype=self.dtype)

        first_key = available_keys[0]
        example_tensor = a[first_key]
        batch_dim_size = example_tensor.shape[0] if example_tensor.numel() > 0 else 0
        if batch_dim_size == 0:
            for key in available_keys:
                if a[key].numel() > 0:
                    batch_dim_size = a[key].shape[0]
                    example_tensor = a[key]
                    break

        reconstruction = torch.zeros((batch_dim_size, self.config.d_model), device=self.device, dtype=self.dtype)

        for src_layer in range(layer_idx + 1):
            if src_layer in a:
                activation_tensor = a[src_layer].to(device=self.device, dtype=self.dtype)

                if activation_tensor.numel() == 0:
                    continue
                if activation_tensor.shape[-1] != self.config.num_features:
                    logger.warning(
                        f"Rank {self.rank}: Activation tensor for layer {src_layer} has incorrect feature dimension {activation_tensor.shape[-1]}, expected {self.config.num_features}. Skipping decode contribution."
                    )
                    continue

                if self.config.decoder_tying == "per_source":
                    # Use tied decoder for the source layer
                    decoder = self.decoders[src_layer]
                    decoded = decoder(activation_tensor)
                    
                    # Apply per-target scale and bias if enabled
                    if self.per_target_scale is not None:
                        decoded = decoded * self.per_target_scale[src_layer, layer_idx]
                    if self.per_target_bias is not None:
                        decoded = decoded + self.per_target_bias[src_layer, layer_idx]
                else:
                    # Use untied decoder for (src, tgt) pair
                    decoder = self.decoders[f"{src_layer}->{layer_idx}"]
                    decoded = decoder(activation_tensor)
                    
                reconstruction += decoded
        return reconstruction

    def get_decoder_norms(self) -> torch.Tensor:
        """Get L2 norms of all decoder matrices for each feature (gathered across ranks).

        The decoders are of type `RowParallelLinear`. Their weights are sharded across the
        input feature dimension (CLT features). Each feature's decoder weight vector
        (across all target layers) resides on a single rank.

        The computation proceeds as follows:
        1. For each source CLT layer (`src_layer`):
           a. Initialize a local accumulator for squared norms (`local_norms_sq_accum`)
              for all features, matching the model's device and float32 for precision.
           b. For each target model layer (`tgt_layer`) that this `src_layer` decodes to:
              i. Get the corresponding `RowParallelLinear` decoder module.
              ii. Access its local weight shard (`decoder.weight`, shape [d_model, local_num_features]).
              iii. Compute L2 norm squared for each column (feature) in this local shard.
              iv. Determine the global indices for the features this rank owns.
              v. Add these squared norms to the corresponding global slice in `local_norms_sq_accum`.
           c. All-reduce `local_norms_sq_accum` across all ranks using SUM operation.
              This sums the squared norm contributions for each feature from the rank that owns it.
           d. Take the square root of the summed squared norms and cast to the model's dtype.
              Store this in the `full_decoder_norms` tensor for the current `src_layer`.
        2. Cache and return `full_decoder_norms`.

        The norms are cached in `self._cached_decoder_norms` to avoid recomputation.

        Returns:
            Tensor of shape [num_layers, num_features] containing L2 norms of decoder
            weights for each feature, applicable for sparsity calculations.
        """
        if self._cached_decoder_norms is not None:
            return self._cached_decoder_norms

        full_decoder_norms = torch.zeros(
            self.config.num_layers, self.config.num_features, device=self.device, dtype=self.dtype
        )

        for src_layer in range(self.config.num_layers):
            local_norms_sq_accum = torch.zeros(self.config.num_features, device=self.device, dtype=torch.float32)

            if self.config.decoder_tying == "per_source":
                # For tied decoders, compute norms once per source layer
                decoder = self.decoders[src_layer]
                assert isinstance(decoder, RowParallelLinear), f"Decoder {src_layer} is not RowParallelLinear"

                current_norms_sq = torch.norm(decoder.weight, dim=0).pow(2).to(torch.float32)

                full_dim = decoder.full_in_features
                features_per_rank = (full_dim + self.world_size - 1) // self.world_size
                start_idx = self.rank * features_per_rank
                end_idx = min(start_idx + features_per_rank, full_dim)
                actual_local_dim = max(0, end_idx - start_idx)
                local_dim_padded = decoder.local_in_features

                if local_dim_padded != features_per_rank and self.rank == self.world_size - 1:
                    pass
                elif local_dim_padded != actual_local_dim and local_dim_padded != features_per_rank:
                    logger.warning(
                        f"Rank {self.rank}: Padded local dim ({local_dim_padded}) doesn't match calculated actual local dim ({actual_local_dim}) or features_per_rank ({features_per_rank}) for decoder {src_layer}. This might indicate an issue with RowParallelLinear partitioning."
                    )

                if actual_local_dim > 0:
                    valid_norms_sq = current_norms_sq[:actual_local_dim]
                    if valid_norms_sq.shape[0] == actual_local_dim:
                        global_slice = slice(start_idx, end_idx)
                        local_norms_sq_accum[global_slice] += valid_norms_sq
                    else:
                        logger.warning(
                            f"Rank {self.rank}: Shape mismatch in decoder norm calculation for decoder {src_layer}. "
                            f"Valid norms shape {valid_norms_sq.shape}, expected size {actual_local_dim}."
                        )
            else:
                # For untied decoders, accumulate norms from all target layers
                for tgt_layer in range(src_layer, self.config.num_layers):
                    decoder_key = f"{src_layer}->{tgt_layer}"
                    decoder = self.decoders[decoder_key]
                    assert isinstance(decoder, RowParallelLinear), f"Decoder {decoder_key} is not RowParallelLinear"

                    current_norms_sq = torch.norm(decoder.weight, dim=0).pow(2).to(torch.float32)

                    full_dim = decoder.full_in_features
                    features_per_rank = (full_dim + self.world_size - 1) // self.world_size
                    start_idx = self.rank * features_per_rank
                    end_idx = min(start_idx + features_per_rank, full_dim)
                    actual_local_dim = max(0, end_idx - start_idx)
                    local_dim_padded = decoder.local_in_features

                    if local_dim_padded != features_per_rank and self.rank == self.world_size - 1:
                        pass
                    elif local_dim_padded != actual_local_dim and local_dim_padded != features_per_rank:
                        logger.warning(
                            f"Rank {self.rank}: Padded local dim ({local_dim_padded}) doesn't match calculated actual local dim ({actual_local_dim}) or features_per_rank ({features_per_rank}) for {decoder_key}. This might indicate an issue with RowParallelLinear partitioning."
                        )

                    if actual_local_dim > 0:
                        valid_norms_sq = current_norms_sq[:actual_local_dim]
                        if valid_norms_sq.shape[0] == actual_local_dim:
                            global_slice = slice(start_idx, end_idx)
                            local_norms_sq_accum[global_slice] += valid_norms_sq
                        else:
                            logger.warning(
                                f"Rank {self.rank}: Shape mismatch in decoder norm calculation for {decoder_key}. "
                                f"Valid norms shape {valid_norms_sq.shape}, expected size {actual_local_dim}."
                            )

            if self.process_group is not None and dist_ops.is_dist_initialized_and_available():
                dist_ops.all_reduce(local_norms_sq_accum, op=dist_ops.SUM, group=self.process_group)

            full_decoder_norms[src_layer] = torch.sqrt(local_norms_sq_accum).to(self.dtype)

        self._cached_decoder_norms = full_decoder_norms
        return full_decoder_norms
