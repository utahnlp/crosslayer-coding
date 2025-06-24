import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging

from clt.config import CLTConfig
from clt.models.parallel import ColumnParallelLinear
from clt.parallel import ops as dist_ops
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    """
    Encapsulates the encoder functionality of the CrossLayerTranscoder.
    It holds the stack of encoder layers and provides methods to get
    pre-activations.
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

        self.world_size = dist_ops.get_world_size(process_group)
        self.rank = dist_ops.get_rank(process_group)

        self.encoders = nn.ModuleList(
            [
                ColumnParallelLinear(
                    in_features=config.d_model,
                    out_features=config.num_features,
                    bias=True,
                    process_group=self.process_group,
                    device=self.device,
                    dtype=self.dtype,
                )
                for _ in range(config.num_layers)
            ]
        )
        
        # Initialize theta_bias and theta_scale parameters if enabled
        # These are per-layer, per-feature parameters
        # Note: For tensor parallelism, each rank only holds a shard of features
        features_per_rank = config.num_features // self.world_size
        
        if config.enable_feature_offset:
            # Initialize feature_offset for each layer
            self.feature_offset = nn.ParameterList([
                nn.Parameter(torch.zeros(features_per_rank, device=self.device, dtype=self.dtype))
                for _ in range(config.num_layers)
            ])
        else:
            self.feature_offset = None
            
        if config.enable_feature_scale:
            # Initialize feature_scale for each layer
            self.feature_scale = nn.ParameterList([
                nn.Parameter(torch.ones(features_per_rank, device=self.device, dtype=self.dtype))
                for _ in range(config.num_layers)
            ])
        else:
            self.feature_scale = None

    def get_preactivations(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Get pre-activation values (full tensor) for features at the specified layer."""
        result: Optional[torch.Tensor] = None
        fallback_shape: Optional[Tuple[int, int]] = None
        input_for_linear: Optional[torch.Tensor] = None

        # Ensure input is on the correct device and dtype
        x = x.to(device=self.device, dtype=self.dtype)

        try:
            # 1. Check input shape and reshape if necessary
            if x.dim() == 2:
                input_for_linear = x
            elif x.dim() == 3:
                batch, seq_len, d_model = x.shape
                if d_model != self.config.d_model:
                    logger.warning(
                        f"Rank {self.rank}: Input d_model {d_model} != config {self.config.d_model} layer {layer_idx}"
                    )
                    fallback_shape = (batch * seq_len, self.config.num_features)
                else:
                    input_for_linear = x.reshape(-1, d_model)
            else:
                logger.warning(
                    f"Rank {self.rank}: Cannot handle input shape {x.shape} for preactivations layer {layer_idx}"
                )
                fallback_batch_dim = x.shape[0] if x.dim() > 0 else 0
                fallback_shape = (fallback_batch_dim, self.config.num_features)

            # 2. Check d_model match if not already done and input_for_linear was set
            if fallback_shape is None and input_for_linear is not None:
                if input_for_linear.shape[1] != self.config.d_model:
                    logger.warning(
                        f"Rank {self.rank}: Input d_model {input_for_linear.shape[1]} != config {self.config.d_model} layer {layer_idx}"
                    )
                    fallback_shape = (input_for_linear.shape[0], self.config.num_features)
            elif fallback_shape is None and input_for_linear is None:
                logger.error(
                    f"Rank {self.rank}: Could not determine input for linear layer {layer_idx} and no fallback shape set. Input x.shape: {x.shape}"
                )
                fallback_batch_dim = x.shape[0] if x.dim() > 0 else 0
                fallback_shape = (fallback_batch_dim, self.config.num_features)

            # 3. Proceed if no errors so far (i.e. fallback_shape is still None)
            if fallback_shape is None and input_for_linear is not None:
                # The input_for_linear is already on self.device and self.dtype due to the .to() call at the start of the function
                # or because it's derived from x which was moved.
                result = self.encoders[layer_idx](input_for_linear)
            elif fallback_shape is None and input_for_linear is None:
                logger.error(
                    f"Rank {self.rank}: Critical logic error in get_preactivations for layer {layer_idx}. input_for_linear is None and fallback_shape is None. Input x.shape: {x.shape}"
                )
                fallback_batch_dim = x.shape[0] if x.dim() > 0 else 0
                fallback_shape = (fallback_batch_dim, self.config.num_features)

        except IndexError:
            logger.error(
                f"Rank {self.rank}: Invalid layer index {layer_idx} requested for encoder. Max index is {len(self.encoders) - 1}."
            )
            if x.dim() == 2:
                fallback_batch_dim = x.shape[0]
            elif x.dim() == 3:
                fallback_batch_dim = x.shape[0] * x.shape[1]
            elif x.dim() > 0:
                fallback_batch_dim = x.shape[0]
            else:
                fallback_batch_dim = 0
            fallback_shape = (fallback_batch_dim, self.config.num_features)

        if result is not None:
            return result
        else:
            if fallback_shape is None:
                logger.error(
                    f"Rank {self.rank}: Fallback shape not determined for layer {layer_idx}, and no result. Input x.shape: {x.shape}. Returning empty tensor."
                )
                fallback_shape = (0, self.config.num_features)
            return torch.zeros(fallback_shape, device=self.device, dtype=self.dtype)

    def encode_all_layers(
        self, inputs: Dict[int, torch.Tensor]
    ) -> Tuple[Dict[int, torch.Tensor], List[Tuple[int, int, int]]]:
        """
        Encodes inputs for all layers using the stored encoders.
        Assumes input tensors in `inputs` will be moved to the correct device/dtype
        by the `get_preactivations` method.

        Returns:
            A tuple containing:
                - preactivations_dict: Dictionary mapping layer indices to pre-activation tensors.
                - original_shapes_info: List of tuples storing (layer_idx, batch_size, seq_len)
                                        for restoring original 3D shapes if needed.
        """
        preactivations_dict: Dict[int, torch.Tensor] = {}
        original_shapes_info: List[Tuple[int, int, int]] = []

        # Iterate in a deterministic layer order
        for layer_idx in sorted(inputs.keys()):
            x = inputs[layer_idx]  # x will be moved to device/dtype in get_preactivations

            if x.dim() == 3:
                batch_size, seq_len, _ = x.shape
                original_shapes_info.append((layer_idx, batch_size, seq_len))
            elif x.dim() == 2:
                batch_size, _ = x.shape
                original_shapes_info.append((layer_idx, batch_size, 1))  # seq_len is 1 for 2D

            preact = self.get_preactivations(x, layer_idx)
            preactivations_dict[layer_idx] = preact

        return preactivations_dict, original_shapes_info
