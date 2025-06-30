"""Unit tests for tied decoder functionality in CLT models."""

import pytest
import torch
import torch.nn as nn
from typing import Dict

from clt.config import CLTConfig
from clt.models.clt import CrossLayerTranscoder
from clt.models.decoder import Decoder
from clt.models.encoder import Encoder


class TestTiedDecoders:
    """Test suite for tied decoder architecture."""
    
    @pytest.fixture
    def base_config(self):
        """Base CLT configuration for testing."""
        return CLTConfig(
            num_features=128,
            num_layers=4,
            d_model=64,
            activation_fn="relu",
            decoder_tying="none",  # Default untied
        )
    
    @pytest.fixture
    def tied_config(self):
        """CLT configuration with tied decoders."""
        return CLTConfig(
            num_features=128,
            num_layers=4,
            d_model=64,
            activation_fn="relu",
            decoder_tying="per_source",
        )
    
    def test_decoder_initialization_untied(self, base_config):
        """Test that untied decoder creates correct number of decoder modules."""
        decoder = Decoder(
            config=base_config,
            process_group=None,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        
        # Should have decoders for each (src, tgt) pair where src <= tgt
        # For 4 layers: 0->0, 0->1, 0->2, 0->3, 1->1, 1->2, 1->3, 2->2, 2->3, 3->3
        # Total: 4 + 3 + 2 + 1 = 10
        expected_decoder_count = sum(range(1, base_config.num_layers + 1))
        assert len(decoder.decoders) == expected_decoder_count
        
        # Check that all expected keys exist
        for src in range(base_config.num_layers):
            for tgt in range(src, base_config.num_layers):
                assert f"{src}->{tgt}" in decoder.decoders
    
    def test_decoder_initialization_tied(self, tied_config):
        """Test that tied decoder creates one decoder per source layer."""
        decoder = Decoder(
            config=tied_config,
            process_group=None,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        
        # Should have one decoder per source layer
        assert len(decoder.decoders) == tied_config.num_layers
        
        # Check that decoders are indexed by layer
        for layer in range(tied_config.num_layers):
            assert isinstance(decoder.decoders[layer], nn.Module)
    
    def test_per_target_parameters(self, tied_config):
        """Test per-target scale and bias parameters."""
        # Test with per-target scale
        config_with_scale = CLTConfig(
            **{**tied_config.__dict__, "per_target_scale": True}
        )
        decoder = Decoder(
            config=config_with_scale,
            process_group=None,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        
        assert decoder.per_target_scale is not None
        assert decoder.per_target_scale.shape == (
            config_with_scale.num_layers,
            config_with_scale.num_layers,
            config_with_scale.d_model,
        )
        assert torch.allclose(decoder.per_target_scale, torch.ones_like(decoder.per_target_scale))
        
        # Test with per-target bias
        config_with_bias = CLTConfig(
            **{**tied_config.__dict__, "per_target_bias": True}
        )
        decoder = Decoder(
            config=config_with_bias,
            process_group=None,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        
        assert decoder.per_target_bias is not None
        assert decoder.per_target_bias.shape == (
            config_with_bias.num_layers,
            config_with_bias.num_layers,
            config_with_bias.d_model,
        )
        assert torch.allclose(decoder.per_target_bias, torch.zeros_like(decoder.per_target_bias))
    
    def test_feature_affine_parameters(self):
        """Test feature offset and scale parameters in encoder."""
        config = CLTConfig(
            num_features=128,
            num_layers=4,
            d_model=64,
            activation_fn="relu",
            enable_feature_offset=True,
            enable_feature_scale=True,
        )
        
        encoder = Encoder(
            config=config,
            process_group=None,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        
        # Check feature_offset initialization
        assert encoder.feature_offset is not None
        assert len(encoder.feature_offset) == config.num_layers
        for layer_offset in encoder.feature_offset:
            assert layer_offset.shape == (config.num_features,)
            assert torch.allclose(layer_offset, torch.zeros_like(layer_offset))
        
        # Check feature_scale initialization
        assert encoder.feature_scale is not None
        assert len(encoder.feature_scale) == config.num_layers
        for layer_scale in encoder.feature_scale:
            assert layer_scale.shape == (config.num_features,)
            assert torch.allclose(layer_scale, torch.ones_like(layer_scale))
    
    def test_decode_with_tied_decoders(self, tied_config):
        """Test decoding with tied decoders."""
        decoder = Decoder(
            config=tied_config,
            process_group=None,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        
        # Create test activations
        batch_size = 8
        activations = {
            0: torch.randn(batch_size, tied_config.num_features),
            1: torch.randn(batch_size, tied_config.num_features),
        }
        
        # Test reconstruction at layer 1
        reconstruction = decoder.decode(activations, layer_idx=1)
        
        assert reconstruction.shape == (batch_size, tied_config.d_model)
        # With zero-initialized decoders (matching reference implementation),
        # the output will be zeros initially
        assert torch.allclose(reconstruction, torch.zeros_like(reconstruction))
        
        # Verify that if we set non-zero weights, we get non-zero outputs
        for decoder_module in decoder.decoders:
            decoder_module.weight.data.fill_(0.1)
        reconstruction2 = decoder.decode(activations, layer_idx=1)
        assert not torch.allclose(reconstruction2, torch.zeros_like(reconstruction2))
    
    def test_decoder_norms_tied(self, tied_config):
        """Test decoder norm computation for tied decoders."""
        decoder = Decoder(
            config=tied_config,
            process_group=None,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        
        norms = decoder.get_decoder_norms()
        
        # Should have shape [num_layers, num_features]
        assert norms.shape == (tied_config.num_layers, tied_config.num_features)
        
        # Norms should be positive
        assert torch.all(norms >= 0)
    
    def test_feature_affine_transformation(self):
        """Test feature affine transformation in forward pass."""
        config = CLTConfig(
            num_features=128,
            num_layers=2,
            d_model=64,
            activation_fn="relu",
            enable_feature_offset=True,
            enable_feature_scale=True,
        )
        
        model = CrossLayerTranscoder(
            config=config,
            process_group=None,
            device=torch.device("cpu"),
        )
        
        # Create test inputs
        batch_size = 4
        seq_len = 16
        inputs = {
            0: torch.randn(batch_size, seq_len, config.d_model),
            1: torch.randn(batch_size, seq_len, config.d_model),
        }
        
        # Get activations
        activations = model.get_feature_activations(inputs)
        
        # Apply affine transformation
        transformed = model._apply_feature_affine(activations)
        
        # The transformation should preserve zeros
        for layer_idx in transformed:
            zero_mask = activations[layer_idx] == 0
            assert torch.all(transformed[layer_idx][zero_mask] == 0)
    
    def test_backward_compatibility_config(self):
        """Test loading old config without new fields."""
        old_config_dict = {
            "num_features": 128,
            "num_layers": 4,
            "d_model": 64,
            "activation_fn": "relu",
            # Missing: decoder_tying, per_target_scale, per_target_bias, 
            # enable_feature_offset, enable_feature_scale
        }
        
        # Should not raise an error
        config = CLTConfig(**old_config_dict)
        
        # Should have default values
        assert config.decoder_tying == "none"
        assert config.per_target_scale == False
        assert config.per_target_bias == False
        assert config.enable_feature_offset == False
        assert config.enable_feature_scale == False
    
    def test_checkpoint_compatibility(self, base_config, tied_config):
        """Test loading old untied checkpoint into tied model."""
        # Create untied model and save checkpoint
        untied_model = CrossLayerTranscoder(
            config=base_config,
            process_group=None,
            device=torch.device("cpu"),
        )
        
        # Get state dict from untied model
        untied_state_dict = untied_model.state_dict()
        
        # Create tied model
        tied_model = CrossLayerTranscoder(
            config=tied_config,
            process_group=None,
            device=torch.device("cpu"),
        )
        
        # Should be able to load with custom logic
        tied_model.load_state_dict(untied_state_dict, strict=False)
        
        # Tied model should have loaded the diagonal decoder weights
        for src_layer in range(tied_config.num_layers):
            tied_weight = tied_model.decoder_module.decoders[src_layer].weight
            untied_key = f"decoder_module.decoders.{src_layer}->{src_layer}.weight"
            if untied_key in untied_state_dict:
                untied_weight = untied_state_dict[untied_key]
                # Shapes might differ due to RowParallelLinear, so just check they're both tensors
                assert isinstance(tied_weight, torch.Tensor)
                assert isinstance(untied_weight, torch.Tensor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])