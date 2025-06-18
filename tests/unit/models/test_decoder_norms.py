import torch

from clt.config import CLTConfig
from clt.models.decoder import Decoder


def _create_decoder(num_layers: int = 3, d_model: int = 8, num_features: int = 12):
    config = CLTConfig(num_layers=num_layers, d_model=d_model, num_features=num_features)
    return Decoder(config=config, process_group=None, device=torch.device("cpu"), dtype=torch.float32)


def test_decoder_norms_shape_and_non_negative():
    """Decoder.get_decoder_norms should return a tensor of shape [num_layers, num_features] with non-negative values."""
    decoder = _create_decoder()
    norms = decoder.get_decoder_norms()

    assert norms.shape == (decoder.config.num_layers, decoder.config.num_features)
    # All norms should be >= 0 (L2 norms)
    assert torch.all(norms >= 0), "Decoder norms should be non-negative"


def test_decoder_norms_cached():
    """Subsequent calls to get_decoder_norms should return the cached tensor object (no recomputation)."""
    decoder = _create_decoder()
    norms_first = decoder.get_decoder_norms()
    norms_second = decoder.get_decoder_norms()

    # Should be the *same* tensor object (cached)
    assert norms_first is norms_second, "Decoder norms should be cached and identical object on repeated calls"
