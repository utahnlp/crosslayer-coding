import pytest
import torch

from clt.config import CLTConfig, TrainingConfig
from clt.training.data import ActivationStore
from clt.training.trainer import CLTTrainer
from clt.models.clt import CrossLayerTranscoder


@pytest.fixture
def small_clt_config():
    """Create a minimal CLTConfig for testing."""
    return CLTConfig(
        num_layers=2,
        num_features=8,
        d_model=16,
        activation_fn="jumprelu",
        jumprelu_threshold=0.03,
    )


@pytest.fixture
def small_activation_data():
    """Generate small activation tensors for testing."""
    num_tokens = 20
    d_model = 16  # Must match the config

    # Create small random activation tensors
    mlp_inputs = {
        0: torch.randn(num_tokens, d_model),
        1: torch.randn(num_tokens, d_model),
    }

    # Outputs can be slightly different to simulate MLP transformation
    mlp_outputs = {
        0: torch.randn(num_tokens, d_model),
        1: torch.randn(num_tokens, d_model),
    }

    return mlp_inputs, mlp_outputs


@pytest.fixture
def small_activation_store(small_activation_data):
    """Create a small ActivationStore from generated data."""
    mlp_inputs, mlp_outputs = small_activation_data

    return ActivationStore(
        mlp_inputs=mlp_inputs, mlp_outputs=mlp_outputs, batch_size=4, normalize=True
    )


@pytest.mark.parametrize(
    "optimizer,lr_scheduler",
    [
        ("adam", None),
        ("adam", "linear"),
        ("adam", "cosine"),
        ("adamw", None),
        ("adamw", "linear"),
        ("adamw", "cosine"),
    ],
)
@pytest.mark.integration
def test_config_variants_optimizer_scheduler(
    small_clt_config, small_activation_store, optimizer, lr_scheduler
):
    """Test that various optimizer and scheduler configurations initialize correctly."""
    # Create a training config with the specified optimizer and scheduler
    training_config = TrainingConfig(
        learning_rate=1e-3,
        batch_size=4,
        training_steps=10,
        sparsity_lambda=1e-3,
        sparsity_c=1.0,
        preactivation_coef=3e-6,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    # Initialize trainer with temporary log dir
    trainer = CLTTrainer(
        clt_config=small_clt_config,
        training_config=training_config,
        activation_store=small_activation_store,
        log_dir=f"temp_log_{optimizer}_{lr_scheduler}",
        device="cpu",
    )

    # Verify optimizer type
    if optimizer == "adam":
        assert isinstance(trainer.optimizer, torch.optim.Adam)
    else:  # adamw
        assert isinstance(trainer.optimizer, torch.optim.AdamW)

    # Verify scheduler type
    if lr_scheduler == "linear":
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.LinearLR)
    elif lr_scheduler == "cosine":
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    else:  # None
        assert trainer.scheduler is None


@pytest.mark.parametrize(
    "sparsity_lambda,sparsity_c,preactivation_coef",
    [
        (0.0, 1.0, 0.0),  # No regularization
        (1e-2, 2.0, 0.0),  # Stronger sparsity, no preactivation
        (1e-3, 1.0, 1e-5),  # Normal sparsity, with preactivation
        (1e-2, 2.0, 1e-5),  # Both stronger
    ],
)
@pytest.mark.integration
def test_config_variants_loss_params(
    small_clt_config,
    small_activation_store,
    sparsity_lambda,
    sparsity_c,
    preactivation_coef,
):
    """Test that various loss hyperparameter configurations initialize correctly."""
    # Create training config with the specified loss parameters
    training_config = TrainingConfig(
        learning_rate=1e-3,
        batch_size=4,
        training_steps=10,
        sparsity_lambda=sparsity_lambda,
        sparsity_c=sparsity_c,
        preactivation_coef=preactivation_coef,
        optimizer="adam",
        lr_scheduler=None,
    )

    # Initialize trainer
    trainer = CLTTrainer(
        clt_config=small_clt_config,
        training_config=training_config,
        activation_store=small_activation_store,
        log_dir=(
            f"temp_log_loss_params_{sparsity_lambda}_{sparsity_c}_{preactivation_coef}"
        ),
        device="cpu",
    )

    # Verify loss manager parameters
    assert trainer.loss_manager.config.sparsity_lambda == sparsity_lambda
    assert trainer.loss_manager.config.sparsity_c == sparsity_c
    assert trainer.loss_manager.config.preactivation_coef == preactivation_coef


@pytest.mark.parametrize(
    "num_layers,num_features,d_model",
    [
        (2, 8, 16),  # Small config (default from fixture)
        (3, 16, 32),  # Medium config
        (4, 32, 64),  # Larger config
    ],
)
@pytest.mark.integration
def test_config_variants_model_sizes(
    small_activation_store, num_layers, num_features, d_model
):
    """Test that different model size configurations initialize correctly."""
    # Create a custom CLT config
    clt_config = CLTConfig(
        num_layers=num_layers,
        num_features=num_features,
        d_model=d_model,
        activation_fn="jumprelu",
        jumprelu_threshold=0.03,
    )

    # Create a simple model directly to check the structure
    model = CrossLayerTranscoder(clt_config).to("cpu")

    # Verify model structure
    assert len(model.encoders) == num_layers
    assert model.encoders[0].weight.shape == (num_features, d_model)

    # Expected number of decoder matrices: sum from 1 to num_layers
    expected_decoder_count = (num_layers * (num_layers + 1)) // 2
    assert len(model.decoders) == expected_decoder_count

    # Check a specific decoder's shape
    decoder_key = "0->0"  # From layer 0 to layer 0
    assert decoder_key in model.decoders
    assert model.decoders[decoder_key].weight.shape == (d_model, num_features)

    # Verify threshold parameter
    assert model.threshold.shape == (num_features,)
    assert torch.allclose(model.threshold, torch.ones(num_features) * 0.03)

    # Create a training config and initialize a trainer to ensure compatibility
    training_config = TrainingConfig(
        learning_rate=1e-3,
        batch_size=4,
        training_steps=5,
        sparsity_lambda=1e-3,
        sparsity_c=1.0,
        preactivation_coef=3e-6,
        optimizer="adam",
        lr_scheduler=None,
    )

    # Initialize trainer - this should not raise errors
    trainer = CLTTrainer(
        clt_config=clt_config,
        training_config=training_config,
        activation_store=small_activation_store,
        log_dir=f"temp_log_model_{num_layers}_{num_features}_{d_model}",
        device="cpu",
    )

    # Verify initialization succeeded (using the trainer)
    assert trainer.clt_config.num_layers == num_layers
    assert trainer.clt_config.num_features == num_features
    assert trainer.clt_config.d_model == d_model
