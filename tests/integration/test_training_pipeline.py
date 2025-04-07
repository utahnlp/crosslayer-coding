import pytest
import torch
import os
import tempfile
import shutil

from clt.config import CLTConfig, TrainingConfig
from clt.training.data import ActivationStore
from clt.training.trainer import CLTTrainer
from clt.models.clt import CrossLayerTranscoder


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for logs and clean it up after the test."""
    temp_dir = tempfile.mkdtemp(prefix="clt_integration_test_")
    yield temp_dir
    shutil.rmtree(temp_dir)


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
def small_training_config():
    """Create a minimal TrainingConfig for testing."""
    return TrainingConfig(
        learning_rate=1e-3,
        batch_size=4,
        training_steps=5,  # Very small number of steps for quick testing
        sparsity_lambda=1e-3,
        sparsity_c=1.0,
        preactivation_coef=3e-6,
        optimizer="adam",
        lr_scheduler="linear",
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


@pytest.mark.integration
def test_training_pipeline_runs(
    small_clt_config, small_training_config, small_activation_store, temp_log_dir
):
    """Test that the training pipeline runs end-to-end without errors."""
    # Initialize the trainer
    trainer = CLTTrainer(
        clt_config=small_clt_config,
        training_config=small_training_config,
        activation_store=small_activation_store,
        log_dir=temp_log_dir,
        device="cpu",  # Use CPU for testing
    )

    # Run training for a small number of steps
    trained_model = trainer.train(eval_every=2)  # Evaluate every 2 steps

    # Assertions to verify training completed successfully

    # 1. Check that the model is returned
    assert isinstance(trained_model, CrossLayerTranscoder)

    # 2. Check that training metrics were recorded
    assert len(trainer.metrics["train_losses"]) == small_training_config.training_steps

    # 3. Check that L0 stats were collected (should be 3 times: steps 0, 2, 4)
    assert len(trainer.metrics["l0_stats"]) == 3

    # 4. Check that model files were created
    final_model_path = os.path.join(temp_log_dir, "clt_final.pt")
    assert os.path.exists(final_model_path)

    # 5. Check that metrics were saved
    metrics_path = os.path.join(temp_log_dir, "metrics.json")
    assert os.path.exists(metrics_path)

    # 6. Verify the model has expected attributes based on config
    assert trained_model.config == small_clt_config
    assert len(trained_model.encoders) == small_clt_config.num_layers
    assert (
        len(trained_model.decoders)
        == (small_clt_config.num_layers * (small_clt_config.num_layers + 1)) // 2
    )


@pytest.mark.integration
def test_model_save_load_integration(
    small_clt_config, small_training_config, small_activation_store, temp_log_dir
):
    """Test saving a trained model and loading it back."""
    # Initialize the trainer
    trainer = CLTTrainer(
        clt_config=small_clt_config,
        training_config=small_training_config,
        activation_store=small_activation_store,
        log_dir=temp_log_dir,
        device="cpu",
    )

    # Train for a few steps
    trained_model = trainer.train(eval_every=5)  # Only evaluate at the end

    # Get path to saved model
    saved_model_path = os.path.join(temp_log_dir, "clt_final.pt")
    assert os.path.exists(saved_model_path)

    # Load the model back
    loaded_model = CrossLayerTranscoder.load(saved_model_path)

    # Assertions
    assert isinstance(loaded_model, CrossLayerTranscoder)
    assert loaded_model.config.num_layers == small_clt_config.num_layers
    assert loaded_model.config.num_features == small_clt_config.num_features
    assert loaded_model.config.d_model == small_clt_config.d_model

    # Check that the model can perform a forward pass
    inputs = {
        0: torch.randn(1, 1, small_clt_config.d_model),  # Batch=1, Seq=1
        1: torch.randn(1, 1, small_clt_config.d_model),
    }

    with torch.no_grad():
        outputs = loaded_model(inputs)

    # Check output structure
    assert isinstance(outputs, dict)
    assert set(outputs.keys()) == set(inputs.keys())
    assert outputs[0].shape == (1, 1, small_clt_config.d_model)
    assert outputs[1].shape == (1, 1, small_clt_config.d_model)
