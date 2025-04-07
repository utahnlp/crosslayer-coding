"""Unit tests for the CLTTrainer class."""

import os
import json
from unittest.mock import Mock, patch, MagicMock, call
import pytest
import torch
import torch.nn as nn

from clt.config import CLTConfig, TrainingConfig
from clt.models.clt import CrossLayerTranscoder
from clt.training.trainer import CLTTrainer
from clt.training.data import ActivationStore
from clt.nnsight.extractor import ActivationExtractorCLT


@pytest.fixture
def clt_config():
    """Fixture for CLTConfig."""
    return CLTConfig(
        num_features=300,
        num_layers=12,
        d_model=768,
        activation_fn="jumprelu",
        jumprelu_threshold=0.03,
    )


@pytest.fixture
def training_config():
    """Fixture for TrainingConfig."""
    return TrainingConfig(
        model_name="gpt2",
        dataset_path="test_dataset",
        dataset_split="train",
        dataset_text_column="text",
        learning_rate=1e-4,
        optimizer="adam",
        lr_scheduler="linear",
        training_steps=100,
        n_batches_in_buffer=5,
        train_batch_size_tokens=128,
        store_batch_size_prompts=1,
        context_size=128,
        normalization_method="none",
        normalization_estimation_batches=2,
        prepend_bos=True,
        exclude_special_tokens=True,
        streaming=False,
        dataset_trust_remote_code=False,
        cache_path=None,
    )


@pytest.fixture
def temp_log_dir(tmpdir):
    """Fixture for temporary log directory."""
    log_dir = tmpdir.mkdir("test_logs")
    return str(log_dir)


@pytest.fixture
def mock_model():
    """Fixture for mock CrossLayerTranscoder."""
    model = MagicMock(spec=CrossLayerTranscoder)
    # Configure mock for parameters - needed for optimizer initialization
    mock_param = torch.nn.Parameter(torch.randn(1))
    model.parameters.return_value = [mock_param]

    model.get_feature_activations.return_value = {
        0: torch.randn(10, 300),
        1: torch.randn(10, 300),
    }
    return model


@pytest.fixture
def mock_activation_extractor():
    """Fixture for mock ActivationExtractorCLT."""
    extractor = MagicMock(spec=ActivationExtractorCLT)
    # Configure mock to only be called once with stream_activations
    extractor.stream_activations.return_value = MagicMock()
    return extractor


@pytest.fixture
def mock_activation_store():
    """Fixture for mock ActivationStore."""
    store = MagicMock(spec=ActivationStore)
    # Setup for iteration
    store.__iter__.return_value = store
    store.__next__.return_value = (
        {"inputs": torch.randn(10, 768)},
        {"targets": torch.randn(10, 768)},
    )
    return store


@pytest.fixture
def mock_loss_manager():
    """Fixture for mock LossManager."""
    loss_manager = MagicMock()
    loss_manager.compute_total_loss.return_value = (
        torch.tensor(0.5, requires_grad=True),
        {"total": 0.5, "reconstruction": 0.4, "sparsity": 0.1},
    )
    return loss_manager


def test_init(clt_config, training_config, temp_log_dir):
    """Test CLTTrainer initialization."""
    # Create proper mocks for model parameters - needs real tensor
    mock_model = MagicMock()
    mock_param = torch.nn.Parameter(torch.randn(1))
    mock_model.parameters.return_value = [mock_param]
    mock_model.to.return_value = mock_model

    with patch(
        "clt.training.trainer.CrossLayerTranscoder", return_value=mock_model
    ), patch("clt.training.trainer.LossManager"), patch.object(
        CLTTrainer, "_create_activation_extractor"
    ) as mock_create_extractor, patch.object(
        CLTTrainer, "_create_activation_store"
    ) as mock_create_store:

        mock_create_extractor.return_value = MagicMock()
        mock_create_store.return_value = MagicMock()

        trainer = CLTTrainer(clt_config, training_config, log_dir=temp_log_dir)

        # Check initialization
        assert trainer.clt_config == clt_config
        assert trainer.training_config == training_config
        assert trainer.log_dir == temp_log_dir
        assert isinstance(trainer.device, torch.device)

        # Check if activation extractor and store were created
        mock_create_extractor.assert_called_once()
        mock_create_store.assert_called_once()


def test_create_activation_extractor():
    """Test _create_activation_extractor method."""
    with patch("clt.training.trainer.ActivationExtractorCLT") as mock_extractor_cls:
        # Create a trainer instance manually to avoid initialization issues
        trainer = CLTTrainer.__new__(CLTTrainer)
        trainer.training_config = MagicMock()
        trainer.device = torch.device("cpu")

        # Set up expected attributes on training_config mock
        trainer.training_config.model_name = "gpt2"
        trainer.training_config.context_size = 128
        trainer.training_config.store_batch_size_prompts = 1
        trainer.training_config.exclude_special_tokens = True
        trainer.training_config.prepend_bos = True

        # Call the method directly
        result = trainer._create_activation_extractor()

        # Now check the call
        mock_extractor_cls.assert_called_once_with(
            model_name=trainer.training_config.model_name,
            device=trainer.device,
            context_size=trainer.training_config.context_size,
            store_batch_size_prompts=trainer.training_config.store_batch_size_prompts,
            exclude_special_tokens=trainer.training_config.exclude_special_tokens,
            prepend_bos=trainer.training_config.prepend_bos,
        )

        # Ensure the result is what we expect
        assert result == mock_extractor_cls.return_value


def test_create_activation_store(mock_activation_extractor):
    """Test _create_activation_store method."""
    with patch("clt.training.trainer.ActivationStore") as mock_store_cls:
        # Create a trainer instance manually to avoid initialization issues
        trainer = CLTTrainer.__new__(CLTTrainer)
        trainer.training_config = MagicMock()
        trainer.device = torch.device("cpu")
        trainer.activation_extractor = mock_activation_extractor

        # Set up expected attributes on training_config mock
        trainer.training_config.dataset_path = "test_dataset"
        trainer.training_config.dataset_split = "train"
        trainer.training_config.dataset_text_column = "text"
        trainer.training_config.streaming = False
        trainer.training_config.dataset_trust_remote_code = False
        trainer.training_config.cache_path = None
        trainer.training_config.n_batches_in_buffer = 5
        trainer.training_config.train_batch_size_tokens = 128
        trainer.training_config.normalization_method = "none"
        trainer.training_config.normalization_estimation_batches = 2

        # Set up the mock for stream_activations
        mock_activation_generator = MagicMock()
        mock_activation_extractor.stream_activations.return_value = (
            mock_activation_generator
        )

        # Call the method directly
        result = trainer._create_activation_store()

        # Check if extractor's stream_activations was called correctly
        mock_activation_extractor.stream_activations.assert_called_once_with(
            dataset_path=trainer.training_config.dataset_path,
            dataset_split=trainer.training_config.dataset_split,
            dataset_text_column=trainer.training_config.dataset_text_column,
            streaming=trainer.training_config.streaming,
            dataset_trust_remote_code=trainer.training_config.dataset_trust_remote_code,
            cache_path=trainer.training_config.cache_path,
        )

        # Check if ActivationStore was initialized correctly
        mock_store_cls.assert_called_once_with(
            activation_generator=mock_activation_generator,
            n_batches_in_buffer=trainer.training_config.n_batches_in_buffer,
            train_batch_size_tokens=trainer.training_config.train_batch_size_tokens,
            normalization_method=trainer.training_config.normalization_method,
            normalization_estimation_batches=trainer.training_config.normalization_estimation_batches,
            device=trainer.device,
        )

        # Ensure the result is what we expect
        assert result == mock_store_cls.return_value


def test_log_metrics(temp_log_dir):
    """Test _log_metrics method."""
    with patch.object(CLTTrainer, "_save_metrics") as mock_save_metrics:
        # Create a trainer instance manually to avoid initialization issues
        trainer = CLTTrainer.__new__(CLTTrainer)
        trainer.log_dir = temp_log_dir
        trainer.metrics = {"train_losses": []}
        trainer.training_config = MagicMock()

        # Default log_interval
        trainer.training_config.log_interval = 100

        loss_dict = {"total": 0.5, "reconstruction": 0.4, "sparsity": 0.1}
        trainer._log_metrics(50, loss_dict)

        # Check if metrics were updated
        assert len(trainer.metrics["train_losses"]) == 1
        assert trainer.metrics["train_losses"][0]["step"] == 50
        assert trainer.metrics["train_losses"][0]["total"] == 0.5

        # By default, we should save metrics every 100 steps
        mock_save_metrics.assert_not_called()

        # Test with a step that is a multiple of log_interval
        trainer.training_config.log_interval = 50
        trainer._log_metrics(50, loss_dict)
        mock_save_metrics.assert_called_once()


def test_save_metrics(temp_log_dir):
    """Test _save_metrics method."""
    # Create a trainer instance manually to avoid initialization issues
    trainer = CLTTrainer.__new__(CLTTrainer)
    trainer.log_dir = temp_log_dir
    trainer.metrics = {
        "train_losses": [{"step": 1, "total": 0.5}],
        "l0_stats": [{"step": 1, "avg_l0": 0.1}],
        "eval_metrics": [],
    }

    trainer._save_metrics()

    # Check if metrics file was created
    metrics_path = os.path.join(temp_log_dir, "metrics.json")
    assert os.path.exists(metrics_path)

    # Check file contents
    with open(metrics_path, "r") as f:
        saved_metrics = json.load(f)
        assert "train_losses" in saved_metrics
        assert saved_metrics["train_losses"][0]["step"] == 1
        assert saved_metrics["train_losses"][0]["total"] == 0.5


def test_compute_l0(clt_config):
    """Test _compute_l0 method."""
    # Create a trainer instance manually to avoid initialization issues
    trainer = CLTTrainer.__new__(CLTTrainer)
    trainer.clt_config = clt_config
    trainer.device = torch.device("cpu")

    # Create a mock model directly here rather than using fixture
    mock_model = MagicMock(spec=CrossLayerTranscoder)
    trainer.model = mock_model

    # Set up mock activation store to return a batch
    mock_store = MagicMock()
    inputs = {"layer_0": torch.randn(10, 768)}
    mock_store.__iter__.return_value = mock_store
    mock_store.__next__.return_value = (inputs, {})
    trainer.activation_store = mock_store

    # Set up mock model to return feature activations - with explicit control
    feature_activations = {
        0: torch.ones(10, 300) * 0.1,  # All below threshold
        1: torch.ones(10, 300) * 0.5,  # All above threshold
    }
    mock_model.get_feature_activations.return_value = feature_activations

    # Patch the StopIteration issue that's causing recursion depth error
    with patch.object(
        trainer.activation_store, "__iter__", return_value=mock_store
    ), patch.object(trainer.activation_store, "__next__", return_value=(inputs, {})):

        l0_stats = trainer._compute_l0(threshold=0.2)

        # Check if l0 stats are computed correctly
        assert "avg_l0" in l0_stats
        assert "total_l0" in l0_stats
        assert "sparsity" in l0_stats
        assert "per_layer" in l0_stats

        # Layer 0 should have 0 features above threshold, layer 1 all features
        per_layer = l0_stats["per_layer"]
        assert per_layer["layer_0"] == 0.0
        assert per_layer["layer_1"] == 300.0

        # Check average and total calculations
        assert l0_stats["total_l0"] == 300.0
        assert l0_stats["avg_l0"] == 150.0  # 300 / 2 layers

        # Check sparsity calculation
        total_features = clt_config.num_layers * clt_config.num_features
        expected_sparsity = 1.0 - (300.0 / total_features)
        assert l0_stats["sparsity"] == expected_sparsity


def test_save_checkpoint(temp_log_dir, mock_model):
    """Test _save_checkpoint method."""
    with patch("torch.save") as mock_torch_save:
        # Create a trainer instance manually to avoid initialization issues
        trainer = CLTTrainer.__new__(CLTTrainer)
        trainer.log_dir = temp_log_dir
        trainer.model = mock_model
        trainer.activation_store = MagicMock()
        trainer.activation_store.state_dict.return_value = {"mock_state": "value"}

        trainer._save_checkpoint(100)

        # Check if model save was called
        mock_model.save.assert_any_call(
            os.path.join(temp_log_dir, "clt_checkpoint_100.pt")
        )

        # Check if latest checkpoint was saved
        mock_model.save.assert_any_call(
            os.path.join(temp_log_dir, "clt_checkpoint_latest.pt")
        )

        # Check if activation store state was saved
        mock_torch_save.assert_any_call(
            {"mock_state": "value"},
            os.path.join(temp_log_dir, "activation_store_checkpoint_100.pt"),
        )
        mock_torch_save.assert_any_call(
            {"mock_state": "value"},
            os.path.join(temp_log_dir, "activation_store_checkpoint_latest.pt"),
        )


def test_load_checkpoint(temp_log_dir, mock_model):
    """Test load_checkpoint method."""
    with patch("os.path.exists", return_value=True), patch(
        "torch.load", return_value={"mock_state": "value"}
    ):

        # Create a trainer instance manually to avoid initialization issues
        trainer = CLTTrainer.__new__(CLTTrainer)
        trainer.log_dir = temp_log_dir
        trainer.model = mock_model
        trainer.device = torch.device("cpu")
        trainer.activation_store = MagicMock()

        checkpoint_path = os.path.join(temp_log_dir, "clt_checkpoint_100.pt")
        store_checkpoint_path = os.path.join(
            temp_log_dir, "activation_store_checkpoint_100.pt"
        )

        trainer.load_checkpoint(checkpoint_path, store_checkpoint_path)

        # Check if model load was called
        mock_model.load.assert_called_once_with(checkpoint_path)

        # Check if activation store load_state_dict was called
        trainer.activation_store.load_state_dict.assert_called_once_with(
            {"mock_state": "value"}
        )


@pytest.mark.parametrize("with_scheduler", [True, False])
def test_train(
    clt_config,
    training_config,
    temp_log_dir,
    mock_model,
    mock_loss_manager,
    with_scheduler,
):
    """Test train method."""
    with patch.object(CLTTrainer, "_log_metrics") as mock_log_metrics, patch.object(
        CLTTrainer, "_compute_l0"
    ) as mock_compute_l0, patch.object(
        CLTTrainer, "_save_checkpoint"
    ) as mock_save_checkpoint, patch.object(
        CLTTrainer, "_save_metrics"
    ) as mock_save_metrics, patch(
        "torch.save"
    ), patch(
        "tqdm.tqdm"
    ):

        # Set up mock store
        mock_store = MagicMock()
        mock_store.__iter__.return_value = mock_store
        mock_store.__next__.return_value = (
            {"layer_0": torch.randn(10, 768)},
            {"layer_0": torch.randn(10, 768)},
        )

        # Set up mock loss
        loss = torch.tensor(0.5, requires_grad=True)
        loss_dict = {"total": 0.5, "reconstruction": 0.4, "sparsity": 0.1}
        mock_loss_manager.compute_total_loss.return_value = (loss, loss_dict)

        # Set up mock l0 stats
        mock_compute_l0.return_value = {
            "avg_l0": 150.0,
            "sparsity": 0.95,
            "per_layer": {"layer_0": 0.0, "layer_1": 300.0},
        }

        # Set up trainer - bypass init
        trainer = CLTTrainer.__new__(CLTTrainer)
        trainer.clt_config = clt_config
        trainer.training_config = training_config
        trainer.log_dir = temp_log_dir
        trainer.device = torch.device("cpu")
        trainer.model = mock_model
        trainer.optimizer = MagicMock()
        trainer.activation_store = mock_store
        trainer.loss_manager = mock_loss_manager
        trainer.metrics = {"train_losses": [], "l0_stats": [], "eval_metrics": []}

        # Configure scheduler for test
        if with_scheduler:
            trainer.scheduler = MagicMock()
        else:
            trainer.scheduler = None

        # Run training
        training_config.training_steps = 5
        training_config.eval_interval = 2
        training_config.checkpoint_interval = 3
        result = trainer.train(eval_every=2)

        # Check if training ran for expected steps
        assert mock_loss_manager.compute_total_loss.call_count == 5
        assert mock_log_metrics.call_count == 5

        # Check if eval was run at expected intervals
        assert mock_compute_l0.call_count == 3  # steps 0, 2, 4

        # Check if checkpoints were saved at expected intervals
        # Steps 3 (checkpoint_interval) and 4 (final) = 2 calls
        assert mock_save_checkpoint.call_count >= 2

        # Check if metrics were saved at the end
        mock_save_metrics.assert_called()

        # Check if scheduler was stepped if provided
        if with_scheduler:
            assert trainer.scheduler.step.call_count == 5

        # Check return value
        assert result == mock_model


def test_train_with_nan_loss():
    """Test train method handling of NaN loss."""
    with patch("torch.isnan", return_value=True), patch("tqdm.tqdm"):

        # Set up mock model and optimizer
        mock_model = MagicMock()
        mock_optimizer = MagicMock()

        # Set up mock loss manager to return NaN loss
        mock_loss_manager = MagicMock()
        nan_tensor = torch.tensor(float("nan"))
        loss_dict = {"total": float("nan")}
        mock_loss_manager.compute_total_loss.return_value = (nan_tensor, loss_dict)

        # Set up mock store
        mock_store = MagicMock()
        mock_store.__iter__.return_value = mock_store
        mock_store.__next__.return_value = (
            {"layer_0": torch.randn(10, 768)},
            {"layer_0": torch.randn(10, 768)},
        )

        # Set up trainer - bypass init
        trainer = CLTTrainer.__new__(CLTTrainer)
        trainer.clt_config = MagicMock()
        trainer.training_config = MagicMock()
        trainer.training_config.training_steps = 3
        trainer.log_dir = "mock_log_dir"
        trainer.device = torch.device("cpu")
        trainer.model = mock_model
        trainer.optimizer = mock_optimizer
        trainer.activation_store = mock_store
        trainer.loss_manager = mock_loss_manager
        trainer.metrics = {"train_losses": [], "l0_stats": [], "eval_metrics": []}
        trainer.scheduler = None

        # Run training
        trainer.train()

        # Check that backward was not called due to NaN loss
        mock_optimizer.step.assert_not_called()


def test_train_with_error_in_backward():
    """Test train method handling of error in backward pass."""
    with patch("torch.isnan", return_value=False), patch("tqdm.tqdm"):

        # Set up mock model
        mock_model = MagicMock()

        # Set up mock optimizer
        mock_optimizer = MagicMock()

        # Set up mock loss with a single call to backward raising error
        mock_loss = torch.tensor(0.5, requires_grad=True)
        # Make a side_effect that raises once then returns normally
        side_effect = [RuntimeError("Test error in backward")]
        mock_loss.backward = MagicMock(side_effect=side_effect)

        # Set up mock loss manager
        mock_loss_manager = MagicMock()
        loss_dict = {"total": 0.5}
        mock_loss_manager.compute_total_loss.return_value = (mock_loss, loss_dict)

        # Set up mock store that only returns one batch then stops
        mock_store = MagicMock()
        mock_store.__iter__.return_value = mock_store
        # Return one batch then stop iteration
        mock_store.__next__.side_effect = [
            ({"layer_0": torch.randn(10, 768)}, {"layer_0": torch.randn(10, 768)}),
            StopIteration(),
        ]

        # Set up trainer - bypass init
        trainer = CLTTrainer.__new__(CLTTrainer)
        trainer.clt_config = MagicMock()
        trainer.training_config = MagicMock()
        trainer.training_config.training_steps = 3
        trainer.log_dir = "mock_log_dir"
        trainer.device = torch.device("cpu")
        trainer.model = mock_model
        trainer.optimizer = mock_optimizer
        trainer.activation_store = mock_store
        trainer.loss_manager = mock_loss_manager
        trainer.metrics = {"train_losses": [], "l0_stats": [], "eval_metrics": []}
        trainer.scheduler = None

        # Run training
        trainer.train()

        # Check that backward was called but optimizer step was not
        mock_loss.backward.assert_called_once()
        mock_optimizer.step.assert_not_called()


def test_activation_store_exception_handling():
    """Test handling of exceptions from the activation store."""
    with patch("tqdm.tqdm"):

        # Set up mock store that raises StopIteration
        mock_store = MagicMock()
        mock_store.__iter__.return_value = mock_store
        mock_store.__next__.side_effect = StopIteration()

        # Set up trainer - bypass init
        trainer = CLTTrainer.__new__(CLTTrainer)
        trainer.clt_config = MagicMock()
        trainer.training_config = MagicMock()
        trainer.training_config.training_steps = (
            100  # High number to ensure early termination
        )
        trainer.log_dir = "mock_log_dir"
        trainer.device = torch.device("cpu")
        trainer.model = MagicMock()
        trainer.optimizer = MagicMock()
        trainer.activation_store = mock_store
        trainer.loss_manager = MagicMock()
        trainer.metrics = {"train_losses": [], "l0_stats": [], "eval_metrics": []}
        trainer.scheduler = None

        # Run training - should exit gracefully when store is exhausted
        trainer.train()

        # Now test with a different exception
        mock_store.__next__.side_effect = ValueError("Test error")

        # Should continue loop despite error
        trainer.train()
