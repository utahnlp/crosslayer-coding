import pytest
import torch
from unittest.mock import MagicMock
from unittest.mock import patch

from clt.training.checkpointing import CheckpointManager
from clt.models.clt import CrossLayerTranscoder
from clt.config import CLTConfig

# --- Fixtures ---


@pytest.fixture
def tiny_model(device) -> CrossLayerTranscoder:
    """Provides a tiny, non-distributed model for testing."""
    config = CLTConfig(num_layers=1, d_model=4, num_features=8)
    return CrossLayerTranscoder(config, process_group=None, device=device)


@pytest.fixture
def mock_activation_store():
    """Mocks the BaseActivationStore."""
    store = MagicMock()
    store.state_dict.return_value = {"sampler_state": "dummy"}
    return store


@pytest.fixture
def mock_wandb_logger():
    """Mocks the WandB logger."""
    logger = MagicMock()
    logger.get_current_wandb_run_id.return_value = "test_run_id_123"
    logger.log_artifact = MagicMock()
    return logger


@pytest.fixture
def checkpoint_manager_components(tmp_path, tiny_model, mock_activation_store, mock_wandb_logger, device):
    """A dictionary of components needed to instantiate CheckpointManager."""
    return {
        "model": tiny_model,
        "activation_store": mock_activation_store,
        "wandb_logger": mock_wandb_logger,
        "log_dir": str(tmp_path),
        "distributed": False,
        "rank": 0,
        "device": device,
        "world_size": 1,
    }


class TestCheckpointManagerNonDistributed:
    def test_save_checkpoint_non_distributed(self, checkpoint_manager_components, tmp_path):
        """
        Verifies that _save_checkpoint (non-distributed) creates the correct files.
        """
        manager = CheckpointManager(**checkpoint_manager_components)

        step = 100
        trainer_state = {
            "step": step,
            "optimizer_state_dict": {"param_groups": []},
            "scheduler_state_dict": None,
            "scaler_state_dict": None,
            "n_forward_passes_since_fired": torch.zeros(1, 8),
            "wandb_run_id": "test_run_id_123",
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": None,
            "python_rng_state": None,
        }

        # --- Call the save method ---
        manager._save_checkpoint(step, trainer_state)

        # --- Assertions ---
        # 1. Check for the step-specific files
        model_path = tmp_path / f"clt_checkpoint_{step}.safetensors"
        trainer_state_path = tmp_path / f"trainer_state_{step}.pt"
        store_path = tmp_path / f"activation_store_{step}.pt"

        assert model_path.exists(), "Model checkpoint file was not created."
        assert trainer_state_path.exists(), "Trainer state checkpoint file was not created."
        assert store_path.exists(), "Activation store checkpoint file was not created."

        # 2. Check for the 'latest' symlinks or copies
        latest_model_path = tmp_path / "clt_checkpoint_latest.safetensors"
        latest_trainer_state_path = tmp_path / "trainer_state_latest.pt"
        latest_store_path = tmp_path / "activation_store_latest.pt"

        assert latest_model_path.exists(), "Latest model checkpoint was not created."
        assert latest_trainer_state_path.exists(), "Latest trainer state was not created."
        assert latest_store_path.exists(), "Latest activation store was not created."

        # 3. Verify content of trainer state can be loaded
        loaded_trainer_state = torch.load(trainer_state_path)
        assert loaded_trainer_state["step"] == step
        assert loaded_trainer_state["wandb_run_id"] == "test_run_id_123"

    def test_save_checkpoint_handles_io_error(self, checkpoint_manager_components, caplog):
        """
        Verifies that _save_checkpoint logs a warning and continues if an IO error occurs.
        """
        manager = CheckpointManager(**checkpoint_manager_components)

        # Patch torch.save to raise an IOError
        with patch("torch.save", side_effect=IOError("Disk full")):
            with caplog.at_level("WARNING"):
                manager._save_checkpoint(step=100, trainer_state_to_save={})

        # Check that a warning was logged
        assert "Warning: Failed to save non-distributed checkpoint" in caplog.text
        assert "Disk full" in caplog.text
