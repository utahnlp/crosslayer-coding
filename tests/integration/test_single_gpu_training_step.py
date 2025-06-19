import pytest
import torch
from pathlib import Path

from clt.config import CLTConfig, TrainingConfig
from clt.training.trainer import CLTTrainer
from tests.helpers.tiny_configs import create_tiny_clt_config, create_tiny_training_config


@pytest.fixture
def tiny_clt_config() -> CLTConfig:
    return create_tiny_clt_config(num_layers=2, d_model=8, num_features=16)


@pytest.fixture
def tiny_training_config(tmp_local_dataset: Path) -> TrainingConfig:
    # Set training steps to a small number for a quick test
    return create_tiny_training_config(
        training_steps=5,
        train_batch_size_tokens=16,
        eval_interval=10_000,  # Disable eval for this test
        checkpoint_interval=10_000,  # Disable checkpointing for this test
        activation_source="local_manifest",  # Specify the source
        activation_path=str(tmp_local_dataset),  # Use the fixture
        activation_dtype="float32",  # Use float32 for CPU test consistency
        precision="fp32",  # Use fp32 for CPU test consistency
    )


class TestSingleDeviceTraining:
    def test_training_loop_runs(self, tiny_clt_config: CLTConfig, tiny_training_config: TrainingConfig, tmp_path: Path):
        """
        Test that the basic training loop can run for a few steps on a single device
        without crashing. This is a basic integration test.
        """
        log_dir = tmp_path / "test_logs"

        trainer = CLTTrainer(
            clt_config=tiny_clt_config,
            training_config=tiny_training_config,
            log_dir=str(log_dir),
            device="cpu",
            distributed=False,
        )

        # Check that model parameters have no gradients initially
        for p in trainer.model.parameters():
            if p.requires_grad:
                assert p.grad is None

        # Run the training for the configured number of steps (5)
        trained_model = trainer.train()

        # --- Assertions ---
        # 1. The model returned should be the trainer's model
        assert trained_model is trainer.model

        # 2. Model parameters should have gradients after training
        grads_found = False
        for p in trainer.model.parameters():
            if p.requires_grad and p.grad is not None:
                # Check that gradients are not all zero
                assert torch.any(p.grad != 0)
                grads_found = True
        assert grads_found, "No gradients were found on model parameters after training."

        # 3. Check that log files were created
        # The trainer logs metrics and checkpoints
        assert log_dir.exists()
        # Check for final checkpoint directory created by the trainer
        final_checkpoint_dir = log_dir / "final"
        assert final_checkpoint_dir.exists()
        # Check that the log directory was created, metrics file is optional
        assert log_dir.exists()

        # 4. Checkpoint saving was disabled, so the step_4 dir should NOT exist
        step_4_checkpoint_dir = log_dir / "step_4"
        assert not step_4_checkpoint_dir.exists(), "Checkpoint at step 4 was created but should have been disabled."
