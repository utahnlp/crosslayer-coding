"""Test to ensure distributed checkpointing saves unique weights per rank."""

import torch
import torch.distributed as dist
import os
from unittest.mock import MagicMock, patch

from clt.config import CLTConfig
from clt.models.clt import CrossLayerTranscoder
from clt.training.checkpointing import CheckpointManager
from clt.training.wandb_logger import DummyWandBLogger


class TestDistributedCheckpointWeights:
    """Tests to prevent regression of the distributed checkpoint weight duplication bug."""

    def test_ranks_save_different_weights(self, tmp_path):
        """
        Verifies that in distributed mode, each rank saves its own unique weights,
        not duplicates of rank 0's weights.
        """
        # This test simulates what would happen in a distributed environment
        # by manually creating models with different weights for each "rank"

        # Note: We use random weights centered around different values (1.0 vs 1.5)
        # rather than exact values (1.0 vs 2.0) to avoid artificial patterns like
        # rank 1 checksum being exactly 2x rank 0 checksum, which could mask bugs.

        config = CLTConfig(
            num_features=128,
            num_layers=2,
            d_model=32,
            activation_fn="jumprelu",
        )

        # Simulate rank 0
        rank0_device = torch.device("cpu")
        rank0_model = CrossLayerTranscoder(config, process_group=None, device=rank0_device)

        # Set rank 0 weights to a specific pattern for identification
        with torch.no_grad():
            for i, param in enumerate(rank0_model.parameters()):
                # Use a pattern based on parameter index to avoid artificial 2x relationships
                torch.manual_seed(42 + i)  # Consistent seed for reproducibility
                param.data = torch.randn_like(param) * 0.1 + 1.0  # Centered around 1.0

        # Simulate rank 1
        rank1_device = torch.device("cpu")
        rank1_model = CrossLayerTranscoder(config, process_group=None, device=rank1_device)

        # Set rank 1 weights to a different pattern for identification
        with torch.no_grad():
            for i, param in enumerate(rank1_model.parameters()):
                # Use a different seed for rank 1 to ensure different weights
                torch.manual_seed(142 + i)  # Different seed
                param.data = torch.randn_like(param) * 0.1 + 1.5  # Centered around 1.5

        # Create mock components
        mock_store = MagicMock()
        mock_store.state_dict.return_value = {"test": "value"}

        mock_wandb = DummyWandBLogger(
            training_config=MagicMock(), clt_config=config, log_dir=str(tmp_path), resume_wandb_id=None
        )

        # Create checkpoint managers for both "ranks"
        manager0 = CheckpointManager(
            model=rank0_model,
            activation_store=mock_store,
            wandb_logger=mock_wandb,
            log_dir=str(tmp_path),
            distributed=True,  # Important: distributed mode
            rank=0,
            device=rank0_device,
            world_size=2,
        )

        manager1 = CheckpointManager(
            model=rank1_model,
            activation_store=mock_store,
            wandb_logger=mock_wandb,
            log_dir=str(tmp_path),
            distributed=True,  # Important: distributed mode
            rank=1,
            device=rank1_device,
            world_size=2,
        )

        # Mock distributed barrier to avoid actual distributed ops
        with patch.object(dist, "barrier"):
            # Save checkpoints from both ranks
            step = 100
            trainer_state = {"step": step, "optimizer_state_dict": {}}

            manager0._save_checkpoint(step, trainer_state)
            manager1._save_checkpoint(step, trainer_state)

        # Verify the saved files exist
        checkpoint_dir = os.path.join(tmp_path, f"step_{step}")
        rank0_path = os.path.join(checkpoint_dir, "rank_0_model.pt")
        rank1_path = os.path.join(checkpoint_dir, "rank_1_model.pt")

        assert os.path.exists(rank0_path), "Rank 0 checkpoint not found"
        assert os.path.exists(rank1_path), "Rank 1 checkpoint not found"

        # Load the saved state dicts
        rank0_saved = torch.load(rank0_path)
        rank1_saved = torch.load(rank1_path)

        # Verify they are different
        encoder_key = "encoder_module.encoders.0.weight"
        assert encoder_key in rank0_saved, "Encoder weight not found in rank 0 checkpoint"
        assert encoder_key in rank1_saved, "Encoder weight not found in rank 1 checkpoint"

        # Check that rank 0 saved weights are centered around 1.0
        rank0_encoder = rank0_saved[encoder_key]
        rank0_mean = rank0_encoder.mean().item()
        assert 0.9 < rank0_mean < 1.1, f"Rank 0 weights not centered around 1.0 (mean={rank0_mean})"

        # Check that rank 1 saved weights are centered around 1.5
        rank1_encoder = rank1_saved[encoder_key]
        rank1_mean = rank1_encoder.mean().item()
        assert 1.4 < rank1_mean < 1.6, f"Rank 1 weights not centered around 1.5 (mean={rank1_mean})"

        # Verify the weights are actually different
        assert not torch.allclose(
            rank0_encoder, rank1_encoder
        ), "CRITICAL: Both ranks saved identical weights! The distributed checkpoint bug has returned."

        # Verify file sizes (they should be similar but content different)
        rank0_size = os.path.getsize(rank0_path)
        rank1_size = os.path.getsize(rank1_path)
        size_diff = abs(rank0_size - rank1_size)

        # Sizes should be very close (same model structure)
        assert size_diff < 1000, f"Checkpoint sizes differ too much: {rank0_size} vs {rank1_size}"

        print(f"✓ Rank 0 checkpoint size: {rank0_size} bytes")
        print(f"✓ Rank 1 checkpoint size: {rank1_size} bytes")
        print(f"✓ Rank 0 encoder checksum: {torch.sum(rank0_encoder).item()}")
        print(f"✓ Rank 1 encoder checksum: {torch.sum(rank1_encoder).item()}")
        print("✓ Verified: Each rank saves its own unique weights")

    def test_merged_weights_contain_all_ranks(self, tmp_path):
        """
        Verifies that when rank 0 merges checkpoints, it correctly combines
        weights from all ranks, not just duplicates of rank 0.
        """
        config = CLTConfig(
            num_features=8,  # Small for easy verification
            num_layers=1,
            d_model=4,
            activation_fn="jumprelu",
        )

        # Create two state dicts representing different ranks
        # For tensor parallel, encoder weights are split along dim 0
        rank0_state = {
            "encoder_module.encoders.0.weight": torch.ones(4, 4),  # First half of features
            "encoder_module.encoders.0.bias_param": torch.ones(4),
            "decoder_module.decoders.0->0.weight": torch.ones(4, 4),  # Decoder split along dim 1
            "decoder_module.decoders.0->0.bias_param": torch.ones(4),  # Replicated
        }

        rank1_state = {
            "encoder_module.encoders.0.weight": torch.full((4, 4), 2.0),  # Second half of features
            "encoder_module.encoders.0.bias_param": torch.full((4,), 2.0),
            "decoder_module.decoders.0->0.weight": torch.full((4, 4), 2.0),
            "decoder_module.decoders.0->0.bias_param": torch.full((4,), 2.0),  # Replicated
        }

        # Save these to files
        checkpoint_dir = os.path.join(tmp_path, "step_100")
        os.makedirs(checkpoint_dir, exist_ok=True)

        torch.save(rank0_state, os.path.join(checkpoint_dir, "rank_0_model.pt"))
        torch.save(rank1_state, os.path.join(checkpoint_dir, "rank_1_model.pt"))

        # Create a checkpoint manager to test the merge function
        device = torch.device("cpu")
        model = CrossLayerTranscoder(config, process_group=None, device=device)

        mock_store = MagicMock()
        mock_wandb = DummyWandBLogger(
            training_config=MagicMock(), clt_config=config, log_dir=str(tmp_path), resume_wandb_id=None
        )

        manager = CheckpointManager(
            model=model,
            activation_store=mock_store,
            wandb_logger=mock_wandb,
            log_dir=str(tmp_path),
            distributed=True,
            rank=0,
            device=device,
            world_size=2,
        )

        # Test the merge function
        merged = manager._merge_tensor_parallel_weights([rank0_state, rank1_state])

        # Verify encoder weights are concatenated along dim 0
        encoder_weight = merged["encoder_module.encoders.0.weight"]
        assert encoder_weight.shape == (8, 4), f"Wrong shape: {encoder_weight.shape}"
        assert torch.allclose(encoder_weight[:4], torch.ones(4, 4)), "First half should be 1s"
        assert torch.allclose(encoder_weight[4:], torch.full((4, 4), 2.0)), "Second half should be 2s"

        # Verify encoder biases are concatenated along dim 0
        encoder_bias = merged["encoder_module.encoders.0.bias_param"]
        assert encoder_bias.shape == (8,), f"Wrong bias shape: {encoder_bias.shape}"
        assert torch.allclose(encoder_bias[:4], torch.ones(4)), "First half of bias should be 1s"
        assert torch.allclose(encoder_bias[4:], torch.full((4,), 2.0)), "Second half of bias should be 2s"

        # Verify decoder weights are concatenated along dim 1
        decoder_weight = merged["decoder_module.decoders.0->0.weight"]
        assert decoder_weight.shape == (4, 8), f"Wrong decoder shape: {decoder_weight.shape}"
        assert torch.allclose(decoder_weight[:, :4], torch.ones(4, 4)), "First columns should be 1s"
        assert torch.allclose(decoder_weight[:, 4:], torch.full((4, 4), 2.0)), "Last columns should be 2s"

        # Verify replicated parameters use rank 0's version
        decoder_bias = merged["decoder_module.decoders.0->0.bias_param"]
        assert torch.allclose(decoder_bias, torch.ones(4)), "Replicated params should use rank 0 version"

        print("✓ Verified: Merge correctly combines weights from all ranks")

    def test_all_tensors_differ_between_ranks(self, tmp_path):
        """
        Comprehensive test that verifies ALL tensors (not just encoder)
        differ between ranks, using hash comparisons.
        """
        config = CLTConfig(
            num_features=64,
            num_layers=2,
            d_model=16,
            activation_fn="jumprelu",
        )

        # Create two models with different weights
        rank0_model = CrossLayerTranscoder(config, process_group=None, device=torch.device("cpu"))
        rank1_model = CrossLayerTranscoder(config, process_group=None, device=torch.device("cpu"))

        # Initialize with different patterns
        with torch.no_grad():
            for i, (p0, p1) in enumerate(zip(rank0_model.parameters(), rank1_model.parameters())):
                torch.manual_seed(42 + i)
                p0.data = torch.randn_like(p0) * 0.1 + 0.5
                torch.manual_seed(142 + i)
                p1.data = torch.randn_like(p1) * 0.1 + 1.0

        # Save checkpoints
        mock_store = MagicMock()
        mock_store.state_dict.return_value = {"test": "value"}
        mock_wandb = DummyWandBLogger(
            training_config=MagicMock(), clt_config=config, log_dir=str(tmp_path), resume_wandb_id=None
        )

        managers = []
        for rank, model in enumerate([rank0_model, rank1_model]):
            manager = CheckpointManager(
                model=model,
                activation_store=mock_store,
                wandb_logger=mock_wandb,
                log_dir=str(tmp_path),
                distributed=True,
                rank=rank,
                device=torch.device("cpu"),
                world_size=2,
            )
            managers.append(manager)

        with patch.object(dist, "barrier"):
            step = 100
            trainer_state = {"step": step, "optimizer_state_dict": {}}
            for manager in managers:
                manager._save_checkpoint(step, trainer_state)

        # Load saved state dicts
        checkpoint_dir = os.path.join(tmp_path, f"step_{step}")
        rank0_state = torch.load(os.path.join(checkpoint_dir, "rank_0_model.pt"))
        rank1_state = torch.load(os.path.join(checkpoint_dir, "rank_1_model.pt"))

        # Compute hashes for all tensors
        import hashlib

        def tensor_hash(tensor):
            """Compute SHA256 hash of tensor data."""
            return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()

        rank0_hashes = {key: tensor_hash(val) for key, val in rank0_state.items()}
        rank1_hashes = {key: tensor_hash(val) for key, val in rank1_state.items()}

        # Verify all keys match
        assert set(rank0_hashes.keys()) == set(rank1_hashes.keys()), "State dicts have different keys!"

        # Check that at least some tensors differ
        differences = []
        for key in rank0_hashes:
            if rank0_hashes[key] != rank1_hashes[key]:
                differences.append(key)

        # We expect most parameters to differ (except maybe some small biases)
        assert len(differences) > len(rank0_hashes) * 0.8, (
            f"Too few differences found ({len(differences)}/{len(rank0_hashes)}). " "Possible weight duplication bug!"
        )

        print(f"✓ Found {len(differences)}/{len(rank0_hashes)} different tensors between ranks")
        print(f"✓ Sample differences: {differences[:3]}")

    def test_negative_identical_weights_detected(self, tmp_path):
        """
        Negative test: Intentionally save identical weights from both ranks
        and verify our test would catch it.
        """
        config = CLTConfig(
            num_features=64,
            num_layers=1,
            d_model=16,
            activation_fn="jumprelu",
        )

        # Create ONE model that both "ranks" will save
        shared_model = CrossLayerTranscoder(config, process_group=None, device=torch.device("cpu"))

        # Set to known values
        with torch.no_grad():
            for param in shared_model.parameters():
                param.fill_(1.234)

        # Create checkpoint managers that will save the SAME model
        mock_store = MagicMock()
        mock_store.state_dict.return_value = {"test": "value"}
        mock_wandb = DummyWandBLogger(
            training_config=MagicMock(), clt_config=config, log_dir=str(tmp_path), resume_wandb_id=None
        )

        managers = []
        for rank in range(2):
            manager = CheckpointManager(
                model=shared_model,  # SAME model for both ranks!
                activation_store=mock_store,
                wandb_logger=mock_wandb,
                log_dir=str(tmp_path),
                distributed=True,
                rank=rank,
                device=torch.device("cpu"),
                world_size=2,
            )
            managers.append(manager)

        with patch.object(dist, "barrier"):
            step = 100
            trainer_state = {"step": step, "optimizer_state_dict": {}}
            for manager in managers:
                manager._save_checkpoint(step, trainer_state)

        # Load and compare
        checkpoint_dir = os.path.join(tmp_path, f"step_{step}")
        rank0_state = torch.load(os.path.join(checkpoint_dir, "rank_0_model.pt"))
        rank1_state = torch.load(os.path.join(checkpoint_dir, "rank_1_model.pt"))

        # This SHOULD show identical weights
        encoder_key = "encoder_module.encoders.0.weight"
        rank0_encoder = rank0_state[encoder_key]
        rank1_encoder = rank1_state[encoder_key]

        # Verify they are identical (this is the bug condition!)
        assert torch.allclose(rank0_encoder, rank1_encoder), "Expected identical weights in negative test!"

        print("✓ Negative test confirmed: Identical weights were saved (bug condition)")
        print("✓ This proves our main test would catch the distributed checkpoint bug")
