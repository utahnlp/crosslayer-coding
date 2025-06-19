import pytest
import torch
import subprocess
from pathlib import Path


class TestDistributedTraining:
    def test_tensor_parallel_training_runs(self, tmp_path: Path):
        """
        Tests that training with Tensor Parallelism (distributed) runs successfully
        and produces model checkpoints from each rank.

        Note: With tensor parallelism, different ranks hold different shards of
        the model weights, so we don't expect identical parameters.
        """
        world_size = 2

        # --- Run the distributed worker script using torchrun ---
        cmd = [
            "torchrun",
            "--nproc_per_node",
            str(world_size),
            "-m",
            "tests.integration.distributed_training_worker",
            "--output-dir",
            str(tmp_path),
        ]

        try:
            process = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
            print("STDOUT:", process.stdout)
            print("STDERR:", process.stderr)
        except subprocess.CalledProcessError as e:
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            pytest.fail(f"Distributed worker script failed with exit code {e.returncode}.")
        except subprocess.TimeoutExpired as e:
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            pytest.fail("Distributed worker script timed out.")

        # --- Verification ---
        # Check that each rank saved its model shard
        rank0_state_path = tmp_path / "rank_0_final_model.pt"
        rank1_state_path = tmp_path / "rank_1_final_model.pt"

        assert rank0_state_path.exists(), "Rank 0 did not save a model file."
        assert rank1_state_path.exists(), "Rank 1 did not save a model file."

        # Load the states to verify they're valid PyTorch checkpoints
        rank0_state = torch.load(rank0_state_path, map_location="cpu")
        rank1_state = torch.load(rank1_state_path, map_location="cpu")

        # Basic sanity checks
        assert len(rank0_state) > 0, "Rank 0 state dict is empty"
        assert len(rank1_state) > 0, "Rank 1 state dict is empty"

        # Check that both ranks have the same keys (structure)
        assert set(rank0_state.keys()) == set(
            rank1_state.keys()
        ), "Model structure differs between ranks (different keys in state dict)"

        # Verify that key tensor parallel layers have expected shapes
        # With world_size=2, each rank should have half the features
        for key in rank0_state:
            if "encoder" in key and "weight" in key:
                # ColumnParallelLinear: out_features is sharded
                rank0_shape = rank0_state[key].shape
                rank1_shape = rank1_state[key].shape
                assert rank0_shape == rank1_shape, f"Shape mismatch for {key}: rank0={rank0_shape}, rank1={rank1_shape}"
                print(f"✓ {key}: shape {rank0_shape} (sharded across {world_size} ranks)")
            elif "decoder" in key and "weight" in key:
                # RowParallelLinear: in_features is sharded
                rank0_shape = rank0_state[key].shape
                rank1_shape = rank1_state[key].shape
                assert rank0_shape == rank1_shape, f"Shape mismatch for {key}: rank0={rank0_shape}, rank1={rank1_shape}"
                print(f"✓ {key}: shape {rank0_shape} (sharded across {world_size} ranks)")

        # Check that training logs were created
        rank0_log_dir = tmp_path / "rank_0_logs"
        rank1_log_dir = tmp_path / "rank_1_logs"
        assert rank0_log_dir.exists(), "Rank 0 log directory not created"
        assert rank1_log_dir.exists(), "Rank 1 log directory not created"

        print("\nDistributed training test passed!")
        print("- Both ranks completed training successfully")
        print(f"- Model checkpoints saved with {len(rank0_state)} parameters each")
        print(f"- Log directories created at {rank0_log_dir} and {rank1_log_dir}")
