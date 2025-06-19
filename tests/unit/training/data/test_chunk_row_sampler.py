import pytest
import numpy as np

from clt.training.data.manifest_activation_store import ChunkRowSampler


@pytest.fixture
def sampler_params():
    """Provides default parameters for creating a ChunkRowSampler."""
    return {
        "chunk_sizes": {0: 30, 1: 25, 2: 40},
        "num_chunks": 3,
        "batch": 10,
        "seed": 42,
        "epoch": 0,
    }


class TestChunkRowSampler:
    def test_initialization(self, sampler_params):
        """Test that the sampler initializes correctly."""
        sampler = ChunkRowSampler(rank=0, world=1, **sampler_params)
        assert sampler.batch == 10
        assert sampler.rank == 0
        assert sampler.world == 1
        assert sampler.epoch == 0
        assert sampler.seed == 42
        assert len(sampler) > 0

    @pytest.mark.parametrize("strategy", ["sequential", "random_chunk"])
    def test_iteration_completeness(self, sampler_params, strategy):
        """Test that the sampler yields all possible rows exactly once per epoch without sharding."""
        sampler = ChunkRowSampler(rank=0, world=1, sampling_strategy=strategy, **sampler_params)

        total_rows = sum(sampler_params["chunk_sizes"].values())
        total_batches = total_rows // sampler_params["batch"]
        assert len(sampler) == total_batches

        yielded_pairs = set()
        for batch_indices in sampler:
            assert batch_indices.shape == (sampler_params["batch"], 2)
            for chunk_id, row_id in batch_indices:
                pair = (chunk_id, row_id)
                assert pair not in yielded_pairs, "Sampler yielded a duplicate (chunk, row) pair"
                yielded_pairs.add(pair)

        assert len(yielded_pairs) == total_batches * sampler_params["batch"]

    def test_sharding_correctness(self, sampler_params):
        """Test that data is correctly sharded across ranks with no overlap."""
        world_size = 4
        all_yielded_pairs = []
        for rank in range(world_size):
            sampler = ChunkRowSampler(rank=rank, world=world_size, **sampler_params)
            for batch_indices in sampler:
                for chunk_id, row_id in batch_indices:
                    # Check that the yielded row_id is valid for this rank
                    assert row_id % world_size == rank
                    all_yielded_pairs.append((chunk_id, row_id))

        # Check for duplicates across all yielded pairs from all ranks
        assert len(all_yielded_pairs) == len(set(all_yielded_pairs)), "Duplicate pairs were yielded across ranks."

    def test_state_dict_roundtrip(self, sampler_params):
        """Test that saving and loading state allows for exact resumption."""
        # 1. Create a sampler and iterate halfway through
        sampler1 = ChunkRowSampler(rank=0, world=1, **sampler_params)
        mid_point = len(sampler1) // 2

        first_half_batches = []
        for i, batch in enumerate(sampler1):
            if i >= mid_point:
                break
            first_half_batches.append(batch)

        # 2. Save its state
        state = sampler1.state_dict()

        # 3. Create a new sampler and load the state
        sampler2 = ChunkRowSampler(rank=0, world=1, **sampler_params)
        sampler2.load_state_dict(state)

        # 4. Ensure the next batch from the new sampler is the same as the one after the midpoint
        next_batch_from_1 = next(sampler1)
        next_batch_from_2 = next(sampler2)

        np.testing.assert_array_equal(next_batch_from_1, next_batch_from_2)

    def test_epoch_determinism(self, sampler_params):
        """Test that different epochs produce different samples, and same epochs produce same samples."""
        # Sampler for epoch 0
        sampler_e0_a = ChunkRowSampler(rank=0, world=1, **sampler_params)
        batches_e0_a = list(sampler_e0_a)

        # A second sampler for epoch 0 should be identical
        sampler_e0_b = ChunkRowSampler(rank=0, world=1, **sampler_params)
        batches_e0_b = list(sampler_e0_b)
        np.testing.assert_array_equal(np.vstack(batches_e0_a), np.vstack(batches_e0_b))

        # A sampler for epoch 1 should be different
        params_e1 = sampler_params.copy()
        params_e1["epoch"] = 1
        sampler_e1 = ChunkRowSampler(rank=0, world=1, **params_e1)
        batches_e1 = list(sampler_e1)

        assert len(batches_e0_a) == len(batches_e1)
        assert not np.array_equal(np.vstack(batches_e0_a), np.vstack(batches_e1))
