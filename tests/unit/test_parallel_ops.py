import torch
import pytest
from unittest.mock import patch

from clt.parallel import ops as dist_ops


# Fixture to simulate dist not being initialized
@pytest.fixture
def mock_dist_not_initialized():
    with patch("torch.distributed.is_available", return_value=True), patch(
        "torch.distributed.is_initialized", return_value=False
    ):
        yield


# Fixture to simulate dist being initialized, single process (world_size=1, rank=0)
@pytest.fixture
def mock_dist_initialized_single_process():
    with patch("torch.distributed.is_available", return_value=True), patch(
        "torch.distributed.is_initialized", return_value=True
    ), patch("torch.distributed.get_rank", return_value=0), patch("torch.distributed.get_world_size", return_value=1):
        yield


# Fixture to simulate dist being initialized, multi-process (world_size=2, rank=1 as example)
@pytest.fixture
def mock_dist_initialized_multi_process():
    with patch("torch.distributed.is_available", return_value=True), patch(
        "torch.distributed.is_initialized", return_value=True
    ), patch("torch.distributed.get_rank", return_value=1), patch("torch.distributed.get_world_size", return_value=2):
        yield


def test_is_dist_initialized_and_available_not_initialized(mock_dist_not_initialized):
    assert not dist_ops.is_dist_initialized_and_available()


def test_is_dist_initialized_and_available_initialized(mock_dist_initialized_single_process):
    assert dist_ops.is_dist_initialized_and_available()


def test_get_rank_not_initialized(mock_dist_not_initialized):
    assert dist_ops.get_rank() == 0


def test_get_rank_initialized_single_process(mock_dist_initialized_single_process):
    assert dist_ops.get_rank() == 0


def test_get_rank_initialized_multi_process(mock_dist_initialized_multi_process):
    # Our mock torch.distributed.get_rank is set to return 1
    assert dist_ops.get_rank() == 1


def test_get_world_size_not_initialized(mock_dist_not_initialized):
    assert dist_ops.get_world_size() == 1


def test_get_world_size_initialized_single_process(mock_dist_initialized_single_process):
    assert dist_ops.get_world_size() == 1


def test_get_world_size_initialized_multi_process(mock_dist_initialized_multi_process):
    # Our mock torch.distributed.get_world_size is set to return 2
    assert dist_ops.get_world_size() == 2


def test_is_main_process_not_initialized(mock_dist_not_initialized):
    assert dist_ops.is_main_process()


def test_is_main_process_initialized_rank_0(mock_dist_initialized_single_process):
    # This fixture sets rank to 0
    assert dist_ops.is_main_process()


def test_is_main_process_initialized_rank_1(mock_dist_initialized_multi_process):
    # This fixture sets rank to 1
    assert not dist_ops.is_main_process()


# Tests for collective wrappers in non-initialized state


def test_all_reduce_not_initialized(mock_dist_not_initialized):
    tensor = torch.tensor([1.0, 2.0])
    original_tensor = tensor.clone()
    work_obj = dist_ops.all_reduce(tensor)
    assert work_obj is None
    assert torch.equal(tensor, original_tensor)  # Should be a no-op


def test_all_reduce_initialized_single_process(mock_dist_initialized_single_process):
    tensor = torch.tensor([1.0, 2.0])
    original_tensor = tensor.clone()
    # We need to mock the actual dist.all_reduce since it might be called if initialized
    with patch("torch.distributed.all_reduce") as mock_actual_all_reduce:
        work_obj = dist_ops.all_reduce(tensor)
        assert work_obj is None  # Our wrapper returns None for world_size = 1
        mock_actual_all_reduce.assert_not_called()  # Should not call actual dist op
    assert torch.equal(tensor, original_tensor)


def test_broadcast_not_initialized(mock_dist_not_initialized):
    tensor = torch.tensor([1.0, 2.0])
    original_tensor = tensor.clone()
    work_obj = dist_ops.broadcast(tensor, src=0)
    assert work_obj is None
    assert torch.equal(tensor, original_tensor)  # Should be a no-op


def test_broadcast_initialized_single_process(mock_dist_initialized_single_process):
    tensor = torch.tensor([1.0, 2.0])
    original_tensor = tensor.clone()
    with patch("torch.distributed.broadcast") as mock_actual_broadcast:
        work_obj = dist_ops.broadcast(tensor, src=0)
        assert work_obj is None
        mock_actual_broadcast.assert_not_called()
    assert torch.equal(tensor, original_tensor)


def test_all_gather_not_initialized(mock_dist_not_initialized):
    tensor = torch.tensor([1.0, 2.0])
    tensor_list = [torch.empty_like(tensor) for _ in range(2)]  # Example list

    work_obj = dist_ops.all_gather(tensor_list, tensor)
    assert work_obj is None
    # In non-initialized case, tensor_list[0] should contain the tensor
    assert torch.equal(tensor_list[0], tensor)
    # Other elements of tensor_list should remain unchanged if not rank 0
    # (assuming rank is 0 in non-initialized state, as per get_rank logic)
    assert torch.equal(tensor_list[1], torch.empty_like(tensor))  # Or its original value


def test_all_gather_initialized_single_process(mock_dist_initialized_single_process):
    tensor = torch.tensor([1.0, 2.0])
    # For world_size = 1, tensor_list should have at least one element
    tensor_list = [torch.empty_like(tensor)]

    with patch("torch.distributed.all_gather") as mock_actual_all_gather:
        dist_ops.all_gather(tensor_list, tensor)
        # In single process, dist.all_gather may or may not be called by the underlying
        # torch.distributed.all_gather depending on its implementation.
        # Our wrapper for world_size=1 would try to call it.
        # The critical part for our wrapper is that it *should* call the underlying if initialized.
        # However, for world_size=1, dist.all_gather itself should effectively
        # behave like a copy from input to tensor_list[0].

        # If dist_ops.all_gather directly handles world_size=1 by not calling dist.all_gather:
        # mock_actual_all_gather.assert_not_called()
        # assert torch.equal(tensor_list[0], tensor)

        # If dist_ops.all_gather calls dist.all_gather which handles world_size=1:
        mock_actual_all_gather.assert_called_once()
        # We can't easily assert tensor_list[0] without knowing mock_actual_all_gather's behavior
        # For now, just ensure our wrapper attempts the call.
        # The behavior of actual dist.all_gather in ws=1 is that it populates tensor_list[rank]

    # Let's refine the logic in dist_ops.all_gather for ws=1 if it's not calling the backend.
    # Current `dist_ops.all_gather` calls `dist.all_gather` if initialized.
    # So, mock_actual_all_gather *should* be called.
    # To test the outcome, we can make the mock_actual_all_gather simulate the copy.
    def mock_all_gather_side_effect(out_list, in_tensor, group=None, async_op=False):
        out_list[0] = in_tensor  # Simulate behavior for rank 0, world_size 1
        return None  # Simulate no Work object for sync op

    with patch("torch.distributed.all_gather", side_effect=mock_all_gather_side_effect) as mock_actual_all_gather_ws1:
        work_obj_ws1 = dist_ops.all_gather(tensor_list, tensor)
        assert work_obj_ws1 is None
        mock_actual_all_gather_ws1.assert_called_once()
        assert torch.equal(tensor_list[0], tensor)


# Example of how one might test a specific ReduceOp re-export
def test_sum_op_export():
    assert dist_ops.SUM == torch.distributed.ReduceOp.SUM
