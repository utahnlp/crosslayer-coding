"""Tests for fused BatchTopK implementations."""

import pytest
import torch
import torch.nn.functional as F
from typing import Optional

from clt.models.activations import BatchTopK
from clt.models.activations_fused import (
    FusedBatchTopK,
    fused_batch_topk,
    TorchCompileBatchTopK,
    get_optimized_batch_topk,
    TRITON_AVAILABLE,
)


class TestFusedBatchTopK:
    """Test suite for fused BatchTopK implementations."""
    
    @pytest.fixture
    def test_tensors(self):
        """Create test tensors."""
        torch.manual_seed(42)
        batch_size = 32
        num_features = 1024
        
        x = torch.randn(batch_size, num_features)
        x_for_ranking = torch.randn(batch_size, num_features)
        
        return {
            'x': x,
            'x_for_ranking': x_for_ranking,
            'batch_size': batch_size,
            'num_features': num_features,
        }
    
    def test_fused_forward_matches_original(self, test_tensors):
        """Test that fused forward pass matches original implementation."""
        x = test_tensors['x']
        x_for_ranking = test_tensors['x_for_ranking']
        k_per_token = 50
        
        # Original implementation
        original_output = BatchTopK.apply(x, k_per_token, True, x_for_ranking)
        
        # Fused implementation
        fused_output = FusedBatchTopK.apply(x, k_per_token, True, x_for_ranking)
        
        # Check that outputs match
        torch.testing.assert_close(original_output, fused_output, rtol=1e-5, atol=1e-5)
        
        # Check sparsity matches
        original_sparsity = (original_output == 0).float().mean()
        fused_sparsity = (fused_output == 0).float().mean()
        assert abs(original_sparsity - fused_sparsity) < 1e-6
    
    def test_fused_backward_straight_through(self, test_tensors):
        """Test backward pass with straight-through estimator."""
        x = test_tensors['x'].clone().requires_grad_(True)
        x_fused = test_tensors['x'].clone().requires_grad_(True)
        k_per_token = 50
        
        # Original implementation
        output = BatchTopK.apply(x, k_per_token, True, None)
        loss = output.sum()
        loss.backward()
        original_grad = x.grad.clone()
        
        # Fused implementation
        output_fused = FusedBatchTopK.apply(x_fused, k_per_token, True, None)
        loss_fused = output_fused.sum()
        loss_fused.backward()
        fused_grad = x_fused.grad
        
        # Gradients should match
        torch.testing.assert_close(original_grad, fused_grad, rtol=1e-5, atol=1e-5)
    
    def test_fused_backward_no_straight_through(self, test_tensors):
        """Test backward pass without straight-through estimator."""
        x = test_tensors['x'].clone().requires_grad_(True)
        x_fused = test_tensors['x'].clone().requires_grad_(True)
        k_per_token = 50
        
        # Original implementation
        output = BatchTopK.apply(x, k_per_token, False, None)
        loss = output.sum()
        loss.backward()
        original_grad = x.grad.clone()
        
        # Fused implementation
        output_fused = FusedBatchTopK.apply(x_fused, k_per_token, False, None)
        loss_fused = output_fused.sum()
        loss_fused.backward()
        fused_grad = x_fused.grad
        
        # Gradients should match
        torch.testing.assert_close(original_grad, fused_grad, rtol=1e-5, atol=1e-5)
    
    def test_torch_compile_batch_topk(self, test_tensors):
        """Test torch.compile optimized version."""
        x = test_tensors['x']
        k_per_token = 50
        
        # Original
        original_output = BatchTopK.apply(x, k_per_token, True, None)
        
        # Torch compile version
        module = TorchCompileBatchTopK(k_per_token, True)
        compile_output = module(x)
        
        # Should have same sparsity pattern
        original_mask = (original_output != 0)
        compile_mask = (compile_output != 0)
        assert torch.equal(original_mask, compile_mask)
    
    def test_fused_batch_topk_helper(self, test_tensors):
        """Test the helper function."""
        x = test_tensors['x']
        k_per_token = 50
        
        # Test with different options
        output1 = fused_batch_topk(x, k_per_token, True, None, use_triton=False)
        output2 = fused_batch_topk(x, k_per_token, True, None, use_triton=True)
        
        # Both should give valid outputs
        assert output1.shape == x.shape
        assert output2.shape == x.shape
        
        # Check sparsity
        sparsity1 = (output1 == 0).float().mean()
        expected_sparsity = 1.0 - (k_per_token / test_tensors['num_features'])
        assert abs(sparsity1 - expected_sparsity) < 0.1
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_performance(self, test_tensors):
        """Test that CUDA version works correctly."""
        x = test_tensors['x'].cuda()
        x_for_ranking = test_tensors['x_for_ranking'].cuda()
        k_per_token = 50
        
        # Original implementation
        original_output = BatchTopK.apply(x, k_per_token, True, x_for_ranking)
        
        # Fused implementation
        fused_output = FusedBatchTopK.apply(x, k_per_token, True, x_for_ranking)
        
        # Check that outputs match
        torch.testing.assert_close(original_output, fused_output, rtol=1e-5, atol=1e-5)
    
    def test_get_optimized_batch_topk(self, test_tensors):
        """Test the factory function for getting optimized modules."""
        x = test_tensors['x']
        k_per_token = 50
        
        # Test different optimization types
        for opt_type in ["fused"]:  # Skip "compile" due to missing g++ in test env
            module = get_optimized_batch_topk(k_per_token, True, opt_type)
            output = module(x)
            
            # Check output properties
            assert output.shape == x.shape
            sparsity = (output == 0).float().mean()
            expected_sparsity = 1.0 - (k_per_token / test_tensors['num_features'])
            assert abs(sparsity - expected_sparsity) < 0.1
    
    def test_different_batch_sizes(self):
        """Test with various batch sizes."""
        torch.manual_seed(42)
        num_features = 512
        k_per_token = 50
        
        for batch_size in [1, 16, 64, 128]:
            x = torch.randn(batch_size, num_features)
            
            # Original
            original = BatchTopK.apply(x, k_per_token, True, None)
            
            # Fused
            fused = FusedBatchTopK.apply(x, k_per_token, True, None)
            
            # Check outputs match
            torch.testing.assert_close(original, fused, rtol=1e-5, atol=1e-5)
    
    def test_edge_cases(self, test_tensors):
        """Test edge cases."""
        batch_size = test_tensors['batch_size']
        num_features = test_tensors['num_features']
        
        # Test with k = num_features (keep all)
        x = torch.randn(batch_size, num_features)
        output = FusedBatchTopK.apply(x, num_features, True, None)
        torch.testing.assert_close(output, x)  # Should be unchanged
        
        # Test with k = 1 (very sparse)
        output = FusedBatchTopK.apply(x, 1, True, None)
        sparsity = (output == 0).float().mean()
        expected_sparsity = 1.0 - (1.0 / num_features)
        assert abs(sparsity - expected_sparsity) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])