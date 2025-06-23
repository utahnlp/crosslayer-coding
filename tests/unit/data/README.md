# Data Integrity Tests

This directory contains comprehensive tests to ensure data integrity across the activation generation and retrieval pipeline.

## Background

We've experienced data mixup issues in the past:
1. **Lexicographic vs Numerical Ordering**: Layers were sorted as strings (layer_10, layer_2, layer_20) instead of numerically
2. **Layer Data Corruption**: Similar issues with layer ordering causing data from one layer to be associated with another

## Test Coverage

### `test_data_integrity.py`

Comprehensive test suite covering:

1. **Layer Ordering** (`test_layer_ordering_numerical_not_lexicographic`)
   - Verifies layers are ordered numerically (1, 2, 10, 20, 100) not lexicographically
   - Creates layers that would be misordered if sorted as strings
   - Validates both HDF5 structure and actual data values

2. **Normalization Application** (`test_normalization_application_correctness`)
   - Tests that normalization statistics are correctly applied during retrieval
   - Creates data with known mean/std, then verifies normalized output
   - Ensures each layer's statistics are applied to the correct layer

3. **Cross-Chunk Token Ordering** (`test_cross_chunk_token_ordering`)
   - Verifies token ordering is preserved across chunk boundaries
   - Uses deterministic patterns to track tokens across multiple chunks
   - Ensures no tokens are duplicated or lost

4. **Manifest Format Compatibility** (`test_manifest_format_compatibility`)
   - Tests both legacy 2-field and new 3-field manifest formats
   - Ensures backward compatibility with existing datasets

### `test_local_activation_store.py`

Includes additional test:
- **Layer Data Integrity** (`test_layer_data_integrity`)
  - Verifies each layer contains distinct, non-mixed data
  - Checks value ranges are layer-specific
  - Ensures targets = inputs + 1 relationship is preserved

## Running the Tests

Run all data integrity tests:
```bash
pytest tests/unit/data/test_data_integrity.py -v
```

Run specific test:
```bash
pytest tests/unit/data/test_data_integrity.py::TestDataIntegrity::test_layer_ordering_numerical_not_lexicographic -v
```

Run with coverage:
```bash
pytest tests/unit/data/test_data_integrity.py --cov=clt.activation_generation --cov=clt.training.data -v
```

## What These Tests Prevent

1. **Silent Data Corruption**: Detects if layers get mixed up during generation or retrieval
2. **Normalization Errors**: Ensures statistics from one layer aren't applied to another
3. **Token Loss**: Verifies all tokens are accessible and in correct order
4. **Format Regressions**: Maintains compatibility with existing activation datasets

## Adding New Tests

When adding features that touch activation generation or retrieval:
1. Add tests that use deterministic, verifiable patterns
2. Test edge cases (empty chunks, single token, many layers)
3. Verify both structure (metadata, manifests) and actual data values
4. Consider cross-component interactions 