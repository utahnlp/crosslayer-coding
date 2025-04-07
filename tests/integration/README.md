# CLT Integration Tests

This directory contains integration tests for the Cross-Layer Transcoder (CLT) library. Unlike unit tests that verify individual components in isolation, these integration tests verify that multiple components work together correctly.

## Testing Strategy

Our integration tests focus on these key integration points:

1. **Configuration Loading & Usage**: Testing that `CLTConfig` and `TrainingConfig` are correctly used by the trainer, models, and loss manager components.

2. **Activation Data Pipeline**: Verifying the flow from activation sources → `ActivationStore` → training components.

3. **End-to-End Training**: Testing that the full training pipeline works with real components (small-scale).

4. **Model Persistence**: Testing save/load functionality for trained models.

5. **Config Variants**: Testing different configuration options and ensuring they properly integrate.

## Test Fixtures

The tests use small-scale fixtures to enable quick testing:

- Small model configurations (few layers, few features)
- Minimal activation datasets
- Short training runs (few steps)

Some tests utilize pre-generated files in the `data/` directory.

## Running the Tests

To run the integration tests only:

```bash
pytest tests/integration -v
```

To run a specific integration test file:

```bash
pytest tests/integration/test_training_pipeline.py -v
```

To run a specific test:

```bash
pytest tests/integration/test_activation_store.py::test_activation_store_from_nnsight -v
```

## Writing New Integration Tests

When adding new integration tests:

1. Use the `@pytest.mark.integration` decorator to mark integration tests
2. Use temporary directories for any file operations
3. Keep model sizes and training steps small to ensure tests run quickly
4. When testing with real components, focus on verifying that they connect correctly, not on the quality of results 