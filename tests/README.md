# CLT Testing Strategy

This directory contains tests for the Cross-Layer Transcoder (CLT) library. The testing strategy is divided into two main categories:

## Unit Tests

Located in `tests/unit/`, these tests verify that individual components of the library work correctly in isolation. Unit tests are designed to:

- Test specific functionality of a single class or function
- Mock dependencies to isolate the component being tested
- Run quickly and provide good coverage
- Help identify where issues originate

## Integration Tests

Located in `tests/integration/`, these tests verify that multiple components work together correctly. Integration tests are designed to:

- Test realistic user workflows
- Verify that components interact properly
- Use real (but small-scale) components instead of mocks
- Test data flow between components

## Running Tests

To run all tests:

```bash
pytest
```

To run unit tests only:

```bash
pytest tests/unit
```

To run integration tests only:

```bash
pytest tests/integration
```

## Test Fixtures

The `tests/integration/data/` directory contains fixtures for integration tests, including:

- Sample activation data
- Pre-trained model files
- Helper scripts to generate test data

These fixtures enable deterministic testing of model loading, inference, and training without requiring external data.

## Writing New Tests

When adding new functionality to the library:

1. Add unit tests for the new component in `tests/unit/`
2. Add integration tests in `tests/integration/` for any new workflows

Use the `@pytest.mark.integration` marker for integration tests:

```python
@pytest.mark.integration
def test_my_integration_test():
    # Test code here
    pass
``` 