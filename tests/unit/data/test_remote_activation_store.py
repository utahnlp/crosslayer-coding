import pytest
import numpy as np
from unittest.mock import MagicMock
import requests

from clt.training.data.remote_activation_store import RemoteActivationStore


# --- Fixtures ---


@pytest.fixture
def mock_metadata() -> dict:
    """Mock metadata.json content."""
    return {
        "num_layers": 2,
        "d_model": 8,
        "total_tokens": 256,
        "chunk_tokens": 128,
        "dtype": "float16",
    }


@pytest.fixture
def mock_manifest_data() -> bytes:
    """Mock index.bin content as bytes."""
    rows = []
    # Create a manifest for 2 chunks, 128 rows each
    for chunk_id in range(2):
        for row_id in range(128):
            rows.append([chunk_id, row_id])
    return np.array(rows, dtype=np.uint32).tobytes()


class TestRemoteActivationStore:
    def test_initialization_success(self, mocker, mock_metadata, mock_manifest_data):
        """
        Tests successful initialization of RemoteActivationStore by mocking
        the HTTP GET requests for metadata and manifest files.
        """
        # --- Mock requests.get to simulate successful server responses ---
        mock_get = mocker.patch("requests.get")

        # Create a mock response for the metadata request
        mock_meta_response = MagicMock()
        mock_meta_response.status_code = 200
        mock_meta_response.ok = True
        mock_meta_response.json.return_value = mock_metadata

        # Create a mock response for the manifest request
        mock_manifest_response = MagicMock()
        mock_manifest_response.status_code = 200
        mock_manifest_response.ok = True
        mock_manifest_response.content = mock_manifest_data

        # Create a mock response for the *optional* norm_stats request
        mock_norm_response = MagicMock()
        mock_norm_response.status_code = 404
        mock_norm_response.ok = False

        # Define the side effects of calling requests.get
        # The store will try to fetch info, then manifest, then norm_stats
        mock_get.side_effect = [mock_meta_response, mock_manifest_response, mock_norm_response]

        # --- Initialize the store, which will trigger the mocked calls ---
        store = RemoteActivationStore(
            server_url="http://fake-server.com",
            dataset_id="test-dataset",
            train_batch_size_tokens=32,
            seed=42,
            # Use a non-default sampling strategy to ensure it's passed down
            sampling_strategy="random_chunk",
        )

        # --- Assertions ---
        # 1. Verify that the store's attributes were set correctly from metadata
        assert store.num_layers == 2
        assert store.d_model == 8
        assert store.total_tokens == 256
        assert store.sampling_strategy == "random_chunk"

        # 2. Verify that requests.get was called for all three resources
        assert mock_get.call_count == 3

        # Check the call arguments
        calls = mock_get.call_args_list
        assert calls[0].args[0] == "http://fake-server.com/datasets/test-dataset/info"
        assert calls[1].args[0] == "http://fake-server.com/datasets/test-dataset/manifest"
        assert calls[2].args[0] == "http://fake-server.com/datasets/test-dataset/norm_stats"

        # 3. Verify the internal sampler was created correctly
        assert store.sampler is not None
        assert store.sampler.num_chunks == 2
        assert store.sampler.total_rows_this_rank == 256  # world_size=1, rank=0

    def test_initialization_metadata_fails(self, mocker):
        """Tests that initialization raises a RuntimeError if metadata fetch fails."""
        mock_get = mocker.patch("requests.get")
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")

        with pytest.raises(RuntimeError, match="Failed to fetch required resource info"):
            RemoteActivationStore(server_url="http://fake-server.com", dataset_id="test-ds")

    def test_initialization_manifest_fails_with_retries(self, mocker, mock_metadata):
        """Tests that initialization raises RuntimeError if manifest fetch fails after all retries."""
        mock_get = mocker.patch("requests.get")

        # Metadata succeeds
        mock_meta_response = MagicMock()
        mock_meta_response.ok = True
        mock_meta_response.json.return_value = mock_metadata

        # Add the norm stats mock response here as well
        mock_norm_response = MagicMock()
        mock_norm_response.status_code = 404
        mock_norm_response.ok = False

        # Manifest fails multiple times
        mock_get.side_effect = [
            mock_meta_response,
            requests.exceptions.RequestException("Connection error"),
            requests.exceptions.RequestException("Connection error"),
            requests.exceptions.RequestException("Connection error"),
            # Note: the norm_stats call won't be reached because manifest fails
        ]

        mocker.patch("time.sleep")  # Don't actually sleep during test

        with pytest.raises(RuntimeError, match="Failed to load or manifest is empty."):
            RemoteActivationStore(server_url="http://fake-server.com", dataset_id="test-ds")

        # Called once for metadata, 3 times for manifest
        assert mock_get.call_count == 4

    def test_fetch_slice_success(self, mocker, mock_metadata, mock_manifest_data):
        """Tests the _fetch_slice method for a successful POST request."""
        # --- Mock the initialization calls ---
        mock_get = mocker.patch("requests.get")
        mock_meta_response = MagicMock()
        mock_meta_response.ok = True
        mock_meta_response.json.return_value = mock_metadata
        mock_manifest_response = MagicMock()
        mock_manifest_response.ok = True
        mock_manifest_response.content = mock_manifest_data
        mock_norm_response = MagicMock()
        mock_norm_response.status_code = 404
        mock_norm_response.ok = False
        mock_get.side_effect = [mock_meta_response, mock_manifest_response, mock_norm_response]
        store = RemoteActivationStore(server_url="http://fake-server.com", dataset_id="test-ds")

        # --- Mock the POST request for the slice ---
        mock_post = mocker.patch("requests.post")
        mock_post_response = MagicMock()
        mock_post_response.status_code = 200
        mock_post_response.content = b"slice_data"
        mock_post.return_value = mock_post_response

        row_indices = np.array([10, 20, 30], dtype=np.uint32)
        result = store._fetch_slice(chunk_id=5, row_indices=row_indices)

        # Assert the POST call was made correctly
        mock_post.assert_called_once_with(
            "http://fake-server.com/datasets/test-ds/slice?chunk=5",
            json={"rows": [10, 20, 30]},
            timeout=60,
        )
        assert result == b"slice_data"

    def test_fetch_slice_http_error(self, mocker, mock_metadata, mock_manifest_data):
        """Tests that _fetch_slice raises a RuntimeError on HTTP error."""
        # --- Mock the initialization calls ---
        mock_get = mocker.patch("requests.get")
        mock_meta_response = MagicMock()
        mock_meta_response.ok = True
        mock_meta_response.json.return_value = mock_metadata
        mock_manifest_response = MagicMock()
        mock_manifest_response.ok = True
        mock_manifest_response.content = mock_manifest_data
        mock_norm_response = MagicMock()
        mock_norm_response.status_code = 404
        mock_norm_response.ok = False
        mock_get.side_effect = [mock_meta_response, mock_manifest_response, mock_norm_response]
        store = RemoteActivationStore(server_url="http://fake-server.com", dataset_id="test-ds")

        # --- Mock the POST request to fail ---
        mock_post = mocker.patch("requests.post")
        mock_post.side_effect = requests.exceptions.RequestException("Server error")

        row_indices = np.array([10, 20, 30], dtype=np.uint32)
        with pytest.raises(RuntimeError, match="Failed to fetch slice for chunk 5"):
            store._fetch_slice(chunk_id=5, row_indices=row_indices)
