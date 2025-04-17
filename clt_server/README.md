# Activation Storage Server

This server provides a RESTful API for storing and retrieving pre-generated model activations, primarily for use with the Cross-Layer Transcoder (CLT) training process.

## Features

- Stores activation chunks uploaded by `ActivationGenerator`.
- Serves dataset metadata (`metadata.json`).
- Serves normalization statistics (`norm_stats.json`).
- Serves random batches of activations for training (`RemoteActivationStore`). (Batch serving logic TBD)

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    # Install PyTorch matching your system/CUDA version (see pytorch.org)
    # e.g., pip install torch torchvision torchaudio
    ```

2.  **Configure Storage:**
    - By default, data is stored in `./server_data`.
    - Set the `STORAGE_BASE_DIR` environment variable to change this location.

3.  **Run the Server:**
    ```bash
    # For development/testing:
    python main.py

    # Or using uvicorn directly:
    uvicorn clt_server.main:app --reload --host 0.0.0.0 --port 8000
    ```
    The `--reload` flag automatically restarts the server when code changes.

## API

The API documentation is available via Swagger UI at `http://<server_address>:8000/docs` when the server is running.

Refer to `ref_docs/activation_server_api.md` in the main project for the detailed API specification.

## TODO

- Implement efficient batch serving logic in `core/storage.py:get_batch`.
- Add robust error handling and logging.
- Consider authentication/authorization.
- Add unit and integration tests.
- Implement alternative storage backends (e.g., S3).
- Optimize chunk storage/retrieval. 