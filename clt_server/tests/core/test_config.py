from pathlib import Path
from unittest.mock import MagicMock  # Removed patch import


# --- Helper to reload settings --- #
def reload_settings():
    """Forces reload of the settings module to pick up env var changes."""
    import importlib
    from clt_server.core import config

    importlib.reload(config)
    return config.settings


# --- Test Cases --- #


def test_default_settings(monkeypatch):
    """Test that default settings are loaded correctly."""
    # Ensure no env vars are set that might override defaults
    monkeypatch.delenv("STORAGE_BASE_DIR", raising=False)
    monkeypatch.delenv("CHUNK_RETRY_ATTEMPTS", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)

    # Reload settings to ensure defaults are picked up
    settings = reload_settings()

    assert settings.STORAGE_BASE_DIR == Path("./server_data")
    assert settings.CHUNK_RETRY_ATTEMPTS == 3
    assert settings.LOG_LEVEL == "info"


def test_override_settings_from_env(monkeypatch, tmp_path):
    """Test that settings can be overridden using environment variables."""
    test_dir = tmp_path / "env_test_data"
    test_retries = 5
    test_log_level = "debug"

    monkeypatch.setenv("STORAGE_BASE_DIR", str(test_dir))
    monkeypatch.setenv("CHUNK_RETRY_ATTEMPTS", str(test_retries))
    monkeypatch.setenv("LOG_LEVEL", test_log_level)

    settings = reload_settings()

    assert settings.STORAGE_BASE_DIR == test_dir
    assert settings.CHUNK_RETRY_ATTEMPTS == test_retries
    assert settings.LOG_LEVEL == test_log_level

    # The directory should be created if overridden and doesn't exist
    assert test_dir.exists()
    assert test_dir.is_dir()


def test_storage_directory_creation(monkeypatch, tmp_path):
    """Test that the storage directory is created if it doesn't exist."""
    test_dir = tmp_path / "creation_test"
    assert not test_dir.exists()

    monkeypatch.setenv("STORAGE_BASE_DIR", str(test_dir))
    reload_settings()  # This triggers the Settings instantiation and directory check

    assert test_dir.exists()
    assert test_dir.is_dir()


def test_storage_directory_exists(monkeypatch, tmp_path):
    """Test that mkdir is not called if the directory already exists."""
    test_dir = tmp_path / "existing_dir"
    # Create the directory BEFORE the settings are potentially reloaded
    test_dir.mkdir(parents=True, exist_ok=True)
    assert test_dir.exists()  # Verify directory creation

    # Set the env var
    monkeypatch.setenv("STORAGE_BASE_DIR", str(test_dir))

    # Explicitly patch Path.mkdir only during the reload_settings call
    mock_mkdir = MagicMock()
    monkeypatch.setattr(Path, "mkdir", mock_mkdir)

    # Reload settings - this should NOT call the mocked mkdir
    try:
        reload_settings()
    finally:
        # Important: undo the patch even if reload_settings fails
        monkeypatch.undo()

    # Assert that the mocked mkdir was NOT called
    mock_mkdir.assert_not_called()
