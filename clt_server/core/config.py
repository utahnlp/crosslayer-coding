import os
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Base directory for storing activation datasets and chunks
    STORAGE_BASE_DIR: Path = Path("./server_data")

    # Number of extra random chunks to try loading if the first fails
    CHUNK_RETRY_ATTEMPTS: int = 3

    # Other potential settings (e.g., logging level, allowed origins for CORS)
    LOG_LEVEL: str = "info"

    class Config:
        # Optional: Load from a .env file if present
        env_file = ".env"
        env_file_encoding = "utf-8"


# Create a single instance of the settings to be imported elsewhere
settings = Settings()

# Ensure the storage directory exists on startup (or handle creation elsewhere)
if not settings.STORAGE_BASE_DIR.exists():
    print(f"Creating storage directory: {settings.STORAGE_BASE_DIR}")
    settings.STORAGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
