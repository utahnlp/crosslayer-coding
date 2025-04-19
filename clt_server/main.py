from fastapi import FastAPI
import uvicorn
import os  # Import os for environment variables

# Import routers (assuming they exist, we will create them)
from .api import health
from .core.config import settings  # Import settings

# Import the low‑level slice server app to expose /datasets endpoints at root
from .core import storage as slice_server

app = FastAPI(
    title="CLT Activation Storage Server",
    description="Stores and serves pre-generated model activations for CLT training.",
    version="0.1.0",
)

# Include API routers
app.include_router(health.router, prefix="/api/v1", tags=["Health"])

# Mount the slice server (raw HDF5 slice endpoints) at root so that
# paths like /datasets/... are served alongside the higher‑level /api/v1 routes.
app.mount("", slice_server.app)


@app.get("/", tags=["Root"])
async def read_root():
    return {
        "message": "Welcome to the CLT Activation Storage Server. See /docs for API details."
    }


# Optional: Add startup/shutdown events later for resource management
# @app.on_event("startup")
# async def startup_event():
#     print("Server starting up...")

# @app.on_event("shutdown")
# async def shutdown_event():
#     print("Server shutting down...")

# Allow running directly for simple testing/development
if __name__ == "__main__":
    # Use environment variables for host/port or defaults
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "info")

    print(f"Starting server on {host}:{port}...")
    print(f"Storage base directory: {settings.STORAGE_BASE_DIR}")  # Log storage dir
    uvicorn.run(app, host=host, port=port, log_level=log_level)
