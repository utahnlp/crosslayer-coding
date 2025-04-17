from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Provides a simple health check endpoint."""
    return {"status": "ok"}
