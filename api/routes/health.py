from datetime import datetime

from fastapi import APIRouter

from api.schemas import HealthResponse
from utils.constants import IST_ZONE

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", timestamp=datetime.now(IST_ZONE))
