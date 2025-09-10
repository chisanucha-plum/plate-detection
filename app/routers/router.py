from fastapi import APIRouter

from app.routers.plate import router as plate_router


def get_router():
    router = APIRouter()
    router.include_router(plate_router, prefix="/api")
    return router
