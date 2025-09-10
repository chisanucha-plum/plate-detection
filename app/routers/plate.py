from fastapi import APIRouter

from app.service.plate import PlateService

router = APIRouter()
plate_service = PlateService(model_path="./train/license_plate_detector.pt")


@router.get("/detect_plate")
async def detect_plate():
    return plate_service.detect_plate()
