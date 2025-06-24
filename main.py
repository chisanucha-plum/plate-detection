from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading
from app.router.plate import router 
from app.service.plate import PlateService

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

plate_service = PlateService(model_path='./train/license_plate_detector.pt')

if __name__ == "__main__":
    t = threading.Thread(target=plate_service.detect_plate)
    t.daemon = True  
    t.start()
    uvicorn.run(app, host="0.0.0.0", port=8001)