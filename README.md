# Plate Project

A real-time license plate detection and recognition system using YOLO, EasyOCR, and FastAPI.  
Supports sending the latest license plate text and plate image (base64) to the frontend via API.

## Features

- Detect license plates from camera or video files
- Recognize license plate text using EasyOCR
- Auto-correct common OCR errors for provinces and characters
- Save only the sharpest license plate image per detection
- Provide API endpoints for retrieving the latest license plate and image (for frontend integration)

## Installation

1. **Install Python 3.8+**
2. **Install required libraries**
    ```
    pip install opencv-python easyocr ultralytics fastapi uvicorn numpy
    ```
3. **Download YOLO model for license plate detection**
    - Place the model file at `./models/license_plate_detector.pt`

## Usage

1. **Run the program**
    ```
    python main.py
    ```
2. **Camera will open and start detection**
    - Press `q` to stop

3. **Get the latest license plate and image via API**
    - Visit [http://localhost:8001/plate](http://localhost:8001/plate)
    - You will get a JSON response like:
      ```json
      {
        "plate": "1ABC1234 Bangkok",
        "image": "<base64 string>"
      }
      ```

## Frontend Integration

Frontend can fetch the latest license plate and image by calling the `/plate` API.  
Example (JavaScript):
```js
fetch("http://localhost:8001/plate")
  .then(res => res.json())
  .then(data => {
    console.log(data.plate);
    // Show image
    document.getElementById("plate-img").src = "data:image/jpeg;base64," + data.image;
  });
```
```html
<img id="plate-img" alt="plate image" />
```

## Notes

- To use with a video file, change `cv2.VideoCapture(0)` to the path of your video file.
- If port 8001 is in use, change it in `main.py`.
- Make sure your camera or video file is available and accessible.

---

**Developed by:**  
- KMUTT