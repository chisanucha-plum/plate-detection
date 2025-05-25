import cv2
import easyocr
from ultralytics import YOLO
import os
import re
import numpy as np
from fastapi import FastAPI
import uvicorn
import threading
import base64

# โหลดโมเดล
LICENSE_MODEL_DETECTION_DIR = './models/license_plate_detector.pt'
license_plate_model = YOLO(LICENSE_MODEL_DETECTION_DIR)

province_corrections = {
    "กรุงเทพฯ": [
        "กรงเทพ", "กรุงเทพ", "กทม", "ก.ท.ม", "กทม.",
        "กรุงทหมนานเค", "กรุงทหม", "กรุงเทพมหานคร", "กรุงเทพมหนคร", "กรุงทหมนานเ ค"
    ],
    "ชลบุรี": ["ชลบรี", "ชลบูรี", "ชลบูร", "ชลบ"],
    "เชียงใหม่": ["เชียงใหม", "เชียงไหม่", "เชีบงใหม่"],
}

def correct_province(text):
    for correct, wrong_list in province_corrections.items():
        for wrong in wrong_list:
            text = re.sub(wrong, correct, text, flags=re.IGNORECASE)
    return text

dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
def correct_plate_characters(text):
    corrected = ""
    for c in text:
        if c in dict_char_to_int:
            corrected += dict_char_to_int[c]
        else:
            corrected += c
    return corrected

reader = easyocr.Reader(['th', 'en'])

app = FastAPI()
latest_plate = {"plate": "", "image": ""}

def detect_plate():
    global latest_plate
    cap = cv2.VideoCapture(0)  # หรือ "video.mp4"
    save_dir = "plate_detection"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    saved_plates = {}  # plate_text: (sharpness, image)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = license_plate_model.predict(frame)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped_plate = frame[y1:y2, x1:x2]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                text = reader.readtext(cropped_plate, detail=0)
                plate_text = " ".join(text)
                plate_text = correct_province(plate_text)
                plate_text = correct_plate_characters(plate_text)

                print(f"ป้ายทะเบียนที่ตรวจพบ: {plate_text}")

                safe_plate_text = "".join(c for c in plate_text if c.isalnum() or c in (' ', '_', '-')).rstrip()
                if not safe_plate_text:
                    continue

                # วัดความคมชัดของป้ายทะเบียน
                gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

                # ถ้ายังไม่เคยบันทึก หรือเจอเฟรมที่คมชัดกว่าเดิม
                if (safe_plate_text not in saved_plates) or (sharpness > saved_plates[safe_plate_text][0]):
                    saved_plates[safe_plate_text] = (sharpness, cropped_plate.copy())
                    # แปลงภาพเป็น base64
                    _, buffer = cv2.imencode('.jpg', cropped_plate)
                    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                    latest_plate["plate"] = plate_text
                    latest_plate["image"] = jpg_as_text

                cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (36, 255, 12), 2)

        cv2.imshow("ผลลัพธ์", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # หลังจากจบ loop ให้บันทึกเฉพาะเฟรมที่คมชัดที่สุด
    for plate, (sharpness, img) in saved_plates.items():
        save_path = os.path.join(save_dir, f"{plate}.jpg")
        cv2.imwrite(save_path, img)

    cap.release()
    cv2.destroyAllWindows()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/plate")
def get_plate():
    return latest_plate

if __name__ == "__main__":
    t = threading.Thread(target=detect_plate)
    t.daemon = True
    t.start()
    uvicorn.run(app, host="0.0.0.0", port=8001)