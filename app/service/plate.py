import cv2
import easyocr
import os 
import re
import base64
from ultralytics import YOLO

province_corrections = {
    "กรุงเทพฯ": [
        "กรุงเทพ", "กรุงเทพ", "กทม", "ก.ท.ม", "กทม.",
        "กรุงทหมนานเค", "กรุงทหม", "กรุงเทพมหานคร", "กรุงเทพมหนคร", "กรุงทหมนานเ ค"
    ],
    "ชลบุรี": ["ชลบรี", "ชลบูรี", "ชลบูร", "ชลบ"],
    "เชียงใหม่": ["เชียงใหม", "เชียงไหม่", "เชีบงใหม่"],
}

dict_char_to_int = {
    'O': '0', 
    'I': '1', 
    'J': '3', 
    'A': '4', 
    'G': '6',
    'S': '5'
}

def correct_province(text):
    for correct, wrong_list in province_corrections.items():
        for wrong in wrong_list:
            text = re.sub(wrong, correct, text, flags=re.IGNORECASE)
    return text

def correct_plate_characters(text):
    corrected = ""
    for char in text:
        corrected += dict_char_to_int.get(char, char)
    return corrected

class PlateService:
    def __init__(self,model_path):
        self.license_plate_model = YOLO(model_path)
        self.reader = easyocr.Reader(['th', 'en'])
        self.latest_plate = {"plate": "", "image": ""}
        self.save_directory = "plate_detection"
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
            
    def detect_plate(self):
        cap = cv2.VideoCapture(0)
        saved_plates = {}
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.license_plate_model.predict(frame)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cropped_plate = frame[y1:y2, x1:x2]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = self.reader.readtext(cropped_plate, detail=0)
                    plate_text = " ".join(text)
                    plate_text = correct_province(plate_text)
                    plate_text = correct_plate_characters(plate_text)
                    safe_plate_text = "".join(c for c in plate_text if c.isalnum() or c in (' ', '_', '-')).rstrip()
                    if not safe_plate_text:
                        continue
                    gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
                    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                    if (safe_plate_text not in saved_plates) or (sharpness > saved_plates[safe_plate_text][0]):
                        saved_plates[safe_plate_text] = (sharpness, cropped_plate.copy())
                        _, buffer = cv2.imencode('.jpg', cropped_plate)
                        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                        self.latest_plate["plate"] = plate_text
                        self.latest_plate["image"] = jpg_as_text
                    cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (36, 255, 12), 2)
            cv2.imshow("ผลลัพธ์", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        for plate, (sharpness, img) in saved_plates.items():
            save_path = os.path.join(self.save_dir, f"{plate}.jpg")
            cv2.imwrite(save_path, img)
        cap.release()
        cv2.destroyAllWindows()

    def get_latest_plate(self):
        return self.latest_plate
