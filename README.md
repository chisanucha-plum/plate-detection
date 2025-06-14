# Plate Project

ระบบตรวจจับและอ่านป้ายทะเบียนรถยนต์แบบ Real-time ด้วย YOLO, EasyOCR และ FastAPI  
รองรับการส่งข้อมูลป้ายทะเบียนล่าสุดและภาพป้ายทะเบียน (base64) ไปยัง Frontend ผ่าน API

## คุณสมบัติ

- ตรวจจับป้ายทะเบียนจากกล้องหรือวิดีโอ
- อ่านตัวอักษรบนป้ายทะเบียนด้วย EasyOCR
- แก้ไขคำผิดของจังหวัดและตัวอักษรที่ OCR มักอ่านผิด
- บันทึกภาพป้ายทะเบียนที่คมชัดที่สุดเท่านั้น
- ให้บริการ API สำหรับดึงป้ายทะเบียนล่าสุดและภาพ (เหมาะสำหรับเชื่อมต่อกับ Frontend)

## วิธีติดตั้ง

1. **ติดตั้ง Python 3.8+**
2. **ติดตั้งไลบรารีที่จำเป็น**
    ```
    pip install opencv-python easyocr ultralytics fastapi uvicorn numpy
    ```
3. **ดาวน์โหลดโมเดล YOLO สำหรับตรวจจับป้ายทะเบียน**
    - วางไฟล์โมเดลไว้ที่ `./models/license_plate_detector.pt`

## วิธีใช้งาน

1. **รันโปรแกรม**
    ```
    python main.py
    ```
2. **เปิดกล้องและเริ่มตรวจจับ**
    - กด `q` เพื่อหยุดการทำงาน

3. **เรียกดูป้ายทะเบียนล่าสุดและภาพผ่าน API**
    - ไปที่ [http://localhost:8001/plate](http://localhost:8001/plate)
    - จะได้ข้อมูล JSON เช่น
      ```json
      {
        "plate": "1กข1234 กรุงเทพฯ",
        "image": "<base64 string>"
      }
      ```

## การนำไปใช้กับ Frontend

Frontend สามารถดึงข้อมูลป้ายทะเบียนล่าสุดและภาพได้โดยการเรียก API `/plate`  
ตัวอย่าง (JavaScript):
```js
fetch("http://localhost:8001/plate")
  .then(res => res.json())
  .then(data => {
    console.log(data.plate);
    // แสดงภาพ
    document.getElementById("plate-img").src = "data:image/jpeg;base64," + data.image;
  });
```
```html
<img id="plate-img" alt="plate image" />
```

## หมายเหตุ

- หากต้องการใช้กับไฟล์วิดีโอ ให้เปลี่ยน `cv2.VideoCapture(0)` เป็น path ของไฟล์วิดีโอ
- หากพอร์ต 8001 ถูกใช้งาน ให้เปลี่ยนเป็นพอร์ตอื่นในไฟล์ `main.py`
- ตรวจสอบให้แน่ใจว่ากล้องหรือไฟล์วิดีโอพร้อมใช้งาน

---

**ผู้พัฒนา:**  
- KMUTT Plate Project Team