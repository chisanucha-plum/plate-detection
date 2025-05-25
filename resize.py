from PIL import Image
from rembg import remove
import os

# รายชื่อไฟล์ภาพต้นฉบับ
image_files = [
    "KMUTT-Helmet-Logo.png",
    # เพิ่มไฟล์อื่นๆที่ต้องการ
]

# ขนาดที่ต้องการ resize (กว้าง, สูง)
sizes = [
    (16, 16),
    (32, 32),
    (48, 48),
    (64, 64),
    (96, 96),
    (128, 128),
    (192, 192),
    (256, 256),
    (512, 512),
    (70, 70),
    (144, 144),
    (150, 150),
    (310, 310)
]

for image_path in image_files:
    image = Image.open(image_path)
    # ลบพื้นหลัง
    image_no_bg = remove(image)
    for size in sizes:
        output_filename = f"ms-icon-{size[0]}x{size[1]}.png"
        resized_image = image_no_bg.resize(size)
        resized_image.save(output_filename)
        print(f"บันทึก {output_filename} แล้ว")
