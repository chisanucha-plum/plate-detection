import cv2
import numpy as np

def convertImage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 100, 200)
    return canny

def process_frame(frame):
    processed_img = convertImage(frame)
    original_img = frame.copy()

    contour_img = processed_img.copy()

    contours, heirarchy = cv2.findContours(contour_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for contour in contours:
        p = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * p, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            license_img = original_img[y:y+h, x:x+w]
            cv2.imshow("License Detected", license_img)
            cv2.drawContours(frame, [contour], -1, (0, 255, 255), 3)

    return frame

# Video capture (0 for webcam, or provide video file path)
cap = cv2.VideoCapture('tests.mp4')  

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Process the current frame
    processed_frame = process_frame(frame)
    
    # Display the processed frame
    cv2.imshow("Video Processing", processed_frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
