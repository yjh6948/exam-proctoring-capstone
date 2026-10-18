from picamera2 import Picamera2
import cv2
import time

def adjust_color_balance(frame):
    frame_BGR = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(frame_BGR, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)  # Fixed typo here
    
    lab_clahe = cv2.merge((l_clahe, a, b))
    frame_corrected = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return frame_corrected

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()
time.sleep(1)

while True:
    frame = picam2.capture_array()
    frame_corrected = adjust_color_balance(frame)  # Apply color correction
    cv2.imshow("PiCam2", frame_corrected)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.close()
