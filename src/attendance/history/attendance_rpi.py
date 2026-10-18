#라즈베리파이 실행 버전
import cv2
import mediapipe as mp
import math
import time
from picamera2 import Picamera2

# === 유클리드 거리 계산 함수 (두 점 사이의 직선 거리) ===
def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def calculate_ear(eye_top, eye_bottom, eye_left, eye_right):
    vertical = euclidean_distance(eye_top, eye_bottom)
    horizontal = euclidean_distance(eye_left, eye_right)
    return vertical / horizontal

def detect_blink_and_drowsiness(ear, state, thresholds):
    blink_text = ""
    drowsiness_text = ""

    if ear < thresholds["blink"]:
        if not state["blinking"]:
            state["blinking"] = True
            blink_text = "Blink detected"
            state["last_blink_time"] = time.time()
            state["no_blink_detected"] = False
    else:
        state["blinking"] = False

    if ear < thresholds["closed"]:
        if state["eye_closed_start_time"] is None:
            state["eye_closed_start_time"] = time.time()
        elif time.time() - state["eye_closed_start_time"] >= thresholds["drowsiness_time"]:
            state["drowsy"] = True
            drowsiness_text = "Drowsiness Detected!"
    else:
        if state["drowsy"]:
            state["last_blink_time"] = time.time()
        state["eye_closed_start_time"] = None
        state["drowsy"] = False

    return blink_text, drowsiness_text

def draw_alerts(frame, alerts):
    y_offset = 50
    for alert in alerts:
        if alert:
            cv2.putText(frame, alert, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            y_offset += 50

def adjust_color_balance(frame):
    frame_BGR = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(frame_BGR, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge((l_clahe, a, b))
    frame_corrected = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return frame_corrected

def process_frame(frame, face_mesh, state, thresholds):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(rgb)

    blink_text = ""
    drowsiness_text = ""
    absence_text = ""

    if face_results.multi_face_landmarks:
        for landmarks in face_results.multi_face_landmarks:
            top = (int(landmarks.landmark[159].x * w), int(landmarks.landmark[159].y * h))
            bottom = (int(landmarks.landmark[145].x * w), int(landmarks.landmark[145].y * h))
            left = (int(landmarks.landmark[33].x * w), int(landmarks.landmark[33].y * h))
            right = (int(landmarks.landmark[133].x * w), int(landmarks.landmark[133].y * h))

            ear = calculate_ear(top, bottom, left, right)

            blink_text, drowsiness_text = detect_blink_and_drowsiness(ear, state, thresholds)

    if not state["drowsy"] and (time.time() - state["last_blink_time"] >= thresholds["no_blink_time"]):
        absence_text = "No blink detected - Possibly absent"

    draw_alerts(frame, [blink_text, drowsiness_text, absence_text])
    return frame

# === 메인 루프 ===
def main():
    mp_face_mesh = mp.solutions.face_mesh

    face_mesh = mp_face_mesh.FaceMesh(
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    thresholds = {
        "blink": 0.21,
        "closed": 0.15,
        "drowsiness_time": 2.0,
        "no_blink_time": 10.0
    }

    state = {
        "blinking": False,
        "eye_closed_start_time": None,
        "last_blink_time": time.time(),
        "drowsy": False,
        "no_blink_detected": False
    }

    # PiCamera2 사용 설정
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration())
    picam2.start()

    time.sleep(1)

    while True:
        # PiCamera2에서 프레임 캡처
        frame = picam2.capture_array()

        # 색 보정 함수 적용
        frame_corrected = adjust_color_balance(frame)

        # 얼굴 추적 및 깜빡임 및 졸음 감지
        frame_corrected = process_frame(frame_corrected, face_mesh, state, thresholds)

        # 결과 화면 표시
        cv2.imshow('Attendance Monitor', frame_corrected)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키로 종료
            break

    cv2.destroyAllWindows()
    picam2.close()

if __name__ == "__main__":
    main()
