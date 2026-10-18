import cv2
import mediapipe as mp
import math
import time
from datetime import datetime

# === 유클리드 거리 계산 함수 ===
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
            drowsiness_text = "Drowsiness Detected!"  # 1번 이벤트
    else:
        if state["drowsy"]:
            state["last_blink_time"] = time.time()
        state["eye_closed_start_time"] = None
        state["drowsy"] = False

    return blink_text, drowsiness_text

# === 경고 텍스트 표시 ===
def draw_alerts(frame, alerts):
    y_offset = 50
    for alert in alerts:
        if alert:
            cv2.putText(frame, alert, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            y_offset += 50

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
        absence_text = "No blink detected - Possibly absent"  # 2번 이벤트

    draw_alerts(frame, [blink_text, drowsiness_text, absence_text])

    return frame, blink_text, drowsiness_text, absence_text

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

    cap = cv2.VideoCapture(0)

    last_print_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, blink_text, drowsiness_text, absence_text = process_frame(frame, face_mesh, state, thresholds)

        # 1초에 한번만 출력
        current_time_sec = time.time()
        if current_time_sec - last_print_time >= 1.0:
            current_time = datetime.now().strftime("%H:%M:%S")
            # 졸음(1번)과 absence(2번)만 출력
            if drowsiness_text:
                print(f"[1] [{current_time}] {drowsiness_text}")
                #  여기서 바로 DB 저장 가능하게
            if absence_text:
                print(f"[2] [{current_time}] {absence_text}")
                #  여기서 바로 DB 저장 가능하게

            last_print_time = current_time_sec

        cv2.imshow('Attendance Monitor', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
