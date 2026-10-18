import os
# --- 라이브러리 임포트 ---
import cv2 
import numpy as np
import math
import time
import threading
from collections import deque
from picamera2 import Picamera2
import mediapipe as mp
import mariadb
from gpiozero import LED
import pigpio
import RPi.GPIO as GPIO
import atexit

# --- GPIO 핀 설정 ---
TRIG = 23            # 초음파 센서 트리거 핀
ECHO = 24            # 초음파 센서 에코 핀
RED_LED_PIN = 17     # 빨간 LED 핀
GREEN_LED_PIN = 18   # 초록 LED 핀

# --- 하드웨어 초기화 ---
red_led = LED(RED_LED_PIN)
green_led = LED(GREEN_LED_PIN)
pi = pigpio.pi()
picam2 = Picamera2()

# --- GPIO 설정 ---
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

# --- MariaDB 연결 함수 ---
def connect_db():
    try:
        conn = mariadb.connect(
            user=os.environ.get("DB_USER", "proctor"),
            password=os.environ.get("DB_PASSWORD", ""),
            host=os.environ.get("DB_HOST", "localhost"),
            port=3306,
            database="AttendanceDB"
        )
        return conn
    except mariadb.Error as e:
        print(f"Error connecting to MariaDB: {e}")
        return None

# --- 초음파 센서를 통한 거리 측정 함수 ---
def measure_distance():
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    start_time = time.time()
    stop_time = time.time()

    while GPIO.input(ECHO) == 0:
        start_time = time.time()
    while GPIO.input(ECHO) == 1:
        stop_time = time.time()

    duration = stop_time - start_time
    distance = duration * 17150  # 왕복 시간 → 거리(cm) 변환
    return distance

# --- DB에 상태 정보 기록 함수 ---
def insert_status(conn, status):
    try:
        print(f"[DB LOG] Trying to insert status: {status}")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO status_table (status_value, timestamp) VALUES (?, NOW())", (status,))
        conn.commit()
        print("[DB LOG] Insert successful.")
    except mariadb.Error as e:
        print(f"Error inserting into MariaDB: {e}")

# --- 종료 시 자원 정리 함수 ---
def cleanup():
    print("프로그램 종료: 모든 LED OFF")
    red_led.off()
    green_led.off()
    GPIO.cleanup()
    pi.stop()

# --- 유클리드 거리 계산 함수 ---
def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# --- 눈 비율(EAR) 계산 함수 ---
def calculate_ear(eye_top, eye_bottom, eye_left, eye_right):
    vertical = euclidean_distance(eye_top, eye_bottom)
    horizontal = euclidean_distance(eye_left, eye_right)
    return vertical / horizontal

# --- 빨간 LED 잠시 켜기 (경고 시 사용) ---
def turn_on_red_led_temporarily():
    red_led.on()
    green_led.off()
    time.sleep(3)
    red_led.off()
    green_led.on()

# --- 깜빡임 및 졸음 감지 함수 ---
def detect_blink_and_drowsiness(ear, state, thresholds, conn):
    blink_text = ""
    drowsiness_text = ""

    # 깜빡임 감지
    if ear < thresholds["blink"]:
        if not state["blinking"]:
            state["blinking"] = True
            blink_text = "Blink detected"
            state["last_blink_time"] = time.time()
            state["no_blink_detected"] = False
    else:
        state["blinking"] = False

    # 졸음 감지
    if ear < thresholds["closed"]:
        if state["eye_closed_start_time"] is None:
            state["eye_closed_start_time"] = time.time()
        elif time.time() - state["eye_closed_start_time"] >= thresholds["drowsiness_time"]:
            if not state["drowsy"]:
                state["drowsy"] = True
                drowsiness_text = "Drowsiness Detected!"
                threading.Thread(target=turn_on_red_led_temporarily).start()
                if conn:
                    insert_status(conn, "졸고 있음")  # 졸음 상태 DB에 기록
    else:
        # 눈이 다시 떠졌을 경우 상태 초기화
        if state["drowsy"]:
            state["last_blink_time"] = time.time()
        state["eye_closed_start_time"] = None
        state["drowsy"] = False

    return blink_text, drowsiness_text

# --- 화면에 경고 메시지 표시 함수 ---
def draw_alerts(frame, alerts):
    y_offset = 50
    for alert in alerts:
        if alert:
            cv2.putText(frame, alert, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            y_offset += 50

# --- 카메라 색감 보정 함수 (CLAHE 적용) ---
def adjust_color_balance(frame):
    frame_BGR = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(frame_BGR, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    frame_corrected = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return frame_corrected

# --- 프레임 처리: 눈 감지 및 경고 표시 ---
def process_frame(frame, face_mesh, state, thresholds, conn):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb)

    blink_text = ""
    drowsiness_text = ""
    absence_text = ""

    if face_results.multi_face_landmarks:
        for landmarks in face_results.multi_face_landmarks:
            # 왼쪽 눈 랜드마크 추출
            top = (int(landmarks.landmark[159].x * w), int(landmarks.landmark[159].y * h))
            bottom = (int(landmarks.landmark[145].x * w), int(landmarks.landmark[145].y * h))
            left = (int(landmarks.landmark[33].x * w), int(landmarks.landmark[33].y * h))
            right = (int(landmarks.landmark[133].x * w), int(landmarks.landmark[133].y * h))

            # EAR 계산 및 상태 판단
            ear = calculate_ear(top, bottom, left, right)
            blink_text, drowsiness_text = detect_blink_and_drowsiness(ear, state, thresholds, conn)

    # 장시간 깜빡임 없을 때 → 부재 판단
    if not state["drowsy"] and (time.time() - state["last_blink_time"] >= thresholds["no_blink_time"]):
        if not state["no_blink_detected"]:
            absence_text = "No blink detected - Possibly absent"
            threading.Thread(target=turn_on_red_led_temporarily).start()
            if conn:
                insert_status(conn, "부재")  # 부재 의심 상태 DB 기록
            state["no_blink_detected"] = True

    # 경고 메시지 표시
    draw_alerts(frame, [blink_text, drowsiness_text, absence_text])
    return frame

# --- 메인 감지 루프 함수 ---
def main():
    # Mediapipe face mesh 초기화
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 감지 임계값 설정
    thresholds = {
        "blink": 0.21,
        "closed": 0.15,
        "drowsiness_time": 2.0,
        "no_blink_time": 10.0
    }

    # 상태 초기화
    state = {
        "blinking": False,
        "eye_closed_start_time": None,
        "last_blink_time": time.time(),
        "drowsy": False,
        "no_blink_detected": False
    }

    # DB 연결
    conn = connect_db()
    if conn:
        print("[DB LOG] MariaDB 연결 성공")
    else:
        print("[DB ERROR] MariaDB 연결 실패. 종료합니다.")
        return

    # LED 상태 초기화
    green_led.on()
    red_led.off()

    # 카메라 설정 및 시작
    picam2.configure(picam2.create_preview_configuration())
    picam2.start()
    time.sleep(1)

    while True:
        frame = picam2.capture_array()                         # 프레임 캡처
        frame_corrected = adjust_color_balance(frame)         # 색감 보정
        frame_corrected = process_frame(frame_corrected, face_mesh, state, thresholds, conn)  # 눈 감지 및 처리
        cv2.imshow('Attendance Monitor', frame_corrected)     # 화면 출력

        if cv2.waitKey(1) & 0xFF == ord('q'):                 # q 키 누르면 종료
            break

    if conn:
        conn.close()
        print("[DB LOG] MariaDB 연결 종료")
    cv2.destroyAllWindows()
    picam2.close()
    green_led.off()

# --- 거리 감지 후 메인 감지 실행 ---
def distance_camera():
    while True:
        print("200cm 이내로 접근을 기다리는 중...")
        distance = measure_distance()
        print(f"측정 거리: {distance:.2f}cm")
        if distance <= 200:
            print("감지됨! 카메라 실행 중...")
            main()
            break  # 한 번만 실행 후 종료

# --- 프로그램 시작 ---
if __name__ == "__main__":
    atexit.register(cleanup)  # 종료 시 자원 정리 등록
    distance_camera()         # 거리 감지부터 시작
import cv2
import numpy as np
import math
import time
import threading
from collections import deque
from picamera2 import Picamera2
import mediapipe as mp
import mariadb
from gpiozero import LED
import pigpio
import RPi.GPIO as GPIO
import atexit

# --- 핀 설정 ---
TRIG = 23
ECHO = 24
RED_LED_PIN = 17
GREEN_LED_PIN = 18

# --- 하드웨어 초기화 ---
red_led = LED(RED_LED_PIN)
green_led = LED(GREEN_LED_PIN)
pi = pigpio.pi()
picam2 = Picamera2()

# --- GPIO 설정 ---
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

# === MariaDB 연결 함수 ===
def connect_db():
    try:
        conn = mariadb.connect(
            user=os.environ.get("DB_USER", "proctor"),
            password=os.environ.get("DB_PASSWORD", ""),
            host=os.environ.get("DB_HOST", "localhost"),
            port=3306,
            database="AttendanceDB"
        )
        return conn
    except mariadb.Error as e:
        print(f"Error connecting to MariaDB: {e}")
        return None

# --- 거리 측정 함수 ---
def measure_distance():
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    start_time = time.time()
    stop_time = time.time()

    while GPIO.input(ECHO) == 0:
        start_time = time.time()
    while GPIO.input(ECHO) == 1:
        stop_time = time.time()

    duration = stop_time - start_time
    distance = duration * 17150
    return distance

def insert_status(conn, status):
    try:
        print(f"[DB LOG] Trying to insert status: {status}")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO status_table (status_value, timestamp) VALUES (?, NOW())", (status,))
        conn.commit()
        print("[DB LOG] Insert successful.")
    except mariadb.Error as e:
        print(f"Error inserting into MariaDB: {e}")

def cleanup():
    print("프로그램 종료: 모든 LED OFF")
    red_led.off()
    green_led.off()
    GPIO.cleanup()
    pi.stop()

# === 유클리드 거리 계산 함수 ===
def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def calculate_ear(eye_top, eye_bottom, eye_left, eye_right):
    vertical = euclidean_distance(eye_top, eye_bottom)
    horizontal = euclidean_distance(eye_left, eye_right)
    return vertical / horizontal

# --- LED 제어 함수 ---
def turn_on_red_led_temporarily():
    red_led.on()
    green_led.off()
    time.sleep(3)
    red_led.off()
    green_led.on()

# --- 눈 깜빡임 및 졸음 감지 ---
def detect_blink_and_drowsiness(ear, state, thresholds, conn):
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
            if not state["drowsy"]:
                state["drowsy"] = True
                drowsiness_text = "Drowsiness Detected!"
                threading.Thread(target=turn_on_red_led_temporarily).start()
                if conn:
                    insert_status(conn, "졸고 있음")  # 졸음: 1
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

def process_frame(frame, face_mesh, state, thresholds, conn):
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
            blink_text, drowsiness_text = detect_blink_and_drowsiness(ear, state, thresholds, conn)

    if not state["drowsy"] and (time.time() - state["last_blink_time"] >= thresholds["no_blink_time"]):
        if not state["no_blink_detected"]:
            absence_text = "No blink detected - Possibly absent"
            threading.Thread(target=turn_on_red_led_temporarily).start()
            if conn:
                insert_status(conn, "부재")  # 부재 의심: 2
            state["no_blink_detected"] = True

    draw_alerts(frame, [blink_text, drowsiness_text, absence_text])
    return frame

# === 메인 카메라 감지 함수 ===
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

    conn = connect_db()
    if conn:
        print("[DB LOG] MariaDB 연결 성공")
    else:
        print("[DB ERROR] MariaDB 연결 실패. 종료합니다.")
        return

    # 초기 초록 LED 켜기
    green_led.on()
    red_led.off()

    picam2.configure(picam2.create_preview_configuration())
    picam2.start()
    time.sleep(1)

    while True:
        frame = picam2.capture_array()
        frame_corrected = adjust_color_balance(frame)
        frame_corrected = process_frame(frame_corrected, face_mesh, state, thresholds, conn)
        cv2.imshow('Attendance Monitor', frame_corrected)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if conn:
        conn.close()
        print("[DB LOG] MariaDB 연결 종료")
    cv2.destroyAllWindows()
    picam2.close()
    
    green_led.off()

# === 거리 측정 후 카메라 실행 ===
def distance_camera():
    while True:
        print("200cm 이내로 접근을 기다리는 중...")
        distance = measure_distance()
        print(f"측정 거리: {distance:.2f}cm")
        if distance <= 200:
            print("감지됨! 카메라 실행 중...")
            main()
            break  # 한 번 실행 후 종료

# === 프로그램 시작 ===
if __name__ == "__main__":
    atexit.register(cleanup)
    distance_camera()
