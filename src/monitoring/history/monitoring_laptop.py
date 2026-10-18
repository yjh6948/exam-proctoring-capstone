import cv2
import mediapipe as mp
import math

# === 유클리드 거리 계산 함수 (두 점 사이의 직선 거리) ===
def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# === 시선 방향 탐지 함수 ===
def detect_gaze_direction(landmarks, w, h):
    # 오른쪽 눈 기준 좌표 (카메라 기준 왼쪽 눈)
    left_eye_outer = landmarks.landmark[33]  # 왼쪽 눈 바깥쪽
    left_eye_inner = landmarks.landmark[133] # 왼쪽 눈 안쪽
    left_iris = landmarks.landmark[468]      # 왼쪽 홍채 중심

    # 오른쪽 눈 기준 좌표
    right_eye_outer = landmarks.landmark[362]  # 오른쪽 눈 바깥쪽
    right_eye_inner = landmarks.landmark[263]  # 오른쪽 눈 안쪽
    right_iris = landmarks.landmark[473]       # 오른쪽 홍채 중심

    # 왼쪽 눈 기준 시선 비율 계산
    left_ratio = (left_iris.x - left_eye_outer.x) / (left_eye_inner.x - left_eye_outer.x)
    # 오른쪽 눈 기준 시선 비율 계산
    right_ratio = (right_iris.x - right_eye_inner.x) / (right_eye_outer.x - right_eye_inner.x)

    # 두 눈의 평균값으로 시선 방향 결정
    avg_ratio = (left_ratio + right_ratio) / 2

    # 시선 방향 판단
    if avg_ratio < 0.35:
        return "Looking Left - Possible Cheating"
    elif avg_ratio > 0.65:
        return "Looking Right - Possible Cheating"
    return ""

# === 손 감지 함수 ===
def detect_hands(hand_results, frame, mp_drawing, mp_hands):
    hand_text = ""
    hand_count = 0

    # 손이 감지되었을 경우
    if hand_results.multi_hand_landmarks:
        hand_count = len(hand_results.multi_hand_landmarks)
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # 손의 랜드마크를 화면에 그림
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 손이 2개 미만일 경우 경고 출력
    if hand_count < 2:
        hand_text = "Hands - Possible cheating"

    return hand_text

# === 경고 문구를 프레임 위에 출력 ===
def draw_alerts(frame, alerts):
    y_offset = 50
    for alert in alerts:
        if alert:
            cv2.putText(frame, alert, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            y_offset += 50

# === 프레임 분석 처리 ===
def process_frame(frame, face_mesh, hands, mp_drawing, mp_hands):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 얼굴과 손 인식
    face_results = face_mesh.process(rgb)
    hand_results = hands.process(rgb)

    warning_text = ""
    hand_text = ""

    # 얼굴이 감지되었을 경우
    if face_results.multi_face_landmarks:
        for landmarks in face_results.multi_face_landmarks:
            # 시선 방향 분석
            warning_text = detect_gaze_direction(landmarks, w, h)

            # 왼쪽 홍채 좌표 표시
            iris_l = landmarks.landmark[468]
            iris_l_coord = (int(iris_l.x * w), int(iris_l.y * h))
            cv2.circle(frame, iris_l_coord, 3, (255, 0, 255), -1)

            # 오른쪽 홍채 좌표 표시
            iris_r = landmarks.landmark[473]
            iris_r_coord = (int(iris_r.x * w), int(iris_r.y * h))
            cv2.circle(frame, iris_r_coord, 3, (255, 0, 255), -1)

    # 손 탐지 결과 분석
    hand_text = detect_hands(hand_results, frame, mp_drawing, mp_hands)

    # 경고 문구 출력
    draw_alerts(frame, [warning_text, hand_text])
    return frame

# === 메인 함수 ===
def main():
    # MediaPipe 솔루션 초기화
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # 얼굴, 손 탐지 모델 초기화
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1,
                                      min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp_hands.Hands()

    # 웹캠 실행
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 처리
        frame = process_frame(frame, face_mesh, hands, mp_drawing, mp_hands)

        # 결과 출력
        cv2.imshow('Test Monitor - Eye and Hand Tracking', frame)

        # ESC 키 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # 리소스 정리
    cap.release()
    cv2.destroyAllWindows()

# === 프로그램 시작점 ===
if __name__ == "__main__":
    main()
