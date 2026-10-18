import mediapipe as mp
import cv2
import numpy as np
import math
import time
from collections import deque

# helper 함수들
relative = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]))
relativeT = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]), 0)

# 전역 변수로 시선 방향 기록과 경고 상태를 저장
gaze_history = deque(maxlen=30)  # 30프레임 저장 (약 1초)
warning_start_time = None
warning_direction = None
start_time = time.time()  # 프로그램 시작 시간

def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def detect_hands(hand_results, frame, mp_drawing, mp_hands):
    hand_text = ""
    hand_count = 0

    if hand_results.multi_hand_landmarks:
        hand_count = len(hand_results.multi_hand_landmarks)
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if hand_count < 2:
        hand_text = "Hands - Possible cheating"
        print(f"{format_time(time.time() - start_time)} - 3")

    return hand_text

def gaze(frame, points):
    """
    The gaze function gets an image and face landmarks from mediapipe framework.
    The function draws the gaze direction into the frame.
    """
    global gaze_history, warning_start_time, warning_direction
    
    try:
        '''
        2D image points.
        relative takes mediapipe points that is normalized to [-1, 1] and returns image points
        at (x,y) format
        '''
        image_points = np.array([
            relative(points.landmark[4], frame.shape),  # Nose tip
            relative(points.landmark[152], frame.shape),  # Chin
            relative(points.landmark[263], frame.shape),  # Left eye left corner
            relative(points.landmark[33], frame.shape),  # Right eye right corner
            relative(points.landmark[287], frame.shape),  # Left Mouth corner
            relative(points.landmark[57], frame.shape)  # Right mouth corner
        ], dtype="double")

        '''
        2D image points.
        relativeT takes mediapipe points that is normalized to [-1, 1] and returns image points
        at (x,y,0) format
        '''
        image_points1 = np.array([
            relativeT(points.landmark[4], frame.shape),  # Nose tip
            relativeT(points.landmark[152], frame.shape),  # Chin
            relativeT(points.landmark[263], frame.shape),  # Left eye, left corner
            relativeT(points.landmark[33], frame.shape),  # Right eye, right corner
            relativeT(points.landmark[287], frame.shape),  # Left Mouth corner
            relativeT(points.landmark[57], frame.shape)  # Right mouth corner
        ], dtype="double")

        # 3D model points.
        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0, -63.6, -12.5),  # Chin
            (-43.3, 32.7, -26),  # Left eye, left corner
            (43.3, 32.7, -26),  # Right eye, right corner
            (-28.9, -28.9, -24.1),  # Left Mouth corner
            (28.9, -28.9, -24.1)  # Right mouth corner
        ])

        '''
        3D model eye points
        The center of the eye ball
        '''
        Eye_ball_center_right = np.array([[-29.05], [32.7], [-39.5]])
        Eye_ball_center_left = np.array([[29.05], [32.7], [-39.5]])  # the center of the left eyeball as a vector.

        '''
        camera matrix estimation
        '''
        focal_length = frame.shape[1]
        center = (frame.shape[1] / 2, frame.shape[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        # 2d pupil location
        left_pupil = relative(points.landmark[468], frame.shape)
        right_pupil = relative(points.landmark[473], frame.shape)

        # Transformation between image point to world point
        _, transformation, _ = cv2.estimateAffine3D(image_points1, model_points)  # image to world transformation

        if transformation is not None:  # if estimateAffine3D secsseded
            # project pupil image point into 3d world point 
            pupil_world_cord = transformation @ np.array([[left_pupil[0], left_pupil[1], 0, 1]]).T

            # 3D gaze point (10 is arbitrary value denoting gaze distance)
            S = Eye_ball_center_left + (pupil_world_cord - Eye_ball_center_left) * 10

            # Project a 3D gaze direction onto the image plane.
            (eye_pupil2D, _) = cv2.projectPoints((int(S[0]), int(S[1]), int(S[2])), rotation_vector,
                                                 translation_vector, camera_matrix, dist_coeffs)
            # project 3D head pose into the image plane
            (head_pose, _) = cv2.projectPoints((int(pupil_world_cord[0]), int(pupil_world_cord[1]), int(40)),
                                               rotation_vector,
                                               translation_vector, camera_matrix, dist_coeffs)
            # correct gaze for head rotation
            gaze = left_pupil + (eye_pupil2D[0][0] - left_pupil) - (head_pose[0][0] - left_pupil)

            # Draw gaze line into screen
            p1 = (int(left_pupil[0]), int(left_pupil[1]))
            p2 = (int(gaze[0]), int(gaze[1]))
            
            # 선의 길이 계산
            line_length = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            # 좌표 차이 계산
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            # 선 그리기
            cv2.line(frame, p1, p2, (0, 0, 255), 2)
            
            # 좌표와 길이 정보 표시
            cv2.putText(frame, f"P1: ({p1[0]}, {p1[1]})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"P2: ({p2[0]}, {p2[1]})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Length: {line_length:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"dx: {dx}, dy: {dy}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 현재 시선 방향 기록
            current_direction = None
            if line_length >= 50:
                if dx >= 50:
                    current_direction = "left"
                    print(f"{format_time(time.time() - start_time)} - 4")
                elif dx <= -80:
                    current_direction = "right"
                    print(f"{format_time(time.time() - start_time)} - 5")
            
            gaze_history.append(current_direction)
            
            # 경고 메시지 처리
            current_time = time.time()
            
            # 경고 상태 확인
            if warning_start_time is not None:
                if current_time - warning_start_time < 3:  # 3초 동안 경고 유지
                    warning_text = f"Looking {warning_direction.capitalize()} - Possible Cheating"
                    cv2.putText(frame, warning_text, (10, 150), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    warning_start_time = None
                    warning_direction = None
            
            # 새로운 경고 조건 확인
            if warning_start_time is None:
                left_count = sum(1 for x in gaze_history if x == "left")
                right_count = sum(1 for x in gaze_history if x == "right")
                
                if left_count >= 2:
                    warning_start_time = current_time
                    warning_direction = "left"
                elif right_count >= 2:
                    warning_start_time = current_time
                    warning_direction = "right"
            
    except Exception as e:
        print(f"Gaze estimation error: {str(e)}")

def main():
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # camera stream:
    cap = cv2.VideoCapture(0)  # 기본 웹캠 사용
    if not cap.isOpened():
        print("카메라를 열 수 없습니다. 카메라가 연결되어 있는지 확인해주세요.")
        exit()

    # 카메라 프레임 크기 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("카메라가 시작되었습니다. 얼굴을 카메라에 보여주세요.")

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh, \
         mp_hands.Hands() as hands:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_results = face_mesh.process(image)
            hand_results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if face_results.multi_face_landmarks:
                gaze(image, face_results.multi_face_landmarks[0])

            # 손 감지 결과 처리
            hand_text = detect_hands(hand_results, image, mp_drawing, mp_hands)
            if hand_text:
                cv2.putText(image, hand_text, (10, 180), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('output window', image)
            if cv2.waitKey(2) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
