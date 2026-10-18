def gaze(frame, points):
    """
    시선 추적 함수
    
    Args:
        frame: 웹캠에서 캡처된 이미지 프레임
        points: MediaPipe에서 감지된 얼굴 랜드마크 포인트들
    
    Returns:
        None (프레임에 시선 방향을 직접 그립니다)
    """
    global gaze_history, warning_start_time, warning_direction

    try:
        # 2D 이미지 포인트 설정
        # MediaPipe의 정규화된 좌표(-1, 1)를 실제 이미지 좌표(x,y)로 변환
        image_points = np.array([
            relative(points.landmark[4], frame.shape),      # 코 끝
            relative(points.landmark[152], frame.shape),    # 턱
            relative(points.landmark[263], frame.shape),    # 왼쪽 눈
            relative(points.landmark[33], frame.shape),     # 오른쪽 눈
            relative(points.landmark[287], frame.shape),    # 왼쪽 입 모서리
            relative(points.landmark[57], frame.shape)      # 오른쪽 입 모서리
        ], dtype="double")

        # 3D 변환을 위한 2D 이미지 포인트 (z=0)
        image_points1 = np.array([
            relativeT(points.landmark[4], frame.shape),
            relativeT(points.landmark[152], frame.shape),
            relativeT(points.landmark[263], frame.shape),
            relativeT(points.landmark[33], frame.shape),
            relativeT(points.landmark[287], frame.shape),
            relativeT(points.landmark[57], frame.shape)
        ], dtype="double")

        # 3D 모델 포인트 (실제 얼굴의 3D 좌표)
        model_points = np.array([
            (0.0, 0.0, 0.0), # 코 끝
            (0, -63.6, -12.5), # 턱
            (-43.3, 32.7, -26), # 왼쪽 눈
            (43.3, 32.7, -26), # 오른쪽 눈
            (-28.9, -28.9, -24.1), # 왼쪽 입 모서리
            (28.9, -28.9, -24.1) # 오른쪽 입 모서리
        ])

        # 3D 눈동자 중심점 좌표
        Eye_ball_center_left = np.array([[29.05], [32.7], [-39.5]])  # 왼쪽 눈동자 중심

        # 카메라 행렬 설정
        focal_length = frame.shape[1]  # 초점 거리
        center = (frame.shape[1] / 2, frame.shape[0] / 2)  # 이미지 중심점
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        # 렌즈 왜곡 계수 (왜곡이 없다고 가정)
        dist_coeffs = np.zeros((4, 1))

        # PnP 알고리즘으로 회전 벡터와 이동 벡터 계산
        success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                     dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        # MediaPipe에서 감지된 왼쪽 눈동자 위치
        left_pupil = relative(points.landmark[468], frame.shape)

        # 2D 이미지 포인트를 3D 월드 좌표로 변환
        _, transformation, _ = cv2.estimateAffine3D(image_points1, model_points)
        if transformation is not None:
            # 눈동자 위치를 3D 월드 좌표로 변환
            pupil_world_cord = transformation @ np.array([[left_pupil[0], left_pupil[1], 0, 1]]).T
            
            # 3D 시선 방향 계산 (10은 임의의 거리 값)
            S = Eye_ball_center_left + (pupil_world_cord - Eye_ball_center_left) * 10

            # 3D 시선 방향을 이미지 평면에 투영
            (eye_pupil2D, _) = cv2.projectPoints((int(S[0]), int(S[1]), int(S[2])), rotation_vector,
                                                 translation_vector, camera_matrix, dist_coeffs)

            # 3D 머리 포즈를 이미지 평면에 투영
            (head_pose, _) = cv2.projectPoints((int(pupil_world_cord[0]), int(pupil_world_cord[1]), int(40)),
                                               rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            # 머리 회전을 고려한 시선 방향 보정
            gaze_point = left_pupil + (eye_pupil2D[0][0] - left_pupil) - (head_pose[0][0] - left_pupil)

            # 시선 방향을 화면에 그리기
            p1 = (int(left_pupil[0]), int(left_pupil[1]))  # 시작점 (눈동자)
            p2 = (int(gaze_point[0]), int(gaze_point[1]))  # 끝점 (시선 방향)

            # 선의 길이와 방향 계산
            line_length = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            dx = p2[0] - p1[0]  # x축 방향 변화량
            dy = p2[1] - p1[1]  # y축 방향 변화량

            # 시선 방향 선 그리기
            cv2.line(frame, p1, p2, (0, 0, 255), 2)
            cv2.putText(frame, f"dx: {dx}, dy: {dy}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # 현재 시선 방향 판단
            current_direction = None
            if line_length >= 50:  # 시선이 충분히 길 때만 방향 판단
                if dx >= 50:
                    current_direction = "left"  # 왼쪽을 보고 있음
                    print(f"{format_time(time.time() - start_time)} - 왼쪽을 보고 있음, 부정행위 가능성")
                    if conn:
                        insert_status(conn, "왼쪽을 보고 있음, 부정행위 가능성" ) #Looking Left 4
                elif dx <= -80:
                    current_direction = "right"  # 오른쪽을 보고 있음
                    print(f"{format_time(time.time() - start_time)} - 오른쪽을 보고 있음, 부정행위 가능성")
                    if conn:
                        insert_status(conn, "오른쪽을 보고 있음, 부정행위 가능성") #Looking Right 5

            # 시선 방향 기록
            gaze_history.append(current_direction)

            # 경고 메시지 처리
            current_time = time.time()
            if warning_start_time is not None:
                if current_time - warning_start_time < 1:  # 1초 동안 경고 유지
                    warning_text = f"Looking {warning_direction.capitalize()} - Possible Cheating"
                    cv2.putText(frame, warning_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    warning_start_time = None
                    warning_direction = None

            # 새로운 경고 조건 확인
            if warning_start_time is None:
                left_count = sum(1 for x in gaze_history if x == "left")
                right_count = sum(1 for x in gaze_history if x == "right")

                # 연속된 방향 감지 시 경고
                if left_count >= 2:
                    warning_start_time = current_time
                    warning_direction = "left"
                elif right_count >= 2:
                    warning_start_time = current_time
                    warning_direction = "right"

    except Exception as e:
        print(f"Gaze estimation error: {str(e)}")
