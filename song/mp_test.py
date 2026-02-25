import cv2
import urllib.request
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. 모델 파일 다운로드 (hand_landmarker.task)
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("모델 파일을 다운로드 중입니다... 잠시만 기다려주세요.")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("다운로드 완료!")

# 2. 제스처 인식 로직 (기존과 동일하지만, 랜드마크 접근 방식이 리스트 인덱싱으로 바뀜)
def get_finger_status(hand_landmarks):
    fingers = []

    # 엄지: 랜드마크 4가 랜드마크 3의 왼/오른쪽에 있는지 판단 (플립 상태 고려)
    if hand_landmarks[4].x < hand_landmarks[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # 나머지 손가락: 팁(8, 12, 16, 20)이 PIP(6, 10, 14, 18)보다 위에(y값이 작은지) 있는지 판단
    tips = [8, 12, 16, 20]
    pip_joints = [6, 10, 14, 18]
    for tip, pip in zip(tips, pip_joints):
        if hand_landmarks[tip].y < hand_landmarks[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

def recognize_gesture(fingers_status):
    if fingers_status == [0, 0, 0, 0, 0]: return 'fist'
    elif fingers_status == [0, 1, 0, 0, 0]: return 'point'
    elif fingers_status == [1, 1, 1, 1, 1]: return 'open'
    elif fingers_status == [0, 1, 1, 0, 0]: return 'peace'
    elif fingers_status == [1, 1, 0, 0, 0]: return 'standby'
    return 'unknown'

# 3. 직접 랜드마크 그리기 함수 (에러 방지용)
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]

def draw_landmarks_cv2(image, landmarks):
    h, w, _ = image.shape
    # 선 그리기
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection[0], connection[1]
        start_lm, end_lm = landmarks[start_idx], landmarks[end_idx]
        cx1, cy1 = int(start_lm.x * w), int(start_lm.y * h)
        cx2, cy2 = int(end_lm.x * w), int(end_lm.y * h)
        cv2.line(image, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
    
    # 점 그리기
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

# 4. MediaPipe Tasks 설정
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# 5. 실행부
print("웹캠을 시작합니다... 종료하려면 'ESC'를 누르세요.")
with vision.HandLandmarker.create_from_options(options) as landmarker:
    video = cv2.VideoCapture(0)

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # 프레임 전처리
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 새로운 MediaPipe Image 포맷으로 변환
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 랜드마크 추론
        result = landmarker.detect(mp_image)

        # 결과 처리
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                # 상태 확인 및 제스처 인식
                fingers_status = get_finger_status(hand_landmarks)
                gesture = recognize_gesture(fingers_status)
                
                print(f"인식된 제스처: {gesture} | 상태: {fingers_status}")

                # 화면에 제스처 이름 띄우기
                cv2.putText(frame, gesture, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

                # 손 랜드마크 그리기
                draw_landmarks_cv2(frame, hand_landmarks)

        cv2.imshow('Hand Gesture', frame)
        if cv2.waitKey(1) == 27:  # ESC 키로 종료
            break

    video.release()
    cv2.destroyAllWindows()