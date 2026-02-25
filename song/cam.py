import cv2
import mediapipe as mp
import torch
import time
from models.model import HandPoseModel

# 12지신 라벨 이름
class_names = ['rat', 'ox', 'tiger', 'rabbit', 'dragon', 'snake', 
               'horse', 'sheep', 'monkey', 'rooster', 'dog', 'boar']

# 튜토리얼 3가지 기술 커맨드 정의
jutsus = {
    "1. Chidori": ["ox", "rabbit", "monkey"],
    "2. Rasengan": ["monkey", "dragon", "rat", "rooster", "monkey"],
    "3. Fireball": ["snake", "sheep", "monkey", "boar", "horse", "tiger"]
}

# 모델 불러오기
model = HandPoseModel()
model.load_state_dict(torch.load('models/saved_weights.pth'))
model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# 튜토리얼 상태 관리 변수
current_mission_name = "1. Chidori"
target_sequence = jutsus[current_mission_name]
current_step = 0 # 현재 몇 번째 동작까지 성공했는지 추적
last_pose = None
pose_hold_time = 0


while cap.isOpened():
    success, image = cap.read()
    if not success: break
    image = cv2.flip(image, 1)
    
    # 1. 인공지능 예측 파트
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    predicted_name = "None"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y])
            
            x_tensor = torch.tensor([coords], dtype=torch.float32)
            with torch.no_grad():
                pred = model(x_tensor)
                # 모델이 80% 이상 확신할 때만 인정
                probabilities = torch.nn.functional.softmax(pred, dim=1)[0]
                max_prob, predicted_class = torch.max(probabilities, 0)
                
                if max_prob.item() > 0.8:
                    predicted_name = class_names[predicted_class.item()]

    # 2. 튜토리얼 커맨드 로직 파트
    if predicted_name == target_sequence[current_step]:
        pose_hold_time += 1
    else:
        pose_hold_time = 0

    if pose_hold_time > 10:
        current_step += 1
        pose_hold_time = 0
        print(f"{predicted_name} 성공!")

    # 기술 발동 체크
    if current_step >= len(target_sequence):
        cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), (0, 255, 255), -1)
        cv2.putText(image, f"JUTSU ACTIVATED: {current_mission_name}!!", (50, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        cv2.imshow('Naruto Tutorial', image)
        cv2.waitKey(2000)
        current_step = 0 

    # 3. 화면 UI 그리기 파트
    # 하단 튜토리얼 박스
    cv2.rectangle(image, (0, 400), (640, 480), (0, 0, 0), -1)
    cv2.putText(image, f"Mission: {current_mission_name}", (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # 커맨드 진행도 표시
    display_seq = ""
    for i, pose in enumerate(target_sequence):
        if i < current_step:
            display_seq += f"[{pose.upper()}] -> " 
        else:
            display_seq += f"{pose} -> "
    
    cv2.putText(image, display_seq[:-4], (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 현재 인식 중인 동작 표시 (좌측 상단)
    cv2.putText(image, f'Current: {predicted_name}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Naruto Tutorial', image)
    
    # 키보드 입력으로 미션 변경
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('1'): 
        current_mission_name = "1. Chidori"
        target_sequence = jutsus[current_mission_name]; current_step = 0
    elif key == ord('2'): 
        current_mission_name = "2. Rasengan"
        target_sequence = jutsus[current_mission_name]; current_step = 0
    elif key == ord('3'): 
        current_mission_name = "3. Fireball"
        target_sequence = jutsus[current_mission_name]; current_step = 0

cap.release()
cv2.destroyAllWindows()