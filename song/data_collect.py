import cv2
import csv
import os
import mediapipe as mp

os.makedirs('data', exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

pose_labels = {
    'a': (0, 'rat'), 'b': (1, 'ox'), 'c': (2, 'tiger'), 'd': (3, 'rabbit'),
    'e': (4, 'dragon'), 'f': (5, 'snake'), 'g': (6, 'horse'), 'h': (7, 'sheep'),
    'i': (8, 'monkey'), 'j': (9, 'rooster'), 'k': (10, 'dog'), 'l': (11, 'boar')
}

with open('data/raw_data.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    cap = cv2.VideoCapture(0)
    
    print("키보드 a~l 키를 눌러 12가지 동작을 저장하세요. (종료: q)")
    for key, (num, name) in pose_labels.items():
        print(f"[{key}] 키 : {name} ({num})")

    while cap.isOpened():
        success, image = cap.read()
        if not success: break

        image = cv2.flip(image, 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        coords = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y])

        cv2.putText(image, "Press 'a'~'l' to save. 'q' to quit", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Data Collection', image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            
        key_char = chr(key)
        if key_char in pose_labels and len(coords) == 42:
            label_num = pose_labels[key_char][0]
            label_name = pose_labels[key_char][1]
            writer.writerow(coords + [label_num])
            print(f"[{label_name}] 저장 완료")

    cap.release()
    cv2.destroyAllWindows()