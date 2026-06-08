import cv2
import csv
import os
import time

import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_draw

os.makedirs('data', exist_ok=True)

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

pose_labels = {
    'a': (0, 'rat'), 'b': (1, 'ox'), 'c': (2, 'tiger'), 'd': (3, 'rabbit'),
    'e': (4, 'dragon'), 'f': (5, 'snake'), 'g': (6, 'horse'), 'h': (7, 'sheep'),
    'i': (8, 'monkey'), 'j': (9, 'chicken'), 'k': (10, 'dog'), 'l': (11, 'pig'),
    'n': (12, 'None')
}

def get_normalized_coords(multi_hand_landmarks):
    sorted_hands = sorted(multi_hand_landmarks, key=lambda h: h.landmark[0].x)

    base_x = sorted_hands[0].landmark[0].x
    base_y = sorted_hands[0].landmark[0].y

    coords = []
    for hand_landmarks in sorted_hands:
        for lm in hand_landmarks.landmark:
            coords.extend([lm.x - base_x, lm.y - base_y])

    if len(coords) == 42:
        coords.extend([0.0] * 42)

    return coords

is_recording = False
record_label_num = 1
record_label_name = ""
frames_recorded = 0
MAX_FRAMES = 20

with open('data/raw_data.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    cap = cv2.VideoCapture(0)

    print("a~l ?몃쾿, n ?됱긽??(醫낅즺: q)")

    while cap.isOpened():
        success, image = cap.read()
        if not success: break

        image = cv2.flip(image, 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        coords = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            coords = get_normalized_coords(results.multi_hand_landmarks)

        if is_recording:
            cv2.putText(image, f"Recording [{record_label_name}] : {frames_recorded} / {MAX_FRAMES}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            if len(coords) == 84:
                writer.writerow(coords + [record_label_num])
                frames_recorded += 1

            if frames_recorded >= MAX_FRAMES:
                is_recording = False
                print(f"[{record_label_name}] {MAX_FRAMES}???섏쭛!")
        else:
            cv2.putText(image, "Press 'a'~'l', 'n' to save. 'q' to quit", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Data Collection', image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if not is_recording:
            key_char = chr(key)
            if key_char in pose_labels:
                record_label_num = pose_labels[key_char][0]
                record_label_name = pose_labels[key_char][1]
                is_recording = True
                frames_recorded = 0
                print("珥ъ쁺?쒖옉??)

    cap.release()
    cv2.destroyAllWindows()
