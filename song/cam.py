import cv2
import torch
import time
import mediapipe as mp
import pygame
import numpy as np
from pathlib import Path

import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_draw
import mediapipe.python.solutions.selfie_segmentation as mp_selfie
from models.model import HandPoseModel

from PIL import ImageFont, ImageDraw, Image
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"

class_names = ['rat', 'ox', 'tiger', 'rabbit', 'dragon', 'snake',
               'horse', 'sheep', 'monkey', 'chicken', 'dog', 'pig', 'None']

# ?쒗넗由ъ뼹 3媛吏 湲곗닠 而ㅻ㎤???뺤쓽
jutsus = {
    "1. 移섎룄由?: ["ox", "rabbit", "monkey"],
    "2. ?섏꽑??: ["monkey", "dragon", "rat", "chicken", "monkey"],
    "3. ?명솕援ъ쓽 ??: ["snake", "sheep", "monkey", "pig", "horse", "tiger"],
    "4. 洹몃┝??遺꾩떊??: ["sheep", "snake", "tiger"]
}

video_paths = {
    "1. 移섎룄由?: ASSETS_DIR / "chidori.mp4",
    "2. ?섏꽑??: ASSETS_DIR / "rasengan.mp4",
    "3. ?명솕援ъ쓽 ??: ASSETS_DIR / "fireball.mp4",
}

pygame.mixer.init()
sounds = {}
try:
    sounds["1. 移섎룄由?] = pygame.mixer.Sound(str(ASSETS_DIR / "chidori_sound.mp3"))
    sounds["2. ?섏꽑??] = pygame.mixer.Sound(str(ASSETS_DIR / "rasengan_sound.mp3"))
    sounds["3. ?명솕援ъ쓽 ??] = pygame.mixer.Sound(str(ASSETS_DIR / "fireball_sound.mp3"))
    sounds["4. 洹몃┝??遺꾩떊??] = pygame.mixer.Sound(str(ASSETS_DIR / "shadow.mp3"))
except Exception as e:
    print("?ъ슫??濡쒕뱶 ?ㅽ뙣:", e)

def overlay_video_additive(background, effect_frame, x, y):
    bg_h, bg_w = background.shape[:2]
    ef_h, ef_w = effect_frame.shape[:2]

    # ?대?吏媛 ?붾㈃ 以묒븰(x, y)???ㅻ룄濡?醫뚯긽??醫뚰몴 怨꾩궛
    x1, y1 = int(x - ef_w / 2), int(y - ef_h / 2)
    x2, y2 = x1 + ef_w, y1 + ef_h

    bg_x1, bg_y1 = max(0, x1), max(0, y1)
    bg_x2, bg_y2 = min(bg_w, x2), min(bg_h, y2)
    ef_x1, ef_y1 = max(0, -x1), max(0, -y1)
    ef_x2, ef_y2 = ef_w - max(0, x2 - bg_w), ef_h - max(0, y2 - bg_h)

    # ?대?吏媛 ?붾㈃ 諛뽰쑝濡??섍???寃쎌슦 ?섎씪?닿린 (?먮윭 諛⑹?)
    if bg_x1 >= bg_x2 or bg_y1 >= bg_y2:
        return background

    bg_roi = background[bg_y1:bg_y2, bg_x1:bg_x2]
    ef_roi = effect_frame[ef_y1:ef_y2, ef_x1:ef_x2]

    blended =cv2.add(bg_roi, ef_roi)
    background[bg_y1:bg_y2, bg_x1:bg_x2] = blended
    return background

def overlay_clone(background, clone_img, x, y, scale=0.6):
    h, w = clone_img.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    resized_clone = cv2.resize(clone_img, (new_w, new_h))

    clone_gray = cv2.cvtColor(resized_clone, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(clone_gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    bg_h, bg_w = background.shape[:2]
    x1, y1 = int(x - new_w / 2), int(y - new_h / 2)
    x2, y2 = x1 + new_w, y1 + new_h

    bg_x1, bg_y1 = max(0, x1), max(0, y1)
    bg_x2, bg_y2 = min(bg_w, x2), min(bg_h, y2)
    cl_x1, cl_y1 = max(0, -x1), max(0, -y1)
    cl_x2, cl_y2 = new_w - max(0, x2 - bg_w), new_h - max(0, y2 - bg_h)

    if bg_x1 >= bg_x2 or bg_y1 >= bg_y2:
        return background

    bg_roi = background[bg_y1:bg_y2, bg_x1:bg_x2]
    cl_roi = resized_clone[cl_y1:cl_y2, cl_x1:cl_x2]
    mask_roi = mask[cl_y1:cl_y2, cl_x1:cl_x2]
    mask_inv_roi = mask_inv[cl_y1:cl_y2, cl_x1:cl_x2]

    bg_roi_bg = cv2.bitwise_and(bg_roi, bg_roi, mask=mask_inv_roi)
    cl_roi_fg = cv2.bitwise_and(cl_roi, cl_roi, mask=mask_roi)
    dst = cv2.add(bg_roi_bg, cl_roi_fg)
    background[bg_y1:bg_y2, bg_x1:bg_x2] = dst
    return background

def put_korean_text(img, text, pos, font_path="C:/Windows/Fonts/malgun.ttf", font_size=28, color=(255,255,255)):
    # OpenCV(BGR) -> PIL(RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(pos, text, font=font, fill=color)

    # PIL(RGB) -> OpenCV(BGR)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


model = HandPoseModel()
try:
    model.load_state_dict(
        torch.load(BASE_DIR / "models" / "saved_weights.pth", map_location="cpu")
    )
    model.eval()
except Exception as e:
    raise RuntimeError(f"紐⑤뜽 濡쒕뱶 ?ㅽ뙣: {e}") from e

jutsu_caps = {}
for jutsu_name, path in video_paths.items():
    cap_obj = cv2.VideoCapture(str(path))
    if not cap_obj.isOpened():
        print("?곸긽 濡쒕뱶 ?ㅽ뙣", path)
    jutsu_caps[jutsu_name] = cap_obj

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
selfie_segmentation = mp_selfie.SelfieSegmentation(model_selection=1)
cap = cv2.VideoCapture(0)

# ?쒗넗由ъ뼹 ?곹깭 愿由?蹂??
current_mission_name = "1. 移섎룄由?
target_sequence = jutsus[current_mission_name]
current_step = 0 # ?꾩옱 紐?踰덉㎏ ?숈옉源뚯? ?깃났?덈뒗吏 異붿쟻
pose_hold_time = 0
active_jutsu_frames = 0
clone_image = None

clone_settings = [
    {"delay":0, "x": 0.2, "y": 0.6, "scale": 0.7},
    {"delay":8, "x": 0.8, "y": 0.6, "scale": 0.7},
    {"delay":16, "x": 0.1, "y": 0.4, "scale": 0.5},
    {"delay":24, "x": 0.9, "y": 0.4, "scale": 0.5},
    {"delay":32, "x": 0.5, "y": 0.2, "scale": 0.4}
]

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

while cap.isOpened():
    success, image = cap.read()
    if not success: break
    image = cv2.flip(image, 1)

    # 1. ?멸났吏???덉륫 ?뚰듃
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    predicted_name = "None"
    hand_center_x, hand_center_y = 0, 0

    if results.multi_hand_landmarks:
        h, w, c = image.shape
        hand_center_x = int(results.multi_hand_landmarks[0].landmark[9].x * w)
        hand_center_y = int(results.multi_hand_landmarks[0].landmark[9].y * h)

        for hl in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hl, mp_hands.HAND_CONNECTIONS)

        coords = get_normalized_coords(results.multi_hand_landmarks)

        if len(coords) == 84:
            x_tensor = torch.tensor([coords], dtype=torch.float32)
            with torch.no_grad():
                pred = model(x_tensor)
                probabilities = torch.nn.functional.softmax(pred, dim=1)[0]
                max_prob, predicted_class = torch.max(probabilities, 0)

                if max_prob.item() > 0.8:
                    predicted_name = class_names[predicted_class.item()]

    # 2. ?쒗넗由ъ뼹 而ㅻ㎤??濡쒖쭅 ?뚰듃
    if active_jutsu_frames  == 0 and current_step < len(target_sequence):
        if predicted_name != "None" and predicted_name == target_sequence[current_step]:
            pose_hold_time += 1
        else:
            pose_hold_time = 0

        if pose_hold_time > 10:
            current_step += 1
            pose_hold_time = 0
            print(f"{predicted_name} ?깃났!")

    # 湲곗닠 諛쒕룞 泥댄겕
    if current_step >= len(target_sequence):
        print(f"{current_mission_name} 諛쒕룞!")
        active_jutsu_frames = 120 # 4珥??곸긽 -> ?곸긽湲몄씠??留욊쾶 蹂寃??꾩슂
        current_step = 0

        if current_mission_name in sounds:
            sounds[current_mission_name].play()

        if current_mission_name == "4. 洹몃┝??遺꾩떊??:
            results_seg = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            mask = results_seg.segmentation_mask > 0.5
            clone_image = np.zeros_like(image)
            clone_image[mask] = image[mask]

        elif current_mission_name in jutsu_caps:
            current_cap = jutsu_caps[current_mission_name]
            if current_cap.isOpened():
                current_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if active_jutsu_frames > 0:
        image = put_korean_text(
    image,
    f"?몃쾿 : {current_mission_name}!!",
    (50, 60),
    font_size=40,
    color=(255, 0, 0)
        )

        if current_mission_name in jutsu_caps:
             current_cap = jutsu_caps[current_mission_name]
             if hand_center_x != 0 and current_cap.isOpened():
                ret, effect_frame = current_cap.read()

                if not ret:
                    current_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, effect_frame = current_cap.read()

                if ret:
                    effect_frame = cv2.resize(effect_frame, (300, 300))
                    image = overlay_video_additive(image, effect_frame, hand_center_x, hand_center_y)

        elif current_mission_name == "4. 洹몃┝??遺꾩떊?? and clone_image is not None:
            elapsed_frames = 120 - active_jutsu_frames
            h, w = image.shape[:2]
            for config in clone_settings:
                if elapsed_frames >= config["delay"]:
                    if elapsed_frames == config["delay"] and "4. Shadow" in sounds:
                        sounds["4. Shadow"].play()

                    image = overlay_clone(image, clone_image, x = w*config["x"], y = h*config["y"], scale = config["scale"])
        active_jutsu_frames -= 1

    # 3. ?붾㈃ UI 洹몃━湲??뚰듃
    cv2.rectangle(image, (0, 400), (640, 480), (0, 0, 0), -1)
    image = put_korean_text(
    image,
    f"Mission: {current_mission_name}",
    (10, 405),
    font_size=28,
    color=(255, 255, 255)
    )

    # 而ㅻ㎤??吏꾪뻾???쒖떆
    display_seq = ""
    for i, pose in enumerate(target_sequence):
        if i < current_step:
            display_seq += f"[{pose.upper()}] -> "
        else:
            display_seq += f"{pose} -> "

    cv2.putText(image, display_seq[:-4], (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ?꾩옱 ?몄떇 以묒씤 ?숈옉 ?쒖떆 (醫뚯륫 ?곷떒)
    cv2.putText(image, f'Current: {predicted_name}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Naruto Tutorial', image)

    # ?ㅻ낫???낅젰?쇰줈 誘몄뀡 蹂寃?
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('1'): current_mission_name, target_sequence, current_step = "1. 移섎룄由?, jutsus["1. 移섎룄由?], 0
    elif key == ord('2'): current_mission_name, target_sequence, current_step = "2. ?섏꽑??, jutsus["2. ?섏꽑??], 0
    elif key == ord('3'): current_mission_name, target_sequence, current_step = "3. ?명솕援ъ쓽 ??, jutsus["3. ?명솕援ъ쓽 ??], 0
    elif key == ord('4'): current_mission_name, target_sequence, current_step = "4. 洹몃┝??遺꾩떊??, jutsus["4. 洹몃┝??遺꾩떊??], 0

cap.release()
for cap_obj in jutsu_caps.values():
    if cap_obj.isOpened():
        cap_obj.release()
pygame.quit()
cv2.destroyAllWindows()
