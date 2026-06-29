import os
import cv2
from tqdm import tqdm
import mediapipe as mp
from clean_utils import extract_hand

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE_DIR, "Letter", "asl_alphabet_train")
DST = os.path.join(BASE_DIR, "Processed_Dataset", "Letter")

os.makedirs(DST, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

for cls in os.listdir(SRC):
    cls_src = os.path.join(SRC, cls)
    cls_dst = os.path.join(DST, cls)
    os.makedirs(cls_dst, exist_ok=True)

    for img_name in tqdm(os.listdir(cls_src), desc=f"Letter {cls}"):
        img = cv2.imread(os.path.join(cls_src, img_name))
        if img is None:
            continue

        hand = extract_hand(img, hands)
        if hand is not None:
            cv2.imwrite(os.path.join(cls_dst, img_name), hand)

hands.close()
print("✅ Letters dataset cleaned")
