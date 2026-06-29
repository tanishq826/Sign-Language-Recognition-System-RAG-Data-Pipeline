import os
import cv2
import mediapipe as mp
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from clean_utils import extract_hand

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE_DIR, "Gesture")
DST = os.path.join(BASE_DIR, "Processed_Dataset", "Gesture")

os.makedirs(DST, exist_ok=True)

def process_image(args):
    img_path, dst = args
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        img = cv2.imread(img_path)
        if img is None:
            return
        hand = extract_hand(img, hands)
        if hand is not None:
            cv2.imwrite(os.path.join(dst, os.path.basename(img_path)), hand)

tasks = []
for cls in os.listdir(SRC):
    cls_src = os.path.join(SRC, cls)
    cls_dst = os.path.join(DST, cls)
    os.makedirs(cls_dst, exist_ok=True)

    for img in os.listdir(cls_src):
        tasks.append((os.path.join(cls_src, img), cls_dst))

with ThreadPoolExecutor() as executor:
    list(tqdm(executor.map(process_image, tasks), total=len(tasks)))

print("✅ Gestures dataset cleaned")
