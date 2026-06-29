import cv2
import mediapipe as mp

IMG_SIZE = 64
MARGIN = 20

def extract_hand(img, hands):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        return None

    h, w, _ = img.shape
    landmarks = result.multi_hand_landmarks[0].landmark

    x_min = int(min(p.x for p in landmarks) * w) - MARGIN
    x_max = int(max(p.x for p in landmarks) * w) + MARGIN
    y_min = int(min(p.y for p in landmarks) * h) - MARGIN
    y_max = int(max(p.y for p in landmarks) * h) + MARGIN

    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(w, x_max), min(h, y_max)

    hand = img[y_min:y_max, x_min:x_max]
    if hand.size == 0:
        return None

    return cv2.resize(hand, (IMG_SIZE, IMG_SIZE))
