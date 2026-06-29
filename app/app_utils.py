import cv2
import numpy as np

IMG_SIZE = 64
CONFIDENCE_THRESHOLD = 0.6

def extract_hand_roi(frame, hand_landmarks):
    h, w, _ = frame.shape
    x_min, y_min = w, h
    x_max = y_max = 0

    for lm in hand_landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x)
        y_max = max(y_max, y)

    hand_w = x_max - x_min
    hand_h = y_max - y_min
    pad = int(0.4 * max(hand_w, hand_h))

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    half = max(hand_w, hand_h) // 2 + pad

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)

    hand = frame[y1:y2, x1:x2]
    if hand.size == 0:
        return None

    hand = cv2.cvtColor(hand, cv2.COLOR_BGR2RGB)
    hand = cv2.resize(hand, (IMG_SIZE, IMG_SIZE))
    hand = hand.reshape(1, IMG_SIZE, IMG_SIZE, 3) / 255.0
    return hand


def predict_sign(model, hand_img, class_names, valid_indices):
    probs = model.predict(hand_img, verbose=0)
    idx = np.argmax(probs)
    conf = probs[0][idx]

    if idx in valid_indices and conf > CONFIDENCE_THRESHOLD:
        return class_names[idx], conf

    return None, None
