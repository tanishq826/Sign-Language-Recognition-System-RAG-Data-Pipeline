import os
import cv2
import json
import mediapipe as mp
import tensorflow as tf

from app_utils import extract_hand_roi, predict_sign

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "sign_language_71class_cnn.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")

# ----------------------------
# Load model & classes
# ----------------------------
model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_NAMES_PATH) as f:
    class_names = json.load(f)

# Indices
LETTER_INDICES = list(range(0, 26))
NUMBER_INDICES = list(range(26, 46))
GESTURE_INDICES = list(range(46, 71))

MODE_MAP = {
    "letters": LETTER_INDICES,
    "numbers": NUMBER_INDICES,
    "gestures": GESTURE_INDICES
}

current_mode = "letters"

# ----------------------------
# MediaPipe
# ----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# ----------------------------
# Webcam
# ----------------------------
cap = cv2.VideoCapture(0)
print("Press L / N / G to switch mode, Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    prediction = ""

    if result.multi_hand_landmarks:
        hand_lm = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

        hand_img = extract_hand_roi(frame, hand_lm)
        if hand_img is not None:
            label, conf = predict_sign(
                model, hand_img, class_names, MODE_MAP[current_mode]
            )
            if label:
                prediction = label

    cv2.putText(frame, f"Mode: {current_mode.upper()}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if prediction:
        cv2.putText(frame, f"Prediction: {prediction}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Sign Language Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('l'):
        current_mode = "letters"
    elif key == ord('n'):
        current_mode = "numbers"
    elif key == ord('g'):
        current_mode = "gestures"
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
