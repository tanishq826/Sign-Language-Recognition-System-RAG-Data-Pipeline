import os
import cv2
import json
import streamlit as st
import mediapipe as mp
import tensorflow as tf

from app_utils import extract_hand_roi, predict_sign

# ----------------------------
# Streamlit setup
# ----------------------------
st.set_page_config(page_title="Sign Language Recognition", layout="wide")
st.title("🤟 Sign Language Recognition System")

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

LETTER_INDICES = list(range(0, 26))
NUMBER_INDICES = list(range(26, 46))
GESTURE_INDICES = list(range(46, 71))

mode = st.sidebar.radio("Select Mode", ["letters", "numbers", "gestures"])

MODE_MAP = {
    "letters": LETTER_INDICES,
    "numbers": NUMBER_INDICES,
    "gestures": GESTURE_INDICES
}

# ----------------------------
# MediaPipe
# ----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

frame_window = st.image([])

cap = cv2.VideoCapture(0)
st.info("Press STOP to end webcam")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    prediction = ""

    if result.multi_hand_landmarks:
        hand_lm = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

        hand_img = extract_hand_roi(frame, hand_lm)
        if hand_img is not None:
            label, conf = predict_sign(
                model, hand_img, class_names, MODE_MAP[mode]
            )
            if label:
                prediction = label

    cv2.putText(frame, f"Mode: {mode.upper()}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if prediction:
        cv2.putText(frame, f"Prediction: {prediction}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()
hands.close()
