import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(BASE_DIR, "Processed_Dataset")
IMG_SIZE = 64

def load_dataset(path, numeric=False):
    classes = sorted(os.listdir(path), key=lambda x: int(x) if numeric else x)
    X, y = [], []

    for idx, cls in enumerate(classes):
        cls_path = os.path.join(path, cls)
        for img in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img)
            im = cv2.imread(img_path)
            if im is None:
                continue
            im = cv2.resize(im, (IMG_SIZE, IMG_SIZE)) / 255.0
            X.append(im)
            y.append(idx)

    return np.array(X), to_categorical(y), classes

Xl, yl, cl = load_dataset(os.path.join(DATA_ROOT, "Letter"))
Xn, yn, cn = load_dataset(os.path.join(DATA_ROOT, "Number"), numeric=True)
Xg, yg, cg = load_dataset(os.path.join(DATA_ROOT, "Gesture"))

X = np.concatenate([Xl, Xn, Xg])
y = np.concatenate([
    yl,
    to_categorical(np.argmax(yn, 1) + len(cl), len(cl) + len(cn) + len(cg)),
    to_categorical(np.argmax(yg, 1) + len(cl) + len(cn), len(cl) + len(cn) + len(cg))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential([
    Conv2D(32, 3, activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(),
    Conv2D(64, 3, activation='relu'),
    MaxPooling2D(),
    Conv2D(128, 3, activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y.shape[1], activation='softmax')
])

model.compile(Adam(0.001), 'categorical_crossentropy', ['accuracy'])
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
