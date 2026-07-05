# Sign Language Recognition System — RAG Data Pipeline

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-brightgreen.svg)](https://streamlit.io/)

---

## 🚀 Project Overview
A **real-time Sign Language Recognition System** that detects hand gestures and translates them into **letters, numbers, and gestures**.  
Works on both images and webcam input.  

- **Hand detection**: MediaPipe  
- **Classification**: TensorFlow CNN  
- **Real-time recognition**: Streamlit or Python script  

---

## ✨ Features
- Detects **letters, numbers, and gestures** from hand movements.
- Real-time recognition with webcam.
- Supports **image input** for testing.
- Lightweight and easy to extend with new gestures.


----------------------------------------------------
INSTALLATION STEPS
----------------------------------------------------
Step 1: Clone the repository
Command:
git clone <your-repo-url>
cd Sign_Language_Recognition

Step 2: Create virtual environment
Command:
python -m venv venv

Step 3: Activate virtual environment
- Windows:
venv\Scripts\activate
- Mac/Linux:
source venv/bin/activate

Step 4: Install dependencies
Command:
pip install -r requirements.txt

----------------------------------------------------
USAGE
----------------------------------------------------
Run the Streamlit App:
Command:
streamlit run app.py

Run via Python Script:
Command:
python main.py

- Place images in the 'data/' folder for recognition
- Processed dataset is saved in 'Processed_Dataset/'

----------------------------------------------------
PROJECT STRUCTURE
----------------------------------------------------
Sign_Language_Recognition/
│
├─ Letter/                 (Raw letter images)
├─ Number/                 (Raw number images)
├─ Gesture/                (Raw gesture images)
├─ Processed_Dataset/      (Cleaned dataset using MediaPipe)
├─ models/                 (Saved trained models .keras, .h5)
├─ app.py                  (Streamlit app)
├─ main.py                 (Main recognition script)
├─ requirements.txt        (Python dependencies)
├─ .gitignore
└─ README.txt

----------------------------------------------------
AUTHOR
----------------------------------------------------
Tanishq Saini
| CDAC Bangalore | DBDA | Aug 2025 - Feb 2026

----------------------------------------------------
LICENSE
----------------------------------------------------
This project is licensed under the MIT License
