import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ==========================================
# SETUP HALAMAN
# ==========================================
st.set_page_config(page_title="Deteksi Kantuk + Delay Cerdas", layout="wide")

# ==========================================
# LOAD MODEL TFLITE
# ==========================================
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="deteksi-kantuk.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ==========================================
# CASCADE
# ==========================================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# ==========================================
# ALARM
# ==========================================
def play_alarm():
    st.components.v1.html(
        """
        <audio autoplay>
        <source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3">
        </audio>
        """,
        height=0,
    )

# ==========================================
# PREDIKSI MATA
# ==========================================
def predict_eye(eye_img):
    img = cv2.resize(eye_img, (64, 64))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])[0][0]

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("🔧 Pengaturan")
alarm_threshold = st.sidebar.slider(
    "Waktu Tunggu (Detik) sebelum Alarm", 1.0, 10.0, 3.0, 0.5
)

st.title("👁️ Deteksi Kantuk Pengemudi (WebCam Browser)")
st.info(
    f"Alarm berbunyi jika mata tertutup lebih dari **{alarm_threshold} detik**"
)

status_text = st.empty()
kpi_text = st.empty()
timer_text = st.empty()

# ==========================================
# VIDEO PROCESSOR (INTI PERBAIKAN)
# ==========================================
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.start_time_closed = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        status = "TIDAK_TAHU"
        score_display = 0
        duration_closed = 0

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            probs = []

            for (ex, ey, ew, eh) in eyes:
                eye_img = roi_color[ey:ey+eh, ex:ex+ew]
                pred = predict_eye(eye_img)
                probs.append(pred)

                color = (0,255,0) if pred > 0.5 else (0,0,255)
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), color, 2)

            if probs:
                avg = sum(probs) / len(probs)
                score_display = int(avg * 100)

                if avg < 0.5:
                    status = "TERTUTUP"
                else:
                    status = "TERBUKA"

            cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,0), 2)

        # TIMER PINTAR
        if status == "TERTUTUP":
            if self.start_time_closed is None:
                self.start_time_closed = time.time()
            duration_closed = time.time() - self.start_time_closed

        elif status == "TERBUKA":
            self.start_time_closed = None

        else:
            if self.start_time_closed:
                duration_closed = time.time() - self.start_time_closed

        # UI UPDATE
        if duration_closed > alarm_threshold:
            status_text.error("⚠️ BAHAYA: NGANTUK!")
            play_alarm()
        elif status == "TERTUTUP":
            status_text.warning("Mata Tertutup...")
        elif status == "TERBUKA":
            status_text.success("✅ AMAN")
        else:
            status_text.info("Mencari Wajah...")

        kpi_text.metric("Skor Mata", f"{score_display}%")
        timer_text.metric("Timer", f"{duration_closed:.2f}s")

        return img

# ==========================================
# START CAMERA
# ==========================================
webrtc_streamer(
    key="kantuk",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
