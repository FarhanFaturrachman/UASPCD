import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

st.set_page_config(page_title="Deteksi Kantuk Realtime", layout="centered")

# ======================
# LOAD MODEL TFLITE
# ======================
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="deteksi-kantuk.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ======================
# DETEKSI WAJAH & MATA
# ======================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# ======================
# FUNGSI PREDIKSI
# ======================
def predict_eye(eye_img):
    img = cv2.resize(eye_img, (64, 64))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0][0]

# ======================
# VIDEO PROCESSOR
# ======================
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        status = "AMAN"
        color = (0, 255, 0)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            scores = []

            for (ex, ey, ew, eh) in eyes:
                eye_img = roi_color[ey:ey+eh, ex:ex+ew]
                pred = predict_eye(eye_img)
                scores.append(pred)

                ec = (0,255,0) if pred > 0.5 else (0,0,255)
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), ec, 2)

            if scores and np.mean(scores) < 0.5:
                status = "NGANTUK"
                color = (0, 0, 255)

            cv2.rectangle(img, (x,y), (x+w,y+h), color, 3)

        cv2.putText(
            img, status, (20,40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ======================
# UI
# ======================
st.title("👁️ Deteksi Kantuk Realtime (WebRTC)")
st.info("Gunakan kamera browser. Model berjalan realtime di server.")

webrtc_streamer(
    key="kantuk",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
