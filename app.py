import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import time
import tempfile

# ==========================================
# SETUP HALAMAN
# ==========================================
st.set_page_config(page_title="Deteksi Kantuk", layout="wide")

# ==========================================
# LOAD MODEL TFLITE
# ==========================================
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="deteksi-kantuk.tflite")
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except:
    st.error("❌ Model deteksi-kantuk.tflite tidak ditemukan")
    st.stop()

# ==========================================
# LOAD CASCADE
# ==========================================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

# ==========================================
# ALARM
# ==========================================
def play_alarm():
    st.components.v1.html("""
    <audio autoplay>
        <source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3">
    </audio>
    """, height=0)

# ==========================================
# PREDIKSI MATA
# ==========================================
def predict_eye(eye_img):
    img = cv2.resize(eye_img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img.astype(np.float32), axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0][0]

# ==========================================
# PROSES 1 FRAME
# ==========================================
def process_frame(frame, alarm_threshold, start_time_closed):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    status = "TIDAK_TAHU"
    score_display = 0

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = rgb[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        probs = []
        for (ex, ey, ew, eh) in eyes:
            eye_img = roi_color[ey:ey+eh, ex:ex+ew]
            pred = predict_eye(eye_img)
            probs.append(pred)

            color = (0,255,0) if pred > 0.5 else (255,0,0)
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), color, 2)

        if probs:
            avg = sum(probs) / len(probs)
            score_display = int(avg * 100)
            status = "TERBUKA" if avg > 0.5 else "TERTUTUP"

        cv2.rectangle(rgb, (x,y), (x+w,y+h), (0,255,0), 2)

    # TIMER
    duration = 0
    if status == "TERTUTUP":
        if start_time_closed is None:
            start_time_closed = time.time()
        duration = time.time() - start_time_closed
    elif status == "TERBUKA":
        start_time_closed = None

    alarm = duration > alarm_threshold
    return rgb, status, score_display, duration, start_time_closed, alarm

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("⚙️ Pengaturan")

mode = st.sidebar.radio(
    "Pilih Mode Input",
    ["📷 Kamera Realtime", "🖼️ Upload Foto", "🎥 Upload Video"]
)

alarm_threshold = st.sidebar.slider(
    "Alarm setelah (detik)", 1.0, 10.0, 3.0, 0.5
)

# ==========================================
# UI
# ==========================================
st.title("👁️ Deteksi Kantuk Pengemudi")

cam_status = st.empty()
status_text = st.empty()
kpi_text = st.empty()
timer_text = st.empty()
frame_window = st.image([])

# ==========================================
# MODE 1: KAMERA REALTIME
# ==========================================
if mode == "📷 Kamera Realtime":
    run = st.checkbox("Buka Kamera")

    if run:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            cam_status.error("❌ Kamera TIDAK bisa dibuka")
            st.stop()
        else:
            cam_status.success("✅ Kamera berhasil dibuka")

        start_time_closed = None

        while run:
            ret, frame = cap.read()
            if not ret:
                cam_status.error("❌ Gagal membaca frame kamera")
                break

            frame = cv2.flip(frame, 1)

            frame, status, score, duration, start_time_closed, alarm = process_frame(
                frame, alarm_threshold, start_time_closed
            )

            if alarm:
                play_alarm()
                status_text.error("⚠️ BAHAYA NGANTUK")
            else:
                status_text.success(status)

            frame_window.image(frame)
            kpi_text.metric("Skor Mata", f"{score}%")
            timer_text.metric("Timer", f"{duration:.2f}s")

        cap.release()
    else:
        cam_status.info("📷 Kamera belum dibuka")

# ==========================================
# MODE 2: FOTO
# ==========================================
elif mode == "🖼️ Upload Foto":
    uploaded = st.file_uploader("Upload Foto", type=["jpg","png","jpeg"])

    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        frame, status, score, duration, _, _ = process_frame(
            frame, alarm_threshold, None
        )

        frame_window.image(frame)
        status_text.success(status)
        kpi_text.metric("Skor Mata", f"{score}%")
        timer_text.metric("Timer", f"{duration:.2f}s")

# ==========================================
# MODE 3: VIDEO
# ==========================================
elif mode == "🎥 Upload Video":
    uploaded = st.file_uploader("Upload Video", type=["mp4","avi","mov"])

    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded.read())

        cap = cv2.VideoCapture(tfile.name)

        if not cap.isOpened():
            st.error("❌ Video tidak bisa dibuka")
            st.stop()
        else:
            st.success("🎥 Video berhasil dimuat")

        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = 1 / fps if fps > 0 else 0.03

        start_time_closed = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.info("✅ Video selesai")
                break

            frame, status, score, duration, start_time_closed, alarm = process_frame(
                frame, alarm_threshold, start_time_closed
            )

            if alarm:
                play_alarm()
                status_text.error("⚠️ BAHAYA NGANTUK")
            else:
                status_text.success(status)

            frame_window.image(frame)
            kpi_text.metric("Skor Mata", f"{score}%")
            timer_text.metric("Timer", f"{duration:.2f}s")

            time.sleep(delay)  # 🔑 bikin video jalan

        cap.release()
