import streamlit as st
import numpy as np
import tensorflow as tf
import time
from PIL import Image

# ==========================================
# SETUP HALAMAN
# ==========================================
st.set_page_config(page_title="Deteksi Kantuk", layout="wide")
st.title("👁️ Deteksi Kantuk Pengemudi (Cloud Version)")

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
# FUNGSI ALARM
# ==========================================
def play_alarm():
    st.components.v1.html("""
    <audio autoplay>
        <source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3" type="audio/mp3">
    </audio>
    """, height=0)

# ==========================================
# FUNGSI PREDIKSI
# ==========================================
def predict_eye(img):
    img = img.resize((64, 64))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    return output[0][0]

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("🔧 Pengaturan")
alarm_threshold = st.sidebar.slider(
    "Waktu Tunggu Alarm (detik)",
    min_value=1.0,
    max_value=10.0,
    value=3.0,
    step=0.5
)

# ==========================================
# UI STATUS
# ==========================================
col1, col2, col3 = st.columns(3)

status_box = col1.empty()
score_box = col2.empty()
timer_box = col3.empty()

# ==========================================
# STATE TIMER
# ==========================================
if "start_time" not in st.session_state:
    st.session_state.start_time = None

# ==========================================
# INPUT KAMERA (CLOUD)
# ==========================================
image_file = st.camera_input("📸 Ambil gambar mata/wajah")

if image_file is not None:
    image = Image.open(image_file).convert("RGB")
    st.image(image, caption="Input", width=400)

    score = predict_eye(image)
    score_percent = int(score * 100)

    # ==============================
    # LOGIKA STATUS
    # ==============================
    status = "TIDAK_TAHU"
    duration = 0

    if score < 0.5:
        status = "TERTUTUP"
        if st.session_state.start_time is None:
            st.session_state.start_time = time.time()
        duration = time.time() - st.session_state.start_time
    else:
        status = "TERBUKA"
        st.session_state.start_time = None

    # ==============================
    # TAMPILKAN STATUS
    # ==============================
    if status == "TERTUTUP":
        status_box.warning("😴 Mata Tertutup")
    else:
        status_box.success("✅ Mata Terbuka")

    score_box.metric("Skor Mata Terbuka", f"{score_percent}%")

    if duration > 0:
        timer_box.metric("Timer", f"{duration:.2f} s")
    else:
        timer_box.metric("Timer", "0.00 s")

    # ==============================
    # ALARM
    # ==============================
    if duration >= alarm_threshold:
        st.error("🚨 BAHAYA! ANDA MENGANTUK!")
        play_alarm()

else:
    st.info("Silakan ambil gambar menggunakan kamera")
