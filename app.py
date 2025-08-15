import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import sqlite3
import pandas as pd
import gdown

# ===== KONFIGURASI =====
MODEL_PATH = "model_cnn.keras"
DRIVE_FILE_ID = "16rx9wvXlJB0PlgZkcO-uXB1SPhUOBnlN"
DB_PATH = "riwayat.db"

# ===== FUNGSI DOWNLOAD & LOAD MODEL =====
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Mengunduh model dari Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# ===== INISIALISASI DATABASE =====
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS riwayat (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            prediksi TEXT,
            waktu TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def simpan_riwayat(filename, prediksi):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO riwayat (filename, prediksi) VALUES (?, ?)", (filename, prediksi))
    conn.commit()
    conn.close()

def ambil_riwayat():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM riwayat ORDER BY waktu DESC", conn)
    conn.close()
    return df

# ===== MULAI STREAMLIT APP =====
st.set_page_config(page_title="Deteksi Penyakit Jeruk", layout="wide")
init_db()

model = load_model()
class_names = ["Busuk", "Sehat"]  # sesuaikan dengan model kamu

tab1, tab2, tab3 = st.tabs(["📷 Deteksi", "📜 Riwayat", "📊 Laporan"])

# ===== TAB 1: DETEKSI =====
with tab1:
    st.header("Deteksi Penyakit Jeruk 🍊")

    uploaded_file = st.file_uploader("Unggah gambar jeruk", type=["jpg", "jpeg", "png"])
    if uploaded_file and model is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img_resized = img.resize((128, 128))
        img_array = np.array(img_resized) / 255.0
        reshaped_img = np.expand_dims(img_array, axis=0)

        prediction = model.predict(reshaped_img)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.image(img, caption="Gambar yang diunggah", width=300)
        st.success(f"Prediksi: **{predicted_class}** ({confidence:.2f}%)")

        simpan_riwayat(uploaded_file.name, predicted_class)

# ===== TAB 2: RIWAYAT =====
with tab2:
    st.header("Riwayat Deteksi")
    riwayat_df = ambil_riwayat()
    if not riwayat_df.empty:
        st.dataframe(riwayat_df)
    else:
        st.info("Belum ada riwayat deteksi.")

# ===== TAB 3: LAPORAN =====
with tab3:
    st.header("Laporan Training Model")
    
    if os.path.exists("training_accuracy.png"):
        st.image("training_accuracy.png", caption="Training Accuracy", width=400)
    else:
        st.warning("Gambar training_accuracy.png tidak ditemukan.")

    if os.path.exists("training_loss.png"):
        st.image("training_loss.png", caption="Training Loss", width=400)
    else:
        st.warning("Gambar training_loss.png tidak ditemukan.")
