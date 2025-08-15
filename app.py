import streamlit as st
st.set_page_config(page_title="Deteksi Penyakit Jeruk", layout="wide")

import os
import io
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from fpdf import FPDF
import gdown

# ==========================
# KONFIGURASI
# ==========================
MODEL_PATH = "model_cnn.keras"
DRIVE_FILE_ID = "16rx9wvXlJB0PlgZkcO-uXB1SPhUOBnlN"
DB_PATH = "riwayat.db"
CLASS_NAMES = ["blackspot", "canker", "fresh", "greening"]  # ≤— sesuaikan urutan output model
TARGET_SIZE = (256, 256)  # ≤— sesuaikan dengan input size saat training

# ==========================
# UTIL DB
# ==========================
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

# ==========================
# LOAD MODEL (CACHED)
# ==========================
# LOAD MODEL (CACHED)
# ==========================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Mengunduh model dari Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

# Inisialisasi
init_db()

# Load model
try:
    model = load_model()
    _model_ok = True
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    model = None
    _model_ok = False


# ==========================
# UI
# ==========================
tab1, tab2, tab3 = st.tabs(["📷 Deteksi", "📜 Riwayat", "📊 Laporan"])

# -------- TAB 1: DETEKSI --------
with tab1:
    st.header("Deteksi Penyakit Jeruk 🍊")

    uploaded_file = st.file_uploader("Unggah gambar jeruk", type=["jpg", "jpeg", "png"])
    deteksi_btn = st.button("Deteksi", disabled=(not _model_ok or uploaded_file is None))

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Gambar yang diunggah", width=320)

    if deteksi_btn:
        try:
            # Preprocess
            img_resized = img.resize(TARGET_SIZE)
            img_array = np.asarray(img_resized, dtype=np.float32) / 255.0
            batch = np.expand_dims(img_array, axis=0)

            # Predict
            preds = model.predict(batch)
            idx = int(np.argmax(preds, axis=1)[0])
            predicted_class = CLASS_NAMES[idx]
            confidence = float(np.max(preds)) * 100.0

            st.success(f"Prediksi: **{predicted_class}** ({confidence:.2f}%)")

            # Simpan riwayat
            simpan_riwayat(uploaded_file.name, predicted_class)

        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")
            st.stop()

# -------- TAB 2: RIWAYAT --------
with tab2:
    st.header("Riwayat Deteksi")
    df = ambil_riwayat()
    if df.empty:
        st.info("Belum ada riwayat deteksi.")
    else:
        st.dataframe(df, use_container_width=True)

# -------- TAB 3: LAPORAN --------
with tab3:
    st.header("Laporan Training Model")
    acc_path = "training_accuracy.png"
    loss_path = "training_loss.png"

    if os.path.exists(acc_path):
        st.image(acc_path, caption="Training Accuracy", width=400)
    else:
        st.warning(f"Gambar {acc_path} tidak ditemukan.")

    if os.path.exists(loss_path):
        st.image(loss_path, caption="Training Loss", width=400)
    else:
        st.warning(f"Gambar {loss_path} tidak ditemukan.")

