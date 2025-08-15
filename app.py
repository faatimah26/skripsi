import streamlit as st
st.set_page_config(page_title="Deteksi Jeruk", layout="centered")  

import tensorflow as tf
import numpy as np
from PIL import Image
import sqlite3
import os
import io
from datetime import datetime
from fpdf import FPDF
from database import init_db, simpan_riwayat, ambil_riwayat, hapus_riwayat
import requests
from PIL import Image

# ==========================
# Download & Load Model
# ==========================

MODEL_PATH = "model_cnn.keras"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = "https://drive.google.com/uc?export=download&id=16rx9wvXlJB0PlgZkcO-uXB1SPhUOBnlN"
        r = requests.get(url)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
    return tf.keras.models.load_model(MODEL_PATH)

# Load model
model = load_model()


# ==========================
# Inisialisasi
# ==========================
st.title("Sistem Deteksi Penyakit Buah Jeruk")
init_db()

# Navigasi Sidebar
st.sidebar.title("Navigasi")
if "selected_tab" not in st.session_state:
    st.session_state.selected_tab = "Deteksi"

if st.sidebar.button("Deteksi"):
    st.session_state.selected_tab = "Deteksi"
if st.sidebar.button("Riwayat Deteksi"):
    st.session_state.selected_tab = "Riwayat Deteksi"
if st.sidebar.button("Laporan"):
    st.session_state.selected_tab = "Laporan"

# ==========================
# Fungsi Buat PDF
# ==========================
def buat_laporan_pdf(result, deskripsi, image):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    waktu = datetime.now().strftime("%d-%m-%Y %H:%M")
    pdf.cell(200, 10, txt="Hasil Deteksi Penyakit Jeruk", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Waktu: {waktu}", ln=True)
    pdf.cell(200, 10, txt=f"Hasil Deteksi: {result.upper()}", ln=True)
    pdf.multi_cell(0, 10, txt=f"Penjelasan: {deskripsi}")
    pdf.ln(5)

    temp_img_path = "temp_image.png"
    image.save(temp_img_path)
    pdf.image(temp_img_path, x=10, y=None, w=100)
    pdf.ln(15)
    pdf.cell(200, 10, txt="Penulis : Fatimah Azahrah", ln=True, align='C')

    pdf_output = pdf.output(dest='S').encode('latin1')
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)

    return pdf_output

# ==========================
# Tab Deteksi
# ==========================
if st.session_state.selected_tab == "Deteksi":
    st.subheader("Upload Gambar Buah Jeruk")
    uploaded_file = st.file_uploader("Pilih Gambar (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Gambar yang Diunggah', width=300)

        if st.button("Deteksi Buah Jeruk"):
            with st.spinner("Memproses..."):
                img = image.resize((256, 256))
                img_array = np.asarray(img) / 255.0
                reshaped_img = np.expand_dims(img_array, axis=0)

                prediction = model.predict(reshaped_img)
                result = class_names[np.argmax(prediction)]

                # Deskripsi
                if result == "blackspot":
                    deskripsi = "Black Spot adalah penyakit akibat jamur Guignardia Citricarpa..."
                elif result == "canker":
                    deskripsi = "Canker adalah penyakit akibat bakteri X. axonopodis pv. citri..."
                elif result == "fresh":
                    deskripsi = "Buah jeruk dalam kondisi segar dan sehat."
                elif result == "greening":
                    deskripsi = "Greening adalah penyakit serius akibat bakteri Candidatus Liberibacter spp."
                else:
                    deskripsi = "Kategori tidak dikenali."

                st.success(f"Hasil Deteksi: {result.upper()}")
                st.markdown(deskripsi)

                # Simpan ke database
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                simpan_riwayat(uploaded_file.name, result, img_byte_arr.getvalue())

                # Download PDF
                laporan_pdf = buat_laporan_pdf(result, deskripsi, image)
                st.download_button(
                    label="Download Hasil Deteksi (PDF)",
                    data=laporan_pdf,
                    file_name="hasil_deteksi.pdf",
                    mime="application/pdf"
                )

# ==========================
# Tab Riwayat
# ==========================
elif st.session_state.selected_tab == "Riwayat Deteksi":
    st.subheader("Riwayat Gambar Deteksi")
    data_riwayat = ambil_riwayat()

    if data_riwayat:
        for id_data, nama_file, hasil, gambar_blob in data_riwayat:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"**{nama_file}** — Hasil: **{hasil}**")
                img = Image.open(io.BytesIO(gambar_blob))
                st.image(img, width=250)
            with col2:
                if st.button("Hapus", key=f"hapus_{id_data}"):
                    hapus_riwayat(id_data)
                    st.success(f"Riwayat '{nama_file}' berhasil dihapus.")
                    st.experimental_rerun()
            st.markdown("---")
    else:
        st.info("Belum ada riwayat deteksi.")

# ==========================
# Tab Laporan
# ==========================
elif st.session_state.selected_tab == "Laporan":
    st.subheader("Laporan Hasil Pelatihan Model CNN")
    st.markdown(
        "Berikut hasil pelatihan model berupa grafik akurasi dan loss..."
    )

    if os.path.exists("accuracy.png"):
        st.image("accuracy.png", caption="Training Accuracy", width=400)
    if os.path.exists("loss.png"):
        st.image("loss.png", caption="Training Loss", width=400)
    if os.path.exists("reportt.png"):
        st.image("reportt.png", caption="Classification Report", width=400)


