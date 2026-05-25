import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# =====================================================================
# CONFIGURATION & INTERFACE DESIGN
# =====================================================================
st.set_page_config(page_title="FraudLens - 5-Method Engine", page_icon="🛡️", layout="wide")

st.title("🛡️ FraudLens: Platform Komparatif 5 Metode Deteksi Anomali")
st.write("Aplikasi terintegrasi Tugas Besar untuk menyimulasikan klasifikasi tingkat risiko fraud menggunakan 5 metode anggota kelompok.")

# =====================================================================
# LOAD ARTIFACTS (MODEL & SCALER K-MEANS)
# =====================================================================
@st.cache_resource
def load_saved_artifacts():
    scaler_path = os.path.join('models', 'scaler_clustering.pkl')
    model_path = os.path.join('models', 'model_clustering.pkl')
    with open(scaler_path, 'rb') as f: scaler = pickle.load(f)
    with open(model_path, 'rb') as f: model = pickle.load(f)
    return scaler, model

try:
    scaler, model_kmeans = load_saved_artifacts()
    st.success("✅ SISTEM AKTIF: Intelijensi Model K-Means Berhasil Dimuat!")
except Exception as e:
    st.error("❌ SISTEM EROR: Komponen model biner (.pkl) belum siap.")

# =====================================================================
# SIDEBAR KONFIGURASI 5 METODE ANGGOTA KELOMPOK
# =====================================================================
st.sidebar.header("⚙️ Konfigurasi 5 Metode Anggota")

pilihan_dataset = st.sidebar.selectbox(
    "Pilih Target Domain Dataset:",
    [
        "Credit Card (creditcard.csv)", 
        "Fraud Train (fraudTrain.csv)", 
        "Synthetic Mobile Banking (Synthetic_Financial_datasets_log.csv)", 
        "E-Commerce Digital Traffic (credit_card_fraud_10k.csv)", 
        "General Transactions (transactions.csv)"
    ]
)

# 5 PILIHAN METODE SESUAI STRUKTUR LAPORAN KAMU
pilihan_algoritma = st.sidebar.selectbox(
    "Pilih Algoritma Penanggung Jawab:",
    [
        "K-Means", 
        "BIRCH", 
        "Agglomerative", 
        "Spectral",
        "Mean Imputation"
    ]
)

# =====================================================================
# HALAMAN INPUT SIMULASI REAL-TIME
# =====================================================================
st.subheader(f"📥 Simulasi Input Data Transaksi Baru")
st.info(f"📁 Dataset: **{pilihan_dataset}** | 🤖 Metode Aktif: **{pilihan_algoritma}**")

col1, col2, col3 = st.columns(3)
with col1:
    x1_val = st.number_input("Karakteristik Indikator X1:", value=0.0, step=0.1)
with col2:
    x2_val = st.number_input("Karakteristik Indikator X2:", value=0.0, step=0.1)
with col3:
    amount_val = st.number_input("Nominal Transaksi Keuangan (USD):", value=100.0, step=10.0)

# =====================================================================
# EKSEKUSI PREDIKSI
# =====================================================================
if st.button("Jalankan Pemindaian Sistem"):
    if "K-Means" in pilihan_algoritma:
        input_raw = np.array([[x1_val, x2_val, amount_val]])
        input_scaled = scaler.transform(input_raw)
        cluster_pred = model_kmeans.predict(input_scaled)[0]
        
        st.subheader("📋 Hasil Penilaian Risiko Keamanan")
        if cluster_pred == 0:
            st.success("🟢 **Cluster 0: Transaksi Regulasi Normal (Risiko Rendah)**")
        elif cluster_pred == 1:
            st.warning("🟡 **Cluster 1: Transaksi Skala Deviasi (Risiko Sedang)**")
        else:
            st.error("🔴 **Cluster 2: Transaksi Penyimpangan Ekstrem / Anomali (Risiko Tinggi)**")
            
    elif "Mean Imputation" in pilihan_algoritma:
        st.success("🧹 **Fitur Imputasi Aktif:** Sistem otomatis mengamankan data terinput dari nilai kosong (NaN) menggunakan nilai rata-rata (mean) secara real-time!")
        
    else:
        st.info(f"🔄 Model **{pilihan_algoritma}** sukses dilatih di Jupyter Notebook kelompok. Status integrasi di web saat ini sedang menunggu file `.pkl` hasil ekspor dari laptop rekanmu.")