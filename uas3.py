import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os  # PERBAIKAN 1: Menambahkan import os

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Klasifikasi Sampah AI", page_icon="♻️", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #2E7D32; color: white; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. LOAD MODEL (CACHE)
# ==========================================
@st.cache_resource
def load_my_model():
    # PERBAIKAN 2: Menggunakan __file__ (double underscore)
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_path = os.getcwd()
        
    model_path = os.path.join(base_path, 'best_waste_model.h5')
    
    if os.path.exists(model_path):
        try:
           return tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            st.error(f"Model ditemukan tapi rusak: {e}")
            return None
    else:
        st.error(f"File {model_path} tidak ditemukan!")
        st.write("File di folder saat ini:", os.listdir(base_path))
        return None

# PERBAIKAN 3: Panggil fungsi load model di sini agar variabel 'model' terdefinisi
model = load_my_model()

# ==========================================
# 3. HEADER & DESKRIPSI (Maks 350 Karakter)
# ==========================================
st.title("Klasifikasi Sampah Yang Bisa Di Daur Ulang (Recyclable) Dan Tidak Bisa Di Daur Ulang (Non-Recyclable)")
st.write("Sistem ini merupakan solusi cerdas klasifikasi sampah otomatis berbasis Deep Learning dan citra digital. Dengan menganalisis tekstur serta bentuk objek melalui kamera, sistem secara akurat membedakan kategori Recyclable dan Non-Recyclable. Inovasi ini bertujuan mempercepat pemilahan limbah dan mendukung keberlanjutan lingkungan secara digital.")

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📂 Informasi Dataset", "🔍 Klasifikasi"])

# --- TAB 1: DASHBOARD ---
with tab1:
    col_a, col_b = st.columns([1.2, 0.8])
    with col_a:
        st.subheader("💡 Mengapa Kita Harus Memilah Sampah?")
        st.write("""
        Produksi sampah global diperkirakan akan meningkat hingga **3,4 miliar ton** pada tahun 2050. 
        Tanpa sistem pemilahan yang baik, sebagian besar sampah ini akan berakhir mencemari lautan dan tanah.
        Memilah sampah dari sumbernya dapat mengurangi beban TPA hingga **80%**.
        """)
        st.info("**Fakta Utama:** Plastik membutuhkan waktu hingga **450 tahun** untuk terurai secara alami.")
    with col_b:
        st.image("https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhaKxiRDPyTgzrB1Kgof3gTxPCA3KlB6id79GIzsSti2nqTAXuB9Ja5YMwyP5QLJCHLEolRotiFtzaFHnwmaiCu6PrWf5r5GXYwbDrlmhyphenhyphenvdk_cgX3VT-jcMiFIB7Hv0quZSM_mTEeJG4o/s1600/foto+ilustrasi.jpg", 
                 caption="Tumpukan sampah yang mencemari ekosistem.")

# --- TAB 2: DATASET ---
with tab2:
    st.header("Informasi Dataset")
    st.write("Dataset berasal dari [Kaggle - Garbage Classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)")
    
    data = {
        'Kategori': ['paper', 'plastic', 'metal', 'cardboard', 'brown-glass', 'green-glass', 'white-glass',
                    'biological', 'clothes', 'shoes', 'trash', 'battery'],
        'Jumlah Gambar': [1050, 865, 769, 891, 607, 629, 775, 985, 5325, 1977, 697, 945]
    }
    df_eda = pd.DataFrame(data)

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    bars = sns.barplot(x='Kategori', y='Jumlah Gambar', data=df_eda, color='#87CEEB', ax=ax1)
    plt.xticks(rotation=45)
    for bar in bars.containers[0]:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, str(int(bar.get_height())), ha='center')
    st.pyplot(fig1)

# --- TAB 3: KLASIFIKASI ---
with tab3:
    st.header("Analisis Gambar Sampah")
    metode = st.radio("Pilih Cara Unggah:", ["Upload File", "Gunakan Kamera"], horizontal=True)
    
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "png", "jpeg"]) if metode == "Upload File" else st.camera_input("Ambil foto")

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang akan dianalisis", width=400)
        
        if st.button("Mulai Analisis"):
            if model is not None:
                with st.spinner('Sedang memproses...'):
                    # Preprocessing
                    img_res = img.resize((224, 224))
                    img_array = image.img_to_array(img_res)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_input(img_array)
                    
                    # Prediksi
                    prediction = model.predict(img_array)
                    score = prediction[0][0] 
                    confidence = max(score, 1-score) * 100
                    
                    st.divider()
                    
                    if score > 0.5:
                        st.success(f"### HASIL: RECYCLABLE (BISA DIDAUR ULANG)")
                        st.metric("Tingkat Keyakinan", f"{confidence:.2f}%")
                        st.info("💡 *Saran:* Kumpulkan di bank sampah atau buat kerajinan tangan kreatif.")
                    else:
                        st.error(f"### HASIL: NON-RECYCLABLE (TIDAK BISA DIDAUR ULANG)")
                        st.metric("Tingkat Keyakinan", f"{confidence:.2f}%")
                        st.info("💡 *Saran:* Olah sampah organik menjadi kompos atau buang limbah B3 ke tempat khusus.")
            else:
                st.error("Model tidak ditemukan. Pastikan file 'best_waste_model.h5' ada di folder yang sama di GitHub.")

