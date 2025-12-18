import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Klasifikasi Sampah Yang Bisa Di Daur Ulang (Recyclable) Dan Tidak Bisa Di Daur Ulang (Non-Recyclable)", page_icon="♻️", layout="wide")

# Custom CSS untuk mempercantik tampilan
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #2E7D32; color: white; }
    .result-box { padding: 20px; border-radius: 10px; margin-top: 20px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. LOAD MODEL (CACHE)
# ==========================================
@st.cache_resource
def load_my_model():
    # Mendapatkan path absolut folder tempat file .py berada
    base_path = os.path.dirname(_file_)
    model_path = os.path.join(base_path, 'best_waste_model.h5') # Pastikan nama file sesuai
    
    if os.path.exists(model_path):
        try:
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            st.error(f"Model ditemukan tapi rusak: {e}")
            return None
    else:
        st.error("File model.h5 tidak ditemukan!")
        # Menampilkan daftar file untuk memantau apa yang di-upload ke GitHub
        st.write("File di folder aplikasi:", os.listdir(base_path))
        return None

# ==========================================
# 3. NAVIGASI UTAMA (TABS) - MENGGANTIKAN SIDEBAR
# ==========================================
st.title("Klasifikasi Sampah Yang Bisa Di Daur Ulang (Recyclable) Dan Tidak Bisa Di Daur Ulang (Non-Recyclable)")
st.write("Solusi cerdas pemilahan sampah otomatis: Mengidentifikasi potensi daur ulang melalui pemrosesan citra digital")

tab1, tab2, tab3 = st.tabs([" Dashboard", " Informasi dataset", " Klasifikasi "])

# --- TAB 1: DASHBOARD ---
with tab1:
    
    # Grid Utama: Pengantar & Gambar
    col_a, col_b = st.columns([1.2, 0.8])
    
    with col_a:
        st.subheader("💡 Mengapa Kita Harus Memilah Sampah?")
        st.write("""
        Produksi sampah global diperkirakan akan meningkat hingga **3,4 miliar ton** pada tahun 2050. 
        Tanpa sistem pemilahan yang baik, sebagian besar sampah ini akan berakhir mencemari lautan dan tanah.
        Memilah sampah dari sumbernya dapat mengurangi beban TPA hingga **80%** dan meningkatkan efisiensi daur ulang.
        """)
        
        st.info("""
        **Fakta Utama:** Plastik membutuhkan waktu hingga **450 tahun** untuk terurai secara alami. 
        """)
    
    with col_b:
        st.image("https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhaKxiRDPyTgzrB1Kgof3gTxPCA3KlB6id79GIzsSti2nqTAXuB9Ja5YMwyP5QLJCHLEolRotiFtzaFHnwmaiCu6PrWf5r5GXYwbDrlmhyphenhyphenvdk_cgX3VT-jcMiFIB7Hv0quZSM_mTEeJG4o/s1600/foto+ilustrasi.jpg", 
                 caption="Tumpukan sampah yang tidak terkelola dengan baik mencemari ekosistem.")

    st.divider()

    # Bagian Fakta Mendalam (Material Specific)
    st.subheader(" Fakta Berdasarkan Jenis Material")
    
    fakta_1, fakta_2, fakta_3 = st.columns(3)
    
    with fakta_1:
        st.markdown("### ♻️ Recyclable")
        with st.expander("Kertas & Kardus", expanded=True):
            st.write("""
            Mendaur ulang **1 ton kertas** dapat menyelamatkan **17 pohon dewasa**, 7.000 galon air, dan menghemat energi yang cukup untuk menghidupi rata-rata rumah selama 6 bulan.
            """)
        with st.expander("Logam & Aluminium"):
            st.write("""
            Mendaur ulang aluminium menghemat **95% energi** dibandingkan memproduksi dari bahan mentah. Kaleng yang Anda daur ulang hari ini bisa kembali ke rak toko dalam 60 hari.
            """)
            
    with fakta_2:
        st.markdown("### 🚯 Non-Recyclable")
        with st.expander("Sampah Organik", expanded=True):
            st.write("""
            Jika sampah organik menumpuk di TPA tanpa oksigen, ia menghasilkan **gas metana**—gas rumah kaca yang **25 kali lebih kuat** daripada CO2 dalam memicu pemanasan global.
            """)
        with st.expander("Limbah Tekstil"):
            st.write("""
            Industri fashion bertanggung jawab atas **10% emisi karbon dunia**. Pakaian berbahan sintetik (poliester) melepaskan mikroplastik yang meracuni rantai makanan laut.
            """)

    with fakta_3:
        st.markdown("### ⚠️ Limbah B3")
        with st.expander("Baterai & Elektronik", expanded=True):
            st.write("""
            Satu baterai kecil dapat mencemari **400.000 liter air** karena kandungan logam beratnya (merkuri, timbal, kadmium) yang bersifat karsinogenik bagi manusia.
            """)
        with st.expander("Kaca"):
            st.write("""
            Kaca adalah material unik yang dapat didaur ulang **100% tanpa henti** tanpa kehilangan kualitasnya. Namun, jika dibuang, kaca butuh **1 juta tahun** untuk terurai.
            """)

    st.divider()
    
    # Edukasi Hierarki Sampah
    st.subheader("♻️ Prinsip Pengelolaan Sampah (5R)")
    st.markdown("""
    Sistem klasifikasi ini membantu Anda pada tahap **Recycle** (Daur Ulang). Namun, penting untuk selalu mengingat urutan prioritas:
    1. **Refuse:** Menolak penggunaan plastik sekali pakai.
    2. **Reduce:** Mengurangi timbulan sampah harian.
    3. **Reuse:** Menggunakan kembali wadah yang masih layak.
    4. **Recycle:** Memilah sampah (Gunakan aplikasi ini!).
    5. **Rot:** Mengolah sampah organik menjadi kompos.
    """)

# --- TAB 2: DATASET ---
with tab2:
    st.header("Informasi Dataset")
    st.write("Dataset yang digunakan pada project ini berasal dari Public Dataset Kaggle berjudul “Garbage Classification Dataset” yang dapat di akses pada laman web [Kaggle - Garbage Classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)")
    # Simulasi Data Frame (Sesuaikan dengan data riil Anda jika perlu)
    data = {
        'Kategori': [
            'paper', 'plastic', 'metal', 'cardboard', 'brown-glass', 'green-glass', 'white-glass',
            'biological', 'clothes', 'shoes', 'trash', 'battery'
        ],
        'Jumlah Gambar':  [1050, 865, 769, 891, 607, 629, 775, 985, 5325, 1977, 697, 945]
    }
    df_eda = pd.DataFrame(data)

    # 1. Bar Chart (Sesuai Snippet Anda)
    st.markdown("#### Distribusi Per Kelas (Sub-Kategori)")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    bars = sns.barplot(x='Kategori', y='Jumlah Gambar', data=df_eda, color='#87CEEB', ax=ax1)
    plt.xticks(rotation=45)
    plt.title("Distribusi Per Kelas (Sub-Kategori)")
    
    # Menambahkan label angka di atas bar
    for bar in bars.containers[0]:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 10, str(int(height)), ha='center', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig1)

    st.divider()

    # 2. Pie Chart (Sesuai Snippet Sebelumnya)
    st.markdown("#### Proporsi Kelas: Recyclable vs Non-Recyclable")
    
    recyclable = ['paper', 'plastic', 'metal', 'cardboard', 'brown-glass', 'green-glass', 'white-glass']
    non_recyclable = ['biological', 'clothes', 'shoes', 'trash', 'battery']

    recyclable_count = sum(df_eda[df_eda["Kategori"].isin(recyclable)]["Jumlah Gambar"])
    non_recyclable_count = sum(df_eda[df_eda["Kategori"].isin(non_recyclable)]["Jumlah Gambar"])

    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.pie(
        [recyclable_count, non_recyclable_count],
        labels=["Recyclable", "Non Recyclable"],
        autopct="%1.1f%%",
        explode=[0.05, 0.05],
        startangle=90,
        colors=['#4CAF50', '#FF9999']
    )
    plt.title("Proporsi Kelas Biner")
    st.pyplot(fig2)

# --- TAB 3: KLASIFIKASI AI (PERBAIKAN UTAMA) ---
with tab3:
    st.header("Analisis Gambar Sampah")
    
    # Pilihan Input
    metode = st.radio("Pilih Cara Unggah:", ["Upload File", "Gunakan Kamera"], horizontal=True)
    
    uploaded_file = None
    if metode == "Upload File":
        uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "png", "jpeg"])
    else:
        uploaded_file = st.camera_input("Ambil foto")

    # JIKA GAMBAR SUDAH DIUNGGAH
    if uploaded_file is not None:
        # Menampilkan gambar yang diunggah
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang akan dianalisis", width=400)
        
        # TOMBOL ANALISIS (Muncul hanya jika gambar ada)
        if st.button("Mulai Analisis"):
            if model is not None:
                with st.spinner('Sedang memproses...'):
                    # 1. Preprocessing
                    img_res = img.resize((224, 224))
                    img_array = image.img_to_array(img_res)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_input(img_array)
                    
                    # 2. Prediksi
                    prediction = model.predict(img_array)
                    score = prediction[0][0] # Sesuaikan index dengan output model anda
                    confidence = max(score, 1-score) * 100
                    
                    st.divider()
                    
                    # 3. Hasil Tampilan
                    if confidence < 60:
                        st.warning(f"Hasil kurang meyakinkan ({confidence:.2f}%). Coba ambil foto lebih jelas.")
                    
                    if score > 0.5:
                        st.success(f"### HASIL: RECYCLABLE (BISA DIDAUR ULANG)")
                        st.metric("Tingkat Keyakinan", f"{confidence:.2f}%")
                        st.info("""
            *💡 Saran Daur Ulang:*
            - Jika berupa *Botol Plastik*, buatlah menjadi pot tanaman atau kerajinan tangan.
            - Jika berupa *Kertas/Kardus*, kumpulkan untuk dijual ke pengepul atau buatlah bubur kertas untuk seni.
            - Jika berupa *Logam*, bersihkan dan bawa ke bank sampah terdekat agar bisa dilebur kembali.
            """)
                    else:
                        st.error(f"### HASIL: NON-RECYCLABLE (TIDAK BISA DIDAUR ULANG)")
                        st.metric("Tingkat Keyakinan", f"{confidence:.2f}%")
                        st.info("""
            *💡 Saran Penanganan:*
            - Jika berupa *Sampah Organik*, olah menjadi pupuk kompos di rumah.
            - Jika berupa *Limbah B3 (Baterai/Elektronik)*, jangan dibuang ke tempat sampah biasa! Cari tempat pengumpulan limbah elektronik khusus.
            - Jika berupa *Pakaian Bekas*, pertimbangkan untuk didonasikan jika masih layak atau dijadikan kain lap.
            """)
            else:

                st.error("Model tidak tersedia.")


