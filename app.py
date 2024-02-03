import streamlit as st
import numpy as np
import pickle

# Fungsi untuk memuat model
def load_model(model_name):
    with open(model_name, 'rb') as f:
        return pickle.load(f)

# Memuat model
dt_model = load_model('DecisionTree_best_model.pkl')
rf_model = load_model('RandomForest_best_model.pkl')
ada_model = load_model('AdaBoost_best_model.pkl')

st.set_page_config(page_title="Prediksi Harga Emas", page_icon=":money_with_wings:")

st.title("Prediksi Harga Emas")
st.write("Prediksi harga emas berdasarkan IHSG, Inflasi, dan Kurs Dollar menggunakan model Decision Tree, Random Forest, atau AdaBoost.")

# Input pengguna
ihsg = st.number_input('Masukkan Nilai IHSG', format="%.2f")
kurs_jual = st.number_input('Masukkan Kurs Jual', format="%.2f")
data_inflasi = st.number_input('Masukkan Data Inflasi (dalam persen, contoh: masukkan 3.5 untuk 3,5%)', format="%.2f")

model_option = st.selectbox("Pilih Model untuk Prediksi:", ['Decision Tree', 'Random Forest', 'AdaBoost'])

# Memilih model berdasarkan pilihan pengguna
model = {'Decision Tree': dt_model, 'Random Forest': rf_model, 'AdaBoost': ada_model}[model_option]

# Tombol prediksi
if st.button("Prediksi Harga"):
    inflasi_desimal = data_inflasi / 100  # Konversi inflasi ke bentuk desimal
    inputs = np.array([[ihsg, kurs_jual, inflasi_desimal]])
    predicted_price = model.predict(inputs)[0]
    
    st.write("")
    st.subheader(f"Harga Emas yang Diprediksi: Rp {predicted_price:,.2f} menggunakan model {model_option}")
