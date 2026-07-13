from flask import Flask, request, render_template
import re
import string
import numpy as np
import joblib
import nltk
import json
from nltk.corpus import stopwords
from datetime import datetime
from db_mysql import simpan_prediksi

app = Flask(__name__, template_folder='templates')

# === Unduh stopwords bahasa Indonesia ===
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

# === Daftar kata tekanan (pressure terms) ===
pressure_words = {
    "urgency": ["sekarang", "segera", "hari ini", "1x24 jam", "batas waktu", "terakhir", "dalam 24 jam", "limit waktu", "sekarang juga"],
    "threat": ["dibekukan", "rekening diblokir", "akun dibekukan", "hangus", "tidak bisa digunakan", "diblokir", "dihapus", "pemblokiran", "penahanan", "akan disita", "penutupan akun", "ancaman hukum", "akan kami tindak", "pengadilan", "diblokir permanen", "dikenakan denda"],
    "command": ["transfer", "bayar", "kirim", "klik", "buka link", "isi", "konfirmasi", "aktifkan", "hubungi", "laporkan", "verifikasi segera", "konfirmasi segera", "wajib bayar"],
    "manipulation": ["khusus anda", "rahasia", "penting", "jangan abaikan", "untuk anda saja"]
}
all_pressure_terms = set(term for group in pressure_words.values() for term in group)

# === Fungsi preprocessing ===
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text.strip()

def count_pressure_terms(text):
    text = text.lower()
    return sum(1 for term in all_pressure_terms if term in text)

# === Load Model dan Vectorizer ===
model_kategori = joblib.load("models/model_rf_kategori.pkl")
model_jenis = joblib.load("models/model_rf_jenis.pkl")
vectorizer = joblib.load("models/vectorizer_kategori.pkl")

# === Load Laporan Evaluasi ===
with open("reports/report_kategori.json") as f:
    report_kategori = json.load(f)

with open("reports/report_jenis.json") as f:
    report_jenis = json.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    pesan = request.form['pesan']
    cleaned = clean_text(pesan)
    tfidf = vectorizer.transform([cleaned])
    pressure = count_pressure_terms(pesan)
    final_input = np.hstack((tfidf.toarray(), [[pressure]]))

    # Prediksi Kategori dan Jenis Spam
    kategori = model_kategori.predict(final_input)[0]
    jenis = model_jenis.predict(final_input)[0] if kategori in ['Spam_Baik', 'Spam_Merugikan'] else "-"

    # Ambil metrik evaluasi berdasarkan hasil prediksi
    metrik_kategori = report_kategori.get(kategori, {})
    metrik_jenis = report_jenis.get(jenis, {}) if kategori != "Non_Spam" else {}

    akurasi_kategori = report_kategori.get("__accuracy__", 0)
    akurasi_jenis = report_jenis.get("__accuracy__", 0) if kategori != "Non_Spam" else 0

    # Hitung tingkat keyakinan model terhadap semua kelas
    proba_kategori = model_kategori.predict_proba(final_input)[0]
    label_kategori = model_kategori.classes_
    confidence_dict = dict(zip(label_kategori, proba_kategori))

    # === Simpan ke Database MySQL ===
    simpan_prediksi(pesan, kategori, jenis, confidence_dict)

    return render_template(
        'index.html',
        pesan=pesan,
        hasil_kategori=kategori,
        hasil_jenis=jenis,
        metrik_kategori=metrik_kategori,
        metrik_jenis=metrik_jenis,
        akurasi_kategori=akurasi_kategori,
        akurasi_jenis=akurasi_jenis,
        confidence_kategori=confidence_dict
    )

if __name__ == '__main__':
    app.run(debug=True)
