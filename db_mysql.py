import mysql.connector
from datetime import datetime
import json

# === Koneksi ke MySQL ===
def get_connection():
    return mysql.connector.connect(
        host="localhost",     # Ganti jika pakai server lain
        user="root",          # Ganti sesuai usermu
        password="",          # Password usermu
        database="spam_system"
    )

# === Simpan Prediksi ke DB ===
def simpan_prediksi(pesan, kategori, jenis_spam, confidence_dict):
    conn = get_connection()
    cursor = conn.cursor()
    query = '''
        INSERT INTO hasil_prediksi (pesan, kategori, jenis_spam, confidence_json, waktu_prediksi)
        VALUES (%s, %s, %s, %s, %s)
    '''
    values = (
        pesan,
        kategori,
        jenis_spam,
        json.dumps(confidence_dict),
        datetime.now()
    )
    cursor.execute(query, values)
    conn.commit()
    cursor.close()
    conn.close()
