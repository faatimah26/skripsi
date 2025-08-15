import sqlite3
def init_db():
    conn = sqlite3.connect('riwayat.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS riwayat_deteksi (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nama_file TEXT,
            hasil TEXT,
            gambar BLOB
        )
    ''')
    conn.commit()
    conn.close()

def simpan_riwayat(nama_file, hasil, gambar):
    conn = sqlite3.connect('riwayat.db')
    c = conn.cursor()
    c.execute('INSERT INTO riwayat_deteksi (nama_file, hasil, gambar) VALUES (?, ?, ?)', (nama_file, hasil, gambar))
    conn.commit()
    conn.close()

def ambil_riwayat():
    conn = sqlite3.connect('riwayat.db')
    c = conn.cursor()
    c.execute('SELECT rowid, nama_file, hasil, gambar FROM riwayat_deteksi')
    data = c.fetchall()
    conn.close()
    return data

def hapus_riwayat(id_data):
    conn = sqlite3.connect('riwayat.db')
    c = conn.cursor()
    c.execute('DELETE FROM riwayat_deteksi WHERE rowid = ?', (id_data,))
    conn.commit()
    conn.close()
