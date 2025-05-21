import sqlite3

# Membuat database dan tabel pengguna
def create_database():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Membuat tabel pengguna jika belum ada
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password TEXT)''')
    conn.commit()
    conn.close()

create_database()
