import sqlite3
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import pickle
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Konfigurasi Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

import pickle

# Load model dan tools ML
model = None
scaler = None
imputer = None
le = None

try:
    with open('alzheimer_model.pkl', 'rb') as file:
        artifacts = pickle.load(file)

    model = artifacts['classifier']
    scaler = artifacts['scaler']
    le_gender = artifacts['le_gender']
    le_memory = artifacts['le_memory']
    le_brain = artifacts['le_brain']
    imputer = artifacts.get('imputer')  # Pastikan ini ada jika imputer digunakan




except FileNotFoundError as e:
    print(f"File tidak ditemukan: {e}")
except Exception as e:
    print(f"Terjadi kesalahan saat loading: {e}")

from sklearn.preprocessing import LabelEncoder

# Melatih LabelEncoder dengan label yang ada
le_gender = LabelEncoder()
le_gender.fit(['Male', 'Female'])  # Latih dengan label yang ada di dataset pelatihan

le_memory = LabelEncoder()
le_memory.fit(['Yes', 'No'])  # Latih dengan label yang ada di dataset pelatihan

le_brain = LabelEncoder()
le_brain.fit(['Yes', 'No'])  # Latih dengan label yang ada di dataset pelatihan


# Fungsi untuk koneksi ke database SQLite
def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row  # Agar hasil query berupa dictionary
    return conn

# User Class
class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

# Fungsi untuk memuat user dari database
@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    
    if user:
        return User(id=user['id'], username=user['username'])
    return None

# Halaman Home
@app.route('/')
def home():
    return render_template('index.html')

# Halaman Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Cek apakah username ada di database
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            user_obj = User(id=user['id'], username=user['username'])
            login_user(user_obj)
            flash('Login berhasil!', 'success')
            return render_template('index.html')
        else:
            flash('Username atau password salah', 'danger')
    return render_template('login.html')


# Halaman Register
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Cek apakah username sudah ada
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        
        if user:
            flash('Username sudah terdaftar. Silakan gunakan username lain.', 'danger')
            return redirect(url_for('register'))
        else:
            # Simpan user baru
            hashed_password = generate_password_hash(password)
            conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
            conn.commit()
            conn.close()
            flash('Registrasi berhasil! Silakan login.', 'success')
            return redirect(url_for('login'))
            
    return render_template('register.html')

# Route untuk Logout
@app.route('/logout')
@login_required  # Pastikan pengguna harus login untuk logout
def logout():
    logout_user()  # Menghapus session user yang sedang login
    flash('Anda telah berhasil logout.', 'success')
    return redirect(url_for('login'))  # Arahkan ke halaman login setelah logout

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/edukasi')
def edukasi():
    return render_template('edukasi.html')


@app.route('/tentang')
def tentang():
       return render_template('tentang.html')

@app.route('/klasifikasi', methods=['GET', 'POST'])
def klasifikasi():
    if request.method == 'POST':
        input_data = request.form.to_dict()
        expected_columns = ['age', 'gender', 'education', 'memory_loss', 'brain_scan']

        # Kirim ulang form_data ke template agar inputan tetap terisi
        form_data = {key: input_data.get(key, '') for key in expected_columns}

        # Cek kelengkapan data
        if not all(input_data.get(col) for col in expected_columns):
            flash('Data input tidak lengkap atau format salah', 'danger')
            return render_template('klasifikasi.html', form_data=form_data)

        try:
            # Siapkan data untuk model
            data = pd.DataFrame([form_data])

            data['age'] = data['age'].astype(int)
            data['education'] = data['education'].astype(int)

            data['gender'] = data['gender'].str.capitalize()
            data['memory_loss'] = data['memory_loss'].str.capitalize()
            data['brain_scan'] = data['brain_scan'].str.capitalize()

            data['gender'] = le_gender.transform(data['gender'])
            data['memory_loss'] = le_memory.transform(data['memory_loss'])
            data['brain_scan'] = le_brain.transform(data['brain_scan'])

            if imputer is not None:
                data_imputed = imputer.transform(data)
            else:
                data_imputed = data.values

            if scaler is not None:
                data_processed = scaler.transform(data_imputed)
            else:
                data_processed = data_imputed

            prediction = model.predict(data_processed)
            output = prediction[0]

            if output == 0:
                prediction_text = "Tidak ada tanda demensia."
            else:
                prediction_text = "Tanda demensia terdeteksi."

            return render_template('klasifikasi.html', 
                                   prediction_text=f'Prediksi: {prediction_text}',
                                   form_data=form_data)

        except Exception as e:
            flash(f'Terjadi kesalahan saat memproses data: {e}', 'danger')
            return render_template('klasifikasi.html', form_data=form_data)

    # Untuk GET request, form kosong
    return render_template('klasifikasi.html', form_data={})


if __name__ == "__main__":
    app.run(debug=True)

