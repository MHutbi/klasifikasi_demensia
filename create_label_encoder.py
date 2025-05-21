import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Contoh data
data = pd.DataFrame({
    'category': ['apple', 'banana', 'orange', 'banana', 'apple']
})

# Membuat objek LabelEncoder
label_encoder = LabelEncoder()

# Melatih LabelEncoder dengan data
label_encoder.fit(data['category'])

# Simpan LabelEncoder ke file label_encoder.pkl menggunakan pickle
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("File label_encoder.pkl berhasil dibuat.")
