import pickle
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Contoh data training dengan nilai hilang
data = pd.DataFrame({
    'feature1': [1, 2, np.nan, 4, 5],
    'feature2': [np.nan, 1, 1, 2, 2],
    'feature3': [5, 3, 4, np.nan, 1]
})

# Membuat objek imputer, misalnya median
imputer = SimpleImputer(strategy='median')

# Melatih imputer dengan data
imputer.fit(data)

# Simpan imputer ke file imputer.pkl menggunakan pickle
with open('imputer.pkl', 'wb') as f:
    pickle.dump(imputer, f)

print("File imputer.pkl berhasil dibuat.")
