# train_model.py

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Load dataset
data = pd.read_csv('dataset.csv')

# Pisahkan fitur dan label (ganti 'target' dengan nama kolom label kamu)
X = data.drop('target', axis=1)
y = data['target']

# Encode kolom kategorikal (misalnya kolom gender)
le = LabelEncoder()
if X['gender'].dtype == 'object':
    X['gender'] = le.fit_transform(X['gender'])

# Imputasi missing value
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluasi
accuracy = model.score(X_test, y_test)
print(f"Akurasi model: {accuracy:.2f}")

# Simpan model, imputer, dan label encoder
with open('model.pkl', 'wb') as f_model:
    pickle.dump(model, f_model)

with open('imputer.pkl', 'wb') as f_imputer:
    pickle.dump(imputer, f_imputer)

with open('label_encoder.pkl', 'wb') as f_encoder:
    pickle.dump(le, f_encoder)

print("Model, imputer, dan label encoder berhasil disimpan.")
