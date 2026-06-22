# 🚗 AutoValuasi — Prediksi Harga Mobil Bekas

Dashboard Streamlit untuk proyek Data Science **Used Car Price Prediction**
(Kaggle Playground Series S4E9), dilengkapi model **XGBoost (tuned)** yang
mereplikasi penuh pipeline cleaning, feature engineering, dan modeling dari
notebook `Final_Project_DS59.ipynb`.

## Isi Folder

```
.
├── app.py                     # Aplikasi Streamlit utama
├── train_model.py             # Script reproduksi pipeline + training model
├── requirements.txt           # Daftar dependency
├── .streamlit/config.toml     # Tema warna (dealer mobil: merah/emas/asphalt)
└── artifacts/                 # Model & encoder yang sudah dilatih
    ├── model.joblib            # Model XGBoost final
    ├── preprocessing.joblib    # Encoder (LabelEncoder, OHE, target/frequency encoding map)
    ├── metadata.json           # Metrik, range slider, daftar dropdown
    ├── eda_data.csv            # Sampel data bersih untuk tab eksplorasi data
    ├── feature_importance.csv  # Feature importance model
    └── val_predictions.csv     # Hasil prediksi vs aktual (data validasi)
```

## Cara Menjalankan

1. Buat virtual environment (opsional tapi disarankan):
   ```bash
   python -m venv venv
   source venv/bin/activate    # Windows: venv\Scripts\activate
   ```

2. Install dependency:
   ```bash
   pip install -r requirements.txt
   ```

3. Jalankan aplikasi (pastikan folder `artifacts/` tetap berada di samping `app.py`):
   ```bash
   streamlit run app.py
   ```

4. Buka browser ke `http://localhost:8501`.

## Fitur Aplikasi

- **🏠 Beranda** — ringkasan pipeline & statistik dataset.
- **📊 Jelajahi Data** — distribusi, korelasi, breakdown kategorikal, dan
  brand explorer interaktif.
- **🧮 Prediksi Harga** — form input lengkap dengan **slider** untuk tahun
  produksi, jarak tempuh, tenaga mesin (HP), kapasitas mesin, jumlah
  silinder, dan jumlah percepatan transmisi, plus dropdown brand/model/warna/
  kondisi. Hasil prediksi ditampilkan beserta kisaran harga dan perbandingan
  terhadap sebaran harga brand yang sama.
- **📈 Performa Model** — metrik R²/RMSE/MAE, grafik actual vs predicted,
  feature importance, dan perbandingan 8 kandidat model dari notebook
  eksperimen.
- **ℹ️ Tentang Proyek** — penjelasan strategi penanganan noise dataset.

## Melatih Ulang Model (opsional)

Jika ingin melatih ulang dari data mentah:

1. Salin `train.csv` (dari kompetisi Kaggle S4E9) ke folder ini.
2. Jalankan:
   ```bash
   python train_model.py
   ```
3. Artefak baru akan menimpa isi folder `artifacts/`.

## Catatan Model

- Algoritma final: **XGBoost** (`n_estimators=600, max_depth=7,
  learning_rate=0.03, subsample=0.8, colsample_bytree=0.8`).
- Target di-log-transform (`log1p`) sebelum training, lalu di-*inverse*
  (`expm1`) saat prediksi.
- Noise ditangani lewat dua tahap: pembuangan baris brand-model mislabel
  (majority-brand rule) dan pembuangan baris harga placeholder (rasio harga
  vs median brand+model di luar 0.2x–5x).
