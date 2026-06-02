# 🛡️ Face Anti-Spoofing Dashboard

Dashboard interaktif Streamlit untuk Capstone Project Data Science — DAC Find-IT 2026.

## 📌 Fitur

| Halaman | Deskripsi |
|---------|-----------|
| 📊 Overview | Business understanding, pipeline project, jenis spoofing |
| 🔍 EDA | Distribusi kelas, analisis ukuran gambar, format file |
| ⚙️ Preprocessing | Pipeline cleaning, deduplication, stratified split |
| 🧠 Modeling | Arsitektur model, training strategy, riwayat eksperimen |
| 📈 Evaluation | Confusion matrix, classification report, error analysis |
| 🎯 Prediction Demo | Upload gambar & simulasi inferensi ensemble |

##  Cara Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

##  Deploy ke Streamlit Cloud

1. Push repo ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Pilih repo → branch → `app.py`
4. Deploy!

##  Struktur Project

```
face-antispoofing-dashboard/
├── app.py                  # Entry point
├── requirements.txt
├── assets/
│   └── style.css           # Custom styling
├── components/
│   └── sidebar.py          # Navigasi sidebar
├── pages/
│   ├── overview.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── prediction.py
└── utils/
    └── helpers.py          # Komponen reusable
```

##  Model

- **EfficientNet-B3** (Base model)
- **ConvNeXt-Base** (Ensemble)
- **EfficientNetV2-M** (Ensemble)
- **EfficientNet-B4** (Ensemble)
- **Best Accuracy: 67.4%** (Soft Voting Ensemble)

##  Tech Stack

- Streamlit · Plotly · Pandas · NumPy · Pillow
- PyTorch · timm · Focal Loss
- Dataset: DAC Find-IT 2026
