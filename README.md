# Tugas05 — Transfer Learning untuk Klasifikasi Citra dengan Data Terbatas

Implementasi Tugas Take-Home Sesi 05 (Advanced CNNs and Transfer Learning). Membandingkan tiga paradigma
pelatihan — *from scratch*, *feature extraction*, dan *progressive fine-tuning* — pada **ResNet-18**
menggunakan subset **10 kelas Oxford-IIIT Pet** (skenario *small-data*, ~2.000 citra), ditambah uji
kontrol augmentasi dan arsitektur pembanding EfficientNet-B0.

## Struktur Repositori

```
.
├── Tugas05_TransferLearning.ipynb   # Notebook utama (jalankan Run all)
├── laporan.md                       # Laporan analisis (Bagian A + Bagian C) — ekspor ke PDF
├── requirements.txt                 # Dependensi tersemat versi
├── README.md                        # Berkas ini
├── logs/                            # Artefak log metrik (dibuat saat notebook dijalankan)
│   ├── metrics_log.csv
│   ├── metrics_log.json
│   └── failure_cases.csv
└── figures/                         # Visualisasi (dibuat saat notebook dijalankan)
    ├── sample_grid.png
    ├── loss_curves_B1_B2_B3.png
    ├── macro_f1_bar.png
    ├── failure_cases.png
    └── gradcam_before_after.png
```

## Cara Menjalankan (Google Colab — disarankan)

1. Unggah `Tugas05_TransferLearning.ipynb` ke Google Colab.
2. Set runtime: **Runtime → Change runtime type → GPU (T4)**.
3. **Runtime → Run all**. Dataset Oxford-IIIT Pet terunduh otomatis (tanpa autentikasi).

## Cara Menjalankan (Lokal)

```bash
pip install -r requirements.txt
# buka & jalankan notebook via Jupyter / VS Code
jupyter notebook Tugas05_TransferLearning.ipynb
```

## Protokol Eksperimen (seragam untuk semua kondisi)

| Aspek | Nilai |
|-------|-------|
| Arsitektur utama | ResNet-18 (`ResNet18_Weights.IMAGENET1K_V1`) |
| Input | 224×224, normalisasi ImageNet (mean/std standar) |
| Optimizer | AdamW, `weight_decay=1e-2` |
| Loss | CrossEntropyLoss |
| Batch size | 32 |
| Early stopping | monitor validation loss, `patience=4` |
| Seed | 42 (Python, NumPy, PyTorch) |
| Split | Stratified 70% train / 15% val / 15% test |

## Matriks Kondisi

| Kode | Kondisi | Deskripsi |
|------|---------|-----------|
| B1 | Baseline from Scratch | ResNet-18 `weights=None` |
| B2 | Feature Extraction | Backbone beku, latih `fc` saja (`nn.Linear(512,10)`) |
| B3 | Progressive Fine-Tuning | Unfreeze `layer4` + `fc`, differential LR (1e-4 / 1e-3) |
| B4 | Uji Kontrol Augmentasi | B3 tanpa vs dengan augmentasi, bandingkan *generalization gap* |
| B5 | Arsitektur Pembanding (bonus) | EfficientNet-B0 fine-tune, trade-off F1 vs params vs waktu |

