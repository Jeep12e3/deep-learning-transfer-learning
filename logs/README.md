# logs/

Folder ini menampung **log metrik mentah** yang dihasilkan otomatis saat notebook
`Tugas05_TransferLearning.ipynb` dijalankan (`Run all`):

- `metrics_log.csv` — tabel konsolidasi B1–B5 (trainable params, LR, epoch konvergen, durasi, val acc, test Macro F1).
- `metrics_log.json` — versi JSON lengkap dengan metadata (seed, daftar kelas).
- `failure_cases.csv` — ringkasan citra test yang salah diprediksi oleh model terbaik.

> Berkas belum ada di sini secara sengaja: angka **tidak difabrikasi**. Jalankan notebook untuk
> menghasilkan log asli, lalu lampirkan berkas-berkas ini saat pengumpulan sesuai
> *Catatan Penilaian Data Empiris* pada soal.
