# Laporan Tugas Take-Home Sesi 05
## Transfer Learning untuk Klasifikasi Citra dengan Data Terbatas

- **Nama / NIM:** Darrell Richie Wibawa / 2802409533
- **Mata Kuliah:** Deep Learning
- **Dataset:** Oxford-IIIT Pet (subset 10 kelas: 5 ras kucing + 5 ras anjing, ~2.000 citra)
- **Arsitektur utama:** ResNet-18 (`ResNet18_Weights.IMAGENET1K_V1`)
- **Seed:** 42 · **Platform:** Google Colab (GPU T4)

> **Catatan integritas data:** Seluruh angka pada laporan ini bersumber langsung dari berkas log mentah
> `logs/metrics_log.csv` / `logs/metrics_log.json` dan `logs/failure_cases.csv` hasil eksekusi notebook
> (seed 42, GPU T4). Angka dapat direproduksi dengan menjalankan ulang notebook `Run all`.

---

## Bagian A — Analisis Konseptual (Bobot 20%)

### A1. Efek Regularisasi pada Batch Normalization (Slide 6–8)

Batch Normalization menormalkan aktivasi tiap mini-batch menggunakan rerata (μ_B) dan variansi (σ²_B)
yang **dihitung dari batch tersebut**, bukan dari seluruh populasi data. Karena komposisi setiap
mini-batch bersifat acak, estimasi μ_B dan σ²_B mengandung *noise* stokastik yang berbeda-beda tiap
iterasi. Noise ini menyuntikkan gangguan kecil pada aktivasi — mirip Dropout — sehingga jaringan tidak
dapat terlalu bergantung pada pola spesifik satu sampel dan terdorong belajar representasi yang lebih
robust (Slide 8). Inilah "mild regularization" yang sering membuat Dropout tidak lagi diperlukan.
Namun asumsi ini **gugur ketika batch size sangat kecil** (misal 2–4): estimasi statistik menjadi
sangat berisik dan bias, mengakibatkan instabilitas pelatihan serta ketidakcocokan antara statistik
train (per-batch) dan inferensi (*running statistics*). Pada kondisi tersebut, GroupNorm/LayerNorm atau
tetap memakai Dropout lebih aman. [Ioffe & Szegedy, 2015]

### A2. Degradasi Jaringan Dalam dan Formulasi Residual (Slide 10–12)

Secara teori, jaringan dalam minimal dapat menyamai jaringan dangkal dengan menyetel lapisan ekstra
menjadi *identity mapping*. Faktanya, *plain CNN* yang sangat dalam justru memperoleh **akurasi
pelatihan lebih rendah** (bukan sekadar overfitting) — inilah *degradation problem* (Slide 10). Penyebabnya
bukan kapasitas, melainkan **sulitnya optimizer mempelajari fungsi identitas** melalui tumpukan
transformasi non-linear, diperburuk oleh gradien yang melemah saat merambat mundur. *Skip connection*
mengubah target pembelajaran dari H(x) menjadi residual F(x) = H(x) − x, sehingga output blok menjadi
F(x) + x. Saat backpropagation, turunan mengandung suku **+1** dari jalur pintas: ∂(F(x)+x)/∂x = F'(x) + 1.
Suku konstan ini menjamin gradien tetap mengalir meski F'(x) mengecil, sehingga **memitigasi vanishing
gradient** (dibahas Sesi 02) dan membuat identity mapping mudah dipelajari (cukup F(x)→0). [He et al., 2015]

### A3. Feature Extraction vs. Fine-Tuning (Slide 14–16)

Ditinjau empat dimensi: **(1) Trainable parameters** — Feature Extraction (FE) hanya melatih kepala
classifier (pada ResNet-18 ≈ 5.130 parameter), sedangkan Fine-Tuning (FT) membuka sebagian/seluruh
backbone sehingga parameter terlatih jauh lebih banyak. **(2) Kebutuhan data** — FE memadai untuk data
sangat terbatas karena ruang hipotesisnya kecil; FT membutuhkan lebih banyak data agar bobot yang dibuka
tidak overfit. **(3) Beban komputasi/memori per epoch** — FE lebih murah (backbone cukup *forward pass*,
tanpa menyimpan gradien), FT lebih mahal karena backward menjangkau lapisan yang dibuka. **(4) Risiko
destructive update** — FE nol (fitur bawaan terjaga total); FT berisiko merusak representasi pretrained bila
LR terlalu besar (Slide 16). **Kriteria praktis:** pilih FE saat data sangat sedikit, domain mirip ImageNet,
dan anggaran minim; pilih FT (progresif, differential LR) saat data cukup atau domain bergeser jauh.

### A4. Kaidah Penerapan Data Augmentation (Slide 20, 21, 23)

Augmentasi hanya diaktifkan pada **train set** karena perannya adalah *regularizer*: memperkaya variasi
sampel latih agar model belajar fitur invarian dan tidak menghafal. Pada **validation/test set**,
transformasi acak harus **dinonaktifkan** demi menjaga evaluasi yang deterministik, dapat direproduksi,
dan merepresentasikan distribusi nyata. Mengaugmentasi data evaluasi akan membuat metrik berfluktuasi
antar-run dan **mencemari estimasi generalisasi** (bentuk kebocoran metodologis). Contoh transformasi
**tidak layak** untuk klasifikasi ras hewan peliharaan: **Random Vertical Flip** (membalik citra atas-bawah).
Ras hewan tidak pernah muncul terbalik secara alami; flip vertikal menciptakan citra yang tidak
merepresentasikan distribusi nyata dan **tidak mempertahankan invariansi semantik label** — orientasi
tidak mengubah identitas ras, namun citra terbalik justru menyesatkan model. Contoh lain yang berisiko:
rotasi ekstrem (>90°) atau grayscale penuh bila warna bulu adalah fitur pembeda ras.

### A5. Mitigasi Catastrophic Forgetting (Slide 28–29)

*Catastrophic forgetting* terjadi ketika adaptasi domain berlangsung terlalu agresif: gradien besar dari
data target "menimpa" bobot pretrained sehingga **representasi visual bawaan ImageNet yang bernilai hilang**
sebelum model sempat memanfaatkannya. Hal ini lazim bila seluruh backbone dibuka dengan learning rate
seragam yang tinggi pada data yang sedikit. **Strategi mitigasi:** (1) **Differential learning rate** —
beri LR kecil untuk lapisan awal/tengah (fitur umum, perlu dijaga) dan LR besar untuk kepala baru; secara
teoritis ini membatasi besar langkah pembaruan pada bobot bernilai sehingga perubahannya halus. (2)
**Gradual unfreezing** — buka lapisan secara bertahap dari atas ke bawah; kepala classifier distabilkan
dulu sehingga gradien yang merambat ke backbone sudah "sehat" dan tidak merusak. (Alternatif: *linear
warmup* yang menaikkan LR perlahan di awal agar langkah awal tidak liar.) [Howard & Ruder, 2018]

---

## Bagian C — Laporan dan Analisis (Bobot 35%)

### C1. Tabel Konsolidasi dan Visualisasi Pembelajaran

Tabel berikut merangkum seluruh kondisi, diambil langsung dari `logs/metrics_log.csv` (seed 42, GPU T4).

| Kode | Strategi | Trainable Params | Learning Rate | Epoch Konvergen | Waktu Latih (s) | Val Acc | Test Macro F1 |
|------|----------|------------------|---------------|-----------------|-----------------|---------|---------------|
| B1 | From Scratch | 11.181.642 | all=1e-3 | 4 | 115,8 | 0,4209 | **0,3048** |
| B2 | Feature Extraction | **5.130** | fc=1e-3 | 18 | 286,2 | 0,9259 | **0,9201** |
| B3 | Progressive Fine-Tuning | 8.398.858 | layer4=1e-4; fc=1e-3 | 5 | 113,6 | 0,9529 | **0,9363** |
| B4-noaug | B3 tanpa augmentasi | 8.398.858 | layer4=1e-4; fc=1e-3 | 4 | 70,2 | 0,9529 | 0,9160 |
| B4-aug | B3 dengan augmentasi | 8.398.858 | layer4=1e-4; fc=1e-3 | 5 | 113,6 | 0,9529 | 0,9363 |
| B5 | EfficientNet-B0 (fine-tune) | 424.970 | last_block+head=5e-4 | 8 | 156,0 | 0,9293 | 0,9190 |

Catatan: `total_params` ResNet-18 = 11.181.642; EfficientNet-B0 = 4.020.358. Nilai `trainable_params` B2
tepat **5.130** (= 512×10 + 10) sesuai *Tips Implementasi* PDF, mengonfirmasi backbone benar-benar beku.

**Visualisasi (otomatis dihasilkan notebook pada `figures/`):**
- `loss_curves_B1_B2_B3.png` — kurva training vs validation loss untuk B1, B2, B3.
- `macro_f1_bar.png` — perbandingan Test Macro F1 antar kondisi.

**Pembacaan kurva loss (`loss_curves_B1_B2_B3.png`):**
- **B1 (From Scratch):** validation loss **berfluktuasi tajam** (naik-turun antara ~1,7–2,7) sementara
  train loss turun perlahan. Pola tak stabil ini khas model acak pada data terbatas — tidak ada
  representasi awal yang menuntun optimasi, sehingga val loss tak kunjung turun mulus.
- **B2 (Feature Extraction):** train & val loss **turun mulus dan nyaris berimpit** hingga ~0,2.
  Gap kecil ini menandakan generalisasi sehat; karena hanya kepala yang dilatih, model sulit overfit.
- **B3 (Progressive Fine-Tuning):** train loss **anjlok cepat mendekati 0** sedangkan val loss datar di
  ~0,20 lalu **sedikit menanjak** di epoch akhir — gejala awal overfitting. Inilah alasan early stopping
  memilih epoch ke-5 sebagai titik terbaik (val loss minimum) sebelum model mulai menghafal.

### C2. Pembahasan Analisis Komparatif

**(1) Disparitas B1 vs B2/B3 (Slide 10–12, 16).** Hasil menunjukkan jurang performa yang tajam: B1
(from scratch) hanya mencapai **Test Macro F1 0,3048**, sementara B2 melonjak ke **0,9201** dan B3 ke
**0,9363** — selisih sekitar **+0,61 hingga +0,63** poin F1. Dengan hanya ~2.000 citra, ResNet-18 acak tidak
punya cukup sinyal untuk membangun hierarki fitur visual (tepi → tekstur → bagian objek) dari nol;
early stopping bahkan menghentikan B1 pada epoch ke-4 karena validation loss berhenti membaik (val acc
mentok di 0,4209), indikasi kuat *underfitting/overfitting* pada data terbatas. Sebaliknya B2/B3 mewarisi
representasi ImageNet yang matang, sehingga tugas menyusut menjadi memetakan fitur siap-pakai ke 10 kelas —
jauh lebih mudah dan terbukti efektif pada *small-data regime*.

**(2) Trade-off FE murni vs FT bertahap (Slide 16).** B3 menang tipis namun konsisten atas B2
(**0,9363 vs 0,9201**, selisih **+0,0162** F1). Menariknya B3 juga **lebih cepat konvergen** (epoch 5,
113,6 s) dibanding B2 (epoch 18, 286,2 s): membuka `layer4` memberi kapasitas adaptasi domain sehingga
model menemukan solusi baik lebih awal, sementara B2 yang hanya melatih kepala butuh lebih banyak epoch
untuk "memeras" fitur beku. Dengan LR kecil (1e-4) pada `layer4`, fine-tuning tetap stabil tanpa merusak
backbone. Kesimpulan: pada data terbatas ini, **B3 adalah titik optimal** antara performa dan stabilitas;
B2 tetap pilihan aman bila ingin trainable params minimal (hanya 5.130) dan risiko nol.

**(3) Peran augmentasi B4 (Slide 20).** Perbandingan menunjukkan pola yang perlu dibaca hati-hati.
*Generalization gap* (train_acc − val_acc) justru tercatat **lebih kecil tanpa augmentasi (0,0471)**
dibanding **dengan augmentasi (0,0777)**. Namun **Test Macro F1 lebih tinggi dengan augmentasi
(0,9363 vs 0,9160, selisih +0,0203)**. Interpretasinya: tanpa augmentasi model konvergen sangat cepat
(berhenti epoch 4, hanya 70,2 s) dan train acc belum sempat memuncak sehingga gap terlihat kecil secara
artifisial; dengan augmentasi, tugas latih lebih sulit (train acc lebih rendah → gap tampak lebih besar)
tetapi model **menggeneralisasi lebih baik ke test set**. Jadi manfaat augmentasi di sini terbukti pada
**kenaikan performa test**, bukan sekadar angka gap. Ini menegaskan augmentasi sebagai *regularizer* yang
berguna meski efek pada data berlabel-sedikit tidak selalu terlihat lewat satu metrik tunggal.

### C3. Failure Analysis (Analisis Misklasifikasi)

Model terbaik dipilih otomatis berdasarkan Macro F1 tertinggi, yaitu **B3 (Progressive Fine-Tuning,
F1 0,9363)**. Berikut 8 misklasifikasi ber-*confidence* tertinggi pada test set (sumber:
`logs/failure_cases.csv`, gambar: `figures/failure_cases.png`):

| # | Ground Truth | Prediksi | Confidence | Hipotesis Penyebab (berdasarkan citra aktual) |
|---|--------------|----------|------------|--------------------|
| 1 | Abyssinian | Bengal | 0,978 | Wajah *tabby* cokelat dengan pola bulu *ticked* yang sangat menyerupai Bengal; latar hijau mencolok + watermark menambah distraksi tekstur |
| 2 | British Shorthair | Bombay | 0,960 | Kucing abu-kebiruan pada **pencahayaan kontras tinggi** sehingga sebagian tubuh jadi siluet gelap — model menebak warna solid gelap ala Bombay |
| 3 | Abyssinian | Bengal | 0,927 | Sama seperti #1: tekstur bulu *reddish/spotted* beririsan erat dengan Bengal |
| 4 | British Shorthair | Bombay | 0,911 | Kucing abu gelap pose frontal; warna bulu gelap dominan menutupi corak khas British Shorthair |
| 5 | American Bulldog | Beagle | 0,903 | Anak anjing *brindle* putih-cokelat di rumput; pola & ukuran tubuh menyerupai Beagle muda, fitur rahang khas Bulldog belum menonjol |
| 6 | Beagle | Basset Hound | 0,848 | **Pose ekstrem (kepala terbalik)** dengan telinga panjang terjulur; proporsi wajah jadi mirip Basset Hound |
| 7 | Abyssinian | Bengal | 0,822 | Wajah frontal *tabby* bermata besar — pengulangan kerancuan Abyssinian↔Bengal |
| 8 | Beagle | American Pit Bull Terrier | 0,815 | Citra **grayscale/hitam-putih** + anjing mengenakan baju; hilangnya informasi warna & oklusi tekstil menyembunyikan ciri khas Beagle |

**Pola yang terbaca jelas dari citra:** (a) **Ambiguitas antar-ras** mendominasi — pasangan
**Abyssinian ↔ Bengal** muncul 3× dan **British Shorthair ↔ Bombay** 2×, keduanya ras berpola/warna bulu
amat mirip. (b) **Faktor akuisisi citra** memperparah: pencahayaan kontras tinggi (#2), pose ekstrem (#6),
dan **citra grayscale + oklusi pakaian** (#8) yang menghapus fitur warna yang biasanya diskriminatif. Jadi
sumber error adalah kombinasi **batas diskriminatif alami antar ras serupa** dan **kondisi citra non-ideal**,
bukan noise acak. Implikasinya: perbaikan menuntut data tambahan khusus kelas rancu dan augmentasi yang
mewakili variasi pencahayaan/warna, bukan sekadar menambah epoch.

### C4. Rekomendasi Rekayasa dan Batasan Eksperimen

Berdasarkan bukti empiris (B3 = Macro F1 **0,9363**, tertinggi; B2 = 0,9201; B1 = 0,3048),
**rekomendasi untuk tim rekayasa industri pada skenario data terbatas:**
- **Strategi optimal: Progressive Fine-Tuning (B3)** dengan **differential LR** (`layer4`=1e-4, `fc`=1e-3) +
  augmentasi. Pada studi ini B3 unggul atas B2 (+0,0162 F1) **dan** lebih cepat konvergen (epoch 5 vs 18).
  Berlaku saat tersedia validation set memadai untuk memantau early stopping.
- **Alternatif aman: Feature Extraction (B2)** bila data lebih sedikit lagi, domain sangat dekat ImageNet,
  atau trainable params ingin ditekan ekstrem (hanya 5.130) — tetap kompetitif (0,9201) dengan risiko nol
  merusak backbone.
- **Hindari training from scratch (B1)** pada regime data terbatas: F1 hanya 0,3048, tidak layak produksi.
- Aktifkan **augmentasi yang menjaga invariansi label** (horizontal flip, rotasi ringan, color jitter);
  hindari transformasi yang melanggar semantik (vertical flip, rotasi ekstrem).
- Jika efisiensi inferensi/ukuran model kritis, pertimbangkan **EfficientNet-B0** (B5): F1 0,9190 nyaris
  setara B2 namun dengan **total params ~3× lebih kecil** (4,02 juta vs 11,18 juta) — trade-off menarik
  untuk *deployment* edge (Slide 24).

**Batasan (limitasi) studi — disampaikan jujur:**
- Pengujian berbasis **single seed (42)**; belum ada rerata ± simpangan baku lintas seed.
- Hanya **10 kelas** dan ~2.000 citra — generalisasi ke 37 kelas penuh belum diverifikasi.
- Anggaran komputasi terbatas (GPU T4, early stopping agresif); *hyperparameter* belum di-*tune* menyeluruh.
- B5 hanya satu arsitektur pembanding; klaim trade-off bersifat indikatif.

### C5. Eksplorasi Tambahan (Opsional) — Grad-CAM

Notebook memvisualisasikan **Grad-CAM pada blok `layer4`** sebelum (B2, backbone beku) dan sesudah (B3,
fine-tuned) pada empat citra test yang sama (`figures/gradcam_before_after.png`). Pengamatan aktual dari
heatmap:

- **Bengal (kolom 1 & 3):** fokus B2 dan B3 relatif mirip, keduanya menyoroti tubuh/pola bulu
  *spotted* — wajar karena pola bulu memang fitur paling diskriminatif untuk ras ini, dan fitur ImageNet
  sudah menangkapnya dengan baik.
- **Bombay (kolom 2):** terlihat **pergeseran paling jelas** — B2 fokusnya menyebar ke area tubuh/latar,
  sedangkan B3 **mengetat ke kepala/telinga** kucing. Ini contoh fine-tuning mempertajam perhatian ke
  bagian subjek yang relevan.
- **American Bulldog (kolom 4):** B2 menyoroti sekitar moncong namun agak terpengaruh latar (selimut putih);
  B3 **menurunkan fokus ke area wajah anjing** sehingga lebih terkonsentrasi pada subjek.

Secara umum, setelah fine-tuning (B3) peta aktivasi cenderung **lebih terfokus pada bagian diskriminatif
subjek** (wajah/kepala/pola bulu) dan sedikit menjauh dari latar belakang, meski pada citra ber-pola kuat
(Bengal) perbedaannya tipis. Pergeseran ini konsisten dengan kenaikan Macro F1 B3 dan menunjukkan `layer4`
menyesuaikan diri terhadap domain pet (merujuk best practice Slide 29).

---

## Lampiran — Pernyataan Penggunaan AI (AI Disclosure Statement)

- **Nama model/alat AI:** **Kiro** (asisten AI agentic di dalam IDE).
- **Ruang lingkup bantuan:** penstrukturan kerangka notebook & laporan, penulisan boilerplate pipeline
  PyTorch/torchvision (transforms, training loop, logging), referensi sintaksis, dan klarifikasi konsep
  (BatchNorm, residual connection, transfer learning). Kiro **tidak** menjalankan eksperimen; seluruh
  pelatihan dan angka hasil berasal dari eksekusi mandiri saya di Google Colab (GPU T4, seed 42).
- **Verifikasi mandiri:** notebook dijalankan `Run all` sampai selesai tanpa error; seluruh angka pada
  laporan dicocokkan langsung dengan `logs/metrics_log.csv` dan `logs/failure_cases.csv`; interpretasi
  kurva loss, failure analysis, dan Grad-CAM diverifikasi dengan mengamati gambar aktual pada `figures/`;
  penjelasan konseptual diperiksa silang dengan slide Sesi 05 dan literatur (Ioffe & Szegedy 2015;
  He et al. 2015; Howard & Ruder 2018).

## Referensi

1. Ioffe, S., & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.* ICML.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep Residual Learning for Image Recognition.* CVPR.
3. Howard, J., & Ruder, S. (2018). *Universal Language Model Fine-tuning for Text Classification (ULMFiT).* ACL.
4. Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for CNNs.* ICML.
5. Parkhi, O. M., Vedaldi, A., Zisserman, A., & Jawahar, C. V. (2012). *Cats and Dogs (Oxford-IIIT Pet Dataset).* CVPR.
6. Bahan tayang Sesi 05 — Advanced CNNs and Transfer Learning (Slide 5–30).
