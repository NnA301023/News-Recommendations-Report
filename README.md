# Laporan Project Machine Learning Terapan - Alif Ramadhan  

Apabla terjadi kesalahan dalam load jupyter file, berikut penulis bagikan versi online menggunakan google collabs :

[News Recommendation System](https://colab.research.google.com/drive/13G-Z-awm2AoOsfh36vLO3G_4Wd4R4CoJ?usp=sharing)
****

# Domain Proyek

  Mengetahui perkembangan dan segala informasi yang terjadi di seluruh penjuru dunia saat ini tidaklah sulit. Ada berbagai media yang bisa menjadi sarana untuk Anda mendapatkan informasi tentang hal-hal yang sedang terjadi di seluruh penjuru dunia. Inilah yang dinamakan berita. Anda mungkin sering melihat tayangan berita di berbagai media, seperti televisi, internet atau membaca di media cetak. 

  Berikut beberapa syarat suatu teks di dapat disebut sebagai berita :

  1. Berdasarkan Fakta 
  
  Informasi dalam berita yang disampaikan haruslah sesuai fakta yang sesungguhnya terjadi di lapangan. Berita tidak boleh dibuat berdasarkan karangan atau cerita fiktif.

  2. Aktual

  Informasi dalam berita yang disampaikan adalah informasi terkini atau terbaru. Hal ini bisa dibuktikan dengan jarak waktu antara berita disiarkan dengan kejadian yang diberitakan tidak berbeda terlalu jauh.

  3. Berimbang

  Dalam berita, informasi yang disampaikan tidak hanya harus berupa fakta namun juga berimbang. Berimbang maksudnya, fakta atau informasi yang disampaikan adalah informasi yang sebenarnya serta tidak memihak maupun memojokkan salah satu pihak. Dengan begitu masyarakat yang membaca atau melihat juga tidak akan terpengaruh.

  4. Lengkap
 
  Unsur terakhir yang harus ada dalam sebuah berita adalah lengkap. Artinya setiap informasi dalam berita harus disajikan secara lengkap, tidak ada yang disembunyikan atau dikurangi. Dengan begitu masyarakat atau khalayak luas yang membaca atau mendengarkan berita tidak menjadi bingung atas informasi yang disampaikan.

Berdasarkan pengertian dan syarat sebuah berita diatas, dapat dipahami gambaran berita secara umum, oleh karena itu penulis berinisiatif untuk membuat sistem rekomendasi berita sehingga dapat mempermudah pengguna dalam membaca berita berdasarkan beberapa metode yang dipelajari di [Dicoding](https://www.dicoding.com/academies/319/tutorials/17114).

  Menurut ([Nurkinan, 2017](https://journal.unsika.ac.id/index.php/politikomindonesiana/article/download/962/792/2748)) Media massa sebagai sarana penyampai informasi menyajikan berita-berita hangat dan aktual kepada khalayak, karena dalam pengaruh berita yang disajikan, media massa dapat membangun kontrol sosial yang ada di masyarakat. Baik dalam mengubah opini atau pandangan seseorang, mengubah sikap dan perilaku, membangun kepercayaan, bahkan mengubah paradigma kehidupan masyarakat. Kontrol sosial yang dibangun media, tujuannya ialah untuk mengawasi segala tindak tanduk pemerintah dalam menjalankan kewajibannya. 

  Kajian diatas juga memperkuat intuisi penulis dalam mengembangkan model sistem rekomendasi untuk menyajikan berita yang sekiranya sesuai dengan konten, riwayat pengguna, maupun kombinasi dari beragam fitur sehingga dampat pola pikir masyarakat dapat dipengaruhi oleh topik bacaan sehari-hari. 
****

# Pemahaman Bisnis 

## Problem Statement
1. Bagaimana cara mengimplementasikan model content based untuk rekomendasi berita ?
2. Bagaimana cara mengimplementasikan model collaborative filtering untuk rekomendasi berita ?
3. Bagaimana cara mengimplementasikan model hybrid untuk rekomendasi berita ?

## Goals Statement
1. Menyajikan rekomendasi berita menggunakan metode yang beragam
2. Mengevaluasi metode yang terbaik dalam penyajian berita

## Solution Statement
1. **Content Based**
  
  Pendekatan ini digunakan untuk metode yang akan mengambil informasi yang berguna dari item yang telah diekstraksi. Informasi ini harus dipastikan merupakan informasi yang baik dan dapat dipastikan akan menjadi relevan terhadap pengguna. Proses ektraksi terhadap item yang digunakan akan memperbesar kemungkinan munculnya item baru yang belum pernah terlihat sebelumnya. Pada dasarnya metode ini sangat bergantung pada perilaku pengguna. Asumsi utama di bawah pendekatan berbasis konten adalah bahwa item atau dokumen dapat diidentifikasi oleh serangkaian fitur yang diekstraksi langsung dari konten mereka.

2. **Collaborative Filtering**

  collaborative filtering adalah suatu konsep dimana opini dari pengguna lain yang ada digunakan untuk memprediksi item yang mungkin disukai/diminati oleh seorang pengguna.

3. **Hybrid Model**

  Hybrid recommender system adalah suatu teknik yang menggabukan beberapa teknik rekomendasi.

4. **TF-IDF (term frequency–inverse document frequency)**

  Term Frequency merupakan frekuensi kemunculan term i pada dokumen j dibagi dengan total term pada dokumen j.

5. **Cossine Similarity** 

  Metode Cosine Similarity adalah mengukur kemiripan antara dua dokumen atau teks. Pada Cosine Similarity dokumen atau teks dianggap sebagai vector. Pada penelitian ini, Cosine Similarity digunakan untuk menghitung jumlah kata istilah yang muncul pada halaman-halaman yang diacu pada daftar indeks.

6. **Matrix Factorization**

  Matrix Factorization adalah penguraian suatu matriks menjadi beberapa buah matriks. membentuk suatu himpunan konveks.

7. **Train test split**

  Train/test split adalah salah satu metode yang dapat digunakan untuk mengevaluasi performa model machine learning. Metode evaluasi model ini membagi dataset menjadi dua bagian yakni bagian yang digunakan untuk training data dan untuk testing data dengan proporsi tertentu.

8. **Feature Scaling ( Min Max )**
    
    Normalisasi data adalah proses membuat beberapa variabel memiliki rentang nilai yang sama, tidak ada yang terlalu besar maupun terlalu kecil sehingga dapat membuat analisis statistik menjadi lebih mudah. Perhatikan dua tabel berikut.

9. **Singular-Value Decomposition (SVD)**

  metode dekomposisi matriks untuk mereduksi suatu matriks menjadi bagian-bagian penyusunnya agar perhitungan matriks berikutnya menjadi lebih sederhana.
  
## Workflow Diagram

Untuk mempermudah pendefinisian langkah-langkah yang diperlukan dalam penelitian kali ini, penulis membuat visualisasi diagram tentang alur kerja secara keseluruhan.

(image here)
****

# Dataset Understanding

(image here)

Dataset [Article News](https://www.kaggle.com/gspmoreira/articles-sharing-reading-from-cit-deskdrop) berisi 2 file berbentuk csv yang mendukung pengimplementasian model sistem rekomendasi content based dan collaborative filtering. Dataset dipublikasikan oleh [Gabriel Moreira](http://about.me/gspmoreira) pada tahun 2017.

**Konteks**

  [Deskdrop](https://deskdrop.co/) adalah platform komunikasi internal yang dikembangkan oleh CI&T, yang berfokus pada perusahaan yang menggunakan Google G Suite. Di antara fitur-fitur lainnya, platform ini memungkinkan karyawan perusahaan untuk berbagi artikel yang relevan dengan rekan-rekan mereka, dan berkolaborasi di sekitar mereka.

**Konten**

  Kumpulan data yang kaya dan langka ini berisi sampel nyata dari log 12 bulan (Maret 2016 - Februari 2017) dari platform Komunikasi Internal CI&T (DeskDrop).
Saya berisi sekitar 73 ribu interaksi pengguna yang tercatat di lebih dari 3 ribu artikel publik yang dibagikan di platform.

Dataset ini memiliki beberapa karakteristik khusus: 

  1. Atribut item: URL asli artikel, judul, dan teks biasa konten tersedia dalam dua bahasa (Inggris dan Portugis).
  2. Informasi kontekstual: Konteks kunjungan pengguna, seperti tanggal/waktu, klien (aplikasi/browser asli seluler) dan geolokasi.
  3. Pengguna yang dicatat: Semua pengguna diharuskan untuk masuk ke platform, menyediakan pelacakan preferensi pengguna jangka panjang (tidak tergantung pada cookie di perangkat).
  4. Umpan balik implisit yang kaya: Jenis interaksi yang berbeda dicatat, sehingga memungkinkan untuk menyimpulkan tingkat minat pengguna pada artikel (mis. komentar > suka > tampilan).
  5. Multi-platform: Interaksi pengguna dilacak di berbagai platform (browser web dan aplikasi asli seluler)


# Exploratory Data Analysis

# Preprocessing

# Model Development

# Evaluation Metrics

# Conclusion & Future Work(s)

# References
