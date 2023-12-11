# **E-Commerce Churn Classification and Data Analysis**

#### contributor : Anastasia Istimedelin & Kennaldy Lukman


## **Business Problem Understanding**

Dalam era digital yang terus berkembang, industri e-commerce menjadi pilar utama perekonomian global. Meskipun memberikan kenyamanan bagi pelanggan, perusahaan e-commerce menghadapi tantangan Churn, yaitu kehilangan pelanggan. Churn dapat disebabkan oleh berbagai faktor seperti harga tinggi, kualitas buruk, pelayanan pelanggan buruk, persaingan ketat, atau perubahan kebutuhan pelanggan. Fenomena ini dapat mengakibatkan kehilangan pendapatan, biaya tambahan untuk mendapatkan pelanggan baru, dan penurunan kepuasan pelanggan.

Untuk mengatasi Churn, perusahaan e-commerce perlu memahami faktor-faktor yang mempengaruhi keputusan pelanggan untuk berhenti menggunakan layanan. Dengan pemahaman ini, perusahaan dapat mengambil langkah-langkah pemasaran atau inovasi yang sesuai untuk mengurangi tingkat Churn. Hal ini akan membantu perusahaan meningkatkan retensi pelanggan, kepuasan pelanggan, dan mencapai pertumbuhan berkelanjutan di pasar yang semakin kompetitif.

Sehingga terdapat beberapa pertanyaan yang perlu dipecahkan :

1. **Identifikasi Faktor Churn**<br>
Apa saja faktor-faktor yang mempengaruhi keputusan pelanggan untuk meninggalkan platform E-commerce (Churn)?
2. **Pengembangan Model Prediksi**<br>
Bagaimana perusahaan dapat mengembangkan model prediksi yang efektif untuk menidentifikasi pelanggan yang perpotensi Churn?

### **Metric Evaluation**

<img width="574" alt="image" src="https://github.com/tasiamdln/E_Commerce_Churn_Analysis_ML/assets/139005822/a267c08e-b62b-4e2f-a4dd-ff4f45fa718e">

**Type 1 Error (False Positive):**
<br>
Model memprediksi Churn tanpa sebab, mengakibatkan langkah-langkah pencegahan tidak perlu.
Dampak: Biaya tambahan perusahaan tanpa manfaat sepadan.

**Type 2 Error (False Negative):**
<br>
Model tidak memprediksi Churn, menyebabkan kehilangan pelanggan dan pendapatan yang dapat dipertahankan.
Biaya terlibat: Tinggi karena kehilangan pelanggan yang seharusnya dipertahankan.

**Fokus Evaluasi Model:**
<br>
Pemilihan metrik: F2-Score dengan bobot lebih besar pada False Negative.
Alasan pemilihan: Biaya False Negative lebih dari 2x lebih besar dibandingkan dengan False Positive.
Manfaat: Meningkatkan keakuratan dalam mengidentifikasi pelanggan berpotensi Churn, memungkinkan tindakan preventif dan retensi pelanggan.

## **Data Overview**

Dataset source : <https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction/data>


**Atribute Information**

| **Data** | **Variable** | **Description** |
|:---:|:---:|:---:|
| E Comm | CustomerID | ID unik pelanggan |
| E Comm | Churn | Tanda Churn |
| E Comm | Tenure | Masa pelanggan bergabung dalam organisasi (satuan bulan)|
| E Comm | PreferredLoginDevice | Perangkat login yang disukai pelanggan |
| E Comm | CityTier | Tingkatan kota |
| E Comm | WarehouseToHome | Jarak antara gudang ke rumah pelanggan (km)|
| E Comm | PreferredPaymentMode | Metode pembayaran yang disukai pelanggan |
| E Comm | Gender | Jenis kelamin pelanggan |
| E Comm | HourSpendOnApp | Jumlah jam yang dihabiskan pelanggan di aplikasi seluler atau situs web |
| E Comm | NumberOfDeviceRegistered | Total jumlah perangkat yang terdaftar pada pelanggan tertentu |
| E Comm | PreferedOrderCat | Kategori pesanan yang disukai pelanggan dalam sebulan terakhir |
| E Comm | SatisfactionScore | Skor kepuasan pelanggan terhadap layanan |
| E Comm | MaritalStatus | Status pernikahan pelanggan |
| E Comm | NumberOfAddress | Total jumlah alamat yang ditambahkan pada pelanggan tertentu |
| E Comm | Complain | Ada/ tidak keluhan yang diajukan dalam sebulan terakhir |
| E Comm | OrderAmountHikeFromlastYear | Persentase peningkatan pesanan dari tahun lalu |
| E Comm | CouponUsed | Total jumlah kupon yang digunakan dalam sebulan terakhir |
| E Comm | OrderCount | Total jumlah pesanan dalam sebulan terakhir |
| E Comm | DaySinceLastOrder | Hari sejak pesanan terakhir oleh pelanggan |
| E Comm | CashbackAmount | Rata-rata cashback dalam sebulan terakhir (Rupee) |

Note :
- Dataset tidak seimbang antara yang Churn dengan yang Tidak Churn
- Setiap baris merepresentasikan informasi terkait pelanggan yang berbelanja di platform E-Commerce


## **Exploratory Data Analysis**

1. Data yang kita miliki tidak seimbang: lebih banyak user yang **stay/tidak churn (4682)** daripada yang **Churn (948)**
2. Dalam korelasi fitur numerik ditemukan adanya korelasi yang kuat antara CuponUsed vs OrderCount. 
3. Waktu pelanggan bergabung ke dalam E-Commerce (Tenure) mempengaruhi pelanggan akan Churn/ Tidak. Pelanggan yang bergabung selama 0-21 bulan cenderung untuk Churn. Pelanggan yang paling signifikan untuk Churn adalah pelanggan yang baru bergabung 0-1 bulan, dengan persentase Churn sebesar 51-54%.
4. Jarak antara Gudang ke rumah pelanggan (WarehouseToHome) mempengaruhi pelanggan akan Churn/ Tidak. Pelanggan yang memiliki jarak rumah dengan gudang mencapai >= 21 km lebih cenderung untuk Churn, dengan persentase 20% - 21%. Pelanggan yang hanya memiliki jarak dari rumah ke Gudang <= 10 km memiliki persentase Churn yang rendah, sebesar 15%. 
5. Jumlah device yang didaftarkan ke platform E-Commerce (NumberofDevice) mempengaruhi pelanggan akan Churn/ Tidak. Pelanggan yang mendaftarkan banyak device lebih banyak yang Churn dibandingkan dengan yang sedikit.
6. Jumlah alamat yang didaftarkan ke platform E-Commerce (NumberofAdress) mempengaruhi pelanggan akan Chrun/Tidak. Namun, grafik menunjukkan data yang fluktuatif. Sehingga semua pelanggan, baik yang mendaftarkan banyak alamat maupun sedikit tetap berpotensi untuk Churn. Akantetapi, jumlah alamat yang didaftarkan memiliki tingkat Churn yang berbeda-beda.
7. Waktu/ tanggal pelanggan melakukan pesanan (DaySinceLastOrder) mempengaruhi pelanggan akan Churn/Tidak. Pelanggan yang melakukan order di akhir bulan cenderung untuk Churn, dibandingkan dengan awal bulan.
8. Jumlah oder/ pesanan (OrderCount) mempengaruhi pelanggan akan Churn/Tidak. Grafik yang ditunjukkan fluktuatif, sehingga pelanggan yang melakukan order dengan jumlah kecil belum tentu akan Churn, begitu juga sebaliknya.
9. Jumlah Cashback yang diterima (CashbackAmmount) mempengaruhi pelanggan akan Churn/ Tidak. Pelanggan yang mendapatkan Cashback sedikit cenderung Churn, dibandingkan dengan yang mendapatkan banyak Cashback.
10. Terdapat beberapa fitur yang tidak memiliki pengaruh signifikan terhadap Churn, yaitu : HourSpendOnApp, OrderAmountHikeFromlastYear, dan CouponUsed.
11. Seluruh fitur kategorikal data kita (PreferredLoginDevice, PrefferedPaymentMode, Gender, PreferedOrderCat, MaritalStatus, Complain, CityTier, dan StatisfactionScore) memiliki keterikatan dengan Churn/ Tidak.
	- Pelanggan yang menggunakan Computer lebih banyak Churn dibandingkan yang menggunakan Mobile Phone
	- Metode pembayaran COD memiliki nilai Churn yang paling tinggi dibandingkan dengan metode pembayaran lainnya.
	- Pelanggan laki-laki lebih banyak Chrun dibandingkan pelanggan perempuan
	- Pelanggan yang membeli produk Mobile Phone merupakan pelanggan yang paling banyak Churn dibandingkan dengan produk lainnya
	- Pelanggan dengan status Single merupakan pelanggan yang paling banyak Churn
	- Pelanggan yang pernah mengajukan Complain lebih cenderung Churn dibanding yang tidak mengajukan Complain
	- Pelanggan yang berada di CityTier 3 paling banyak melakukan Churn dibanding CityTier lainnya.
	- Pelanggan yang memberikan nilai 5 pada SatisfactionScore justru lebih banyak Churn dibandingkan dengan pelanggan yang memberikan nilai < 5


## **Machine Learning**

**X (Feature)** : Churn<br>
**y (Target)** : Semua kolom selain CustomerID

### **Preprocessing**

**Encoding**
- OneHotEncoder: PreferredLoginDevice, Gender, MaritalStatus
    - Kita gunakan OneHotEncoder karena terdapat sedikit unique values (2-3) untuk kolom tersebut.
- BinaryEncoder: PreferredPaymentMode, PreferedOrderCat
    - Kita gunakan BinaryEncoder karena terdapat banyak unique values (5) untuk kedua kolom tersebut.

**Scaling**
- RobustScaler: semua kolom numerikal, kecuali CityTier, Complain, dan SatisfactionScore
    - CityTier, Complain dan SatisfactionScore merupakan kolom kategorikal yang ordinal. Tidak perlu discale.
    - Kita gunakan RobustScaler karena semua data numerik yang kita gunakan bersifat tidak normal.

### **Cross Validation**

Kita akan menguji semua algorithm ini dan memilih yang terbaik untuk model kita. <br>
Beberapa model yang akan kita gunakan:
1. LogisticRegression
2. DecisionTreeClassifier
3. KNeighborsClassifier
4. Ensemble Models:
    - VotingClassifier
    - StackingClassifier
    - BaggingClassifier
    - RandomForestClassifier
    - AdaBoostClassifier
    - XGBoostClassifier
    - GradientBoostingClassifier

Model terbaik adalah XGBClassifier, karena:
1. Mean F2-score yang tertinggi: **0.8487**
2. Standard deviation yang terendah: **0.0168**

### **Hyperparameter Tuning**

1. **f2-score naik dari 0.9674 ke 0.9704**
2. **ada penurunan False Positive (FP) menjadi True Negative (TN)**, dari 7 FP ke 4 FP.
3. **ada penurunan jauh dari loss tanpa modeling ke loss menggunakan tuned model.** <br>

Maka kita akan menggunakan model yang sudah dituning.

### **Imbalanced Data Treatment**

Menggunakan 4 metode resampling dan melihat apakah resampling akan berpengaruh ke model kita:
1. **RandomUnderSampler**
    - menghapus dataset dari kelas mayoritas secara random
2. **RandomOverSampler**
    - menduplikasi dataset dari kelas minoritas secara random
3. **NearMiss**
    - mencari data dari kelas mayoritas yang dekat dengan kelas minoritas.
4. **SMOTE**
    - singkatan dari **S**ynthetic **M**inority **O**ver-sampling **TE**chnique
    - mencari data dari kelas minoritas, menjadikan mereka "contoh", dan membuat data baru berdasarkan "contoh".

Hasilnya :

1. **f2-score naik dari 0.9704 ke 0.9726**
2. **ada sedikit penurunan dari False Negative (FN) ke True Positive (TP)**, dari 6 FN ke 5 FN.
3. **ada kenaikan dari True Negative (TN) ke False Positive (FP)**, dari 4 FP ke 6 FP.
4. **ada penurunan loss dari tuned model menjadi model yang sudah diresampling.**


### **Feature Selection**

Ada peningkatan signifikan dari performa model ketika k-value = 22. <br>
Tidak ada perbedaan jauh dari f2-score trainset dan testset ketika k-value = 22. <br>
Artinya, ada beberapa (4) feature yang boleh dihapus untuk meningkatkan performa model kita.<br>
(Ada total 26 features di model kita setelah preprocessing.)

Berdasarkan feature selection dari SelectKBest, beberapa insight bisa didapatkan:
- Feature yang paling berpengaruh adalah:
    1. Tenure
    2. Complain
    3. CashbackAmount
    4. MaritalStatus
    5. PreferedOrderCat
<br><br>
- Feature yang paling tidak berpengaruh dan boleh diignore untuk modeling adalah:
    1. CouponUsed
    2. OrderAmountHikeFromLastYear
    3. PreferredPaymentMode

Hasilnya :

1. **f2-score naik secara signifikan, dari 0.9726 ke 0.9926**
2. **ada penurunan drastis dari False Negative (FN) ke True Positive (TP)**, dari 5 FN ke 0 FN.
3. **ada penurunan sedikit dari True Negative (TN) ke False Positive (FP)**, dari 6 FP ke 7 FP.
4. **ada penurunan drastis dari loss saat menggunakan selection model.**

## **Kesimpulan & Rekomendasi**

### **Kesimpulan**
Model machine learning kita, dengan nilai f2-score yang sangat tinggi (**99.26%**) cocok dipakai untuk memprediksi apakah customer akan churn atau tidak, dan bisa mengurangi loss yang akan dialami oleh perusahaan e-commerce.

### **Rekomendasi**

**Berdasarkan Analysis**
1. **Onboarding Pelanggan Baru:**
    - Sediakan panduan dan tutorial interaktif.
    - Berikan diskon khusus untuk pembelian pertama.
    - Fokus pada pemeliharaan pelanggan baru selama 2 tahun.
2. **Optimalisasi Pengiriman:**
    - Tawarkan opsi pengiriman cepat dan diskon ongkos kirim.
    - Perhatikan khusus pada pembayaran COD.
3. **Pengelolaan Multi-Device:**
    - Pastikan konsistensi UI/UX di semua perangkat.
    - Fasilitasi login yang mudah dan responsivitas platform.
4. **Promosi & Cashback Tepat Sasaran:**
    - Sesuaikan diskon dengan preferensi pelanggan.
    - Lakukan promosi di CityTier 3 dan untuk pelanggan Single.
    - Berikan insentif kepada pelanggan yang berpotensi Churn.
5. **Penanganan Complain Efektif:**
    - Lakukan evaluasi rutin terhadap penanganan keluhan.
    - Kompensasi seperti diskon dapat diberikan.
6. **Strategi Retensi Pelanggan:**
    - Terapkan program loyalitas dan reward.
    - Sediakan penawaran khusus untuk pelanggan setia.
    - Fokus pada pencegahan Churn dan perawatan pelanggan.

**Untuk Bisnis, berdasarkan Model:**
1. Ada 2 faktor yang paling signifikan yang mempengaruhi churn atau tidaknya seorang customer, yaitu **Tenure** dan **Complain**.
    - Perusahaan sebaiknya memberi perhatian yang lebih terhadap customer yang telah menggunakan layanan kita untuk jangka waktu yang lama, agar mempertahankan customer setia dari e-commerce kita.
    - Perusahaan juga harus memprioritaskan penanganan keluhan customer dengan baik, seperti komunikasi yang efektif dan solusi yang memuaskan, agar customer yang mengomplain e-commerce kita tidak churn.
2. Model ini bisa digunakan oleh bisnis untuk mengurangi loss dari churn atau tidaknya seorang customer.
    - Pendapatan tambahan dari model ini boleh digunakan bisnis untuk dialokasikan ke peningkatan jumlah customer e-commerce.

**Untuk Model:**
1. **Data Imbalance**<br>
  Data yang diambil tidak balanced, karena lebih banyak data dari customer yang tidak churn dibandingkan dengan yang churn.
      - Diperlukan lebih banyak data dari customer yang churn, agar datanya menjadi lebih balanced dan tidak bias.<br>
2. **Faktor Tidak Berpengaruh**<br>
  Beberapa faktor boleh dihapus dari modeling, seperti **CouponUsed, PreferedPaymentMode dan OrderAmountHikeFromLastYear**, karena tidak berpengaruh besar terhadap performa model kita.
3. **Kekurangan Kuantitas Data**<br>
  Model kita membutuhkan data yang lebih banyak. Dataset kita hanya memiliki 5630 data. Dataset yang lebih besar boleh menciptakan sebuah model yang lebih akurat.
