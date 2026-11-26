# Analiz planı (DONDURULMUŞ)

Sürüm 1.0 — dondurma tarihi 2026-11-22. Bu dosyanın SHA-256 özeti `docs/analiz_plani.sha256` dosyasındadır. Git kurulu olmadığından dondurma, git işlemesi yerine özet ve tarih damgasıyla belgelenmiştir; Git kurulunca ilk işleme bu sürümle yapılacaktır. Bu tarihten sonraki her değişiklik Bölüm 12'ye **"plan dışı"** olarak yazılır.

## 1. Soru ve hipotezler
- **H1 (KAN):** KAN içeren bir model (U-Net-KAN veya ConvKAN), en iyi CNN/U-Net temel modeline göre **eşit veya daha az parametreyle**, LUDB iç testinde ortalama F1 bakımından aşağı kalmaz. Aşağı kalmama sınırı: fark −1,0 yüzde puan; karar için farkın %95 bootstrap GA alt sınırı kullanılır. Ayrıca sınır hatası (ortalama |hata|) daha kötü değildir. Öğrenilen tek değişkenli fonksiyonlar görselleştirilir ve yorumlanır (nitel ikincil sonuç).
- **H2 (DEQ):** Örtük türevli DEQ, aynı parametreli açılmış (unrolled, tam geri yayılım) karşılığına göre:
  - (a) daha düşük eğitim belleği kullanır (tepe RSS artışı, aynı batch ve uzunluk);
  - (b) LUDB iç testinde ortalama F1'de aşağı kalmaz (sınır −1,0 puan);
  - (c) zor vakalarda daha çok iterasyon yapar: kayıt başına çıkarım iterasyonu ile zorluk göstergeleri arasındaki Spearman ρ > 0 (tek yönlü, α = 0,05). Zorluk göstergeleri: tahmini gürültü düzeyi (yüksek frekans gücü oranı), patolojik ritim grubu (Mann–Whitney) ve kayıt başına hata.
- **H3 (dış genelleme):** LUDB'de eğitilen önerilen modeller, dış veri tabanlarında patolojik ritimlerde P tespitinde en iyi temel modelden daha iyidir.

## 2. Veri ve sızıntı kuralları
- **Eğitim/ayar:** yalnızca LUDB. **Dış setler (QTDB, BUT PDB, pwave) hiçbir ayar, model seçimi veya eşik seçiminde kullanılmaz;** yalnızca dondurulmuş modellerle bir kez değerlendirilir.
- **Bölme** (`configs/splits.json`, tohum 2026, ritim grubuna göre tabakalı, kişi düzeyi):
  - İç test: 41 kişi.
  - Kalan 159 kişi: 5 kat (30–33 kişi).
  - Aynı kişinin 12 derivasyonu daima aynı bölümdedir.
- **Normalizasyon:** derivasyon-sinyali başına sağlam standartlaştırma (medyan ve IQR/1,349). Başka kayıtların istatistiğini kullanmaz, dolayısıyla sızıntı yoktur. Gerekçe: setlerin birimleri farklı (BUT PDB µV, diğerleri mV), bu yüzden eğitim istatistiğiyle küresel ölçekleme dış setlere taşınamaz. PROMPT'taki "eğitim istatistikleriyle normalizasyon" yerine bu kullanılır; gerekçeli tasarım kararıdır.
- **Ön işleme:** 250 Hz'e polifaz yeniden örnekleme; 0,5–40 Hz Butterworth bant geçiren süzgeç (2. derece, sıfır faz). 500 Hz ve 12 derivasyonlu girdi ablasyondadır.

## 3. Görev, etiket ve son işlem (bütün modellerde aynı)
- **Görev:** örnek başına 4 sınıf (arka plan / P / QRS / T).
- **Etiket maskesi:**
  - LUDB'de her derivasyonun ilk ve son etiketi arasındaki bölge dışında kalan örnekler "yoksay" (−1) olarak işaretlenir; kayıt kenarları etiketsizdir.
  - Başlangıcı veya bitişi eksik dalgalarda tepenin ±100 ms çevresi de yoksay sayılır.
- **Son işlem** (`src/postprocess.py`):
  - Aynı sınıfın 20 ms'den kısa boşlukları birleştirilir.
  - En kısa süreler: P 40, QRS 40, T 60 ms.
  - Başlangıç/bitiş: koşunun (run) uçları.
  - P/T tepesi: uçları birleştiren doğrudan en büyük mutlak sapma.
  - QRS tepesi: en büyük değer (R).
  - P→QRS→T sıra kısıtı **uygulanmaz**; AV blok, AF ve ektopide bu sıra bozulur.
- **Uzun kayıtlar:** 2048 örneklik pencere, 1024 kaydırma, Hann ağırlıklı olasılık ortalaması.
- **Kahin kontrolü:** gerçek maske son işlemden geçirildiğinde başlangıç/bitiş hatası 0,00 ms, Se ≥ %99,5 (`results/oracle_check_ludb.json`).

## 4. Değerlendirme kuralları
- **Eşleştirme:** aynı nokta türü içinde, |Δ| ≤ tolerans koşuluyla, artan |Δ| sırasına göre açgözlü birebir eşleştirme. TP: eşleşen, FN: eşleşmeyen referans, FP: değerlendirme bölgesindeki eşleşmeyen tahmin.
- **Birincil tolerans: ±150 ms** (literatürle karşılaştırma için).
- **İkincil:** P tepesinde ±50 ms. Ayrıca CSE uyumu: hata SD'sinin 2σ_CSE ile karşılaştırılması (P_on 10,2; P_off 12,7; QRS_on 6,5; QRS_off 11,6; T_off 30,6 ms).
- **Metrikler:** Se, PPV, F1 (%), ortalama hata ± SD (ms), ortalama |hata|; hepsi kayıt (kişi) düzeyinde bootstrap ile %95 GA (10 000 tekrar).
- **Değerlendirme bölgeleri:**
  - LUDB: derivasyonun ilk–son etiket aralığı. 9 nokta.
  - QTDB: `q1c`. Referans yalnızca etiketli vuruşlar; FP yalnızca etiketli vuruş pencerelerinde (QRS tepesi −350/+550 ms) sayılır. T_on yalnızca etiketli olanlarda; 7 nokta (P on/peak/off, QRS on/off, T peak/off). `q2c` (11 kayıt, P yok) kullanılmaz.
  - BUT PDB ve pwave: yalnızca P_peak; bölge [ilk P − 0,5 s, son P + 0,5 s].
- **Derivasyon politikası (dış setler):**
  - **Birincil:** "her derivasyon": iki derivasyon ayrı ayrı değerlendirilip havuzlanır. Kahin seçimi yoktur.
  - **İkincil:** "en iyi derivasyon": QTDB'de nokta başına en düşük |hata|, P setlerinde en yüksek F1. Bu literatürle karşılaştırma içindir ve iyimserdir.
- **Çakışma:** BUT PDB'nin 9 kesiti pwave kayıtlarıyla çakışır. İki set ayrı raporlanır; birleşik P analizinde çakışan 9 BUT PDB kesiti çıkarılır.

## 5. Modeller
- **Temel modeller:**
  - Martínez türü dalgacık (NeuroKit2 `dwt`, eğitimsiz)
  - 1B U-Net
  - CNN-BiLSTM (DENS-ECG çizgisi)
  - Yamalı Transformer
- **Önerilen modeller:**
  - U-Net-KAN: darboğazda 2 KAN katmanı ve KAN sınıflandırma başlığı; B-spline, ızgara 8, derece 3.
  - ConvKAN: tamamen KAN evrişimli; RBF tabanı.
  - DEQ: Anderson ileri çözücü (m = 5, tolerans 1e-3, en çok 30 iterasyon), örtük geri yayılım (sabit nokta yinelemesi, en çok 30).
- **H2 kontrolleri:** DEQ-unrolled (aynı blok, 8 iterasyon, tam geri yayılım); açık yığın (ağırlık paylaşımsız 2 blok).
- **İsteğe bağlı:** DEQ gövdesi ve KAN başlığı. Yalnızca ayrı sonuçlar raporlandıktan sonra denenir; ikincil/keşifsel.
- **KAN uygulaması:** kendi B-spline (efficient-KAN formülasyonu) ve RBF (FastKAN) uygulamamız. Hız ve bellek karşılaştırması Faz 4'te verilir. pykan, yavaşlığı ve CPU bütçesi nedeniyle yalnızca hız karşılaştırmasında denenir (kurulabilirse).

## 6. Eğitim protokolü (bütün DL modellerinde aynı)
- Girdi: 1024 örneklik (4,1 s) rastgele kesit, tek derivasyon.
- Veri artırma: genlik ×[0,6; 1,6], taban kayması, beyaz gürültü, 50 Hz girişim.
- Kayıp: çapraz entropi (yoksay = −1). Optimizasyon: AdamW (ağırlık çürümesi 1e-4), OneCycle.
- Bütçe: 40 epoch × 60 adım × batch 32; gradyan kırpma 1,0.
- Model seçimi: 5 epoch'ta bir, doğrulama katında 6 başlangıç/bitiş noktasının ortalama F1'i.
- **Ayar bütçesi (eşit):** her model için öğrenme hızı ∈ {1e-3, 3e-3}, yalnızca **0. katın** doğrulama verisinde seçilir. Başka ayar yok.
- **Nihai modeller:** her model için 5 kat modeli. İç test ve dış setlerde **5 modelin olasılık ortalaması (topluluk)** birincil, kat bazlı sonuçlar değişkenlik için raporlanır. Tohum 1. Zaman kalırsa ana modellerde tohum 2 ve 3 de çalıştırılır (ikincil).
- İşlemci: `torch.set_num_threads(6)`. Her eğitim için duvar süresi, işlemci süresi ve tepe RSS `experiments/log.csv` dosyasına yazılır.

## 7. Uç noktalar
- **Birincil uç nokta:** BUT PDB'de P_peak F1 (±150 ms, her derivasyon havuzlanmış, 50 kayıt, topluluk modeli).
- **Anahtar ikincil uç noktalar:**
  1. pwave'de P_peak F1.
  2. QTDB'de P_on/P_off F1 ve ortalama |hata|.
  3. BUT PDB alt grupları: AV blok (BI/BII/BIII), ektopi (A/V), AF/AFL, diğer.
  4. LUDB iç testinde 9 noktanın Se/PPV/F1/hata değerleri ve 6 başlangıç/bitiş noktasının ortalama F1'i (H1 ve H2b için).
  5. ±50 ms P tepesi ve CSE uyumu.
- **Verimlilik:**
  - Parametre sayısı ve FLOP (10 s kayıt).
  - CPU çıkarım gecikmesi (10 s, tek iş parçacığı ve 6 iş parçacığı, medyan / 50 tekrar).
  - Eğitim tepe belleği (ayrı süreçte, batch 32 × 1024).
  - DEQ iterasyon dağılımı.

## 8. İstatistik
- **Birincil aile (Holm düzeltmesi, 3 karşılaştırma):** U-Net-KAN, ConvKAN ve DEQ'nin her biri, en iyi temel modele karşı birincil uç noktada.
  - **"En iyi temel model":** LUDB CV ortalama doğrulama F1'i en yüksek olan DL temel modeli. Dış setlere bakılmadan seçilir.
  - **Test:** kayıt düzeyinde eşleştirilmiş bootstrap (10 000 tekrar). F1 farkının iki yönlü p değeri ve %95 GA. Destekleyici olarak kayıt başına F1 üzerinde Wilcoxon işaretli sıra testi.
- **İkincil karşılaştırmalar:** aynı yöntemle yapılır, ayrı ailelerde Holm düzeltmesi uygulanır; "keşifsel" diye etiketlenir.
- **H2c:** Spearman ρ ve Mann–Whitney (tek yönlü).
- Bütün GA'lar kayıt/kişi düzeyinde yeniden örneklemeyle hesaplanır.

## 9. Ablasyonlar (LUDB CV, 5 kat; kayıt dışı setlere ancak dondurulmuş ana analizden sonra bakılır)
1. KAN ızgara boyutu {4, 8, 16} ve spline derecesi {1, 3}.
2. KAN konumu: {yalnız başlık, yalnız darboğaz, ikisi}.
3. KAN tabanı: B-spline ve RBF.
4. DEQ çözücü toleransı {1e-2, 1e-3, 1e-4} ve iterasyon sınırı {10, 30, 60} (çıkarımda).
5. DEQ geri yayılımı: örtük ve phantom (1 adım).
6. Derivasyon sayısı: tek derivasyon ve 12 derivasyon girdisi (yalnızca LUDB).
7. Girdi frekansı: 250 ve 500 Hz.

## 10. Olumsuz sonuçlar
Hipotezler desteklenmezse bu açıkça raporlanır. Uç noktalar sonuçlara bakıldıktan sonra değiştirilmez.

## 11. Raporlama
Bütün sayılar `results/` altındaki JSON dosyalarından otomatik tablo üreten betiklerle alınır. Elle sayı kopyalanmaz.

## 12. Plan dışı değişiklikler günlüğü
| Tarih | Değişiklik | Gerekçe |
|---|---|---|
| – | – | – |
