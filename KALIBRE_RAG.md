# Kalibre RAG — fikir ve sonraki adım

Bu belge iki şeyi anlatır: (1) bu projenin RAG'de neyi değiştirebileceğini ve neyin kanıtlanıp neyin
kanıtlanmadığını, (2) bunu gerçek kitaplarda kanıtlamak için sonraki adımın ayrıntılı planını.
Projenin genel anlatımı için `PROJE.md`, tüm sonuç tabloları için `README.md`.

---

## 1. Bu neyi değiştiriyor?

RAG'i baştan icat etmiyor; RAG'in en zayıf halkası olan **arama adımını** değiştiriyor.

```
Normal RAG:
  soru → embedding ile en benzer 5 paragraf → LLM'e ver → LLM cevap yazar
         (cevap belgelerde olmasa bile yine 5 paragraf gider, LLM uydurabilir)

Kalibre RAG:
  soru → her paragraf için "bu soruyu cevaplıyor mu?" olasılığı
       → eşiği (ör. %50) geçenler LLM'e gider (bazen 1, bazen 3 paragraf)
       → hiçbiri geçmezse LLM hiç çağrılmaz: "Bu belgelerde cevap yok"
```

İndeksleme önceden yapılır (embedding gibi); soru gelince model sadece soruyu okur.

| Normal RAG'in sorunu | Kalibre RAG'de |
|---|---|
| Cevap yoksa bile paragraf gönderir → LLM uydurur | "Cevap yok" der, LLM çağrılmaz |
| Her soruda sabit sayıda paragraf → gereksiz token maliyeti | Sadece eşiği geçenler gider |
| Birleşik sorular zor ("hem X'ten bahseden hem Y'yi eleştiren") | Olasılıklar çarpılır |
| Sayamaz ("kaç bölümde geçiyor?") | Olasılıklar toplanır |
| Sadece "benzer mi?" | Karar soruları da sorulabilir ("bu paragraf X'i eleştiriyor mu?") |

## 2. Ne yeni, ne değil?

**Yeni olmayan parçalar:** kelime düzeyi vektörlerle arama (ColBERT, 2020), büyük modelden küçüğe distillation
(Margin-MSE, TAS-B), iki aşamalı arama (PLAID), RAG'de "cevap verilemez" tespiti üzerine araştırmalar,
olasılık benzeri skor veren yeniden sıralayıcılar.

**Yeni olabilecek kısım — birleşim:** embedding hızında önceden indekslenen, ama çıktısı benzerlik değil
**güvenilir (kalibre) bir olasılık** olan ve bu olasılıkla "yok" deme, birleşik sorgu ve sayma yapabilen bir arama
modeli. Bildiğimiz kadarıyla bu kombinasyon pek çalışılmamış — "yeni" demeden önce dikkatli bir literatür taraması
şart (özellikle: calibrated retrieval, selective/abstaining RAG, answerability detection, ColBERT kalibrasyonu).

## 3. Ne kanıtlandı, ne kanıtlanmadı?

| Soru | Durum |
|---|---|
| Sentetik emlak verisinde fikir çalışıyor mu? | **Evet** — görülen sorularda AUC 0,999, ECE 0,003; "sonuç yok" %100 |
| Görülmemiş kavramlara genelleşiyor mu? | **Büyük ölçüde** — v3: 0,903 (v2: 0,665) |
| Gerçek kitaplarda çalışıyor mu? | **Bilinmiyor.** Eğitimsiz denemede tek bir sinyal: 9 soruda "yok" ayrımı doğru (anekdot) |
| bge-m3'ün sıralama kalitesine yetişiyor mu? | **Henüz hayır** — kitap göreviyle hiç eğitilmedi |
| LLM maliyetini ve uydurmayı azaltıyor mu? | **Ölçülmedi** |

**Nerede işe yarar?** Yanlış cevabın pahalı olduğu yerlerde (hukuk, sağlık, şirket içi belgeler, müşteri desteği)
"bilmiyorum" diyebilmek; koşullu aramalarda (emlak, e-ticaret, iş ilanı ↔ CV).

**Dürüst risk:** Sadece "en alakalı paragrafı getir" işinde bge-m3 gibi yüz milyonlarca örnekle eğitilmiş modelleri
geçmek zor. Değerimiz sıralamadan çok kalibrasyonda olacak; kalibrasyon farklı kitaplara/konulara taşınmazsa fikrin
değeri azalır. Sonraki deneyin cevaplaması gereken asıl soru bu.

---

## 4. Sonraki adım: kitap görevinde eğitim ve kanıt

### 4.1 Görev

Tek bir evet/hayır sorusu: **"Bu paragraf bu soruyu cevaplıyor mu?"** — `P(evet | soru, paragraf)`.
Mimari v3 (kelime vektörleri + sorudan kalibrasyon); emlaktaki `noul` sorusunun yerini bu alır.

### 4.2 Veri (eğitim — kitaplar değil, genel Türkçe metin)

| Kaynak | Ne verir | Not |
|---|---|---|
| Türkçe soru-cevap setleri (TQuAD, XQuAD'ın Türkçe bölümü, SQuAD'ın Türkçe çevirisi) | soru + cevabı içeren paragraf (pozitif) | lisans ve erişim kontrol edilecek |
| Aynı belgenin diğer paragrafları | zor negatifler (konu aynı, cevap yok) | ince farkı öğretir |
| bge-m3'ün getirdiği ama yanlış paragraflar | zor negatifler | "benzer ama cevaplamıyor" |
| Türkçe Wikipedia paragrafları + LLM ile üretilmiş sorular | çeşitlilik: tarih, bilim, felsefe, hukuk, sağlık… | konu çeşitliliği genelleme için şart |
| Cevabı olmayan sorular (başka konudan sorular) | "yok" örnekleri | kalibrasyonun asıl testi |

Hedef büyüklük: ilk deney için ~50 bin pozitif + negatiflerle birkaç yüz bin (soru, paragraf) çifti.
Telif hakkı olan kitaplar **eğitimde kullanılmaz**; sadece yerelde, kullanıcının kendi testinde.

### 4.3 Öğretmen

* **Seçenek A — ince ayarlı Laya cross-encoder** (emlaktaki v1 öğretmeninin aynısı): pozitif/negatif çiftlerle ince
  ayar, sonra LLM'in ürettiği etiketsiz çiftleri etiketler. Ücretsiz, bu GPU'da çalışır, yavaş (~150 çift/sn).
* **Seçenek B — büyük bir LLM** etiketler: daha doğru, ama API maliyeti var.
* Emlakta öğrendiğimiz ders: öğretmen **küçük bir etiketli doğrulama setinde** kontrol edilir, güvenilmez kısımlar atılır.

### 4.4 Öğrenci

* v3 mimarisi, hibrit eğitim (tek vektör + kelime vektörleri), çıkarımda kelime yolu.
* İki başlangıç noktası karşılaştırılır:
  1. Laya'nın mmBERT encoder'ı (şimdiki)
  2. **bge-m3** (arama işinde zaten güçlü; üzerine kalibrasyon öğretmek). 4 GB GPU için ağır — sadece üst
     katmanlar eğitilir; gerekirse Kaggle/Colab (Laya'nın ince ayar notebook'u 2×T4 kullanıyor).
* Kayıp: ikili çapraz entropi (strictly proper → kalibrasyonu ödüllendirir).

### 4.5 Değerlendirme (dürüst test)

* **Görülmemiş kitaplar ve konular:** bazı konular (ör. felsefe, hukuk) eğitimden tamamen çıkarılır; test sadece
  onlarda. "Bilim Felsefesine Giriş" (Kadir Çüçen) test kitaplarından biri olur.
* **Test soruları:** görülmemiş kitapların paragraflarından LLM ile üretilir, bir kısmı elle kontrol edilir; doğru
  paragraf = sorunun üretildiği paragraf. Ayrıca kitapta cevabı olmayan sorular.
* **Karşılaştırılanlar:** BM25, bge-m3, bge-m3 + ince ayarlı Laya yeniden sıralama, bizim model.
* **Ölçütler:**

| Ölçüt | Ne ölçer |
|---|---|
| Recall@1 / @5 / @10, MRR@10 | doğru paragrafı ilk sıralarda bulma |
| "Cevap yok" doğruluğu ve AUC | kitapta cevabı olmayan soruyu tanıma |
| ECE | olasılıkların dürüstlüğü (%80 dediği gerçekten %80 mi) |
| Uyarlanabilir k | eşikle LLM'e ortalama kaç paragraf gidiyor, recall ne kadar korunuyor |
| (isteğe bağlı) uçtan uca | LLM'in uydurma oranı ve token maliyeti |

### 4.6 Başarı ölçütleri

| | Hedef |
|---|---|
| Recall@10 | bge-m3'ün en fazla 3 puan gerisinde |
| "Cevap yok" doğruluğu | ≥ %85 ve bge-m3'ün en iyi eşiğinden yüksek |
| ECE (görülmemiş kitaplarda) | < 0,05 |
| Uyarlanabilir k | ortalama ≤ 3 paragrafla, bge-m3 top-5 kadar recall |

**Durdurma ölçütü:** görülmemiş kitaplarda ECE 0,15'in üstünde kalır ve "yok" doğruluğu bge-m3'ü geçemezse, kalibrasyon
alan dışına taşınmıyor demektir — o zaman yaklaşım "her alan için küçük bir ince ayar" yönüne kayar.

### 4.7 Aşamalar

| Aşama | İş | Tahmini süre |
|---|---|---|
| A | Veri: Türkçe QA setlerini indir, paragraflara böl, zor negatifler, LLM soruları, görülmemiş konu ayrımı | 1–2 gün |
| B | Öğretmen: Laya cross-encoder ince ayarı + doğrulama + etiketleme | ~1 gün (GPU süresi dahil) |
| C | Öğrenci: v3 eğitimi (Laya encoder ve bge-m3 başlangıçlı iki deney) | ~1 gün |
| D | Değerlendirme: görülmemiş kitaplarda tüm yöntemler, `book_server` paneline yeni modeli eklemek | yarım gün |
| E | Ölçek: iki aşamalı arama (ön eleme + v3), kelime vektörlerini sıkıştırma (128 → 32 boyut, 8-bit) | sonra |

### 4.8 Açık kararlar

* Öğretmen: Laya (ücretsiz, daha zayıf) mı, LLM (ücretli, daha doğru) mu?
* Eğitim bu bilgisayarda mı (4 GB, yavaş, bge-m3 zorlanır) yoksa Kaggle/Colab'da mı?
* Test için kullanıcının kendi kitaplarından hangileri kullanılacak?

---

## 5. Sonuç

Emlak prototipi fikrin mekanizmasının çalıştığını gösterdi. Kitap deneyi, bunun **gerçek, açık uçlu sorularda ve
görülmemiş belgelerde** de çalışıp çalışmadığını gösterecek. Çalışırsa bu, "kalibre RAG" adıyla paylaşılabilir bir
yöntem olur: embedding hızında arama, güvenilir olasılık, "cevap yok" diyebilme ve daha az LLM maliyeti.
