# layavektor — Proje Açıklaması

Bu belge projenin ne olduğunu, neden yapıldığını, şimdiye kadar ne denendiğini, sonuçları, dersleri ve sonraki
adımları anlatır. Teknik ayrıntılar ve tüm tablolar için `README.md`'ye bakın.

---

## 1. Tek cümlede proje

**Laya adlı karar modelini, embedding modelleri gibi önceden indekslenebilir hale getirmek:** belgeler bir kez
işlenip saklanır, soru gelince milisaniyeler içinde her belge için *kalibre bir olasılık* ("bu belge bu soruya
uyuyor mu? %92") döner.

## 2. Arka plan: üç tür model

| | Nasıl çalışır | Hız | Ne verir |
|---|---|---|---|
| **Embedding** (bge-m3, RAG'de kullanılan) | Her belge önceden bir vektöre çevrilir; soru da vektöre çevrilir; en yakın vektörler bulunur | Çok hızlı | Benzerlik skoru (0,73 gibi) — olasılık değil |
| **Cross-encoder** (Laya) | Soru ve belge her seferinde birlikte okunur | Yavaş (her belge için ayrı model çalışması) | Kalibre olasılık, "evet/hayır" kararı |
| **Bu proje: kalibre karar indeksi** | Belge önceden kodlanıp saklanır (embedding gibi), ama çıktı Laya gibi olasılık | Embedding hızına yakın | Kalibre olasılık |

**Laya** (convaiinnovations/laya): metin üretmeyen, tek geçişte karar veren bir model. Üç soru tipi var:
`noul` (evet/hayır olasılığı), `choice` (seçenekler), `score` (ölçek). Olasılıkları kalibre olacak şekilde
(strictly proper scoring rule'larla) eğitilmiş. Sorunu: her (belge, soru) çifti için modeli ayrı çalıştırmak
gerekir — 1 milyon belgede bir soru saatler sürer.

## 3. Neden önemli? Embedding'in yapamadığı şeyler

Embedding "en benzer" olanı getirir ama:
* **"Sonuç yok" diyemez** — cevap hiç yoksa bile en benzer olanı getirir (RAG'de uydurmanın bir sebebi).
* **Olasılık vermez** — skorlar çarpılamaz, eşiklenemez, toplanamaz.
* **Birleşik ve olumsuz sorgularda zayıftır** — "manzaralı, metroya yakın, Kadıköy dışında" gibi.
* **Sayamaz** — "kaç tane?" sorusuna cevap veremez.

Kalibre olasılıkla bunların hepsi basit matematik olur: VE = çarpım, DEĞİL = 1 − P, sayı = olasılıkların toplamı,
"yok" = hiçbir olasılık eşiği geçmiyor.

## 4. Deney ortamı (prototip)

Doğru cevapları bildiğimiz sentetik bir dünya: **emlak ilanları**.

* 5.000 eğitim + 2.000 test ilanı, serbest metin açıklamalar (her özellik 3 farklı biçimde yazılabiliyor).
* 66 evet/hayır sorusu: özellikler (deniz manzarası, asansör…), ilçe, oda, kat, m², fiyat, kira.
* **Genelleme testleri:**
  * 11 soru eğitimde **hiç** sorulmadı (jeneratör, klima, takas, Sarıyer, Pendik, m² > 120…)
  * Her sorunun 3. söylenişi sadece testte kullanıldı
* 400 birleşik sorgu (2–4 koşul, %25 olumsuz, 29'u bilerek boş).
* Donanım: RTX 3050 Ti Laptop, 4 GB — encoder'ın alt katmanları donduruldu, üst 6 katman eğitildi.

**Ölçütler (basitçe):**
* **AUC**: doğru belgeleri yanlışlardan ne kadar iyi ayırıyor (1,0 mükemmel, 0,5 yazı tura).
* **AP**: doğru sonuçlar listenin ne kadar başında.
* **ECE**: olasılıkların ne kadar dürüst olduğu (%80 dediği şeylerin gerçekten %80'i doğru mu; 0'a yakın iyi).
* **"Sonuç yok" doğruluğu**: boş sorgularda "yok", dolu sorgularda "var" diyebiliyor mu.

## 5. Sürümler

### v1 — iki kuleli tek vektör
İlan → 256 boyutlu vektör, soru → vektör + sapma; `P = σ(ilan · soru + sapma)`. Laya'nın encoder'ından başlatıldı,
gerçek cevaplarla 2 dakika eğitildi.
* Görülen sorularda tam Laya kadar iyi (AUC 0,983 vs 0,982), iyi kalibre (ECE 0,009), **~600 kat hızlı**.
* Birleşik sorgularda bge-m3'ü açık ara geçti (AP 0,815 vs 0,249); "sonuç yok" tespitinde bge-m3 şanstan kötü.
* Zayıflık: yeni söylenişler (0,82) ve görülmemiş sorular (0,65).
* Yan bulgu: **hazır Laya bu işte kötü bir öğretmen** — her şeye "evet" diyor. Önce ince ayar gerekiyor.

### v2 — soru çeşitliliği + öğretmenden öğrenme (distillation)
İnce ayarlı Laya cross-encoder öğretmen yapıldı. 118 kavram / 527 söyleniş (63 kavram tamamen yeni: mahalle, oda
aralıkları, fiyat eşikleri); yeni kavramların cevapları öğretmenden geldi. Öğretmen küçük bir doğrulama setinde
kontrol edildi, güvenilmez 11 kavram atıldı ("m²'den küçük mü" sorularını tersine anlıyordu).
* Görülen 0,993, yeni söyleniş 0,927, birleşik AP 0,935 — **öğretmenini geçti**.
* Görülmemiş **sayılara** genelleşiyor (m² > 120: 0,98), görülmemiş **kavramlara** genelleşmiyor (~0,5).
* Sebep: tek vektör, yalnızca eğitimde sorulan bilgiyi tutmayı öğreniyor.

### v3 — çok vektörlü (kelime/token) temsil
İlanın her token'ı için 128 boyutlu bir vektör saklanır (ColBERT tipi); soru kelimeleri ilandaki en benzer kelimeyi
bulur, sonuç sorudan öğrenilen iki sayıyla kalibre olasılığa çevrilir. Hibrit eğitildi (tek vektör + token),
çıkarımda sadece token yolu kullanılıyor.
* **Görülmemiş kavramlar 0,665 → 0,903** (Sarıyer 1,00, oyun parkı 1,00, jeneratör 0,93).
* Görülen 0,999, birleşik AP 0,907 (tam Laya 0,883).
* Bedeli: indeks 512 bayttan 22,8 KB/ilana çıktı.
* Kalan sorunlar: anlam komşusu karışması ("Klimalı mı?" mükemmel, "Klima var mı?" ısıtmalı ilanları getiriyor),
  görülmemiş kavramlarda olasılıklar çok düşük (sıralama doğru ama sayma yanlış).

### Özet tablo

| | v1 | v2 | **v3** | Tam Laya | bge-m3 |
|---|---|---|---|---|---|
| Görülen soru (AUC) | 0,983 | 0,993 | **0,999** | 0,982 | 0,839 |
| Yeni söyleniş | 0,817 | 0,927 | **0,970** | 0,930 | 0,825 |
| Görülmemiş kavram | 0,650 | 0,665 | **0,903** | 0,992 | 0,824 |
| Birleşik sorgu AP | 0,654 | 0,745 | **0,907** | 0,883 | 0,230 |
| İndeks / ilan | 512 B | 512 B | 22,8 KB | — | 2 KB |
| 2.000 ilanda bir soru | 22 ms | 25 ms | 36 ms | 13,8 sn | — |

## 6. Öğrenilen dersler

1. **Kalibre karar + önceden indeksleme mümkün.** Bilinen soru tiplerinde öğrenci, öğretmeninin kalitesine
   embedding hızında ulaşıyor, hatta birleşik sorgularda geçiyor.
2. **Kalibrasyon gerçek bir avantaj.** "Sonuç yok", olasılık çarpımı ve sayma embedding'in yapamadığı şeyler.
3. **Tek vektör bir darboğaz.** Sadece sorulan bilgiyi saklıyor; yeni kavramlar için kelime düzeyi temsil gerekiyor.
4. **Öğretmen yanılabilir, ama kontrol edilebilir.** Küçük bir etiketli doğrulama seti güvenilmez kavramları eliyor.
5. **Eğitim süresi önemli.** v2 3 epoch'ta v1'den kötüydü, 8 epoch'ta en iyisi oldu.
6. **İki yolu ortalamak kötü bir birleştirme.** Görülmemiş kavramda bilgisiz yol ortalamayı aşağı çekiyor.

## 7. Asıl hedef: kitaplarda arama

Kullanıcı eğitimde hiç görülmemiş kitapları (ör. 50 PDF) yükleyip soru soracak, en alakalı paragraflar gelecek —
RAG'deki gibi, ama kalibre olasılık ve "bu kitaplarda cevap yok" diyebilme ile.

**Nasıl çalışır:** Kitaplar bir kez paragraflara bölünüp indekslenir (50 kitap için birkaç dakika). Soru sorulunca
model sadece soruyu okur (~30 ms); kitaplar tekrar modelden geçmez. Büyük ölçekte iki aşama: önce paragraf başına
tek vektörlük küçük indeksten ~200 aday, sonra sadece onların kelime vektörleriyle hassas karşılaştırma.

**Eğitim gerektirmez mi?** Yeni kitap için gerektirmez — model kitapları değil, "bu paragraf bu soruyu cevaplıyor mu?"
becerisini öğrenir. Ama bu beceri için modelin bir kez, genel ve çeşitli Türkçe veriyle eğitilmesi gerekir; şu anki
ağırlıklar sadece emlakla eğitildi.

### İlk eğitimsiz deneme: "Bilim Felsefesine Giriş" (Kadir Çüçen)

`book_server.py` (http://localhost:8766): PDF → 640 paragraf (~120 kelime), dört yöntem yan yana. **Hiçbir yöntem
bu kitapla ya da kitap göreviyle eğitilmedi.** 7 kitap içi + 2 kitap dışı soru (az örnek; anekdot düzeyinde):

| Yöntem | Sıralama | "Kitapta yok" ayrımı (en yüksek skor) |
|---|---|---|
| BM25 (kelime araması) | kelime tutarsa iyi, Türkçe ekler yüzünden kaçırıyor | yok: kitap dışı soruya 9,7 puan (kitap içi 5,9–19) |
| **bge-m3** (standart RAG) | **en tutarlı** (ör. yanlışlanabilirlik → s.172, kuram/yasa → s.128) | zayıf: kitap içi 0,56–0,75, kitap dışı 0,36–0,48 — sabit bir eşik yok |
| **v3** (sadece emlakla eğitildi) | bazı sorularda iyi (yanlışlanabilirlik, uygunluk, paradigma), bazılarında zayıf (bilimsel yöntem) | **şaşırtıcı biçimde var:** kitap içi 0,71–0,99, kitap dışı 0,14–0,23 — 0,5 eşiği 9 sorunun 9'unda doğru |
| bge-m3 + hazır Laya | bazen iyi, bazen alakasız (künye sayfası) | yok: her şeye ~%100 "evet" |

Çıkarım: eğitimsiz haliyle sıralamada bge-m3 önde; ama v3, hiç görmediği bir alanda bile "cevap yok" ayrımına dair
umut verici bir sinyal veriyor. Kitap görevinde eğitilmiş bir v3'ün hedefi: sıralamada bge-m3'e yetişmek,
"yok" tespitinde onu geçmek.

## 8. Sonraki adımlar

Ayrıntılı plan ve "kalibre RAG" değerlendirmesi: `KALIBRE_RAG.md`.

1. **Kitap görevi için eğitim:** "Bu paragraf bu soruyu cevaplıyor mu?" — Türkçe soru-cevap verileri (TQuAD vb.),
   aynı belgeden zor negatifler, Türkçe Wikipedia'dan LLM ile üretilmiş sorular; ince ayarlı Laya öğretmen; v3
   mimarisi (Laya encoder'ından ya da bge-m3'ten başlatma).
2. **Dürüst test:** bazı kitap/konuları eğitimden tamamen saklamak; bge-m3 ve BM25 ile karşılaştırma; ölçütler:
   ilk 10'da doğru paragraf oranı, "cevap yok" doğruluğu.
3. **Görülmemiş kavramlarda kalibrasyon** ve **anlam komşusu karışması** (v3'ün kalan sorunları).
4. **İndeks boyutu ve ölçek:** token vektörlerini sıkıştırma (128 → 32 boyut, 8-bit), iki aşamalı arama.
5. **Genel model:** çok sektörlü, çok çeşitli kavramla eğitilmiş, indirilebilir bir model.

## 9. İlgili çalışmalar

* **ColBERT / ColBERTv2 / PLAID** (Khattab ve ark., 2020–2022): kelime düzeyi vektörlerle "geç etkileşim" araması —
  v3'ün mimari atası. Farkımız: çıktı benzerlik değil kalibre olasılık.
* **DeFormer, PreTTR, MORES** (2020): cross-encoder'ı alt katmanlarda ayırıp belge tarafını önceden hesaplamak.
* **Margin-MSE, TAS-B** (Hofstätter ve ark.): cross-encoder'dan embedding'e distillation.
* **Poly-encoders** (Humeau ve ark., 2020): tek vektör ile cross-encoder arası.
* **Query2Box, BetaE** (Ren ve ark., 2020): birleşik sorguları vektörlerle çözmek (bilgi grafiklerinde).
* **SPLADE, doc2query**: öğrenilmiş seyrek indeksler.

Bildiğimiz kadarıyla "kalibre olasılık veren, birleşik sorguyu olasılık matematiğiyle çözen, önceden indekslenen
karar modeli" kombinasyonu pek çalışılmamış; iddia etmeden önce detaylı literatür taraması gerekir.

## 10. Dosyalar ve çalıştırma

| Dosya | Ne yapar |
|---|---|
| `bank.py`, `build_data.py` | Sentetik ilanlar, soru bankası, doğru cevaplar, birleşik sorgular |
| `qgen.py` | v2/v3 kavram havuzu (118 kavram) |
| `models.py` | Tek vektör (`Student`), çok vektör (`MultiStudent`), yükleme yardımcıları |
| `train_student.py` | Öğrenci eğitimi (`MODEL`, `QSET`, `TARGET`, `INIT`, `EPOCHS`) |
| `train_cross.py`, `teacher_label.py`, `teacher_concepts.py` | Laya öğretmen: ince ayar ve etiketleme |
| `evaluate.py`, `per_question.py`, `baseline_embed.py` | Değerlendirme, bge-m3 karşılaştırması |
| `save_weights.py` | Ağırlıkları `weights/<sürüm>/` altına tek başına yüklenebilir kaydeder |
| `run_v2.sh`, `run_v3.sh` | Deney dizileri |
| `demo_server.py` + `demo.html` | Emlak deneme arayüzü (http://localhost:8765) |
| `book_server.py` + `book.html` | Kitap arama denemesi (http://localhost:8766) |
| `results/` | Tüm değerlendirme raporları |

**Ağırlıklar** (repoda değil, yerelde `weights/`, her sürüm ayrı klasörde ve `SHA256SUMS` ile):
`v1/student`, `v1/laya-cross-ft` (öğretmen, `laya.load` ile açılır), `v2/student`, `v3/student`.
Yüklemek: `models.load_student("weights/v3/student")`.

```bash
python demo_server.py                  # emlak denemesi (v3)
python book_server.py                  # kitap denemesi; BOOKS_DIR içindeki tüm PDF'ler
```

Repo: https://github.com/msariakcali/layavektor
