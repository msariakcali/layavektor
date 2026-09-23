# layavektor — Laya ile karar indeksi

Soru: Laya gibi bir karar modeli, embedding gibi **önceden indekslenebilir** hale getirilebilir mi?
Yani `ilan vektörü · soru vektörü = kalibre P(evet)` veren iki kuleli bir model, tam Laya'nın kalitesine
embedding hızında yaklaşabilir mi?

## Kurulum

| | |
|---|---|
| Veri | 5.000 eğitim + 2.000 test ilanı, serbest metin açıklamalar (`bank.py`, `build_data.py`); doğru cevaplar biliniyor |
| Soru bankası | 66 evet/hayır sorusu (özellik, ilçe, oda, kat, m², fiyat, kira), her biri 3 söyleniş |
| Genelleme testleri | 11 soru eğitimde **hiç** sorulmadı; her sorunun 3. söylenişi ve özelliklerin 3. yazılışı sadece testte |
| Birleşik sorgular | 400 sorgu, 2–4 koşul, %25 olumsuz koşul, 29'u bilerek boş |
| Öğrenci (yeni yöntem) | Laya'nın mmBERT encoder'ı, iki kule, ilan → 256 boyut, soru → 256+1 (sapma), tek iç çarpım; üst 6 katman eğitildi, **2 dakika** |
| Laya cross-encoder | Aynı veri, aynı katmanlar ince ayarlandı (48k çift, 8 dakika) — kalite tavanı |
| Embedding | bge-m3 (hazır, kosinüs benzerliği) |
| Donanım | RTX 3050 Ti Laptop, 4 GB |

Hazır Laya (ince ayarsız) öğretmen olarak denendi ve elendi: 100 ilanlık örnekte her soruya "evet" eğilimli
(ortalama P 0,69, gerçek pozitif oranı 0,19), doğruluk %47, fiyat/kira sorularında AUC 0,39–0,79.

## Sonuçlar

### Tek soru

| | Öğrenci | Laya cross (ft) | bge-m3 |
|---|---|---|---|
| **A. Görülen soru** — AUC / doğruluk / ECE | **0,983** / 0,971 / **0,009** | 0,982 / 0,985 / 0,011 | 0,839 / — / — |
| **B. Görülen soru, yeni söyleniş** — AUC | 0,817 | **0,930** | 0,825 |
| **C. Görülmemiş soru** — AUC | 0,650 | **0,992** | 0,824 |
| **D. Görülmemiş soru, yeni söyleniş** — AUC | 0,616 | **0,946** | 0,810 |
| Sayma hatası (A, göreli) | 0,090 | 0,078 | yapılamaz |

### Birleşik sorgular ("manzaralı, metroya yakın, Kadıköy dışında kiralık daire")

| | Öğrenci | Laya cross (ft) | bge-m3 |
|---|---|---|---|
| AP — sadece görülen sorular | 0,815 | **0,885** | 0,249 |
| AP — görülmemiş soru içeren | 0,403 | **0,880** | 0,200 |
| "Sonuç yok" tespiti, AUC (görülen) | **0,986** | 0,963 | 0,320 |
| "Sonuç yok" doğruluğu, eşik 0,5 (görülen) | 0,901 | **0,951** | eşik yok |
| Eşleşme sayısı hatası (ort., görülen) | 6,7 | 6,1 | yapılamaz |

### Hız

| | Öğrenci | Laya cross (ft) |
|---|---|---|
| 2.000 ilanda bir soru | **22 ms** | 13,8 sn |
| 1M ilanda bir soru | **~3 ms tarama** + ~20 ms soru kodlama | ~1,9 saat (tahmin, doğrusal) |
| İndeksleme | 3 ms/ilan, bir kez | yok (her sorguda baştan okur) |
| İndeks boyutu | 512 bayt/ilan (1M ilan ≈ 512 MB) | — |

## Ne öğrendik

1. **Mekanizma çalışıyor.** Eğitimde görülen sorularda iki kuleli öğrenci, tam cross-encoder ile aynı kalitede
   (AUC 0,983 vs 0,982) ve aynı kalibrasyonda (ECE 0,009 vs 0,011). Buna rağmen 2.000 ilanda ~600 kat,
   1M ilanda milyonlarca kat daha hızlı.
2. **Kalibrasyon embedding'in yapamadığını yapıyor.** Birleşik sorgularda olasılık çarpımı bge-m3'ü açık ara
   geçiyor (AP 0,815 vs 0,249). "Sonuç yok" tespitinde bge-m3 şanstan kötü (0,32), öğrenci 0,986. Sayma da
   yapılabiliyor.
3. **Asıl zayıflık genelleme.** Öğrenci görmediği sorularda (AUC 0,65) ve yeni söylenişlerde (0,82) çöküyor.
   Cross-encoder ise aynı görülmemiş sorularda 0,99. Sebep büyük ihtimalle şu: 55 soruyla eğitilince
   ilan vektörü sadece o 55 sorunun bilgisini tutuyor, soru kulesi de soruları ezberliyor.

Özet: görülen sorular için yöntem zaten "cross-encoder kalitesi, embedding hızı". Açık soru, bunun
**her soruya** genişletilip genişletilemeyeceği.

## v2: soru çeşitliliği + öğretmenden öğrenme

v1'in ince ayarlı cross-encoder'ı öğretmen yapıldı. `qgen.py` 118 kavram / 527 söyleniş üretir: görülen 55 sorunun
yeni söylenişleri + bankada hiç olmayan 63 yeni kavram (mahalle, "en az 2+1", kat/m²/fiyat/kira eşikleri).
Yeni kavramların hedefi gerçek cevaptan değil öğretmenden gelir (`teacher_concepts.py`, 3.000 ilan).
Görülmemiş 11 soru, onlara yakın eşikler ve test söylenişleri hiçbir aşamada kullanılmadı.

**Öğretmenin hataları sistematik:** hiç eğitilmediği mahalle sorularında AUC 1,00, ama "m²'den küçük mü"
sorularını tersine anlıyor (AUC 0,01) ve "üzerinde mi" fiyat/kira sorularında zayıf (0,69) — eğitimde sadece tek
yönlü eşik gördü. Çözüm: öğretmeni 200 ilanlık etiketli bir doğrulama örneğinde kontrol edip AUC < 0,9 olan
11 kavramı atmak (`TARGET=teacher_f`).

| deney | görülen AUC | yeni söyleniş AUC | görülmemiş AUC | birleşik AP (görülen) | "sonuç yok" doğruluğu |
|---|---|---|---|---|---|
| v1 (55 soru, gerçek cevap, 3 ep) | 0,983 | 0,817 | 0,650 | 0,815 | 0,901 |
| E1 çeşitli sorular, gerçek cevap, 3 ep | 0,941 | 0,818 | 0,658 | 0,510 | 0,646 |
| E2 + öğretmen, 3 ep | 0,946 | 0,849 | 0,661 | 0,560 | 0,704 |
| E2f + öğretmen doğrulaması, 3 ep | 0,949 | 0,853 | 0,661 | 0,564 | 0,708 |
| E3 + kelime soruları (öz-denetimli), 3 ep | 0,934 | 0,838 | **0,705** | 0,505 | 0,654 |
| **E4 = E2f, 8 ep → `weights/v2`** | **0,993** | **0,927** | 0,665 | **0,935** | **0,988** |
| *Laya cross-encoder (öğretmen)* | *0,982* | *0,930* | *0,992* | *0,885* | *0,951* |

1. **3 epoch yetersizdi.** Soru sayısı 2 kat, söyleniş 5 kat artınca model eksik eğitildi. 8 epoch'ta v2 öğrenci
   görülen sorularda (0,993) ve birleşik sorgularda (AP 0,935) **öğretmenini geçiyor**, yeni söylenişlerde ona
   yetişiyor (0,927 vs 0,930). Olasılıkları çarpmak cross-encoder'da da mümkün, ama öğrencinin olasılıkları daha iyi
   kalibre (ECE 0,007) olduğu için çarpım daha doğru çıkıyor. Not: v1 3 epoch eğitildi; v1'in 8 epoch'luk hali
   denenmedi, yani görülen sorulardaki artışın ne kadarı soru çeşitliliğinden ne kadarı uzun eğitimden, ayrılmadı.
   Yeni söylenişteki artış (0,82 → 0,93) ise büyük ihtimalle soru çeşitliliğinden.
2. **Öğretmen etiketleri gerçek cevaplardan kötü değil** (E2 0,946 vs E1 0,941; tek tohum, fark gürültü
   seviyesinde). Hatalı bir öğretmenle bile distillation çalışıyor; doğrulamayla filtrelemek (E2f) küçük bir artı.
3. **Görülmemiş kavramlar hâlâ çözülmedi.** Soru bazında (`per_question.py`):

| görülmemiş soru | v1 | E2f | E3 (+kelime) | cross |
|---|---|---|---|---|
| m² > 120, fiyat < 8M, kira < 40k (yeni sayı) | 0,98 | 0,97–0,99 | 0,96–0,99 | 0,99 |
| klima, ebeveyn banyosu, takas, jeneratör, oyun parkı, sahil (yeni kavram) | 0,48–0,63 | 0,46–0,65 | 0,48–0,73 | 0,95–1,00 |
| Sarıyer, Pendik (yeni ilçe) | 0,46–0,51 | 0,40–0,51 | 0,52–0,53 | 1,00 |

Öğrenci **görmediği sayılara genelleşiyor, görmediği kavramlara genelleşmiyor.** Tek 256 boyutlu ilan vektörü
yalnızca eğitimde sorulan bilgiyi tutmayı öğreniyor; "jeneratör" hiç sorulmadığı için vektörde yer bulmuyor.
Cross-encoder metni sorgu anında okuduğu için bu sorunu yaşamıyor. Kelime soruları (E3) biraz yardım etti ama yetmedi.

## v3: çok vektörlü (token) temsil

Tek ilan vektörü yalnızca eğitimde sorulan bilgiyi tutuyordu. v3'te ilanın her token'ı için 128 boyutlu
normalize bir vektör saklanır (ColBERT tipi). Soru token'ları ilandaki en benzer token'ı bulur, sonuçlar
öğrenilen token ağırlıklarıyla toplanır ve sorudan gelen iki sayı (a, b) bu benzerliği **kalibre bir olasılığa**
çevirir: `P = σ(a_q · Σ w_i max_t q_i·d_t + b_q)`. Birleşik sorgu, "sonuç yok" ve sayma aynen çalışır.

Deneyler v2'den başlatıldı, v2 ile aynı veri ve hedeflerle 8 epoch eğitildi (`run_v3.sh`):
* **v3a** (`MODEL=multi`): sadece token vektörleri
* **v3b** (`MODEL=hybrid`): tek vektör + token vektörleri; eğitimde iki yol hem ayrı ayrı hem ortalama olarak
  cevap vermek zorunda (aksi halde tek vektör eğitim sorularını zaten çözdüğü için token yolu hiç öğrenmezdi)

| | v2 | v3a | v3b ortalama | **v3 = v3b token yolu** | Laya cross |
|---|---|---|---|---|---|
| Görülen soru, AUC | 0,993 | 0,999 | 0,998 | **0,999** (ECE 0,003) | 0,982 |
| Yeni söyleniş | 0,927 | 0,958 | 0,964 | **0,970** | 0,930 |
| **Görülmemiş soru** | 0,665 | 0,886 | 0,793 | **0,903** | 0,992 |
| Görülmemiş + yeni söyleniş | 0,660 | 0,898 | 0,793 | **0,938** | 0,946 |
| Birleşik AP, tümü | 0,745 | 0,903 | 0,829 | **0,907** | 0,883 |
| Birleşik AP, görülen | 0,935 | 0,992 | 0,990 | **0,994** | 0,885 |
| Birleşik AP, görülmemiş içeren | 0,449 | 0,765 | 0,578 | **0,773** | 0,880 |
| "Sonuç yok" doğruluğu, görülen | 0,988 | 1,000 | 0,992 | **1,000** | 0,951 |
| İndeks | 512 B/ilan | 22 KB | 22,8 KB | 22,8 KB | — |
| 2.000 ilanda sorgu | 25 ms | 33 ms | 35 ms | 36 ms | 13,8 sn |

Görülmemiş sorular, soru bazında AUC (`per_question.py`):

| | v2 | v3a | v3b ort. | cross |
|---|---|---|---|---|
| Sarıyer | 0,49 | **1,00** | 0,95 | 1,00 |
| oyun parkı | 0,59 | **1,00** | 0,83 | 0,99 |
| jeneratör | 0,54 | **0,93** | 0,60 | 1,00 |
| sahil | 0,55 | **0,88** | 0,72 | 1,00 |
| Pendik | 0,52 | **0,86** | 0,69 | 1,00 |
| takas | 0,48 | **0,85** | 0,65 | 1,00 |
| ebeveyn banyosu | 0,60 | 0,76 | 0,74 | 0,95 |
| klima | 0,59 | 0,57 | 0,61 | 1,00 |
| m² > 120, fiyat < 8M, kira < 40k | 0,97–0,99 | 0,95–0,98 | 0,98–0,99 | 0,99 |

1. **Token temsili görülmemiş kavram sorununu büyük ölçüde çözdü** (0,665 → 0,903). Metindeki kelime bilgisi
   indekste kaldığı için hiç sorulmamış kavramlar bulunabiliyor. Görülen sorularda ve birleşik sorgularda
   cross-encoder'ı geçiyor; görülmemiş kavramlarda hâlâ gerisinde (0,90 vs 0,99).
2. **İki yolun ortalaması kötü bir birleştirme:** görülmemiş kavramda tek vektör yolu yazı tura attığı için
   ortalamayı aşağı çekiyor (0,793). En iyisi hibrit eğitilen modelin yalnızca token yolu (0,903, v3a'dan da iyi;
   tek tohum, fark küçük olabilir). `weights/v3` bu yapıda: `arch.combine = "token"`.
3. **Kalan sorunlar:**
   * **Anlam komşusu karışması:** "Klimalı mı?" mükemmel (ilk 10'un hepsi doğru, P 0,97–1,00), ama "Klima var mı?"
     merkezi ısıtmalı ilanları getiriyor — eğitimdeki çok sayıda ısıtma sorusu "klima"yı o gruba çekmiş.
   * **Görülmemiş kavramlarda kalibrasyon bozuk:** jeneratörde sıralama doğru (ilk 10'un hepsi doğru), ama tahmini
     sayı 19, gerçek 238. Olasılıklar görülmemiş kavramlarda sistematik olarak düşük; "sonuç yok" ve sayma bu
     kavramlarda güvenilir değil (görülmemiş içeren birleşik sorgularda "sonuç yok" doğruluğu 0,80).
   * **İndeks 45 kat büyüdü** (512 B → 22,8 KB/ilan; 1M ilan ≈ 23 GB). ColBERTv2 tipi sıkıştırma ve 1M ölçeğinde
     aday seçme (PLAID gibi) henüz yok; 2.000 ilanda kaba kuvvet tarama yapılıyor.

## Sonraki adım

* **Görülmemiş kavramlarda kalibrasyon:** a_q, b_q soru metninden öğreniliyor ve eğitimdeki kavramlara göre ayarlı.
  Fikir: öğretmenle etiketlenmiş daha çeşitli kavramlar, ya da sorgu anında birkaç aday üzerinde cross-encoder ile
  yeniden kalibrasyon.
* **Genelleme testi başka bir alanda:** asıl hedef (eğitimde görülmemiş kitaplarda soru → alakalı paragraf) için
  görev "bu paragraf bu soruyu cevaplıyor mu?" olur; görülmemiş kitap/konu ayrımıyla ve bge-m3 ile karşılaştırmalı.
* **İndeks boyutu:** token vektörlerini sıkıştırmak (ör. 128 → 32 boyut, 8-bit), gereksiz token'ları atmak.

## Gereksinimler

* Python 3.11+, CUDA'lı bir GPU (geliştirme RTX 3050 Ti 4 GB üzerinde yapıldı)
* `pip install torch transformers safetensors huggingface_hub fastapi uvicorn scikit-learn sentence-transformers`
* Laya: `pip install laya` ya da [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) klonu. Klon bu
  klasörün yanında `../laya-check` değilse yolu `LAYA_PATH` ile verin.
* Temel model (`convaiinnovations/laya`, `multilingual` alt klasörü) ilk çalıştırmada Hugging Face'ten iner.

Model ağırlıkları (`weights/`, sürüm başına ~620 MB) ve üretilen veri (`data/`) repoda değil. Veri
`build_data.py` ile aynı tohumlarla yeniden üretilir. Ağırlıklar aşağıdaki adımlarla yeniden eğitilir.
`results/` içindeki raporlar README'deki tabloların kaynağıdır.

## Deneme arayüzü

```bash
python demo_server.py       # http://localhost:8765 — weights/v3/student ve weights/v1/laya-cross-ft gerekir
STUDENT=weights/v2/student python demo_server.py   # v2 ile karşılaştırmak için
```

## Çalıştırma

```bash
python build_data.py        # veri + doğru cevaplar + birleşik sorgular
python train_student.py     # v1 öğrenci (QSET=bank TARGET=truth)
python train_cross.py       # karşılaştırma: ince ayarlı Laya cross-encoder + test etiketleri (~40 dk)
python baseline_embed.py    # bge-m3
python evaluate.py          # -> results/report_<ckpt>.json
python qgen.py              # v2 kavram havuzu
python teacher_concepts.py  # yeni kavramları öğretmene etiketlet (~15 dk)
bash run_v2.sh              # E1-E3
QSET=concepts TARGET=teacher_f EPOCHS=8 OUT=ckpt/e4.pt python train_student.py   # v2
bash run_v3.sh              # v3a / v3b (v2'den başlatılır)
python per_question.py ckpt/a.pt ckpt/b.pt   # görülmemiş sorularda soru bazında AUC
python save_weights.py      # ckpt -> weights/<VERSION>/ (tek başına yüklenebilir)
```

Kayıtlı ağırlıklar (her sürüm ayrı klasörde, `SHA256SUMS` ile): `weights/v1/student`, `weights/v1/laya-cross-ft`
(`laya.load` ile açılır), `weights/v2/student` (tek vektör), `weights/v3/student` (token vektörleri).
Öğrenciyi yüklemek: `models.load_student("weights/v3/student")`.
`teacher_label.py`: hazır Laya ile öğretmen etiketleri (distillation için; `LIMIT=100` ile hızlı ölçüm).
