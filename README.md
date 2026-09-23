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

## Sonraki adım: yeni kavramlar

* **Çok vektörlü ilan temsili (ColBERT tipi, kalibre karar kafasıyla):** ilanı tek vektör yerine token veya
  birkaç özet vektörle saklamak. Metnin kelime düzeyi bilgisi korunur, hâlâ indekslenebilir. Bedel: indeks
  512 bayttan ~10-40 KB/ilana çıkar.
* **Kavram çeşitliliğini büyütmek (genel model yolu):** binlerce kavram ve birden çok sektör. İlan vektörü o zaman
  genel içerik tutmak zorunda kalır. Bu sentetik tek sektörde test edilemez.
* **Hibrit:** soru vektörü eğitimdeki soru dağılımından uzaksa (yeni kavram), cevabı az sayıda aday üzerinde
  cross-encoder'a bırakmak.

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
python demo_server.py       # http://localhost:8765 — weights/v2/student ve weights/v1/laya-cross-ft gerekir
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
python per_question.py ckpt/a.pt ckpt/b.pt   # görülmemiş sorularda soru bazında AUC
python save_weights.py      # ckpt -> weights/<VERSION>/ (tek başına yüklenebilir)
```

Kayıtlı ağırlıklar: `weights/v1/student`, `weights/v1/laya-cross-ft` (`laya.load` ile açılır), `weights/v2/student`.
Öğrenciyi yüklemek: `models.load_student("weights/v2/student")`.
`teacher_label.py`: hazır Laya ile öğretmen etiketleri (distillation için; `LIMIT=100` ile hızlı ölçüm).
