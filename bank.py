"""Prototip verisi: ilanlar, serbest metin açıklamalar, soru bankası ve doğru cevaplar.

İlanlar yapısal alanlardan üretildiği için her (ilan, soru) çiftinin gerçek cevabı bilinir.
Açıklamalar serbest metindir; model özellikleri alan adlarından değil metinden okumak zorundadır.

Genelleme testleri için iki şey bilerek eğitimden saklanır:
  * ifade 2: her özelliğin/sorunun üçüncü söylenişi sadece test ilanlarında ve değerlendirmede kullanılır
  * heldout=True sorular: eğitimde hiç sorulmaz
"""
import random

ILCELER = {
    "Kadıköy": ["Moda", "Caddebostan", "Fenerbahçe", "Göztepe", "Suadiye"],
    "Beşiktaş": ["Levent", "Etiler", "Bebek", "Ortaköy", "Arnavutköy"],
    "Şişli": ["Nişantaşı", "Mecidiyeköy", "Teşvikiye", "Fulya"],
    "Üsküdar": ["Çengelköy", "Kuzguncuk", "Altunizade", "Bulgurlu"],
    "Beylikdüzü": ["Adnan Kahveci", "Yakuplu", "Gürpınar"],
    "Bakırköy": ["Ataköy", "Yeşilköy", "Yeşilyurt"],
    "Maltepe": ["Bağlarbaşı", "Cevizli", "İdealtepe"],
    "Ataşehir": ["Barbaros", "İçerenköy", "Küçükbakkalköy"],
    "Sarıyer": ["Tarabya", "Yeniköy", "Emirgan"],
    "Pendik": ["Kurtköy", "Güzelyalı", "Çamçeşme"],
}
ODA_TIPLERI = ["1+0", "1+1", "2+1", "3+1", "4+1", "5+1"]
KATLAR = ["bahçe katı", "zemin kat", "ara kat"] + ["%d. kat" % k for k in range(1, 15)]

# özellik -> metinde söylenişleri (0,1 eğitim ilanlarında; 2 sadece test ilanlarında)
FEATURES = {
    "deniz manzaralı": ["deniz manzaralı", "salondan denizi gören", "denize nazır"],
    "boğaz manzaralı": ["boğaz manzaralı", "Boğaz'ı gören", "Boğaz'a nazır"],
    "asansörlü": ["asansörlü binada", "binada asansör mevcut", "asansörü olan bir binada"],
    "otoparklı": ["otoparklı", "açık otoparkı bulunan", "araç park yeri olan"],
    "balkonlu": ["balkonlu", "geniş balkonu olan", "balkonunda kahvenizi içebileceğiniz"],
    "eşyalı": ["eşyalı", "tüm eşyalarıyla teslim edilecek", "mobilyalı"],
    "klimalı": ["klimalı", "odalarında klima bulunan", "split klima mevcut"],
    "güvenlikli site içinde": ["güvenlikli site içinde", "7/24 güvenlikli sitede", "güvenlik görevlisi bulunan sitede"],
    "havuzlu sitede": ["havuzlu sitede", "site içinde yüzme havuzu var", "havuzu olan bir sitede"],
    "yeni yapı": ["yeni yapı", "sıfır binada", "yeni inşa edilmiş binada"],
    "merkezi ısıtmalı": ["merkezi ısıtmalı", "merkezi sistem ısınma", "ısınması merkezi sistemle sağlanan"],
    "kombili": ["kombili", "bireysel kombi ile ısınan", "kombi ısıtmalı"],
    "geniş teraslı": ["geniş teraslı", "büyük terası olan", "terası bulunan"],
    "bahçe katı": ["bahçe katı", "bahçe kullanımlı", "kendi bahçesi olan"],
    "çatı katı": ["çatı katı", "çatı dubleksi", "en üstte çatı katında"],
    "ebeveyn banyolu": ["ebeveyn banyolu", "yatak odasında ayrı banyo bulunan", "ebeveyn banyosu mevcut"],
    "giyinme odalı": ["giyinme odalı", "giyinme odası bulunan", "walk-in giyinme alanlı"],
    "kapalı otoparklı": ["kapalı otoparklı", "kapalı garajı olan", "yer altı otoparkı bulunan"],
    "asansörsüz": ["asansörsüz binada", "binada asansör bulunmuyor", "asansörü olmayan bir binada"],
    "öğrenciye uygun": ["öğrenciye uygun", "öğrenciler için ideal", "üniversite öğrencisine uygun"],
    "evcil hayvana uygun": ["evcil hayvana uygun", "evcil hayvan kabul edilir", "kedi ve köpek dostu"],
    "krediye uygun": ["krediye uygun", "konut kredisine uygun", "banka kredisi kullanılabilir"],
    "takasa uygun": ["takasa uygun", "araç veya daire ile takas olur", "takas teklifleri değerlendirilir"],
    "yeni tadilatlı": ["yeni tadilatlı", "baştan sona yenilenmiş", "komple tadilattan geçmiş"],
    "boş teslim": ["boş teslim", "hemen taşınmaya hazır ve boş", "boş olarak teslim edilecek"],
    "doğalgaz tesisatlı": ["doğalgaz tesisatlı", "doğalgaz tesisatı hazır", "doğalgaz bağlantısı mevcut"],
    "jeneratörlü": ["jeneratörlü", "binada jeneratör var", "elektrik kesintisine karşı jeneratör mevcut"],
    "spor salonlu sitede": ["spor salonlu sitede", "site içinde fitness salonu var", "spor salonu bulunan sitede"],
    "oyun parklı sitede": ["oyun parklı sitede", "çocuk oyun parkı bulunan sitede", "site içinde çocuk parkı mevcut"],
    "metroya yakın": ["metroya yakın", "metro istasyonuna yürüme mesafesinde", "metroya 5 dakika"],
    "sahile yakın": ["sahile yakın", "sahile yürüme mesafesinde", "sahil yoluna birkaç dakika"],
}

FILLER = [
    "Detaylı bilgi için arayınız.", "Kaçırılmayacak bir fırsat.", "Ferah ve aydınlık bir yaşam alanı.",
    "Aile yaşamına uygun bir daire.", "Randevu ile gezilebilir.", "Emlakçıdan değil, sahibinden.",
    "Bakımlı bir binada yer alıyor.", "Mutfak ve banyo kullanışlı.",
]


def random_listing(rng, i, prefix):
    ilce = rng.choice(list(ILCELER))
    feats = rng.sample(list(FEATURES), rng.randint(2, 5))
    if "asansörlü" in feats and "asansörsüz" in feats:
        feats.remove("asansörsüz")
    durum = rng.choice(["satılık", "kiralık"])
    fiyat = rng.randint(15, 200) * 100_000 if durum == "satılık" else rng.randint(8, 90) * 1_000
    return {
        "id": "%s_%05d" % (prefix, i),
        "ilce": ilce,
        "mahalle": rng.choice(ILCELER[ilce]),
        "oda": rng.choice(ODA_TIPLERI),
        "m2": rng.randint(45, 260),
        "kat": rng.choice(KATLAR),
        "ozellikler": feats,
        "durum": durum,
        "fiyat": fiyat,
    }


def _fmt_price(l, rng):
    f = l["fiyat"]
    if l["durum"] == "satılık":
        if rng.random() < 0.5:
            return "Satış fiyatı %s TL." % format(f, ",").replace(",", ".")
        m = f / 1e6
        s = ("%d" % m) if m == int(m) else ("%.1f" % m).replace(".", ",")
        return "Fiyat: %s milyon TL." % s
    if rng.random() < 0.5:
        return "Aylık kira %s TL." % format(f, ",").replace(",", ".")
    return "Kirası ayda %d bin TL." % (f // 1000)


def describe(l, rng, variants):
    """Serbest metin ilan açıklaması. variants: kullanılabilecek söyleniş indeksleri."""
    oda = "stüdyo (1+0)" if l["oda"] == "1+0" else l["oda"]
    m2 = rng.choice(["%d m²" % l["m2"], "%d metrekare" % l["m2"], "brüt %d m²" % l["m2"]])
    opening = rng.choice([
        "%s ilçesinde, %s mahallesinde %s %s daire." % (l["ilce"], l["mahalle"], l["durum"], oda),
        "%s mahallesi (%s) içinde %s, %s %s daire." % (l["mahalle"], l["ilce"], m2, oda, l["durum"]),
        "%s ilçesi %s mahallesinde yer alan %s daire %s." % (l["ilce"], l["mahalle"], oda, l["durum"]),
    ])
    parts = [opening]
    if m2 not in opening:
        parts.append("Daire %s kullanım alanına sahip." % m2)
    parts.append("Daire %s konumunda." % l["kat"])
    phr = [FEATURES[f][rng.choice(variants)] for f in l["ozellikler"]]
    rng.shuffle(phr)
    if len(phr) > 2 and rng.random() < 0.5:
        parts.append("Öne çıkanlar: %s ve %s." % (", ".join(phr[:-1]), phr[-1]))
    else:
        parts.append("Ev %s." % phr[0])
        parts += ["Ayrıca %s." % p for p in phr[1:]]
    parts.append(_fmt_price(l, rng))
    if rng.random() < 0.7:
        parts.insert(rng.randint(1, len(parts)), rng.choice(FILLER))
    return " ".join(parts)


# ----------------------------------------------------------------- soru bankası
# Her soru: id, 3 söyleniş (0,1 eğitim; 2 sadece değerlendirme), doğru cevap fonksiyonu,
# grup (birleşik sorguda çelişki önlemek için), sorgu ifadesi ve olumsuz ifade (varsa).
FEATURE_Q = {
    "deniz manzaralı": ["Evin deniz manzarası var mı?", "Daireden deniz görünüyor mu?", "Bu ev deniz manzaralı mı?"],
    "boğaz manzaralı": ["Evin Boğaz manzarası var mı?", "Daireden Boğaz görünüyor mu?", "Bu ev Boğaz manzaralı mı?"],
    "asansörlü": ["Binada asansör var mı?", "Asansörlü bir binada mı?", "Daireye asansörle çıkılabiliyor mu?"],
    "otoparklı": ["Otopark var mı?", "Araba park yeri var mı?", "Otoparklı mı?"],
    "balkonlu": ["Balkonu var mı?", "Balkonlu mu?", "Evde balkon bulunuyor mu?"],
    "eşyalı": ["Ev eşyalı mı?", "Mobilyalı olarak mı veriliyor?", "Eşyalarıyla birlikte mi teslim ediliyor?"],
    "klimalı": ["Klima var mı?", "Evde klima bulunuyor mu?", "Klimalı mı?"],
    "güvenlikli site içinde": ["Güvenlikli bir sitede mi?", "Sitede güvenlik görevlisi var mı?", "7/24 güvenlik var mı?"],
    "havuzlu sitede": ["Sitede havuz var mı?", "Havuzlu bir sitede mi?", "Yüzme havuzu var mı?"],
    "yeni yapı": ["Yeni yapı mı?", "Bina yeni mi inşa edilmiş?", "Sıfır bir binada mı?"],
    "merkezi ısıtmalı": ["Merkezi ısıtma var mı?", "Isınma merkezi sistemle mi?", "Merkezi ısıtmalı mı?"],
    "kombili": ["Kombili mi?", "Isınma kombi ile mi?", "Evde kombi var mı?"],
    "geniş teraslı": ["Terası var mı?", "Geniş bir terası var mı?", "Teraslı mı?"],
    "bahçe katı": ["Bahçe katı mı?", "Bahçe kullanımı var mı?", "Evin bahçesi var mı?"],
    "çatı katı": ["Çatı katı mı?", "Çatı dubleksi mi?", "Daire çatı katında mı?"],
    "ebeveyn banyolu": ["Ebeveyn banyosu var mı?", "Yatak odasında ayrı banyo var mı?", "Ebeveyn banyolu mu?"],
    "giyinme odalı": ["Giyinme odası var mı?", "Giyinme odalı mı?", "Ayrı bir giyinme alanı var mı?"],
    "kapalı otoparklı": ["Kapalı otopark var mı?", "Kapalı garajı var mı?", "Kapalı otoparklı mı?"],
    "asansörsüz": ["Binada asansör yok mu?", "Asansörsüz bir bina mı?", "Bina asansörden yoksun mu?"],
    "öğrenciye uygun": ["Öğrenciye uygun mu?", "Öğrenciler için uygun mu?", "Üniversite öğrencisine kiralanabilir mi?"],
    "evcil hayvana uygun": ["Evcil hayvana uygun mu?", "Evcil hayvan kabul ediliyor mu?", "Kedi veya köpekle oturulabilir mi?"],
    "krediye uygun": ["Krediye uygun mu?", "Konut kredisi kullanılabilir mi?", "Banka kredisiyle alınabilir mi?"],
    "takasa uygun": ["Takasa uygun mu?", "Takas kabul ediliyor mu?", "Takas teklifleri değerlendiriliyor mu?"],
    "yeni tadilatlı": ["Yeni tadilatlı mı?", "Ev yakın zamanda yenilenmiş mi?", "Tadilattan geçmiş mi?"],
    "boş teslim": ["Boş teslim mi?", "Ev boş olarak mı teslim ediliyor?", "Hemen taşınılabilir durumda boş mu?"],
    "doğalgaz tesisatlı": ["Doğalgaz tesisatı var mı?", "Doğalgaz bağlantısı mevcut mu?", "Doğalgaz tesisatlı mı?"],
    "jeneratörlü": ["Jeneratör var mı?", "Binada jeneratör bulunuyor mu?", "Jeneratörlü mü?"],
    "spor salonlu sitede": ["Sitede spor salonu var mı?", "Fitness salonu var mı?", "Spor salonlu bir sitede mi?"],
    "oyun parklı sitede": ["Sitede çocuk oyun parkı var mı?", "Çocuk parkı var mı?", "Oyun parklı bir sitede mi?"],
    "metroya yakın": ["Metroya yakın mı?", "Metro istasyonuna yürüme mesafesinde mi?", "Metroya yakın bir konumda mı?"],
    "sahile yakın": ["Sahile yakın mı?", "Deniz kenarına yürüme mesafesinde mi?", "Sahile yakın bir konumda mı?"],
}
HELDOUT_FEATURES = {"jeneratörlü", "takasa uygun", "ebeveyn banyolu", "sahile yakın", "klimalı", "oyun parklı sitede"}
HELDOUT_ILCE = {"Sarıyer", "Pendik"}


def _has(l, f):
    if f == "otoparklı":
        return "otoparklı" in l["ozellikler"] or "kapalı otoparklı" in l["ozellikler"]
    if f == "bahçe katı":
        return "bahçe katı" in l["ozellikler"] or l["kat"] == "bahçe katı"
    return f in l["ozellikler"]


def _kat_no(l):
    return int(l["kat"].split(".")[0]) if l["kat"][0].isdigit() else 0


def build_questions():
    qs = []

    def add(qid, texts, fn, group, phrase, neg=None, heldout=False):
        qs.append({"id": qid, "texts": texts, "fn": fn, "group": group, "phrase": phrase,
                   "neg": neg, "heldout": heldout})

    for f, texts in FEATURE_Q.items():
        add("ozellik:" + f, texts, lambda l, f=f: _has(l, f), "ozellik:" + f, f, f + " olmayan",
            f in HELDOUT_FEATURES)
    add("durum:kiralık", ["Ev kiralık mı?", "Bu ilan kiralık mı?", "Daire kiraya mı veriliyor?"],
        lambda l: l["durum"] == "kiralık", "durum", "kiralık")
    add("durum:satılık", ["Ev satılık mı?", "Bu ilan satılık mı?", "Daire satışa mı çıkarılmış?"],
        lambda l: l["durum"] == "satılık", "durum", "satılık")
    for ilce in ILCELER:
        add("ilce:" + ilce,
            ["Ev %s ilçesinde mi?" % ilce, "İlan %s ilçesine ait mi?" % ilce, "Daire %s ilçesinde mi bulunuyor?" % ilce],
            lambda l, v=ilce: l["ilce"] == v, "ilce", "%s ilçesinde" % ilce, "%s dışında" % ilce, ilce in HELDOUT_ILCE)
    add("oda:1+0", ["Stüdyo daire mi?", "Daire 1+0 mı?", "Tek odalı stüdyo mu?"],
        lambda l: l["oda"] == "1+0", "oda", "stüdyo")
    for oda in ODA_TIPLERI[1:]:
        add("oda:" + oda, ["Daire %s mi?" % oda, "Oda sayısı %s mi?" % oda, "Ev %s olarak mı tasarlanmış?" % oda],
            lambda l, v=oda: l["oda"] == v, "oda", oda)
    add("oda:>=3", ["Daire en az 3+1 mi?", "Üç veya daha fazla odası var mı?", "Kalabalık bir aile için en az üç odalı mı?"],
        lambda l: int(l["oda"][0]) >= 3, "oda", "en az 3+1")
    add("kat:zemin", ["Daire zemin katta mı?", "Zemin kat mı?", "Giriş katında mı?"],
        lambda l: l["kat"] == "zemin kat", "kat", "zemin katta", "zemin kat olmayan")
    add("kat:>=10", ["Daire 10. kat veya üstünde mi?", "Yüksek katta mı, 10 ve üzeri?", "En az onuncu katta mı?"],
        lambda l: _kat_no(l) >= 10, "kat", "10. kat ve üstünde")
    for t in (80, 100, 120, 150, 200):
        add("m2:>%d" % t,
            ["Daire %d metrekareden büyük mü?" % t, "Evin alanı %d m²'den fazla mı?" % t, "%d metrekarenin üzerinde mi?" % t],
            lambda l, t=t: l["m2"] > t, "m2", "%d m²'den büyük" % t, heldout=t == 120)
    for t in (3, 5, 8, 10, 15):
        add("fiyat:<%dM" % t,
            ["Satılık ve fiyatı %d milyon TL'nin altında mı?" % t, "Satış fiyatı %d milyon TL'den düşük mü?" % t,
             "%d milyon TL altına satın alınabilir mi?" % t],
            lambda l, t=t: l["durum"] == "satılık" and l["fiyat"] < t * 1_000_000, "durum",
            "%d milyon TL altında satılık" % t, heldout=t == 8)
    for t in (15, 30, 40, 50):
        add("kira:<%dk" % t,
            ["Kiralık ve aylık kirası %d bin TL'nin altında mı?" % t, "Aylık kira %d bin TL'den az mı?" % t,
             "%d bin TL altına kiralanabilir mi?" % t],
            lambda l, t=t: l["durum"] == "kiralık" and l["fiyat"] < t * 1_000, "durum",
            "%d bin TL altında kiralık" % t, heldout=t == 40)
    return qs


QUESTIONS = build_questions()
TRAIN_VARIANTS = (0, 1)
EVAL_VARIANT = 2
