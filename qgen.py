"""v2 eğitim soru havuzu: görülen sorulara yeni söylenişler + bankada olmayan yeni kavramlar.

Test dürüstlüğü için:
  * görülmemiş 11 soru ve onlara yakın eşikler hiç yok (m² 110-130, fiyat 7-9 M, kira 35-45 bin)
  * bankanın test söylenişi (ifade 2) ve onun şablonu hiçbir kavramda kullanılmaz (assert ile kontrol)
  * Sarıyer/Pendik (görülmemiş ilçeler) mahalleleri sorulmaz

Kavram: {"id", "texts", "fn", "bank": bank indeksi ya da None}. bank=None olanlar yeni kavramdır;
bunların hedefi TARGET=teacher modunda ince ayarlı cross-encoder'dan gelir.
"""
from bank import HELDOUT_FEATURES, HELDOUT_ILCE, ILCELER, ODA_TIPLERI, QUESTIONS, _kat_no

EXTRA_FEATURE_Q = {
    "deniz manzaralı": ["Denize bakıyor mu?", "Manzarasında deniz var mı?", "Pencereden deniz görülebiliyor mu?",
                        "Deniz gören bir daire mi?", "Deniz manzarası mevcut mu?"],
    "boğaz manzaralı": ["Boğaz'a bakıyor mu?", "Manzarasında Boğaz var mı?", "Pencereden Boğaz görülebiliyor mu?",
                        "Boğaz gören bir daire mi?", "Boğaz manzarası mevcut mu?"],
    "asansörlü": ["Asansör mevcut mu?", "Binanın asansörü var mı?", "Asansörlü mü?", "Kat çıkmak için asansör kullanılabilir mi?"],
    "otoparklı": ["Park yeri mevcut mu?", "Arabamı park edebileceğim bir yer var mı?", "Otopark imkanı var mı?",
                  "Araç parkı için yer var mı?"],
    "balkonlu": ["Balkon mevcut mu?", "Dairenin balkonu var mı?", "Balkonu olan bir ev mi?", "Açık havaya çıkılan bir balkon var mı?"],
    "eşyalı": ["Mobilyalı mı?", "Eşyalı olarak mı kiraya veriliyor?", "İçinde mobilya var mı?", "Eşya dahil mi?"],
    "güvenlikli site içinde": ["Site güvenlikli mi?", "Güvenlik hizmeti olan bir site mi?", "Güvenlikli site mi?",
                               "Sitenin güvenliği var mı?"],
    "havuzlu sitede": ["Havuz mevcut mu?", "Sitenin havuzu var mı?", "Havuzlu mu?", "Havuzu olan bir site mi?"],
    "yeni yapı": ["Yeni bir bina mı?", "Bina sıfır mı?", "Yeni inşa edilmiş mi?", "Binanın yaşı yeni mi?"],
    "merkezi ısıtmalı": ["Merkezi sistemle mi ısınıyor?", "Merkezi ısınma mevcut mu?", "Isıtma sistemi merkezi mi?",
                         "Merkezi ısıtması var mı?"],
    "kombili": ["Kombisi var mı?", "Kombi mevcut mu?", "Bireysel kombi var mı?", "Kombi ile mi ısınıyor?"],
    "geniş teraslı": ["Teras mevcut mu?", "Dairenin terası var mı?", "Terası olan bir ev mi?", "Büyük bir terası var mı?"],
    "bahçe katı": ["Bahçesi var mı?", "Bahçeye çıkışı var mı?", "Bahçe katında mı?", "Bahçeli mi?"],
    "çatı katı": ["Çatı katı dairesi mi?", "Dubleks çatı katı mı?", "En üst kattaki çatı dairesi mi?"],
    "giyinme odalı": ["Giyinme odası mevcut mu?", "Giyinme odası bulunuyor mu?", "Giyinme odasına sahip mi?"],
    "kapalı otoparklı": ["Kapalı otopark mevcut mu?", "Kapalı park yeri var mı?", "Garajı var mı?", "Kapalı otoparka sahip mi?"],
    "asansörsüz": ["Asansörü yok mu?", "Asansör bulunmuyor mu?", "Asansörsüz mü?", "Binanın asansörü eksik mi?"],
    "öğrenciye uygun": ["Öğrenci için uygun mu?", "Öğrenciye kiralanır mı?", "Öğrenci evi olarak uygun mu?", "Öğrencilere uygun mu?"],
    "evcil hayvana uygun": ["Evcil hayvanla oturulabilir mi?", "Evcil hayvan kabul edilir mi?", "Evcil hayvan dostu mu?",
                            "Evcil hayvan beslenebilir mi?"],
    "krediye uygun": ["Konut kredisine uygun mu?", "Krediyle satın alınabilir mi?", "Banka kredisi çekilebilir mi?",
                      "Kredi ile alınabilir mi?"],
    "yeni tadilatlı": ["Tadilatı yeni mi?", "Yenilenmiş bir daire mi?", "Yeni tadilat yapılmış mı?", "Tadilatlı mı?"],
    "boş teslim": ["Boş mu teslim ediliyor?", "Ev boş mu?", "Taşınmaya hazır boş bir daire mi?", "Kimse oturmuyor, boş mu?"],
    "doğalgaz tesisatlı": ["Doğalgaz var mı?", "Doğalgaz tesisatı mevcut mu?", "Doğalgazlı mı?", "Doğalgaz kullanılabiliyor mu?"],
    "spor salonlu sitede": ["Spor salonu mevcut mu?", "Sitenin spor salonu var mı?", "Spor yapılacak salon var mı?",
                            "Spor salonlu mu?"],
    "metroya yakın": ["Metro yakında mı?", "Metro durağı yakın mı?", "Metroya yürüyerek gidilebilir mi?",
                      "Metroya kolay ulaşım var mı?"],
}
EXTRA_DURUM = {
    "durum:kiralık": ["Kiralık bir ev mi?", "Kiraya verilen bir daire mi?", "Kiralanabilir mi?", "Kira ilanı mı?"],
    "durum:satılık": ["Satılık bir ev mi?", "Satın alınabilir mi?", "Satış ilanı mı?", "Satılıyor mu?"],
}
ILCE_T = ["{X} ilçesinde mi?", "Bu daire {X} ilçesinde yer alıyor mu?", "Konumu {X} ilçesi mi?", "{X} ilçesindeki bir ev mi?"]
MAHALLE_T = ["Ev {X} mahallesinde mi?", "{X} mahallesinde mi?", "İlan {X} mahallesine ait mi?",
             "Daire {X} mahallesinde yer alıyor mu?"]
ODA_T = ["{X} bir daire mi?", "Daire {X} planlı mı?", "{X} mi?"]
M2_GT_T = ["Daire {X} metrekareden büyük mü?", "Evin alanı {X} m²'den fazla mı?", "Alanı {X} m²'nin üstünde mi?",
           "{X} metrekareden geniş mi?"]
M2_LT_T = ["Daire {X} metrekareden küçük mü?", "Evin alanı {X} m²'den az mı?", "Alanı {X} m²'nin altında mı?"]
FIYAT_LT_T = ["Satılık ve fiyatı {X} milyon TL'nin altında mı?", "Satış fiyatı {X} milyon TL'den düşük mü?",
              "Fiyatı {X} milyon TL'den az olan satılık bir ev mi?", "{X} milyon TL altı satılık mı?"]
FIYAT_GT_T = ["Satılık ve fiyatı {X} milyon TL'nin üzerinde mi?", "Satış fiyatı {X} milyon TL'den yüksek mi?"]
KIRA_LT_T = ["Kiralık ve aylık kirası {X} bin TL'nin altında mı?", "Aylık kira {X} bin TL'den az mı?",
             "Kirası ayda {X} bin TL'den düşük mü?", "{X} bin TL altı kiralık mı?"]
KIRA_GT_T = ["Kiralık ve aylık kirası {X} bin TL'nin üzerinde mi?", "Aylık kira {X} bin TL'den fazla mı?"]

BANK = {q["id"]: i for i, q in enumerate(QUESTIONS)}
FORBIDDEN = {q["texts"][2] for q in QUESTIONS}   # test söylenişleri


def build_concepts():
    cs = []

    def add(cid, texts, fn):
        bi = BANK.get(cid)
        if bi is not None:
            assert not QUESTIONS[bi]["heldout"], cid
            texts = QUESTIONS[bi]["texts"][:2] + [t for t in texts if t not in QUESTIONS[bi]["texts"][:2]]
        cs.append({"id": cid, "texts": texts, "fn": fn, "bank": bi})

    for f, extra in EXTRA_FEATURE_Q.items():
        assert f not in HELDOUT_FEATURES
        add("ozellik:" + f, extra, QUESTIONS[BANK["ozellik:" + f]]["fn"])
    for cid, extra in EXTRA_DURUM.items():
        add(cid, extra, QUESTIONS[BANK[cid]]["fn"])
    for ilce, mahalleler in ILCELER.items():
        if ilce in HELDOUT_ILCE:
            continue
        add("ilce:" + ilce, [t.format(X=ilce) for t in ILCE_T], lambda l, v=ilce: l["ilce"] == v)
        for m in mahalleler:
            add("mahalle:" + m, [t.format(X=m) for t in MAHALLE_T], lambda l, v=m: l["mahalle"] == v)
    add("oda:1+0", ["Stüdyo mu?", "1+0 stüdyo bir daire mi?"], lambda l: l["oda"] == "1+0")
    for oda in ODA_TIPLERI[1:]:
        add("oda:" + oda, [t.format(X=oda) for t in ODA_T], lambda l, v=oda: l["oda"] == v)
    add("oda:>=3", ["En az üç odalı mı?", "Üç oda ve üzeri mi?"], lambda l: int(l["oda"][0]) >= 3)
    add("oda:>=2", ["Daire en az 2+1 mi?", "İki veya daha fazla odası var mı?", "En az iki odalı mı?"],
        lambda l: int(l["oda"][0]) >= 2)
    add("oda:>=4", ["Daire en az 4+1 mi?", "Dört veya daha fazla odası var mı?", "En az dört odalı mı?"],
        lambda l: int(l["oda"][0]) >= 4)
    add("oda:<=1", ["Daire en fazla 1+1 mi?", "Bir odalı ya da stüdyo mu?", "Küçük bir daire mi, en fazla 1+1?"],
        lambda l: int(l["oda"][0]) <= 1)
    add("kat:zemin", ["Daire zemin kat mı?", "Zemin kattaki bir daire mi?"], lambda l: l["kat"] == "zemin kat")
    add("kat:>=10", ["Daire 10 ve üzeri bir katta mı?", "Kat numarası 10 veya daha büyük mü?"], lambda l: _kat_no(l) >= 10)
    add("kat:>=5", ["Daire 5. kat veya üstünde mi?", "Kat numarası 5 veya daha büyük mü?", "Daire 5 ve üzeri bir katta mı?"],
        lambda l: _kat_no(l) >= 5)
    add("kat:ara", ["Daire ara katta mı?", "Ara kat mı?", "Ne en alt ne en üst, ara katta mı?"], lambda l: l["kat"] == "ara kat")
    for t in (60, 70, 80, 90, 100, 140, 150, 160, 180, 200, 220, 240):
        add("m2:>%d" % t, [x.format(X=t) for x in M2_GT_T], lambda l, t=t: l["m2"] > t)
    for t in (70, 90, 100, 150, 200):
        add("m2:<%d" % t, [x.format(X=t) for x in M2_LT_T], lambda l, t=t: l["m2"] < t)
    for t in (2, 3, 4, 5, 6, 10, 12, 15, 18):
        add("fiyat:<%dM" % t, [x.format(X=t) for x in FIYAT_LT_T],
            lambda l, t=t: l["durum"] == "satılık" and l["fiyat"] < t * 1_000_000)
    for t in (5, 10, 15):
        add("fiyat:>%dM" % t, [x.format(X=t) for x in FIYAT_GT_T],
            lambda l, t=t: l["durum"] == "satılık" and l["fiyat"] > t * 1_000_000)
    for t in (10, 15, 20, 25, 30, 50, 60, 70):
        add("kira:<%dk" % t, [x.format(X=t) for x in KIRA_LT_T],
            lambda l, t=t: l["durum"] == "kiralık" and l["fiyat"] < t * 1_000)
    for t in (20, 50):
        add("kira:>%dk" % t, [x.format(X=t) for x in KIRA_GT_T],
            lambda l, t=t: l["durum"] == "kiralık" and l["fiyat"] > t * 1_000)

    for c in cs:
        assert not FORBIDDEN & set(c["texts"]), (c["id"], FORBIDDEN & set(c["texts"]))
        assert c["texts"], c["id"]
    ids = [c["id"] for c in cs]
    assert len(ids) == len(set(ids)), "tekrarlanan kavram"
    return cs


CONCEPTS = build_concepts()

if __name__ == "__main__":
    import json

    import numpy as np

    docs = [json.loads(l) for l in open("data/train.jsonl", encoding="utf-8")]
    y = np.array([[c["fn"](d) for c in CONCEPTS] for d in docs], dtype=np.uint8)
    np.save("data/y_concepts_train.npy", y)
    json.dump([{k: v for k, v in c.items() if k != "fn"} for c in CONCEPTS],
              open("data/concepts.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    new = [c for c in CONCEPTS if c["bank"] is None]
    print("kavram: %d (bankadan %d, yeni %d) | söyleniş: %d | pozitif oranı %.3f" % (
        len(CONCEPTS), len(CONCEPTS) - len(new), len(new), sum(len(c["texts"]) for c in CONCEPTS), y.mean()))
