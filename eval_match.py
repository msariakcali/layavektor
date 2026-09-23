"""Kitap eşleştirme ölçümü: "doğru kitap kaçıncı sırada?" (match_server.py çalışırken).

  python eval_match.py            ->  results/match_eval.json

Doğru cevaplar kitap başlıklarından: her istek bir kitabın (ya da aynı konudaki birkaç kitabın) konusunu,
başlıktaki kelimeleri mümkün olduğunca kullanmadan anlatır. Kütüphanede olmayan konular "yok" testi içindir.
Karşılaştırılan sıralamalar:
  v3     kitap raporundaki sıra (uyan paragraf oranı, eşit olanlarda olasılık ortalaması)
  bge    bge-m3: kitabın isteğe en benzer 10 paragrafının kosinüs ortalaması
Ölçütler: doğru kitabın sırası (birden fazla doğru varsa en iyisi), ilk 1 / ilk 3 isabet, MRR;
"yok" testinde v3'ün "uygun kitap yok" deyip demediği.
"""
import json
import os
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
URL = os.environ.get("URL", "http://127.0.0.1:8767")

CASES = [   # (istek, doğru kitap id'leri)
    ("oruç tutmanın hükümleri", ["HHRO_TBY_1b", "ramazan_mektubu_full"]),
    ("kalbin sertleşmesi ve katılaşması", ["kalpkatiligi_full"]),
    ("sufilik ve tarikat yolu", ["Tasavvuf_TBY_Full", "Rabita_Full"]),
    ("şeyhi zihinde canlandırarak bağlanma", ["Rabita_Full"]),
    ("alışverişte ve ticarette ahlak", ["TEMN_TBY"]),
    ("bir dilden başka bir dile çeviri yapmak", ["Tercume_Bilimi"]),
    ("kadınlara öğütler", ["MHN_TBY_full"]),
    ("gençlerle sohbet", ["Genclerle_Hasbihal", "7_İbrahimi_Genc"]),
    ("Allah'ın güzel isimlerinin anlamları", ["EsmaTBY_1_2_Full"]),
    ("En'am suresinin açıklaması", ["VREST_TBY_1b_full"]),
    ("âlimler arasındaki görüş ayrılıkları", ["10_İhtilaf_fikhi"]),
    ("bilimsel yöntem ve bilginin doğası", ["Kadir Çüçen - Bilim Felsefesine Giriş"]),
    ("peygamberlerin hepsinin ortak daveti", ["17_TROM_full"]),
    ("Müslümanların kardeşlik hakları", ["3_MBKS"]),
    ("iman esasları dersleri", ["AD_6_Baski_full"]),
    ("dini doğru anlamak için temel kurallar", ["12_Kavaidul_Erba"]),
    ("günlük hayattan kısa hikâyeler", ["Hayatin_icinden_TBY_1b_2000_adet_Full", "4_YMO"]),
    ("tevhid kelimesinin anlamı ve şartları", ["Lailaheillallah_14b"]),
    ("sahabelerin hayatından örnekler", ["14_SCT"]),
    ("tek bir kişinin rivayet ettiği hadisler", ["AHAD"]),
]
NONE_CASES = ["futbol taktikleri", "yemek tarifleri", "kuantum fiziği", "aşk romanı", "borsa ve yatırım"]


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode("utf-8"), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


def run(wish):
    crits = post("/api/parse", {"wish": wish})["criteria"] or [{"text": wish, "neg": False}]
    return post("/api/match", {"wish": wish, "criteria": crits}), crits


rows, out = [], {"cases": [], "none": []}
for wish, gold in CASES:
    d, crits = run(wish)
    v3 = [b["book"] for b in d["books"]]
    bge = [b["book"] for b in d["bge"]["book_scores"]]
    rv = min(v3.index(g) for g in gold) + 1
    rb = min(bge.index(g) for g in gold) + 1
    verdict = d["books"][rv - 1]["verdict"]
    rows.append((rv, rb))
    out["cases"].append({"wish": wish, "criteria": [c["text"] for c in crits], "gold": gold, "v3_rank": rv, "bge_rank": rb,
                         "v3_verdict_gold": verdict, "v3_top3": v3[:3], "bge_top3": bge[:3], "ms": d["ms"],
                         "bge_top_score": d["bge"]["book_scores"][0]["score"]})
    print("%-44s v3 %2d (%s)  bge %2d   v3 ilk: %s" % (wish, rv, verdict, rb, d["books"][0]["title"][:40]))

for wish in NONE_CASES:
    d, _ = run(wish)
    out["none"].append({"wish": wish, "v3_none": d["none"], "v3_top": d["books"][0]["title"],
                        "v3_top_verdict": d["books"][0]["verdict"], "bge_top_score": d["bge"]["book_scores"][0]["score"]})
    print("%-44s yok=%s  (v3 ilk: %s, %s; bge en iyi kitap skoru %.3f)" % (
        wish, d["none"], d["books"][0]["title"][:30], d["books"][0]["verdict"], d["bge"]["book_scores"][0]["score"]))

n = len(rows)
summary = {name: {"top1": sum(r[k] == 1 for r in rows) / n, "top3": sum(r[k] <= 3 for r in rows) / n,
                  "mrr": sum(1 / r[k] for r in rows) / n} for k, name in ((0, "v3"), (1, "bge"))}
summary["v3_gold_not_no"] = sum(c["v3_verdict_gold"] != "no" for c in out["cases"]) / n
summary["v3_none_correct"] = sum(c["v3_none"] for c in out["none"]) / len(NONE_CASES)
# bge için "yok" eşiği var mı? kütüphanede olan konuların en düşük skoru, olmayanların en yüksek skorunu geçiyor mu
summary["bge_min_score_in_library"] = min(c["bge_top_score"] for c in out["cases"])
summary["bge_max_score_not_in_library"] = max(c["bge_top_score"] for c in out["none"])
out["summary"] = summary
print(json.dumps(summary, ensure_ascii=False, indent=1))
os.makedirs("results", exist_ok=True)
json.dump(out, open("results/match_eval.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
