"""data/ altına eğitim/test ilanlarını, doğru cevap matrislerini ve birleşik sorguları yazar."""
import json
import os
import random

import numpy as np

from bank import EVAL_VARIANT, QUESTIONS, TRAIN_VARIANTS, describe, random_listing

N_TRAIN, N_TEST = int(os.environ.get("N_TRAIN", 5000)), int(os.environ.get("N_TEST", 2000))
os.makedirs("data", exist_ok=True)


def make(n, seed, prefix, variants):
    rng = random.Random(seed)
    docs = []
    for i in range(n):
        l = random_listing(rng, i, prefix)
        l["text"] = describe(l, rng, variants)
        docs.append(l)
    truth = np.array([[q["fn"](l) for q in QUESTIONS] for l in docs], dtype=np.uint8)
    return docs, truth


train, y_train = make(N_TRAIN, 1, "tr", TRAIN_VARIANTS)
test, y_test = make(N_TEST, 2, "te", TRAIN_VARIANTS + (EVAL_VARIANT,))   # test ilanlarında görülmemiş söylenişler de var


def composite_queries(docs, n, seed):
    """2-3 koşullu sorgular (bazıları olumsuz). Aynı gruptan tek koşul: 'kiralık' + 'satılık' gibi çelişki olmaz."""
    rng = random.Random(seed)
    truth = np.array([[q["fn"](l) for q in QUESTIONS] for l in docs], dtype=bool)
    out, empty = [], 0
    while len(out) < n:
        k = rng.choice([2, 3, 3, 4])
        conds, groups = [], set()
        for qi in rng.sample(range(len(QUESTIONS)), len(QUESTIONS)):
            q = QUESTIONS[qi]
            if q["group"] in groups:
                continue
            neg = q["neg"] is not None and rng.random() < 0.25
            conds.append((qi, neg))
            groups.add(q["group"])
            if len(conds) == k:
                break
        m = np.ones(len(docs), bool)
        for qi, neg in conds:
            m &= ~truth[:, qi] if neg else truth[:, qi]
        n_match = int(m.sum())
        if n_match == 0:
            if empty >= n // 4:   # sorguların ~%25'i bilerek boş: "belgede yok" testi
                continue
            empty += 1
        phrases = [QUESTIONS[qi]["neg"] if neg else QUESTIONS[qi]["phrase"] for qi, neg in conds]
        out.append({"conds": conds, "text": ", ".join(phrases) + " daire", "n_match": n_match,
                    "heldout": any(QUESTIONS[qi]["heldout"] for qi, _ in conds)})
    return out


queries = composite_queries(test, 400, 3)

for name, docs in (("train", train), ("test", test)):
    with open("data/%s.jsonl" % name, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
np.save("data/y_train.npy", y_train)
np.save("data/y_test.npy", y_test)
with open("data/questions.json", "w", encoding="utf-8") as f:
    json.dump([{k: v for k, v in q.items() if k != "fn"} for q in QUESTIONS], f, ensure_ascii=False, indent=1)
with open("data/queries.json", "w", encoding="utf-8") as f:
    json.dump(queries, f, ensure_ascii=False, indent=1)

print("ilan: %d eğitim, %d test | soru: %d (%d görülmemiş) | birleşik sorgu: %d (%d boş)" % (
    len(train), len(test), len(QUESTIONS), sum(q["heldout"] for q in QUESTIONS),
    len(queries), sum(q["n_match"] == 0 for q in queries)))
print("pozitif oranı (eğitim): %.3f" % y_train.mean())
for d in test[:3]:
    print("-", d["text"])
print("örnek sorgu:", queries[0]["text"], "->", queries[0]["n_match"], "eşleşme")
