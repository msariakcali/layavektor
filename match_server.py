"""Kitap eşleştirme denemesi (eğitimsiz): "bilimle ilgili ama iç dünyamla da alakalı bir şey okumak istiyorum"
gibi bir isteği kriterlere ayırır, her paragraf için kriter başına olasılık hesaplar ve kitap/bölüm düzeyinde
birleştirir.

  python match_server.py  ->  http://localhost:8767
  BOOKS_DIR=<klasör>      içindeki tüm PDF'ler (library.py; tekrar baskılar ve metinsiz PDF'ler atlanır)
İlk açılışta her kitap için v3 (8 bit kelime vektörleri) ve bge-m3 indeksi hesaplanıp data/books/pdf/ altına
kaydedilir; sonraki açılışlarda sadece yeni kitaplar işlenir.

Birleştirme (paragraf i, kriter c; p_ic = v3'ün "bu metin c ile ilgili mi?" olasılığı, istenmeyen kriterde 1 - p):
  birlikte         = p_i = Π_c p_ic                kriterlerin hepsi AYNI paragrafta (bağımsızlık varsayımı)
  COUNT=soft       kapsam = ortalama_i p_ic, uyan sayısı = Σ_i p_i   (kalibre modelde beklenen değer)
  COUNT=hard       kapsam = oran(p_ic >= eşik), uyan = her kriter >= eşik olan paragraf sayısı   (varsayılan)
  karar            = uyan sayısı < 1 ise "uymuyor" (kütüphanede hiçbiri uymuyorsa "uygun kitap yok")
Soft sayım ancak olasılıklar kalibre ise anlamlıdır; v3 sadece emlakla eğitildiği için buradaki oranlar
yön gösterir, doğru sayılmamalı. Soru kalıbına duyarlılığı azaltmak için her kriter 4 kalıpla sorulup ortalanır.
Karşılaştırma: bge-m3 aynı isteğe en benzer paragrafları getirir ama skorundan oran ya da "uymuyor" çıkmaz.
"""
import hashlib
import os
import re
import sys
import threading
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

import library
from library import norm
from models import DOC_LEN, Q_LEN, load_student, tokenize

STUDENT = os.environ.get("STUDENT", "weights/v3/student")
TEMPLATES = ["Bu metin {x} ile ilgili mi?", "Metinde {x} anlatılıyor mu?", "Metin {x} hakkında mı?",
             "Metinde {x} var mı?"]
# Sayım: soft = olasılıkların toplamı (kalibre modelde doğru tahmin), hard = olasılığı eşiği geçen paragraflar.
# Kalibre olmayan modelde (v3, bu alan) 640 paragrafın küçük olasılıkları toplanınca gürültü birikir
# ("futbol" hiçbir paragrafta %11'i geçmediği halde toplam ~4,5 paragraf) -> varsayılan hard.
COUNT = os.environ.get("COUNT", "hard")
MIN_EXPECTED = 1.0      # uyan paragraf sayısı bunun altındaysa kitap isteğe uymuyor
MIN_FIT = 0.03          # uyan paragraf oranı bunun (ve 5 paragrafın) altındaysa "kısmen"
THIN = 0.02             # kapsamı bunun altındaki kriter "neredeyse yok" sayılır
N_EVIDENCE = 5          # kitap başına kanıt paragrafı
N_TOP = 8               # kütüphane genelinde en iyi paragraflar

# ------------------------------------------------------------------ istek -> kriterler (kural tabanlı, yer tutucu)
# Gerçek üründe bu adım küçük bir modelle yapılabilir; burada arayüz kriterleri düzenlenebilir gösterir.
_SPLIT = {"ve", "ama", "ancak", "fakat", "hem", "ayrıca", "alakalı", "ilgili", "hakkında", "üzerine", "konulu"}
_NEG = {"olmasın", "olmayan", "değil", "istemiyorum", "istemem", "hariç", "dışında", "içermesin", "girmesin"}
_FILLER = re.compile(r"(?:ne|okusam|okuyayım|okumak|ist\w*|bir|şey\w*|biraz|daha|çok|az|mesela|ya|yine|şöyle|benim|"
                     r"bana|olsun|olan|kitap\w*|metin\w*|ile|de|da|gibi|tarzı?|türü?|konusu|roman\w*)$")
_CASE = re.compile(r"(?<=\w{3})(?:y?l[ae]|y[ae]|[dt][ae]n)$")   # bilimle -> bilim, felsefeye -> felsefe


def parse(wish):
    """Kelime düzeyinde: bağlaçlarda böl, dolgu kelimelerini at, olumsuzluğu işaretle; özgün yazım (Popper) korunur."""
    crits, seg, neg = [], [], False

    def close():
        if seg:
            if len(seg[-1]) > 4:
                seg[-1] = _CASE.sub("", seg[-1])
            text = " ".join(seg)
            if len(text) >= 3 and text.lower() not in [c["text"].lower() for c in crits]:
                crits.append({"text": text, "neg": neg})

    for w in re.findall(r"[\w']+|[,;.]", wish):
        n = norm(w)
        if n in _SPLIT or not n[0].isalnum():
            close()
            seg, neg = [], False
        elif n in _NEG:
            neg = True
        elif not _FILLER.fullmatch(n):
            seg.append(w)
    close()
    return crits


# ------------------------------------------------------------------ yükleme ve indeksleme
BOOKS, CHUNKS, cache, SKIPPED = library.load()
TEXTS = [c["text"] for c in CHUNKS]
BOOK = {b["id"]: b for b in BOOKS}
BOOK_IDX = {b["id"]: np.array([i for s in b["sections"] for i in s["idx"]]) for b in BOOKS}
print("%d kitap, %d paragraf, %d atlandı" % (len(BOOKS), len(CHUNKS), len(SKIPPED)), flush=True)

student, stok = load_student(STUDENT)
student = student.to(torch.bfloat16)
_tag = hashlib.md5(os.path.abspath(STUDENT).encode()).hexdigest()[:6]


def book_cache(b, name):
    return os.path.join(library.CACHE, "pdf", "%s.%s" % (b["key"], name))


@torch.inference_mode()
def v3_encode(texts):
    """Kelime vektörleri 8 bit saklanır (normalize vektörler, x127): 4 GB GPU'da ~19 bin paragraf ~0,6 GB."""
    out = []
    for i in range(0, len(texts), 32):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            d = student.encode_docs(*tokenize(stok, texts[i:i + 32], DOC_LEN))
        out.append({"tok": (d["tok"] * 127).round().clamp(-127, 127).to(torch.int8).cpu(), "mask": d["mask"].cpu()})
    return out


t0, V3, BGE = time.time(), [], []
from sentence_transformers import SentenceTransformer   # noqa: E402

bge = SentenceTransformer("BAAI/bge-m3", device="cuda", model_kwargs={"dtype": torch.float16})
for n, b in enumerate(BOOKS, 1):
    texts = [TEXTS[i] for i in BOOK_IDX[b["id"]]]
    assert list(BOOK_IDX[b["id"]]) == list(range(BOOK_IDX[b["id"]][0], BOOK_IDX[b["id"]][0] + len(texts)))
    path = book_cache(b, "v3-%s.pt" % _tag)
    if not os.path.exists(path):
        t = time.time()
        torch.save(v3_encode(texts), path)
        print("  [%d/%d] v3 %s: %d paragraf, %.0f sn" % (n, len(BOOKS), b["id"], len(texts), time.time() - t), flush=True)
    V3 += [{k: v.cuda() for k, v in blk.items()} for blk in torch.load(path)]
    path = book_cache(b, "bge.npy")
    if not os.path.exists(path):
        t = time.time()
        np.save(path, bge.encode(texts, batch_size=16, normalize_embeddings=True, convert_to_numpy=True).astype(np.float16))
        print("  [%d/%d] bge %s: %.0f sn" % (n, len(BOOKS), b["id"], time.time() - t), flush=True)
    BGE.append(np.load(path))
BGE_T = torch.tensor(np.concatenate(BGE), device="cuda", dtype=torch.float16)
assert sum(blk["tok"].shape[0] for blk in V3) == len(TEXTS) == BGE_T.shape[0]
print("hazır: %.0f sn, v3 indeksi %.0f MB (8 bit), GPU %.1f GB" % (
    time.time() - t0, sum(blk["tok"].numel() for blk in V3) / 1e6, torch.cuda.memory_allocated() / 1e9), flush=True)

LOCK = threading.Lock()
app = FastAPI()


@torch.inference_mode()
def v3_probs(crits):
    """-> [paragraf, kriter] olasılık; her kriter TEMPLATES kalıplarıyla sorulup olasılıklar ortalanır.
    v3'ün token yolu: logit = a_q * maxsim + b_q (MultiStudent.score, combine="token" ile aynı)."""
    qs = [t.format(x=c) for c in crits for t in TEMPLATES]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        q = student.encode_questions(*tokenize(stok, qs, Q_LEN))
        s = torch.cat([student.maxsim({"tok": blk["tok"].to(torch.bfloat16) / 127, "mask": blk["mask"]}, q)
                       for blk in V3]).float()
        p = torch.sigmoid(q["a"] * s + q["b"])
    return p.view(len(TEXTS), len(crits), len(TEMPLATES)).mean(-1).cpu().numpy()


def para(i, P, joint):
    c = CHUNKS[i]
    return {"i": int(i), "page": c["page"], "page_end": c["page_end"], "text": c["text"],
            "probs": [round(float(x), 3) for x in P[i]], "joint": round(float(joint[i]), 3)}


def book_report(b, P, joint, crits, thr):
    idx = BOOK_IDX[b]
    M, J = (P >= thr, (P >= thr).all(1)) if COUNT == "hard" else (P, joint)   # sayılacak büyüklükler
    cover = M[idx].mean(0)
    expected = float(J[idx].sum())
    top = idx[np.argsort(-joint[idx])[:N_EVIDENCE]]
    secs = [{"title": s["title"], "page": s["page"], "n": len(s["idx"]),
             "cover": [round(float(x), 3) for x in M[s["idx"]].mean(0)],
             "joint": round(float(J[s["idx"]].mean()), 3), "expected": round(float(J[s["idx"]].sum()), 1)}
            for s in BOOK[b]["sections"]]
    thin = [crits[k]["label"] for k in range(len(crits)) if cover[k] < THIN]
    if expected < MIN_EXPECTED:
        verdict = "no"
    elif expected < max(5, MIN_FIT * len(idx)):
        verdict = "part"
    else:
        verdict = "yes"
    return {"book": b, "title": BOOK[b]["title"], "lang": BOOK[b]["lang"], "n": len(idx), "cover": [round(float(x), 3) for x in cover],
            "joint": round(float(J[idx].mean()), 3), "expected": round(expected, 1), "verdict": verdict,
            "soft": round(float(joint[idx].mean()), 4),
            "thin": thin, "sections": secs, "evidence": [para(i, P, joint) for i in top]}


class ParseReq(BaseModel):
    wish: str


class Crit(BaseModel):
    text: str
    neg: bool = False


class MatchReq(BaseModel):
    wish: str = ""
    criteria: list[Crit]
    threshold: float = 0.5


@app.post("/api/parse")
def api_parse(req: ParseReq):
    return {"criteria": parse(req.wish)}


@app.post("/api/match")
def api_match(req: MatchReq):
    crits = [{"text": c.text.strip(), "neg": c.neg} for c in req.criteria if c.text.strip()][:6]
    if not crits:
        return {"error": "En az bir kriter gerekli."}
    for c in crits:
        c["label"] = ("%s olmasın" if c["neg"] else "%s") % c["text"]
    with LOCK:
        t = time.time()
        P = v3_probs([c["text"] for c in crits])
        P = np.where([c["neg"] for c in crits], 1 - P, P)   # DEĞİL = 1 - P
        joint = P.prod(1)
        books = sorted((book_report(b["id"], P, joint, crits, req.threshold) for b in BOOKS),
                       key=lambda r: (-r["joint"], -r["soft"]))   # eşit sayımda olasılık ortalaması
        top = np.argsort(-joint)[:N_TOP]
        torch.cuda.synchronize()
        ms_v3 = round((time.time() - t) * 1000, 1)

        t = time.time()
        wish = req.wish.strip() or " ve ".join(c["label"] for c in crits)
        qv = bge.encode([wish], normalize_embeddings=True, convert_to_tensor=True).to(torch.float16)
        sb = (BGE_T @ qv[0]).float().cpu().numpy()
        top_b = np.argsort(-sb)[:N_EVIDENCE]
        ms_bge = round((time.time() - t) * 1000, 1)
    return {
        "criteria": crits, "books": books, "ms": ms_v3, "count": COUNT, "threshold": req.threshold,
        "none": all(r["verdict"] == "no" for r in books),
        "top": [dict(para(i, P, joint), book=CHUNKS[i]["book"], title=BOOK[CHUNKS[i]["book"]]["title"]) for i in top],
        "bge": {"ms": ms_bge, "query": wish,
                "results": [{"book": CHUNKS[i]["book"], "title": BOOK[CHUNKS[i]["book"]]["title"],
                             "page": CHUNKS[i]["page"], "score": round(float(sb[i]), 3),
                             "text": TEXTS[i]} for i in top_b],
                "book_scores": sorted(({"book": b["id"], "title": b["title"],
                                        "score": round(float(np.sort(sb[BOOK_IDX[b["id"]]])[-10:].mean()), 3)}
                                       for b in BOOKS), key=lambda r: -r["score"])},
    }


@app.get("/api/info")
def info():
    return {"n_chunks": len(CHUNKS), "student": STUDENT, "templates": TEMPLATES, "skipped": SKIPPED,
            "books": [{"book": b["id"], "title": b["title"], "lang": b["lang"], "pages": b["pages"],
                       "n": len(BOOK_IDX[b["id"]]), "sections": len(b["sections"])} for b in BOOKS]}


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "match.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8767)))
