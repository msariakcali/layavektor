"""Kitap arama denemesi (eğitimsiz): PDF -> paragraflar -> dört yöntemle arama, yan yana.

  python book_server.py  ->  http://localhost:8766
  BOOKS_DIR=<klasör>     içindeki tüm PDF'ler indekslenir (varsayılan: bu klasörün bir üstü)

Yöntemler (hiçbiri bu kitapla ya da kitap göreviyle eğitilmedi):
  bm25        kelime araması (Türkçe için kelimelerin ilk 5 harfi)
  bge         bge-m3 embedding, kosinüs benzerliği — standart RAG
  v3          bizim öğrenci (weights/v3), sadece emlak ilanlarıyla eğitildi; olasılıkları bu alanda kalibre değil
  laya        bge-m3'ün ilk 30 sonucu, hazır Laya (ince ayarsız) ile "bu metin soruyu cevaplıyor mu?" sıralaması
Paragraflar ve indeksler data/books/ altında önbelleğe alınır (git dışı).
"""
import glob
import hashlib
import json
import os
import re
import sys
import threading
import time
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

import fitz
import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from models import DOC_LEN, Q_LEN, load_laya, load_student, tokenize

BOOKS_DIR = os.environ.get("BOOKS_DIR", os.path.dirname(HERE))
STUDENT = os.environ.get("STUDENT", "weights/v3/student")
CHUNK_WORDS, STRIDE = 120, 90          # ~250 token: v3'ün 256 token sınırına sığar
RERANK_K = 30

# ------------------------------------------------------------------ PDF -> paragraflar
_TR = str.maketrans({"İ": "i", "I": "ı"})


def norm(s):
    return s.translate(_TR).lower()


def pdf_chunks(path):
    doc = fitz.open(path)
    pages = [p.get_text() for p in doc]
    # sayfa başlığı/altlığı: birçok sayfada tekrar eden satırlar
    freq = Counter(norm(l.strip()) for t in pages for l in set(t.split("\n")) if l.strip())
    headers = {l for l, c in freq.items() if c >= max(5, len(pages) // 10)}
    words = []   # (kelime, sayfa)
    for pno, t in enumerate(pages, 1):
        if len(re.findall(r"(?:\.\s*){5,}", t)) >= 5:   # içindekiler sayfası: başlıklar arama sonucu olarak işe yaramaz
            continue
        lines = []
        for l in t.split("\n"):
            s = l.strip()
            if (not s or s in "•·-–" or re.fullmatch(r"[\divxlcIVXLC]{1,4}", s) or re.search(r"(\.\s*){5,}", s)
                    or norm(s) in headers):
                continue
            lines.append(s)
        text = re.sub(r"­\s*", "", " ".join(lines))          # yumuşak tire: "yürütme­ lere"
        text = re.sub(r"(\w)- (\w)", r"\1\2", text)               # satır sonunda bölünmüş kelimeler
        words += [(w, pno) for w in text.split()]
    chunks = []
    for i in range(0, max(1, len(words) - CHUNK_WORDS // 3), STRIDE):
        part = words[i:i + CHUNK_WORDS]
        text = " ".join(w for w, _ in part)
        if sum(ch.isalpha() for ch in text) < 0.6 * len(text):   # içindekiler / tablo artığı
            continue
        chunks.append({"book": os.path.splitext(os.path.basename(path))[0], "page": part[0][1],
                       "page_end": part[-1][1], "text": text})
    return chunks


def stem(w):
    return w[:5]


def bm25_tokens(t):
    return [stem(w) for w in re.findall(r"\w+", norm(t)) if len(w) > 1]


# ------------------------------------------------------------------ yükleme ve indeksleme
pdfs = sorted(glob.glob(os.path.join(BOOKS_DIR, "*.pdf")))
assert pdfs, "PDF bulunamadı: %s" % BOOKS_DIR
key = hashlib.md5("|".join("%s:%d" % (p, os.path.getsize(p)) for p in pdfs).encode()).hexdigest()[:10]
cache = os.path.join("data", "books", key)
os.makedirs(cache, exist_ok=True)
if os.path.exists(os.path.join(cache, "chunks.json")):
    CHUNKS = json.load(open(os.path.join(cache, "chunks.json"), encoding="utf-8"))
else:
    CHUNKS = [c for p in pdfs for c in pdf_chunks(p)]
    json.dump(CHUNKS, open(os.path.join(cache, "chunks.json"), "w", encoding="utf-8"), ensure_ascii=False)
TEXTS = [c["text"] for c in CHUNKS]
print("%d PDF, %d paragraf" % (len(pdfs), len(CHUNKS)), flush=True)
TIMING = {}

t = time.time()
BM25 = BM25Okapi([bm25_tokens(x) for x in TEXTS])
TIMING["bm25_index_sec"] = time.time() - t

from sentence_transformers import SentenceTransformer   # noqa: E402

t = time.time()
bge = SentenceTransformer("BAAI/bge-m3", device="cuda", model_kwargs={"dtype": torch.float16})
bge_path = os.path.join(cache, "bge.npy")
if os.path.exists(bge_path):
    BGE = np.load(bge_path)
else:
    BGE = bge.encode(TEXTS, batch_size=16, normalize_embeddings=True, convert_to_numpy=True)
    np.save(bge_path, BGE)
BGE_T = torch.tensor(BGE, device="cuda", dtype=torch.float16)
TIMING["bge_index_sec"] = time.time() - t

t = time.time()
student, stok = load_student(STUDENT)
student = student.to(torch.bfloat16)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    V3 = [student.encode_docs(*tokenize(stok, TEXTS[i:i + 32], DOC_LEN)) for i in range(0, len(TEXTS), 32)]
TIMING["v3_index_sec"] = time.time() - t
TIMING["v3_index_mb"] = sum(int(d["mask"].sum()) for d in V3) * student.arch.get("tdim", 128) * 2 / 1e6

laya_agent = load_laya()
laya_agent.model.to(torch.bfloat16)
print("hazır: %s" % {k: round(v, 1) for k, v in TIMING.items()}, flush=True)

LOCK = threading.Lock()
app = FastAPI()


def sync_ms(t0):
    torch.cuda.synchronize()
    return round((time.time() - t0) * 1000, 1)


def pack(idx, scores, prob=False):
    return [{"i": int(i), "score": float(s), "prob": prob, "book": CHUNKS[i]["book"], "page": CHUNKS[i]["page"],
             "page_end": CHUNKS[i]["page_end"], "text": TEXTS[i]} for i, s in zip(idx, scores)]


@torch.inference_mode()
def laya_probs(question, idx):
    from laya.common import QTYPES, build_sequence, collate_items
    a = laya_agent
    q = {"t": "noul", "ins": "Bu metin şu soruyu cevaplıyor mu: " + question, "crit": None}
    items = []
    for i in idx:
        ids, markers = build_sequence(a.tok, TEXTS[i], q, a.cfg["max_len"], a.cfg["head_max_len"])
        items.append({"ids": ids, "markers": markers, "qtype": QTYPES["noul"]})
    probs = []
    for k in range(0, len(items), 16):
        b = collate_items([items[k:k + 16]], a.tok.pad_token_id)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = a.model(b["input_ids"].cuda(), b["attention_mask"].cuda(), b["marker_pos"].cuda(),
                                b["marker_mask"].cuda(), b["qtype"].cuda())
        probs += torch.softmax(logits[:, :2].float(), -1)[:, 1].tolist()
    return probs


class Req(BaseModel):
    question: str
    top_k: int = 8


@app.post("/api/search")
def search(req: Req):
    qtext = req.question.strip()
    k = req.top_k
    out = {}
    with LOCK:
        t = time.time()
        s = BM25.get_scores(bm25_tokens(qtext))
        top = np.argsort(-s)[:k]
        out["bm25"] = {"ms": round((time.time() - t) * 1000, 1), "results": pack(top, s[top])}

        t = time.time()
        qv = bge.encode([qtext], normalize_embeddings=True, convert_to_tensor=True).to(torch.float16)
        sb = (BGE_T @ qv[0]).float()
        top_b = torch.topk(sb, max(k, RERANK_K)).indices.tolist()
        ms_b = sync_ms(t)
        sb = sb.cpu().numpy()
        out["bge"] = {"ms": ms_b, "results": pack(top_b[:k], sb[top_b[:k]])}

        t = time.time()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            qe = student.encode_questions(*tokenize(stok, [qtext], Q_LEN))
            p = torch.sigmoid(torch.cat([student.score(d, qe) for d in V3]).float()).squeeze(1)
        top_v = torch.topk(p, k).indices.tolist()
        ms_v = sync_ms(t)
        p = p.cpu().numpy()
        out["v3"] = {"ms": ms_v, "results": pack(top_v, p[top_v], prob=True)}

        t = time.time()
        lp = laya_probs(qtext, top_b[:RERANK_K])
        order = np.argsort(-np.array(lp))[:k]
        ms_l = sync_ms(t)
        out["laya"] = {"ms": round(ms_b + ms_l, 1), "results": pack([top_b[j] for j in order], [lp[j] for j in order], prob=True)}
    return {"n_chunks": len(CHUNKS), "books": sorted({c["book"] for c in CHUNKS}), "methods": out}


@app.get("/api/info")
def info():
    return {"n_chunks": len(CHUNKS), "books": sorted({c["book"] for c in CHUNKS}), "timing": TIMING, "student": STUDENT}


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "book.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8766)))
