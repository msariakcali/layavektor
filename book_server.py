"""Kitap arama denemesi (eğitimsiz): PDF -> paragraflar -> dört yöntemle arama, yan yana.

  python book_server.py  ->  http://localhost:8766
  BOOKS_DIR=<klasör>     içindeki tüm PDF'ler indekslenir (varsayılan: bu klasörün bir üstündeki PDF'ler)

Yöntemler (hiçbiri bu kitapla ya da kitap göreviyle eğitilmedi):
  bm25        kelime araması (Türkçe için kelimelerin ilk 5 harfi)
  bge         bge-m3 embedding, kosinüs benzerliği — standart RAG
  v3          bizim öğrenci (weights/v3), sadece emlak ilanlarıyla eğitildi; olasılıkları bu alanda kalibre değil
  laya        bge-m3'ün ilk 30 sonucu, hazır Laya (ince ayarsız) ile "bu metin soruyu cevaplıyor mu?" sıralaması
Paragraflar (library.py) ve indeksler data/books/ altında önbelleğe alınır (git dışı).
"""
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
from rank_bm25 import BM25Okapi

import library
from library import norm
from models import DOC_LEN, Q_LEN, load_laya, load_student, tokenize

STUDENT = os.environ.get("STUDENT", "weights/v3/student")
RERANK_K = 30


def stem(w):
    return w[:5]


def bm25_tokens(t):
    return [stem(w) for w in re.findall(r"\w+", norm(t)) if len(w) > 1]


# ------------------------------------------------------------------ yükleme ve indeksleme
# varsayılan: sadece bu klasörün bir üstündeki PDF'ler (eski davranış; tüm kütüphane 4 GB GPU'da bge + v3 + Laya ile
# sığmıyor, onun için match_server.py). BOOKS_DIR ile başka klasör verilebilir.
pdfs, CHUNKS, cache, _ = library.load(pats=None if os.environ.get("BOOKS_DIR") else [library.DEFAULT_PATTERNS[0]])
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
