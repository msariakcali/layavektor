"""Deneme arayüzü: öğrenci modeliyle 2.000 test ilanında (eğitimde görülmemiş) arama.

  python demo_server.py                              ->  http://localhost:8765  (v3, token vektörleri)
  STUDENT=weights/v2/student python demo_server.py   ->  v2 (tek vektör)

Her koşul bir evet/hayır sorusu; koşulların olasılıkları çarpılır (olumsuz koşulda 1-P).
"Laya ile doğrula" gösterilen sonuçları ince ayarlı cross-encoder'a (v1) tekrar sordurur.
"""
import json
import os
import sys
import threading
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from models import DOC_LEN, Q_LEN, load_student, tokenize

STUDENT = os.environ.get("STUDENT", "weights/v3/student")
CROSS = os.environ.get("CROSS", "weights/v1/laya-cross-ft")
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

model, tok = load_student(STUDENT)
DOCS = [json.loads(l) for l in open("data/test.jsonl", encoding="utf-8")]
t0 = time.time()
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    D = [model.encode_docs(*tokenize(tok, [d["text"] for d in DOCS[i:i + 64]], DOC_LEN))   # indeks parçaları
         for i in range(0, len(DOCS), 64)]
    for n in range(1, 6):   # ısınma: ilk sorgular farklı uzunluklarda yavaş çalışıyor
        qe = model.encode_questions(*tokenize(tok, ["Balkonu var mı?"] * n + ["Satılık ve fiyatı 5 milyon TL'nin altında mı?"], Q_LEN))
        torch.cat([model.score(d, qe) for d in D])
INDEX_SEC = time.time() - t0
MODEL_NAME = "%s (%s)" % (STUDENT, model.arch["type"] + ("/" + model.arch["combine"] if "combine" in model.arch else ""))
print("indeks hazır: %s, %d ilan, %.1f sn" % (MODEL_NAME, len(DOCS), INDEX_SEC), flush=True)

LOCK = threading.Lock()
_cross = {}
app = FastAPI()


class Cond(BaseModel):
    text: str
    neg: bool = False


class SearchReq(BaseModel):
    conds: list[Cond]
    top_k: int = 20


class VerifyReq(BaseModel):
    conds: list[Cond]
    ids: list[int]


def sync_ms(t):
    torch.cuda.synchronize()
    return round((time.time() - t) * 1000, 2)


@app.post("/api/search")
def search(req: SearchReq):
    conds = [c for c in req.conds if c.text.strip()]
    if not conds:
        return {"results": []}
    with LOCK, torch.inference_mode():
        torch.cuda.synchronize()
        t = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q = model.encode_questions(*tokenize(tok, [c.text.strip() for c in conds], Q_LEN))
        enc_ms = sync_ms(t)
        t = time.time()
        P = torch.sigmoid(torch.cat([model.score(d, q) for d in D]).float())   # [ilan, koşul]
        neg = torch.tensor([c.neg for c in conds], device=P.device)
        P = torch.where(neg, 1 - P, P)
        score = P.prod(1)
        top = score.topk(min(req.top_k, len(DOCS)))
        scan_ms = sync_ms(t)
    P, score = P.cpu(), score.cpu()
    results = []
    for s, i in zip(top.values.tolist(), top.indices.tolist()):
        d = DOCS[i]
        results.append({"i": i, "score": s, "per": P[i].tolist(), "text": d["text"],
                        "fields": {"ilçe": d["ilce"], "mahalle": d["mahalle"], "oda": d["oda"], "m²": d["m2"],
                                   "kat": d["kat"], "durum": d["durum"], "fiyat": d["fiyat"],
                                   "özellikler": d["ozellikler"]}})
    return {"results": results, "n_docs": len(DOCS), "model": MODEL_NAME, "expected_count": float(score.sum()),
            "n_over_50": int((score >= 0.5).sum()), "max": float(score.max()),
            "enc_ms": enc_ms, "scan_ms": scan_ms}


def cross_agent():
    if "a" not in _cross:
        sys.path.insert(0, os.environ.get("LAYA_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "laya-check")))
        import laya
        _cross["a"] = laya.load(CROSS, device="cuda")
    return _cross["a"]


@app.post("/api/verify")
def verify(req: VerifyReq):
    conds = [c for c in req.conds if c.text.strip()]
    with LOCK:
        a = cross_agent()
        from laya.common import QTYPES, build_sequence, collate_items
        t = time.time()
        items = []
        for i in req.ids:
            for c in conds:
                ids, markers = build_sequence(a.tok, DOCS[i]["text"], {"t": "noul", "ins": c.text.strip(), "crit": None},
                                              a.cfg["max_len"], a.cfg["head_max_len"])
                items.append({"ids": ids, "markers": markers, "qtype": QTYPES["noul"]})
        probs = []
        with torch.inference_mode():
            for k in range(0, len(items), 64):
                b = collate_items([items[k:k + 64]], a.tok.pad_token_id)
                with torch.autocast("cuda", dtype=a.dtype):
                    logits, _ = a.model(b["input_ids"].cuda(), b["attention_mask"].cuda(), b["marker_pos"].cuda(),
                                        b["marker_mask"].cuda(), b["qtype"].cuda())
                probs += torch.softmax(logits[:, :2].float(), -1)[:, 1].tolist()
        ms = sync_ms(t)
    out, k = {}, 0
    for i in req.ids:
        per = []
        for c in conds:
            per.append(1 - probs[k] if c.neg else probs[k])
            k += 1
        score = 1.0
        for p in per:
            score *= p
        out[i] = {"score": score, "per": per}
    return {"cross": out, "ms": ms, "pairs": len(items),
            "full_scan_estimate_sec": ms / 1000 / max(1, len(items)) * len(DOCS) * len(conds)}


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "demo.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8765)))
