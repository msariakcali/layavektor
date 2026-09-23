"""Öğrenci (karar embedding'i) vs ince ayarlı Laya cross-encoder vs bge-m3 — test ilanlarında.

Tek soru:   doğruluk, AUC, AP, ECE, Brier, sayma hatası
            A) görülen soru + görülen söyleniş   B) görülen soru + yeni söyleniş
            C) görülmemiş soru                   D) görülmemiş soru + yeni söyleniş
Birleşik:   AP, P@k, "sonuç yok" tespiti, beklenen eşleşme sayısı
Hız:        indeksleme, sorgu başına süre, 1M ilana ölçekleme
  -> results/report_<ckpt adı>.json
"""
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from models import DIM, DOC_LEN, Q_LEN, load_trained, tokenize

CKPT = os.environ.get("CKPT", "ckpt/student_truth.pt")
Q = json.load(open("data/questions.json", encoding="utf-8"))
Y = np.load("data/y_test.npy").astype(bool)
queries = json.load(open("data/queries.json", encoding="utf-8"))
docs = [json.loads(l)["text"] for l in open("data/test.jsonl", encoding="utf-8")]
heldout = np.array([q["heldout"] for q in Q])


# ------------------------------------------------------------------ öğrenci
@torch.inference_mode()
def student_scores():
    """-> ({sistem adı: {söyleniş: P [ilan, soru]}}, süre ölçümleri). Hybrid'de iki yol ayrıca raporlanır."""
    model, tok = load_trained(CKPT)
    arch = model.arch["type"]

    def encode(fn, texts, max_len, bs=64):
        out = []
        for i in range(0, len(texts), bs):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(fn(*tokenize(tok, texts[i:i + bs], max_len)))
        return out

    def scores(Dc, Qe, fn=model.score):
        return torch.cat([fn(d, Qe) for d in Dc])

    torch.cuda.synchronize()
    t = time.time()
    Dc = encode(model.encode_docs, docs, DOC_LEN)            # ilan parçaları: indeks
    torch.cuda.synchronize()
    index_sec = time.time() - t

    systems = {"student": {}}
    if arch == "hybrid":
        systems.update({"student_single_path": {}, "student_token_path": {}})
    for v in (0, 2):
        Qe = encode(model.encode_questions, [q["texts"][v] for q in Q], Q_LEN, bs=len(Q))[0]
        systems["student"][v] = torch.sigmoid(scores(Dc, Qe)).cpu().numpy()
        if arch == "hybrid":
            single = lambda d, q: model._paths(d, q)[0]
            token = lambda d, q: model._paths(d, q)[1]
            systems["student_single_path"][v] = torch.sigmoid(scores(Dc, Qe, single)).cpu().numpy()
            systems["student_token_path"][v] = torch.sigmoid(scores(Dc, Qe, token)).cpu().numpy()

    # sorgu anı: tek soru kodlama + tüm indeksle skor + ilk 10
    lat = []
    for i in range(30):
        torch.cuda.synchronize()
        t = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            Qe = model.encode_questions(*tokenize(tok, [Q[i % len(Q)]["texts"][0]], Q_LEN))
        torch.sigmoid(scores(Dc, Qe)).squeeze(1).topk(10)
        torch.cuda.synchronize()
        lat.append(time.time() - t)
    timing = {"index_sec_2000_docs": index_sec, "index_ms_per_doc": 1000 * index_sec / len(docs),
              "query_ms_2000_docs": 1000 * float(np.median(lat[5:]))}
    if arch == "single":
        timing["index_bytes_per_doc"] = DIM * 2
        # 1M ilanlık indeks (rastgele vektörler): sadece iç çarpım + ilk 10 süresi
        big = torch.randn(1_000_000, DIM, device="cuda", dtype=torch.float16)
        qh = Qe["vec"].half()
        for _ in range(3):
            (big @ qh[:, :-1].T).squeeze(1).topk(10)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(20):
            (big @ qh[:, :-1].T).squeeze(1).topk(10)
        torch.cuda.synchronize()
        timing["scan_ms_1M_docs"] = 1000 * (time.time() - t) / 20
        del big
    else:
        n_tok = sum(int(d["mask"].sum()) for d in Dc) / len(docs)
        timing["tokens_per_doc"] = n_tok
        timing["index_bytes_per_doc"] = int(n_tok * model.arch["tdim"] * 2 + (DIM * 2 if arch == "hybrid" else 0))
    del model
    torch.cuda.empty_cache()
    return systems, timing


# ------------------------------------------------------------------ metrikler
def ece(p, y, bins=15):
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if s.any():
            e += s.mean() * abs(p[s].mean() - y[s].mean())
    return float(e)


def single(P, cols, calibrated=True):
    ys, ps = Y[:, cols], P[:, cols]
    ok = [j for j in range(len(cols)) if 0 < ys[:, j].sum() < len(ys)]
    r = {"AUC": float(np.mean([roc_auc_score(ys[:, j], ps[:, j]) for j in ok])),
         "AP": float(np.mean([average_precision_score(ys[:, j], ps[:, j]) for j in ok]))}
    if calibrated:
        r.update({"acc": float(((ps >= 0.5) == ys).mean()), "ECE": ece(ps.ravel(), ys.ravel().astype(float)),
                  "Brier": float(((ps - ys) ** 2).mean()),
                  "count_err": float(np.mean(np.abs(ps.sum(0) - ys.sum(0)) / np.maximum(ys.sum(0), 1)))})
    return r


def composite(score_fn, calibrated=True, subset=None):
    ap, pk, mx, nonempty, cnt_err = [], [], [], [], []
    for qi, q in enumerate(queries):
        if subset is not None and not subset(q):
            continue
        truth = np.ones(len(docs), bool)
        for c, neg in q["conds"]:
            truth &= ~Y[:, c] if neg else Y[:, c]
        s = score_fn(qi, q)
        mx.append(s.max())
        nonempty.append(truth.any())
        if truth.any():
            ap.append(average_precision_score(truth, s))
            k = min(10, int(truth.sum()))
            pk.append(truth[np.argsort(-s)[:k]].mean())
        if calibrated:
            cnt_err.append(abs(s.sum() - truth.sum()))
    mx, nonempty = np.array(mx), np.array(nonempty)
    r = {"n": len(mx), "AP": float(np.mean(ap)), "P@k": float(np.mean(pk)),
         "empty_AUC": float(roc_auc_score(nonempty, mx)) if 0 < nonempty.sum() < len(nonempty) else None}
    if calibrated:
        r["empty_acc@0.5"] = float(((mx >= 0.5) == nonempty).mean())
        r["count_MAE"] = float(np.mean(cnt_err))
    return r


def prod_scores(P):
    def fn(qi, q):
        s = np.ones(len(docs))
        for c, neg in q["conds"]:
            s = s * (1 - P[:, c] if neg else P[:, c])
        return s
    return fn


splits = {"A_seen": (~heldout, 0), "B_seen_newphrase": (~heldout, 2), "C_heldout": (heldout, 0),
          "D_heldout_newphrase": (heldout, 2)}
report = {"single": {}, "composite": {}, "timing": {}}

S, report["timing"]["student"] = student_scores()
systems = {name: (P, True) for name, P in S.items()}
if os.path.exists("data/cross_test_v0.npy"):
    systems["laya_cross_ft"] = ({v: np.load("data/cross_test_v%d.npy" % v) for v in (0, 2)}, True)
    report["timing"]["laya_cross_ft"] = json.load(open("data/cross_timing.json"))
if os.path.exists("data/bge_docs.npy"):
    bd = np.load("data/bge_docs.npy")
    systems["bge_m3"] = ({v: bd @ np.load("data/bge_q_v%d.npy" % v).T for v in (0, 2)}, False)
    bq = np.load("data/bge_queries.npy")

for name, (P, cal) in systems.items():
    report["single"][name] = {sp: single(P[v], np.where(mask)[0], cal) for sp, (mask, v) in splits.items()}
    fn = (lambda qi, q: bd @ bq[qi]) if name == "bge_m3" else prod_scores(P[0])
    report["composite"][name] = {
        "all": composite(fn, cal),
        "seen_only": composite(fn, cal, lambda q: not q["heldout"]),
        "with_heldout": composite(fn, cal, lambda q: q["heldout"]),
    }

os.makedirs("results", exist_ok=True)
REPORT = "results/report_%s.json" % os.path.splitext(os.path.basename(CKPT))[0]
json.dump(report, open(REPORT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("rapor:", REPORT)

fmt = lambda d: "  ".join("%s=%s" % (k, ("%.3f" % v) if isinstance(v, float) else v) for k, v in d.items())
for sec in ("single", "composite"):
    print("\n==", sec)
    for name, parts in report[sec].items():
        for sp, d in parts.items():
            print("%-14s %-20s %s" % (name, sp, fmt(d)))
print("\n== timing")
for name, d in report["timing"].items():
    print("%-14s %s" % (name, fmt(d)))
