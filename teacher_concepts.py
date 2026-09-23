"""İnce ayarlı Laya cross-encoder (weights/v1/laya-cross-ft) ile yeni kavramları etiketler.

Her kavram ilk söylenişiyle sorulur; öğrenci aynı hedefi o kavramın tüm söylenişlerinde öğrenir.
  data/teacher_concepts_train.npy  [N_train, K]  etiketlenmemiş hücreler NaN
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.environ.get("LAYA_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "laya-check")))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import laya
from laya.common import QTYPES, build_sequence, collate_items

TEACHER = os.environ.get("TEACHER", "weights/v1/laya-cross-ft")
N_DOCS = int(os.environ.get("N_DOCS", 3000))
CHUNK = int(os.environ.get("CHUNK", 128))

agent = laya.load(TEACHER, device="cuda")
tok, model = agent.tok, agent.model.eval()
MAX_LEN, HEAD_MAX_LEN, QT = agent.cfg["max_len"], agent.cfg["head_max_len"], QTYPES["noul"]

concepts = json.load(open("data/concepts.json", encoding="utf-8"))
docs = [json.loads(l)["text"] for l in open("data/train.jsonl", encoding="utf-8")][:N_DOCS]
doc_ids = [tok(d, add_special_tokens=False)["input_ids"] for d in docs]
todo = [k for k, c in enumerate(concepts) if c["bank"] is None]

items = []
for k in todo:
    ids, markers = build_sequence(tok, "", {"t": "noul", "ins": concepts[k]["texts"][0], "crit": None}, MAX_LEN, HEAD_MAX_LEN)
    p_ids = ids[:-1]
    for di, st in enumerate(doc_ids):
        items.append({"ids": p_ids + st[:MAX_LEN - len(p_ids) - 1] + [tok.sep_token_id], "markers": markers,
                      "qtype": QT, "di": di, "k": k})
items.sort(key=lambda it: len(it["ids"]))
print("%d kavram x %d ilan = %d çift" % (len(todo), len(docs), len(items)), flush=True)

out =np.full((sum(1 for _ in open("data/train.jsonl", encoding="utf-8")), len(concepts)), np.nan, np.float32)
t0 = time.time()
with torch.inference_mode():
    for i in range(0, len(items), CHUNK):
        ch = items[i:i + CHUNK]
        b = collate_items([ch], tok.pad_token_id)
        with torch.autocast("cuda", dtype=agent.dtype):
            logits, _ = model(b["input_ids"].cuda(), b["attention_mask"].cuda(), b["marker_pos"].cuda(),
                              b["marker_mask"].cuda(), b["qtype"].cuda())
        p = torch.softmax(logits[:, :2].float(), -1)[:, 1].cpu().numpy()
        for it, pv in zip(ch, p):
            out[it["di"], it["k"]] = pv
        if (i // CHUNK) % 200 == 0:
            print("  %d/%d (%.0f sn)" % (i, len(items), time.time() - t0), flush=True)
np.save("data/teacher_concepts_train.npy", out)

y = np.load("data/y_concepts_train.npy")[:N_DOCS][:, todo]
p = out[:N_DOCS][:, todo]
print("bitti %.0f sn | öğretmen yeni kavramlarda doğruluk %.3f (gerçek cevaba göre)" % (
    time.time() - t0, ((p >= 0.5) == y).mean()))
