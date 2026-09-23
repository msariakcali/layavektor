"""Öğretmen: tam Laya (cross-encoder). Her (ilan, soru) çifti için P(evet) üretir.

  data/teacher_train.npy  [N_train, Q]  söyleniş 0, görülmemiş sorular NaN (eğitimde kullanılmaz)
  data/teacher_test_v0.npy / _v2.npy  [N_test, Q]  söyleniş 0 ve görülmemiş söyleniş 2
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
from laya.common import QTYPES, build_sequence, collate_items, temp_bucket

CHUNK = int(os.environ.get("CHUNK", 128))
LIMIT = int(os.environ.get("LIMIT", 0))   # >0: ölçüm için sadece ilk LIMIT ilan

agent = laya.load("convaiinnovations/laya", subfolder="multilingual", device="cuda")
tok, DEV = agent.tok, agent.device
MAX_LEN, HEAD_MAX_LEN = agent.cfg["max_len"], agent.cfg["head_max_len"]
QT = QTYPES["noul"]
TEMP = max(1e-3, float(agent.temperature_by_options.get(temp_bucket(QT, 2), agent.temperature[QT])))

QUESTIONS = json.load(open("data/questions.json", encoding="utf-8"))


def load(name):
    docs = [json.loads(l) for l in open("data/%s.jsonl" % name, encoding="utf-8")]
    return docs[:LIMIT] if LIMIT else docs


def prefix(text):
    ids, markers = build_sequence(tok, "", {"t": "noul", "ins": text, "crit": None}, MAX_LEN, HEAD_MAX_LEN)
    return ids[:-1], markers   # sondaki [SEP] ilan metni eklendikten sonra konur


@torch.inference_mode()
def label(docs, variant, q_idx):
    doc_ids = [tok(d["text"], add_special_tokens=False)["input_ids"] for d in docs]
    items = []
    for qi in q_idx:
        p_ids, markers = prefix(QUESTIONS[qi]["texts"][variant])
        room = MAX_LEN - len(p_ids) - 1
        for di, st in enumerate(doc_ids):
            items.append({"ids": p_ids + st[:room] + [tok.sep_token_id], "markers": markers, "qtype": QT,
                          "di": di, "qi": qi})
    items.sort(key=lambda it: len(it["ids"]))
    out = np.full((len(docs), len(QUESTIONS)), np.nan, dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(items), CHUNK):
        chunk = items[i:i + CHUNK]
        b = collate_items([chunk], tok.pad_token_id)
        with torch.autocast(device_type="cuda", dtype=agent.dtype):
            logits, _ = agent.model(b["input_ids"].to(DEV), b["attention_mask"].to(DEV),
                                    b["marker_pos"].to(DEV), b["marker_mask"].to(DEV), b["qtype"].to(DEV))
        p = torch.softmax(logits[:, :2].float() / TEMP, -1)[:, 1].cpu().numpy()
        for it, pv in zip(chunk, p):
            out[it["di"], it["qi"]] = pv
    dt = time.time() - t0
    print("  %d çift, %.1f sn, %.0f çift/sn" % (len(items), dt, len(items) / dt), flush=True)
    return out


seen = [i for i, q in enumerate(QUESTIONS) if not q["heldout"]]
every = list(range(len(QUESTIONS)))
sfx = "_limit%d" % LIMIT if LIMIT else ""

print("test, söyleniş 0 (tüm sorular)")
np.save("data/teacher_test_v0%s.npy" % sfx, label(load("test"), 0, every))
print("test, söyleniş 2 (görülmemiş söyleniş, tüm sorular)")
np.save("data/teacher_test_v2%s.npy" % sfx, label(load("test"), 2, every))
print("eğitim, söyleniş 0 (görülen sorular)")
np.save("data/teacher_train%s.npy" % sfx, label(load("train"), 0, seen))
