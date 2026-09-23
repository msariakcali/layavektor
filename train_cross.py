"""Karşılaştırma: tam Laya cross-encoder'ı aynı gerçek cevaplarla ince ayarlar, sonra test çiftlerini etiketler.

Öğrenciyle aynı koşullar: aynı eğitim ilanları, görülen sorular, söyleniş 0/1, üst TRAIN_LAYERS katman.
Cross-encoder her (ilan, soru) çifti için ayrı bir forward pass ister; bütçe süreyle sınırlanır.

  ckpt/cross.pt, data/cross_test_v0.npy, data/cross_test_v2.npy
"""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from models import TRAIN_LAYERS, freeze_bottom, load_laya, trainable_state
from laya.common import QTYPES, build_sequence, collate_items

PAIRS = int(os.environ.get("PAIRS", 48000))
BATCH = int(os.environ.get("BATCH", 16))
CHUNK = int(os.environ.get("CHUNK", 128))
random.seed(0)
torch.manual_seed(0)

agent = load_laya()
tok, model = agent.tok, agent.model
MAX_LEN, HEAD_MAX_LEN = agent.cfg["max_len"], agent.cfg["head_max_len"]
QT = QTYPES["noul"]
model.float()
freeze_bottom(model.encoder, TRAIN_LAYERS)
for mod in (model.head, model.type_emb, model.scorer):
    for p in mod.parameters():
        p.requires_grad = True

QUESTIONS = json.load(open("data/questions.json", encoding="utf-8"))
seen = [i for i, q in enumerate(QUESTIONS) if not q["heldout"]]
_prefix = {}


def prefix(qi, v):
    if (qi, v) not in _prefix:
        ids, markers = build_sequence(tok, "", {"t": "noul", "ins": QUESTIONS[qi]["texts"][v], "crit": None},
                                      MAX_LEN, HEAD_MAX_LEN)
        _prefix[qi, v] = (ids[:-1], markers)
    return _prefix[qi, v]


def item(doc_ids, qi, v):
    p_ids, markers = prefix(qi, v)
    return {"ids": p_ids + doc_ids[:MAX_LEN - len(p_ids) - 1] + [tok.sep_token_id], "markers": markers, "qtype": QT}


def forward(items):
    b = collate_items([items], tok.pad_token_id)
    logits, _ = model(b["input_ids"].cuda(), b["attention_mask"].cuda(), b["marker_pos"].cuda(),
                      b["marker_mask"].cuda(), b["qtype"].cuda())
    return logits[:, :2]


# ------------------------------------------------------------------ eğitim
train = [tok(json.loads(l)["text"], add_special_tokens=False)["input_ids"] for l in open("data/train.jsonl", encoding="utf-8")]
Y = np.load("data/y_train.npy")
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=3e-5, weight_decay=0.01)
steps = PAIRS // BATCH
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-5, total_steps=steps, pct_start=0.1)
print("cross-encoder eğitimi: %d çift, %d adım, %.1fM parametre" % (PAIRS, steps, sum(p.numel() for p in params) / 1e6), flush=True)
t0, run = time.time(), 0.0
model.train()
for s in range(steps):
    pairs = [(random.randrange(len(train)), random.choice(seen)) for _ in range(BATCH)]
    items = [item(train[di], qi, random.choice((0, 1))) for di, qi in pairs]
    y = torch.tensor([int(Y[di, qi]) for di, qi in pairs]).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = F.cross_entropy(forward(items).float(), y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    sched.step()
    run = 0.98 * run + 0.02 * loss.item() if run else loss.item()
    if (s + 1) % 200 == 0:
        print("  adım %d/%d kayıp %.4f (%.0f sn)" % (s + 1, steps, run, time.time() - t0), flush=True)
train_sec = time.time() - t0
os.makedirs("ckpt", exist_ok=True)
torch.save(trainable_state(model), "ckpt/cross.pt")


# ------------------------------------------------------------------ test etiketleme
@torch.inference_mode()
def label(v):
    docs = [tok(json.loads(l)["text"], add_special_tokens=False)["input_ids"] for l in open("data/test.jsonl", encoding="utf-8")]
    items = [dict(item(d, qi, v), di=di, qi=qi) for qi in range(len(QUESTIONS)) for di, d in enumerate(docs)]
    items.sort(key=lambda it: len(it["ids"]))
    out = np.zeros((len(docs), len(QUESTIONS)), np.float32)
    t = time.time()
    for i in range(0, len(items), CHUNK):
        ch = items[i:i + CHUNK]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.softmax(forward(ch).float(), -1)[:, 1].cpu().numpy()
        for it, pv in zip(ch, p):
            out[it["di"], it["qi"]] = pv
    dt = time.time() - t
    print("  söyleniş %d: %d çift %.0f sn (%.0f çift/sn)" % (v, len(items), dt, len(items) / dt), flush=True)
    return out, dt / len(QUESTIONS)


model.eval()
res = {"train_sec": train_sec}
for v in (0, 2):
    p, per_q = label(v)
    np.save("data/cross_test_v%d.npy" % v, p)
    res["scan_sec_per_question_v%d" % v] = per_q
json.dump(res, open("data/cross_timing.json", "w"))
print("bitti:", res)
