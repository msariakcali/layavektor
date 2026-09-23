"""Öğrenciyi eğitir.

QSET=bank      v1: bankadaki 55 görülen soru, söyleniş 0/1
QSET=concepts  v2: qgen.py havuzu (118 kavram, 527 söyleniş)
TARGET=truth   hedef gerçek cevaplar
TARGET=teacher v2'de yeni kavramların hedefi ince ayarlı cross-encoder'dan (etiketsiz hücreler maskelenir);
               bank kavramları insan etiketli kabul edilir, gerçek cevaptan kalır
LEX=1          öz-denetimli yardımcı sorular: "Metinde 'X' kelimesi geçiyor mu?" (etiket metinden gelir)
MODEL=single   tek vektör (v1/v2) | multi: sadece token vektörleri (v3a) | hybrid: ikisi birlikte (v3b)
INIT=<ckpt>    eğitime bu ckpt'deki ağırlıklardan başla (ör. v2); eşleşmeyen yeni katmanlar sıfırdan

Her adımda bir grup ilan ile soruların tamamı birlikte işlenir: [B ilan x Q soru] logit matrisi,
ikili çapraz entropi (log score, strictly proper -> kalibrasyonu ödüllendirir).
"""
import json
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from models import DOC_LEN, Q_LEN, TRAIN_LAYERS, build, freeze_bottom, load_laya, read_ckpt, tokenize, trainable_state

EPOCHS = int(os.environ.get("EPOCHS", 3))
BATCH = int(os.environ.get("BATCH", 32))
QSET = os.environ.get("QSET", "bank")
TARGET = os.environ.get("TARGET", "truth")
LEX = int(os.environ.get("LEX", 0))
N_LEX, LEX_WEIGHT = 24, 0.5
MODEL = os.environ.get("MODEL", "single")
INIT = os.environ.get("INIT")
OUT = os.environ.get("OUT", "ckpt/student_%s_%s_%s%s.pt" % (MODEL, QSET, TARGET, "_lex" if LEX else ""))
torch.manual_seed(0)
random.seed(0)

agent = load_laya()
tok, enc = agent.tok, agent.model.encoder
del agent.model.head, agent.model.scorer, agent.model.act_head
enc.float()
freeze_bottom(enc, TRAIN_LAYERS)
model = build({"type": MODEL}, enc, tok).cuda()
if INIT:
    own = model.state_dict()
    init = {k: v for k, v in read_ckpt(INIT)[1].items() if k in own}   # multi'de v2'nin tek vektör katmanları yok
    model.load_state_dict(init, strict=False)
    print("başlangıç: %s (%d tensör)" % (INIT, len(init)), flush=True)

docs = [json.loads(l)["text"] for l in open("data/train.jsonl", encoding="utf-8")]
if QSET == "bank":
    QUESTIONS = json.load(open("data/questions.json", encoding="utf-8"))
    cols = [i for i, q in enumerate(QUESTIONS) if not q["heldout"]]
    TEXTS = [QUESTIONS[i]["texts"][:2] for i in cols]
    Y = np.load("data/y_train.npy" if TARGET == "truth" else "data/teacher_train.npy").astype(np.float32)[:, cols]
else:
    concepts = json.load(open("data/concepts.json", encoding="utf-8"))
    TEXTS = [c["texts"] for c in concepts]
    Y = np.load("data/y_concepts_train.npy").astype(np.float32)
    if TARGET in ("teacher", "teacher_f"):
        T = np.load("data/teacher_concepts_train.npy")
        new = [k for k, c in enumerate(concepts) if c["bank"] is None]
        Y[:, new] = T[:, new]   # öğretmenin etiketlemediği ilanlarda NaN -> maskelenir
    if TARGET == "teacher_f":
        # Öğretmeni küçük bir etiketli doğrulama örneğinde (ilk VAL ilan) kontrol et; güvenilmez kavramları at.
        from sklearn.metrics import roc_auc_score
        VAL = 200
        truth = np.load("data/y_concepts_train.npy")[:VAL]
        dropped = []
        for k in new:
            if 0 < truth[:, k].sum() < VAL and roc_auc_score(truth[:, k], T[:VAL, k]) < 0.9:
                Y[:, k] = np.nan
                dropped.append(concepts[k]["id"])
        print("öğretmen doğrulamasında atılan kavramlar (%d): %s" % (len(dropped), ", ".join(dropped)), flush=True)
Y = torch.tensor(Y)

_TR = str.maketrans({"İ": "i", "I": "ı"})
words = lambda t: set(re.findall(r"\w+", t.translate(_TR).lower()))
DOC_WORDS = [words(d) for d in docs]
VOCAB = sorted({w for ws in DOC_WORDS for w in ws if len(w) >= 4 and not w.isdigit()})


def lexical(idx):
    """Yarısı bu gruptaki ilanlardan, yarısı tüm sözlükten örneklenen kelimeler."""
    pool = sorted({w for j in idx for w in DOC_WORDS[j] if len(w) >= 4 and not w.isdigit()})
    ws = random.sample(pool, N_LEX // 2) + random.sample(VOCAB, N_LEX // 2)
    texts = ["Metinde '%s' kelimesi geçiyor mu?" % w for w in ws]
    y = torch.tensor([[w in DOC_WORDS[j] for w in ws] for j in idx], dtype=torch.float32)
    return texts, y


def masked_bce(logits, y):
    m = ~torch.isnan(y)
    return F.binary_cross_entropy_with_logits(logits[m], y[m])


enc_params = [p for p in enc.parameters() if p.requires_grad]
head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("enc.")]
opt = torch.optim.AdamW([{"params": enc_params, "lr": 3e-5}, {"params": head_params, "lr": 1e-3}], weight_decay=0.01)
steps = EPOCHS * (len(docs) // BATCH)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[3e-5, 1e-3], total_steps=steps, pct_start=0.1)
print("eğitilen parametre: %.1fM | adım: %d | soru: %d kavram, %d söyleniş | hedef: %s | LEX=%d" % (
    sum(p.numel() for p in enc_params + head_params) / 1e6, steps, len(TEXTS), sum(map(len, TEXTS)), TARGET, LEX),
    flush=True)

t0, step = time.time(), 0
for ep in range(EPOCHS):
    order = list(range(len(docs)))
    random.shuffle(order)
    run = 0.0
    for i in range(0, len(order) - BATCH + 1, BATCH):
        idx = order[i:i + BATCH]
        q_texts = [random.choice(t) for t in TEXTS]   # her adımda her kavramın rastgele bir söylenişi
        y = Y[idx].cuda()
        if LEX:
            lex_texts, lex_y = lexical(idx)
            q_texts += lex_texts
        with torch.autocast("cuda", dtype=torch.bfloat16):
            d = model.encode_docs(*tokenize(tok, [docs[j] for j in idx], DOC_LEN))
            q = model.encode_questions(*tokenize(tok, q_texts, Q_LEN))
        loss = 0.0
        for logits in model.train_logits(d, q):   # hybrid: tek yol, token yolu ve birleşim ayrı ayrı cevap verir
            loss = loss + masked_bce(logits[:, :len(TEXTS)], y)
            if LEX:
                loss = loss + LEX_WEIGHT * F.binary_cross_entropy_with_logits(logits[:, len(TEXTS):], lex_y.cuda())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc_params + head_params, 1.0)
        opt.step()
        sched.step()
        step += 1
        run = 0.98 * run + 0.02 * loss.item() if run else loss.item()
        if step % 50 == 0:
            print("  ep %d adım %d/%d kayıp %.4f  (%.0f sn)" % (ep + 1, step, steps, run, time.time() - t0), flush=True)

os.makedirs("ckpt", exist_ok=True)
torch.save({"arch": model.arch, "state": trainable_state(model)}, OUT)
print("kaydedildi: %s  toplam %.0f sn" % (OUT, time.time() - t0))
