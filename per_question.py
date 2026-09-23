"""Görülmemiş sorularda soru bazında AUC: hangi tür sorular genelleşiyor, hangileri genelleşmiyor?"""
import json
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from models import DOC_LEN, Q_LEN, load_trained, tokenize

Q = json.load(open("data/questions.json", encoding="utf-8"))
Y = np.load("data/y_test.npy")
docs = [json.loads(l)["text"] for l in open("data/test.jsonl", encoding="utf-8")]
held = [i for i, q in enumerate(Q) if q["heldout"]]
rows = {}
for ck in sys.argv[1:]:
    model, tok = load_trained(ck)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        q = model.encode_questions(*tokenize(tok, [Q[i]["texts"][0] for i in held], Q_LEN))
        P = torch.cat([model.score(model.encode_docs(*tokenize(tok, docs[i:i + 64], DOC_LEN)), q)
                       for i in range(0, len(docs), 64)])
    rows[ck] = [roc_auc_score(Y[:, i], P[:, j].float().cpu().numpy()) for j, i in enumerate(held)]
    del model
    torch.cuda.empty_cache()
cross = np.load("data/cross_test_v0.npy")
print("%-24s %s  cross" % ("soru", "  ".join("%6s" % c.split("/")[-1][:6] for c in rows)))
for j, i in enumerate(held):
    print("%-24s %s  %.3f" % (Q[i]["id"], "  ".join("%6.3f" % rows[c][j] for c in rows), roc_auc_score(Y[:, i], cross[:, i])))
