"""Görülmemiş sorularda soru bazında AUC: hangi tür sorular genelleşiyor, hangileri genelleşmiyor?"""
import json
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from models import DOC_LEN, Q_LEN, Student, load_laya, tokenize

Q = json.load(open("data/questions.json", encoding="utf-8"))
Y = np.load("data/y_test.npy")
docs = [json.loads(l)["text"] for l in open("data/test.jsonl", encoding="utf-8")]
held = [i for i, q in enumerate(Q) if q["heldout"]]
agent = load_laya()
tok, enc = agent.tok, agent.model.encoder.float()
model = Student(enc).cuda().eval()
base = {k: v.clone() for k, v in model.state_dict().items()}
rows = {}
for ck in sys.argv[1:]:
    model.load_state_dict(base)
    model.load_state_dict(torch.load(ck), strict=False)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        D = torch.cat([model.docs(*tokenize(tok, docs[i:i + 64], DOC_LEN)) for i in range(0, len(docs), 64)])
        q = model.questions(*tokenize(tok, [Q[i]["texts"][0] for i in held], Q_LEN))
    P = torch.sigmoid(Student.logits(D, q)).cpu().numpy()
    rows[ck] = [roc_auc_score(Y[:, i], P[:, j]) for j, i in enumerate(held)]
cross = np.load("data/cross_test_v0.npy")
print("%-24s %s  cross" % ("soru", "  ".join("%6s" % c.split("/")[-1][:6] for c in rows)))
for j, i in enumerate(held):
    print("%-24s %s  %.3f" % (Q[i]["id"], "  ".join("%6.3f" % rows[c][j] for c in rows), roc_auc_score(Y[:, i], cross[:, i])))
