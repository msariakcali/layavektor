"""Normal embedding karşılaştırması: bge-m3 ile ilan, soru ve birleşik sorgu vektörleri (kosinüs benzerliği).

  data/bge_docs.npy, data/bge_q_v0.npy, data/bge_q_v2.npy, data/bge_queries.npy
"""
import json
import time

import numpy as np
from sentence_transformers import SentenceTransformer

m = SentenceTransformer("BAAI/bge-m3", device="cuda", model_kwargs={"torch_dtype": "float16"})
docs = [json.loads(l)["text"] for l in open("data/test.jsonl", encoding="utf-8")]
Q = json.load(open("data/questions.json", encoding="utf-8"))
queries = json.load(open("data/queries.json", encoding="utf-8"))

enc = lambda xs: m.encode(xs, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
t = time.time()
np.save("data/bge_docs.npy", enc(docs))
print("ilanlar: %.1f sn" % (time.time() - t))
for v in (0, 2):
    np.save("data/bge_q_v%d.npy" % v, enc([q["texts"][v] for q in Q]))
np.save("data/bge_queries.npy", enc([q["text"] for q in queries]))
print("bitti")
