"""Öğrenci (iki kuleli karar embedding'i) ve karşılaştırma için ince ayarlı Laya cross-encoder.

Öğrenci:  logit(ilan, soru) = d · q[:-1] + q[-1]
  d: ilan vektörü (bir kez hesaplanıp indekste saklanır)
  q: soru vektörü + soruya özgü sapma (son boyut); ilan vektörüne sabit 1 eklenince tek bir iç çarpım olur,
     yani standart bir MIPS/ANN indeksiyle aranabilir.
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.environ.get("LAYA_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "laya-check")))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import laya  # noqa: E402

DIM = 256
TRAIN_LAYERS = int(os.environ.get("TRAIN_LAYERS", 6))   # 4 GB VRAM: alt katmanlar ve embedding tablosu donuk
DOC_LEN, Q_LEN = 256, 48


def load_laya():
    return laya.load("convaiinnovations/laya", subfolder="multilingual", device="cuda")


def freeze_bottom(encoder, train_layers):
    for p in encoder.parameters():
        p.requires_grad = False
    for layer in encoder.layers[-train_layers:]:
        for p in layer.parameters():
            p.requires_grad = True
    for p in encoder.final_norm.parameters():
        p.requires_grad = True


class Student(nn.Module):
    def __init__(self, encoder, dim=DIM):
        super().__init__()
        self.enc = encoder
        h = encoder.config.hidden_size
        self.dproj = nn.Linear(h, dim)
        self.qproj = nn.Linear(h, dim + 1)

    def _pool(self, ids, att):
        hs = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        m = att.unsqueeze(-1).to(hs.dtype)
        return (hs * m).sum(1) / m.sum(1).clamp_min(1)

    def docs(self, ids, att):
        return self.dproj(self._pool(ids, att)).float()

    def questions(self, ids, att):
        return self.qproj(self._pool(ids, att)).float()

    @staticmethod
    def logits(d, q):
        return d @ q[:, :-1].T + q[:, -1]


def trainable_state(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in names}


def load_student(path, device="cuda"):
    """save_weights.py ile kaydedilmiş öğrenciyi Laya/HF'e ihtiyaç duymadan yükler -> (model, tokenizer)."""
    import json
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    meta = json.load(open(os.path.join(path, "meta.json"), encoding="utf-8"))
    enc = AutoModel.from_config(AutoConfig.from_pretrained(os.path.join(path, "encoder")), attn_implementation="sdpa")
    model = Student(enc, meta["dim"])
    model.load_state_dict({k: v.float() for k, v in load_file(os.path.join(path, "student.safetensors")).items()})
    return model.to(device).eval(), AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer"))


def tokenize(tok, texts, max_len):
    b = tok(texts, add_special_tokens=True, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    return b["input_ids"].cuda(), b["attention_mask"].cuda()
