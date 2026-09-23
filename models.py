"""Öğrenci modelleri ve ortak yardımcılar.

Tek vektör (v1/v2, arch "single"):
  logit(ilan, soru) = d · q[:-1] + q[-1]
  d: ilan vektörü (bir kez hesaplanıp indekste saklanır)
  q: soru vektörü + soruya özgü sapma (son boyut); ilan vektörüne sabit 1 eklenince tek bir iç çarpım olur,
     yani standart bir MIPS/ANN indeksiyle aranabilir.

Çok vektörlü (v3, arch "multi" / "hybrid"):
  ilanın her token'ı için normalize edilmiş küçük bir vektör saklanır (ColBERT tipi).
  s(ilan, soru) = Σ_i w_i · max_t (q_i · d_t)        soru token'ı başına en iyi ilan token'ı, ağırlıklı ortalama
  logit_token   = a_q · s + b_q                       a_q, b_q sorudan: benzerliği kalibre olasılığa çevirir
  hybrid: eğitimde iki yol hem ayrı ayrı hem ortalama olarak cevap vermek zorunda; çıkarımda combine="avg"
          ikisinin ortalaması, combine="token" sadece token yolu (görülmemiş kavramlarda tek vektör yolu
          yazı tura attığı için ortalamayı aşağı çekiyor; v3 = hybrid eğitim + token çıkarımı)
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.environ.get("LAYA_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "laya-check")))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import laya  # noqa: E402

DIM, TDIM = 256, 128
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


def _pool(hs, att):
    m = att.unsqueeze(-1).to(hs.dtype)
    return (hs * m).sum(1) / m.sum(1).clamp_min(1)


class Student(nn.Module):
    arch = {"type": "single"}

    def __init__(self, encoder, dim=DIM):
        super().__init__()
        self.enc = encoder
        h = encoder.config.hidden_size
        self.dproj = nn.Linear(h, dim)
        self.qproj = nn.Linear(h, dim + 1)

    def _pool(self, ids, att):
        return _pool(self.enc(input_ids=ids, attention_mask=att).last_hidden_state, att)

    def docs(self, ids, att):
        return self.dproj(self._pool(ids, att)).float()

    def questions(self, ids, att):
        return self.qproj(self._pool(ids, att)).float()

    @staticmethod
    def logits(d, q):
        return d @ q[:, :-1].T + q[:, -1]

    # ortak arayüz (MultiStudent ile aynı): encode_docs / encode_questions / score / train_logits
    def encode_docs(self, ids, att):
        return {"vec": self.docs(ids, att)}

    def encode_questions(self, ids, att):
        return {"vec": self.questions(ids, att)}

    def score(self, D, Q):
        return self.logits(D["vec"], Q["vec"])

    def train_logits(self, D, Q):
        return [self.score(D, Q)]


class MultiStudent(nn.Module):
    def __init__(self, encoder, special_ids, hybrid=True, dim=DIM, tdim=TDIM, combine="avg"):
        super().__init__()
        self.enc = encoder
        self.hybrid = hybrid
        self.combine = combine
        self.arch = {"type": "hybrid" if hybrid else "multi", "tdim": tdim, "combine": combine}
        self.register_buffer("special", torch.tensor(sorted(special_ids)), persistent=False)
        h = encoder.config.hidden_size
        if hybrid:   # v2 ile aynı adlar: v2 ağırlıklarından başlatılabilir
            self.dproj = nn.Linear(h, dim)
            self.qproj = nn.Linear(h, dim + 1)
        self.tproj = nn.Linear(h, tdim)       # ilan ve soru token'ları aynı uzaya
        self.tw = nn.Linear(h, 1)             # soru token'ı önemi ("var mı" gibi kelimeler az saysın)
        self.qcal = nn.Linear(h, 2)           # a_q, b_q
        nn.init.zeros_(self.qcal.weight)
        with torch.no_grad():
            self.qcal.bias.copy_(torch.tensor([10.0, -6.0]))   # s ∈ [-1, 1] -> anlamlı logit aralığı

    def _tokens(self, ids, att):
        hs = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        mask = att.bool() & ~torch.isin(ids, self.special)
        return hs, mask

    def encode_docs(self, ids, att):
        hs, mask = self._tokens(ids, att)
        out = {"tok": F.normalize(self.tproj(hs).float(), dim=-1), "mask": mask}
        if self.hybrid:
            out["vec"] = self.dproj(_pool(hs, att)).float()
        return out

    def encode_questions(self, ids, att):
        hs, mask = self._tokens(ids, att)
        w = F.softplus(self.tw(hs).float().squeeze(-1)) * mask
        pooled = _pool(hs, att)
        cal = self.qcal(pooled).float()
        out = {"tok": F.normalize(self.tproj(hs).float(), dim=-1), "w": w / w.sum(1, keepdim=True).clamp_min(1e-6),
               "a": cal[:, 0], "b": cal[:, 1]}
        if self.hybrid:
            out["vec"] = self.qproj(pooled).float()
        return out

    @staticmethod
    def maxsim(D, Q, q_chunk=32):
        out = []   # soruları parça parça: [soru, soru tok, ilan, ilan tok] tensörü belleği doldurmasın
        for i in range(0, Q["tok"].shape[0], q_chunk):
            sim = torch.einsum("qid,btd->qibt", Q["tok"][i:i + q_chunk], D["tok"])
            sim = sim.masked_fill(~D["mask"][None, None], -2.0)
            out.append((sim.max(-1).values * Q["w"][i:i + q_chunk, :, None]).sum(1))
        return torch.cat(out).T                                              # [ilan, soru]

    def _paths(self, D, Q):
        tok = Q["a"] * self.maxsim(D, Q) + Q["b"]
        single = D["vec"] @ Q["vec"][:, :-1].T + Q["vec"][:, -1] if self.hybrid else None
        return single, tok

    def score(self, D, Q):
        single, tok = self._paths(D, Q)
        return tok if single is None or self.combine == "token" else (single + tok) / 2

    def train_logits(self, D, Q):
        single, tok = self._paths(D, Q)
        return [tok] if single is None else [single, tok, (single + tok) / 2]


def build(arch, encoder, tok):
    if arch["type"] == "single":
        return Student(encoder)
    return MultiStudent(encoder, {tok.cls_token_id, tok.sep_token_id, tok.pad_token_id},
                        hybrid=arch["type"] == "hybrid", tdim=arch.get("tdim", TDIM), combine=arch.get("combine", "avg"))


def read_ckpt(path):
    """ckpt/ dosyası -> (arch, eğitilen parametreler). Eski (v1/v2) dosyalar düz state dict'tir."""
    ck = torch.load(path)
    return (ck["arch"], ck["state"]) if "state" in ck else ({"type": "single"}, ck)


def load_trained(path):
    """Temel Laya encoder'ı + ckpt'deki eğitilmiş katmanlar -> (model, tokenizer); değerlendirme için."""
    arch, state = read_ckpt(path)
    agent = load_laya()
    enc = agent.model.encoder.float()
    model = build(arch, enc, agent.tok).cuda()
    res = model.load_state_dict(state, strict=False)
    assert not res.unexpected_keys, res.unexpected_keys
    return model.eval(), agent.tok


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
    tok = AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer"))
    model = build(meta.get("arch", {"type": "single"}), enc, tok)
    model.load_state_dict({k: v.float() for k, v in load_file(os.path.join(path, "student.safetensors")).items()})
    return model.to(device).eval(), tok


def tokenize(tok, texts, max_len):
    b = tok(texts, add_special_tokens=True, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    return b["input_ids"].cuda(), b["attention_mask"].cuda()
