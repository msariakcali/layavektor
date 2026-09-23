"""ckpt/ içindeki eğitilmiş katmanları temel Laya ile birleştirip tek başına yüklenebilir ağırlıklar yazar.

  weights/<VERSION>/student/            öğrenci: encoder config, tokenizer, student.safetensors, meta.json
  weights/<VERSION>/laya-cross-ft/      Laya checkpoint formatında: laya.load("weights/v1/laya-cross-ft")
  weights/<VERSION>/report.json         o sürümün değerlendirme sonuçları

Yüklemek için: models.load_student("weights/v1/student")
"""
import json
import os
import shutil
import sys

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

from models import DIM, TRAIN_LAYERS, build, load_laya, read_ckpt

VERSION = os.environ.get("VERSION", "v1")
STUDENT_CKPT = os.environ.get("STUDENT_CKPT", "ckpt/student_truth.pt")
CROSS_CKPT = os.environ.get("CROSS_CKPT", "ckpt/cross.pt")
REPORT = os.environ.get("REPORT", "results/report_student_truth.json")
NOTE = os.environ.get("NOTE", "öğrenci gerçek cevaplarla, 55 görülen soru, 3 epoch; cross-encoder 48k çift")
out = os.path.join("weights", VERSION)
if os.path.exists(out) and "--force" not in sys.argv:
    sys.exit("%s zaten var; üzerine yazmak için --force" % out)

base_dir = snapshot_download("convaiinnovations/laya", allow_patterns=["multilingual/*"])
base = {"repo": "convaiinnovations/laya", "subfolder": "multilingual", "snapshot": os.path.basename(base_dir)}


def tensors(sd):
    return {k: v.detach().to(torch.bfloat16 if v.is_floating_point() else v.dtype).contiguous().cpu()
            for k, v in sd.items()}


# ------------------------------------------------------------------ öğrenci
arch, state = read_ckpt(STUDENT_CKPT)
agent = load_laya()
enc = agent.model.encoder.float()
student = build(arch, enc, agent.tok)
res = student.load_state_dict(state, strict=False)
assert not res.unexpected_keys
sdir = os.path.join(out, "student")
os.makedirs(sdir, exist_ok=True)
save_file(tensors(student.state_dict()), os.path.join(sdir, "student.safetensors"))
enc.config.save_pretrained(os.path.join(sdir, "encoder"))
agent.tok.save_pretrained(os.path.join(sdir, "tokenizer"))
json.dump({"arch": student.arch, "dim": DIM, "train_layers": TRAIN_LAYERS, "base": base, "source_ckpt": STUDENT_CKPT,
           "note": NOTE, "model": type(student).__name__},
          open(os.path.join(sdir, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
del agent, student, enc
torch.cuda.empty_cache()

# ------------------------------------------------------------------ cross-encoder (Laya formatı)
if os.path.exists(CROSS_CKPT):
    agent = load_laya()
    model = agent.model.float()
    res = model.load_state_dict(torch.load(CROSS_CKPT), strict=False)
    assert not res.unexpected_keys
    cdir = os.path.join(out, "laya-cross-ft")
    os.makedirs(cdir, exist_ok=True)
    save_file(tensors(model.state_dict()), os.path.join(cdir, "model.safetensors"))
    src = os.path.join(base_dir, "multilingual")
    for sub in ("tokenizer", "encoder"):
        shutil.copytree(os.path.join(src, sub), os.path.join(cdir, sub), dirs_exist_ok=True)
    cfg = dict(agent.cfg, fine_tuned_from=base, fine_tune_note=NOTE)
    json.dump(cfg, open(os.path.join(cdir, "rl_agent_config.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

if os.path.exists(REPORT):
    shutil.copy(REPORT, os.path.join(out, "report.json"))
print("kaydedildi:", out)
