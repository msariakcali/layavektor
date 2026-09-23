#!/usr/bin/env bash
# v3 deneyleri: çok vektörlü (token) temsil. İkisi de v2'den başlar, v2 ile aynı veri ve hedeflerle eğitilir.
set -e
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
quiet() { grep --line-buffered -vE "Warning|warn|symlink|torchao|Redirects|Fetching|developer mode|SMs"; }
V2=ckpt/e4_concepts_teacher_filtered_8ep.pt
COMMON="QSET=concepts TARGET=teacher_f LEX=0 EPOCHS=8 INIT=$V2"

run() {   # $1 = ad, geri kalanı ortam değişkenleri
  name=$1; shift
  echo "=== $name ($*)"
  env $COMMON "$@" OUT="ckpt/$name.pt" python train_student.py 2>&1 | quiet
  CKPT="ckpt/$name.pt" python evaluate.py 2>&1 | quiet | grep -E "^student|^==|rapor"
}

run v3a_multi  MODEL=multi
run v3b_hybrid MODEL=hybrid
echo "=== görülmemiş sorular, soru bazında"
python per_question.py $V2 ckpt/v3a_multi.pt ckpt/v3b_hybrid.pt 2>&1 | quiet
echo "=== bitti"
