#!/usr/bin/env bash
# v2 deneyleri: soru çeşitliliği (E1), + öğretmenden öğrenme (E2), + öğretmen doğrulaması (E2f),
# + öz-denetimli kelime soruları (E3)
set -e
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
quiet() { grep --line-buffered -vE "Warning|warn|symlink|torchao|Redirects|Fetching|developer mode|SMs"; }

run() {   # $1 = ad, geri kalanı ortam değişkenleri
  name=$1; shift
  echo "=== $name ($*)"
  env "$@" OUT="ckpt/$name.pt" python train_student.py 2>&1 | quiet
  CKPT="ckpt/$name.pt" python evaluate.py 2>&1 | quiet | grep -E "^student|^==|rapor"
}

run e1_concepts_truth   QSET=concepts TARGET=truth   LEX=0
run e2_concepts_teacher QSET=concepts TARGET=teacher LEX=0
run e2f_concepts_teacher_filtered QSET=concepts TARGET=teacher_f LEX=0
run e3_concepts_teacher_filtered_lex QSET=concepts TARGET=teacher_f LEX=1
echo "=== bitti"
