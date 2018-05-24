#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=$1

model=$2
test_nist=$3
name=$4

data_dir=corpus
src=cn
trg=en
lang=cn-en

src_vocab=$data_dir/${src}.voc3.pkl
trg_vocab=$data_dir/${trg}.voc3.pkl

test_prefix=$data_dir/${test_nist}/${test_nist}
test_src=${test_prefix}.${src}
test_trg_prefix=${test_prefix}.${trg}
test_trg=${test_trg_prefix}"0"\ ${test_trg_prefix}"1"\ ${test_trg_prefix}"2"\ ${test_trg_prefix}"3"

eval_script=scripts/validate.sh

python translate.py --cuda --src_vocab ${src_vocab} --trg_vocab ${trg_vocab} --test_src ${test_src} --test_trg ${test_trg} --eval_script ${eval_script} --model ${model}  --name ${name} 

