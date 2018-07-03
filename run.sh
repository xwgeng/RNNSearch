#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=$1


model=$2
dist=$3
info="Rmsprop-half_epoch"

data_dir=corpus
src=cn
trg=en
lang=cn-en

nGPU=${CUDA_VISIBLE_DEVICES//,/}
nGPU=${#nGPU}

src_vocab=$data_dir/${src}.voc3.pkl
trg_vocab=$data_dir/${trg}.voc3.pkl

train_src=$data_dir/train.${lang}.${src}
train_trg=$data_dir/train.${lang}.${trg}

valid_nist=nist02
valid_prefix=$data_dir/$valid_nist/${valid_nist}
valid_src=${valid_prefix}.${src}
valid_trg_prefix=${valid_prefix}.${trg}
valid_trg=${valid_trg_prefix}"0"\ ${valid_trg_prefix}"1"\ ${valid_trg_prefix}"2"\ ${valid_trg_prefix}"3"

eval_script=scripts/validate.sh
dist_script="-m torch.distributed.launch --nproc_per_node=${nGPU}"

if [ -z "$dist" ]
then
	dist_script=""
fi

python ${dist_script} train.py --cuda --src_vocab ${src_vocab} --trg_vocab ${trg_vocab} --train_src ${train_src} --train_trg ${train_trg} --valid_src ${valid_src} --valid_trg ${valid_trg} --eval_script ${eval_script} --model ${model} --info ${info} --half_epoch --sfreq 50 --batch_size 80

