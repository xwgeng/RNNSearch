Attention-based Neural Machine Translation
=====================================================================

### Installation
The following packages are needed:
* [Pytorch](https://github.com/pytorch/pytorch) >= 0.4.0
* NLTK

### Preparation
To obtain vocabulary for training, run:
```
python scripts/buildvocab.py --corpus /path/to/train.cn --output /path/to/cn.voc3.pkl \
--limit 30000 --groundhog
python scripts/buildvocab.py --corpus /path/to/train.en --output /path/to/en.voc3.pkl \
--limit 30000 --groundhog
```

### Training
Training the RNNSearch on Chinese-English translation datasets as follows:
```
python train.py \
--src_vocab /path/to/cn.voc3.pkl --trg_vocab /path/to/en.voc3.pkl \
--train_src corpus/train.cn-en.cn --train_trg corpus/train.cn-en.en \
--valid_src corpus/nist02/nist02.cn \
--valid_trg corpus/nist02/nist02.en0 corpus/nist02/nist02.en1 corpus/nist02/nist02.en2 corpus/nist02/nist02.en3 \
--eval_script scripts/validate.sh \
--model RNNSearch \
--optim RMSprop \
--batch_size 80 \
--half_epoch \
--cuda \
--info RMSprop-half_epoch 
```
### Evaluation
```
python translate.py \
--src_vocab /path/to/cn.voc3.pkl --trg_vocab /path/to/en.voc3.pkl \
--test_src corpus/nist03/nist03.cn \
--test_trg corpus/nist03/nist02.en0 corpus/nist03/nist03.en1 corpus/nist03/nist03.en2 corpus/nist03/nist03.en3 \
--eval_script scripts/validate.sh \
--model RNNSearch \
--name RNNSearch.best.pt \
--cuda 
```
The evaluation metric for Chinese-English we use is case-insensitive BLEU. We use the `muti-bleu.perl` script from [Moses](https://github.com/moses-smt/mosesdecoder) to compute the BLEU:
```
perl scripts/multi-bleu.perl -lc corpus/nist03/nist03.en < nist03.translated
```
### Results on Chinese-English translation
The trainining dataset consists of 1.25M billingual sentence pairs extracted from LDC corpora. Use NIST 2002(MT02) as tuning set for hyper-parameter optimization and model selection, and NIST 2003(MT03), 2004 (MT04), 2005(MT05), 2006(MT06) and 2008(MT08) as test sets. The beam size is set to 10.

|MT02|MT03|MT04|MT05|MT06|MT08|Ave.|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|40.16|37.26|40.50|36.67|37.10|28.54|36.01|

### Acknowledgements
My implementation utilizes code from the following:
* [DeepLearnXMU's ABD-NMT repo](https://github.com/DeepLearnXMU/ABD-NMT)
