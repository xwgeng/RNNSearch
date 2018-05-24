An implemented attention-based neural machine translation using Pytorch
=====================================================================

### Installation
The following packages are needed:
* [Pytorch-0.4.0](https://github.com/pytorch/pytorch)
* NLTK

### Preparation
To obtain vocabulary for training, run:
```
python scripts/buildvocab.py --corpus /path/to/train.zh --output /path/to/zh.voc3.pkl --limit 30000 --groundhog
python scripts/buildvocab.py --corpus /path/to/train.en --output /path/to/de.voc3.pkl --limit 30000 --groundhog
```

### Training
```
./run.sh 
```
### Evaluation
```
./test.sh


