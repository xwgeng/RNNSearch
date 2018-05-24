from __future__ import print_function
import argparse
import time
import os
import sys
import subprocess

import torch.utils.data

from dataset import dataset
from util import convert_data, invert_vocab, load_vocab, convert_str

import model
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

parser = argparse.ArgumentParser(description='Testing Sequence-to-Sequence Model')
# data
parser.add_argument('--src_vocab', type=str, help='source vocabulary')
parser.add_argument('--trg_vocab', type=str, help='target vocabulary')
parser.add_argument('--src_max_len', type=int, default=50, help='maximum length of source')
parser.add_argument('--trg_max_len', type=int, default=50, help='maximum length of target')
parser.add_argument('--test_src', type=str, help='source for testing')
parser.add_argument('--test_trg', type=str, nargs='+', help='reference for testing')
# model
parser.add_argument('--model', type=str, default='FMTNMT', help='name of model')
parser.add_argument('--name', type=str, help='name of checkpoint')
parser.add_argument('--enc_ninp', type=int, default=620, help='size of source word embedding')
parser.add_argument('--dec_ninp', type=int, default=620, help='size of target word embedding')
parser.add_argument('--enc_nhid', type=int, default=1000, help='number of source hidden layer')
parser.add_argument('--dec_nhid', type=int, default=1000, help='number of target hidden layer')
parser.add_argument('--dec_natt', type=int, default=1000, help='number of target attention layer')
parser.add_argument('--nreadout', type=int, default=620, help='number of maxout layer')
parser.add_argument('--enc_emb_dropout', type=float, default=0.3, help='dropout rate for encoder embedding')
parser.add_argument('--dec_emb_dropout', type=float, default=0.3, help='dropout rate for decoder embedding')
parser.add_argument('--enc_hid_dropout', type=float, default=0.3, help='dropout rate for encoder hidden state')
parser.add_argument('--readout_dropout', type=float, default=0.3, help='dropout rate for encoder hidden state')
# search
parser.add_argument('--beam_size', type=int, default=10, help='size of beam')
# bookkeeping
parser.add_argument('--seed', type=int, default=123, help='random number seed')
parser.add_argument('--checkpoint', type=str, default='./checkpoint/', help='path to checkpoint')
parser.add_argument('--save', type=str, default='./generation/', help='path to save generated sequence')
# GPU
parser.add_argument('--cuda', action='store_true', help='use cuda')
# Misc
parser.add_argument('--info', type=str, help='info of the model')

opt = parser.parse_args()

# set the random seed manually
torch.manual_seed(opt.seed)

opt.cuda = opt.cuda and torch.cuda.is_available()
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

# load vocabulary for source and target
src_vocab, trg_vocab = {}, {}
src_vocab['stoi'] = load_vocab(opt.src_vocab)
trg_vocab['stoi'] = load_vocab(opt.trg_vocab)
src_vocab['itos'] = invert_vocab(src_vocab['stoi'])
trg_vocab['itos'] = invert_vocab(trg_vocab['stoi'])
UNK = '<unk>'
SOS = '<sos>'
EOS = '<eos>'
PAD = '<pad>'
opt.enc_pad = src_vocab['stoi'][PAD]
opt.dec_sos = trg_vocab['stoi'][SOS]
opt.dec_eos = trg_vocab['stoi'][EOS]
opt.dec_pad = trg_vocab['stoi'][PAD]
opt.enc_ntok = len(src_vocab['stoi'])
opt.dec_ntok = len(trg_vocab['stoi'])

# load dataset for testing
test_dataset = dataset(opt.test_src, opt.test_trg)
test_iter = torch.utils.data.DataLoader(test_dataset, 1, shuffle=False, collate_fn=lambda x: zip(*x))

# create the model
model = getattr(model, opt.model)(opt)
if opt.cuda:
    model.cuda()

state_dict = torch.load(os.path.join(opt.checkpoint, opt.name))
model.load_state_dict(state_dict)
model.eval()


def bleu_script(f):
    ref_stem = opt.valid_trg[0][:-1] + '*'
    cmd = '{eval_script} {refs} {hyp}'.format(eval_script=opt.eval_script, refs=ref_stem, hyp=f)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode > 0:
        sys.stderr.write(err)
        sys.exit(1)
    bleu = float(out)
    return bleu


hyp_list = []
ref_list = []
start_time = time.time()
for ix, batch in enumerate(test_iter, start=1):
    src_raw = batch[0]
    trg_raw = batch[1:]
    src, src_mask = convert_data(src_raw, src_vocab, True, UNK, PAD, SOS, EOS)
    if opt.cuda:
        src = src.cuda()
        src_mask = src_mask.cuda()
    with torch.no_grad():
        output = model.beamsearch(src, src_mask, opt.beam_size, normalize=True)
        best_hyp, best_score = output[0]
        best_hyp = convert_str([best_hyp], trg_vocab)
        hyp_list.append(best_hyp[0])
        ref = map(lambda x: x[0], trg_raw)
        ref_list.append(ref)
elapsed = time.time() - start_time
bleu1 = corpus_bleu(ref_list, hyp_list, smoothing_function=SmoothingFunction().method1)
hyp_list = map(lambda x: ' '.join(x), hyp_list)
name = opt.name.replace('pt', 'beam') + '.' + opt.info
p_pred = os.path.join(opt.save, name)
f_pred = open(p_pred, 'w')
f_pred.write('\n'.join(hyp_list))
f_pred.close()
bleu2 = bleu_script(p_pred)
print('BLEU score for model {} is {}/{}, {}'.format(opt.name, bleu1, bleu2, elapsed))
