import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from beam import Beam

from util import convert_str

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class Encoder(nn.Module):
    """"encode the input sequence with Bi-GRU"""
    def __init__(self, ninp, nhid, ntok, padding_idx, emb_dropout, hid_dropout):
        super(Encoder, self).__init__()
        self.nhid = nhid
        self.emb = nn.Embedding(ntok, ninp, padding_idx=padding_idx)
        self.bi_gru = nn.GRU(ninp, nhid, 1, batch_first=True, bidirectional=True)
        self.enc_emb_dp = nn.Dropout(emb_dropout)
        self.enc_hid_dp = nn.Dropout(hid_dropout)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        h0 = Variable(weight.new(2, batch_size, self.nhid).zero_())
        return h0

    def forward(self, input, mask):
        hidden = self.init_hidden(input.size(0))
        self.bi_gru.flatten_parameters()
        input = self.enc_emb_dp(self.emb(input))
        length = mask.sum(1).tolist()
        input = torch.nn.utils.rnn.pack_padded_sequence(input, length, batch_first=True)
        output, hidden = self.bi_gru(input, hidden)
        output = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]
        output = self.enc_hid_dp(output)
        hidden = torch.cat([hidden[0], hidden[1]], dim=-1)
        return output, hidden


class Attention(nn.Module):
    """Attention Mechanism"""
    def __init__(self, nhid, ncontext, natt):
        super(Attention, self).__init__()
        self.h2s = nn.Linear(nhid, natt)
        self.s2s = nn.Linear(ncontext, natt)
        self.a2o = nn.Linear(natt, 1)

    def forward(self, hidden, mask, context):
        shape = context.size()
        attn_h = self.s2s(context.view(-1, shape[2]))
        attn_h = attn_h.view(shape[0], shape[1], -1)
        attn_h += self.h2s(hidden).unsqueeze(1).expand_as(attn_h)
        logit = self.a2o(F.tanh(attn_h)).view(shape[0], shape[1])
        if mask.any():
            logit.data.masked_fill_(1 - mask, -float('inf'))
        softmax = F.softmax(logit, dim=1)
        output = torch.bmm(softmax.unsqueeze(1), context).squeeze(1)
        return output


class VallinaDecoder(nn.Module):
    def __init__(self, ninp, nhid, enc_ncontext, natt, nreadout, readout_dropout):
        super(VallinaDecoder, self).__init__()
        self.gru1 = nn.GRUCell(ninp, nhid)
        self.gru2 = nn.GRUCell(enc_ncontext, nhid)
        self.enc_attn = Attention(nhid, enc_ncontext, natt)
        self.e2o = nn.Linear(ninp, nreadout)
        self.h2o = nn.Linear(nhid, nreadout)
        self.c2o = nn.Linear(enc_ncontext, nreadout)
        self.readout_dp = nn.Dropout(readout_dropout)

    def forward(self, emb, hidden, enc_mask, enc_context):
        hidden = self.gru1(emb, hidden)
        attn_enc = self.enc_attn(hidden, enc_mask, enc_context)
        hidden = self.gru2(attn_enc, hidden)
        output = F.tanh(self.e2o(emb) + self.h2o(hidden) + self.c2o(attn_enc))
        output = self.readout_dp(output)
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, opt):
        super(Seq2Seq, self).__init__()
        self.dec_nhid = opt.dec_nhid
        self.dec_sos = opt.dec_sos
        self.dec_eos = opt.dec_eos
        self.dec_pad = opt.dec_pad
        self.enc_pad = opt.enc_pad

        self.emb = nn.Embedding(opt.dec_ntok, opt.dec_ninp, padding_idx=opt.dec_pad)
        self.encoder = Encoder(opt.enc_ninp, opt.enc_nhid, opt.enc_ntok, opt.enc_pad, opt.enc_emb_dropout, opt.enc_hid_dropout)
        self.decoder = VallinaDecoder(opt.dec_ninp, opt.dec_nhid, 2 * opt.enc_nhid, opt.dec_natt, opt.nreadout, opt.readout_dropout)
        self.affine = nn.Linear(opt.nreadout, opt.dec_ntok)
        self.init_affine = nn.Linear(2 * opt.enc_nhid, opt.dec_nhid)
        self.dec_emb_dp = nn.Dropout(opt.dec_emb_dropout)

    def forward(self, src, src_mask, f_trg, f_trg_mask, b_trg=None, b_trg_mask=None):
        enc_context, _ = self.encoder(src, src_mask.data)
        enc_context = enc_context.contiguous()

        avg_enc_context = enc_context.sum(1)
        enc_context_len = src_mask.data.long().sum(1).unsqueeze(-1).expand_as(avg_enc_context.data)
        avg_enc_context = avg_enc_context / Variable(enc_context_len.float())

        hidden = F.tanh(self.init_affine(avg_enc_context))

        loss = 0
        for i in xrange(f_trg.size(1) - 1):
            output, hidden = self.decoder(self.dec_emb_dp(self.emb(f_trg[:, i])), hidden, src_mask.data, enc_context)
            loss += F.cross_entropy(self.affine(output), f_trg[:, i+1], reduce=False) * f_trg_mask[:, i+1].float()
        w_loss = loss.sum() / f_trg_mask[:, 1:].data.sum()
        loss = loss.mean()
        return loss, w_loss

    def beamsearch(self, src, src_mask, beam_size=10, normalize=False, max_len=None, min_len=None):
        max_len = src.size(1) * 3 if max_len is None else max_len
        min_len = src.size(1) / 2 if min_len is None else min_len

        enc_context, _ = self.encoder(src, src_mask.data)
        enc_context = enc_context.contiguous()

        avg_enc_context = enc_context.sum(1)
        enc_context_len = src_mask.data.long().sum(1).unsqueeze(-1).expand_as(avg_enc_context.data)
        avg_enc_context = avg_enc_context / Variable(enc_context_len.float())

        hidden = F.tanh(self.init_affine(avg_enc_context))

        prev_beam = Beam(beam_size)
        prev_beam.candidates = [[self.dec_sos]]
        prev_beam.scores = [0]
        f_done = (lambda x: x[-1] == self.dec_eos)

        valid_size = beam_size

        hyp_list = []
        for k in xrange(max_len):
            candidates = prev_beam.candidates
            input = Variable(src.data.new(map(lambda cand: cand[-1], candidates)))
            input = self.dec_emb_dp(self.emb(input))
            output, hidden = self.decoder(input, hidden, src_mask.data, enc_context)
            log_prob = F.log_softmax(self.affine(output), dim=1)
            if k < min_len:
                log_prob[:, self.dec_eos] = -float('inf')
            if k == max_len - 1:
                eos_prob = log_prob[:, self.dec_eos].clone()
                log_prob[:, :] = -float('inf')
                log_prob[:, self.dec_eos] = eos_prob
            next_beam = Beam(valid_size)
            done_list, remain_list = next_beam.step(-log_prob.data, prev_beam, f_done)
            hyp_list.extend(done_list)
            valid_size -= len(done_list)

            if valid_size == 0:
                break
            
            beam_remain_ix = Variable(src.data.new(remain_list))
            enc_context = enc_context.index_select(0, beam_remain_ix)
            src_mask = src_mask.index_select(0, beam_remain_ix)
            hidden = hidden.index_select(0, beam_remain_ix)
            prev_beam = next_beam
        score_list = [hyp[1] for hyp in hyp_list]
        hyp_list = [hyp[0][1: hyp[0].index(self.dec_eos)] if self.dec_eos in hyp[0] else hyp[0][1:] for hyp in hyp_list]
        if normalize:
            for k, (hyp, score) in enumerate(zip(hyp_list, score_list)):
                if len(hyp) > 0:
                    score_list[k] = score_list[k] / len(hyp)
        score = hidden.data.new(score_list)
        sort_score, sort_ix = torch.sort(score)
        output = []
        for ix in sort_ix.tolist():
            output.append((hyp_list[ix], score[ix]))
        return output
