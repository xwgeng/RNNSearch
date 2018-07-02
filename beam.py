import itertools


class Beam(object):
    def __init__(self, beam_size):
        self.beam_size = beam_size

        self.candidates = []
        self.scores = []

    def step(self, prob, prev_beam, f_done):
        pre_score = prob.new_tensor(prev_beam.scores)
        score = prob + pre_score.unsqueeze(-1).expand_as(prob)
        nbest_score, nbest_ix = score.view(-1).topk(self.beam_size, largest=False)
        beam_ix = nbest_ix / prob.size(1)
        token_ix = nbest_ix - beam_ix * prob.size(1)

        done_list, remain_list = [], []
        prev_candidates = prev_beam.candidates
        for b_score, b_ix, t_ix in itertools.izip(nbest_score.tolist(), beam_ix.tolist(), token_ix.tolist()):
            candidate = prev_candidates[b_ix] + [t_ix]

            if f_done(candidate):
                done_list.append([candidate, b_score])
            else:
                remain_list.append(b_ix)
                self.candidates.append(candidate)
                self.scores.append(b_score)
        return done_list, remain_list
