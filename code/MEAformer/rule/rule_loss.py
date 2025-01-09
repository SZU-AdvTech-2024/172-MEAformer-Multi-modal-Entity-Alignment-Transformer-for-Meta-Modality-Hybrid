# -- coding: utf-8 --
# @Author : ZhiliangLong
# @File : rule_loss.py
# @Time : 2024/11/22 9:11
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .cross_graph_completion import CrossGraphCompletion,print_time_info


class RuleDataset(Dataset):
    def __init__(self, cgc, data_name, triples, relations, nega_sample_num):
        self.triples = set(triples)
        assert len(self.triples) == len(triples)
        assert isinstance(cgc, CrossGraphCompletion)
        self.cgc = cgc
        self.data_name = data_name
        self.premise_pad = len(self.triples)
        print_time_info('premise pad number: %d' % self.premise_pad)
        self.nega_sample_num = nega_sample_num
        self.relations = relations
        self.h = []
        self.t = []
        self.pos_r = []
        self.neg_r = []
        self.premises = []

        self.check_p = -100
        self.init()

    @property
    def new_triple_premises(self):
        data = getattr(self.cgc, self.data_name)
        if self.check_p < 0:
            self.check_p = len(data)
        else:
            print('boot check in Rule dataset: ', self.check_p, len(data))
        return data

    def init(self):
        triples = self.triples
        relations = self.relations
        premise_pad = self.premise_pad
        nega_sample_num = self.nega_sample_num
        self.h = []
        self.t = []
        self.pos_r = []
        self.neg_r = []
        self.premises = []
        for new_triple, premises in self.new_triple_premises.items():
            h, t, r = new_triple
            neg_rs = random.sample(relations, k=nega_sample_num)
            neg_rs = [neg_r for neg_r in neg_rs if (h, t, neg_r) not in triples]
            while len(neg_rs) < nega_sample_num:
                neg_r = random.choice(relations)
                if (h, t, neg_r) not in triples:
                    neg_rs.append(neg_r)
            self.neg_r += neg_rs
            self.pos_r += [r] * nega_sample_num
            self.h += [h] * nega_sample_num
            self.t += [t] * nega_sample_num
            if len(premises) == 1:
                premises.append(premise_pad)
            assert len(premises) == 2
            premises = [premise * 2 * nega_sample_num for premise in premises]  # for rule & transe alignment
            self.premises += [premises] * nega_sample_num  #
        assert len(self.h) == len(self.t) == len(self.pos_r) == len(self.neg_r) == len(self.premises)
        return self

    def __len__(self):
        return len(self.h)

    def __getitem__(self, idx):
        h = torch.tensor([self.h[idx]], dtype=torch.int64)
        t = torch.tensor([self.t[idx]], dtype=torch.int64)
        r = torch.tensor([self.pos_r[idx], self.neg_r[idx]], dtype=torch.int64)
        premise = torch.tensor(self.premises[idx], dtype=torch.int64)
        return h, t, r, premise

    def get_all(self):
        h_all = torch.tensor(self.h, dtype=torch.int64).view(-1, 1)
        t_all = torch.tensor(self.t, dtype=torch.int64).view(-1, 1)
        r_all = torch.tensor(list(zip(self.pos_r, self.neg_r)), dtype=torch.int64)
        premise_all = torch.tensor(self.premises, dtype=torch.int64)
        return h_all, t_all, r_all, premise_all


class SpecialLossRule(nn.Module):
    def __init__(self, margin, re_scale=1.0, cuda=True):
        super(SpecialLossRule, self).__init__()
        self.p = 2
        self.re_scale = re_scale
        self.criterion = nn.MarginRankingLoss(margin)
        self.is_cuda = cuda

    def forward(self, score):
        """
        score shape: [batch_size, 1 + nega_sample_num, embedding_dim]
        """
        # distance = torch.abs(score).sum(dim=-1) * self.re_scale
        pos_score = score[:, 0]
        nega_score = score[:, 1]
        y = torch.FloatTensor([1.0])
        if self.is_cuda:
            y = y.cuda()
        loss = self.criterion(pos_score, nega_score, y)
        loss = loss * self.re_scale
        return loss




def trans_e(self, ent_embedding, rel_embedding, triples_data):
    h_list, t_list, r_list = triples_data
    h = ent_embedding[h_list]
    t = ent_embedding[t_list]
    r = rel_embedding[r_list]
    score = h + r - t
    return score

def rule(self, rules_data, transe_tv, ent_embedding, rel_embedding):
    # trans_e_score shape = [num, dim]
    # r_h shape = [num, 1], r_r shape = [num, 2], premises shape = [num, 2]
    r_h, r_t, r_r, premises = rules_data
    pad_value = torch.tensor([[1.0]])
    if self.is_cuda:
        pad_value = pad_value.cuda()
    transe_tv = torch.cat((transe_tv, pad_value), dim=0)  # for padding
    rule_score = self.trans_e(ent_embedding, rel_embedding, (r_h, r_t, r_r))

    # I(t) = 1 − 1  3√d ||ei + rij − ej||2
    rule_score = self.truth_value(rule_score)
    f1_score = transe_tv[premises[:, 0]]
    f2_score = transe_tv[premises[:, 1]]

    # I(ts) = I(ts1 ∧ ts2) = I(ts1) · I(ts2)
    # I(ts ⇒ tc) = I(ts) · I(tc) − I(ts) + 1 = 1 + I(ts)*(I(ts1) · I(ts2) - 1)
    rule_score = 1 + f1_score * f2_score * (rule_score - 1)
    return rule_score

# ---------------------------------------------------------------------------------------- #
#　invoke:
"""
sr_rule_tv = rule(rules_data_sr, sr_transe_tv[:, :1], graph_embedding_sr, rel_embedding_sr)
tg_rule_tv = rule(rules_data_tg, tg_transe_tv[:, 1:], graph_embedding_tg, rel_embedding_tg)
rule_tv = torch.cat((sr_rule_tv, tg_rule_tv), dim=0)
return sr_data_repre, tg_data_repre, sr_rel_repre, tg_rel_repre, transe_tv, rule_tv
"""
