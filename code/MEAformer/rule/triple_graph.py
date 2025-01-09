import random

import rdflib
import torch
from torch.utils.data import Dataset


class TripleGraph(object):
    def __init__(self):
        self.graph = rdflib.Graph()
        self.prefix_query = '\n'.join([
            'BASE <http://www.entity.com#>'
            'PREFIX relation: <http://www.relation.org#>'
        ])
        self.prefix_load = '\n'.join([
            '@base <http://www.entity.com#> .'
            '@prefix relation: <http://www.relation.org#> .'
        ])
        self.triples = set()

    def load(self, triples):
        data = self.prefix_load
        for head, tail, relation in triples:
            data += ('<' + str(head) + '> ' + 'relation:' +
                     str(relation) + '<' + str(tail) + '> .\n')
        self.graph.parse(data=data, format='turtle')
        triples = set(triples)
        self.triples.update(triples)

    def query(self, query):
        return list(self.graph.query(self.prefix_query + query))

    def inference_by_rule2(self, rule):
        query = ''
        premises, hypothesis, conf = rule
        inferred_relation = hypothesis[2]
        for head, tail, relation in premises:
            query += '?' + head + ' relation:' + relation + ' ?' + tail + ' .\n'
        # query = 'select distinct ?a ?b where {%s}' % query
        query = "select distinct * where {%s}" % query
        query_result = self.query(self.prefix_query + query)
        new_triple_confs = []
        for a, b in query_result:
            a = a.split('/')[-1]
            b = b.split('/')[-1]
            new_triple_confs.append(((a, b, inferred_relation), conf))
        return new_triple_confs
# TODO=============================================================
    def inference_by_rule(self, rule):
        query = ''
        premises, hypothesis, conf = rule
        inferred_relation = hypothesis[2]
        for head, tail, relation in premises:
            query += '?' + str(head) + ' relation:' + str(relation) + ' ?' + str(tail) + ' .\n'
        # query = 'select distinct ?a ?b where {%s}' % query
        query = "select distinct * where {%s}" % query
        query_result = self.graph.query(self.prefix_query + query)
        bindings = [{str(key): str(value).split('/')[-1] for key, value in d.items()} for d in query_result.bindings]
        assert len(list(query_result)) == len(query_result.bindings)
        new_triple_confs_premises = []
        for binding in bindings:
            premise_instances = []
            for head, tail, relation in premises:
                premise_instances.append((int(binding[head]), int(binding[tail]), int(relation)))
            a = int(binding['a'])
            b = int(binding['b'])
            new_triple_confs_premises.append(((a, b, inferred_relation), conf, premise_instances))
        return new_triple_confs_premises




class TripleDataset(Dataset):
    def __init__(self, triples, nega_sapmle_num):
        self.triples = triples
        self.triple_set = set(triples)
        self.nega_sample_num = nega_sapmle_num
        h, t, r = list(zip(*triples))
        self.hs = list(set(h))
        self.ts = list(set(t))
        r2e = {}
        for head, tail, relation in triples:
            if relation not in r2e:
                r2e[relation] = {'h': {head, }, 't': {tail, }}
            else:
                r2e[relation]['h'].add(head)
                r2e[relation]['t'].add(tail)
        self.r2e = {r: {k: list(box) for k, box in es.items()} for r, es in r2e.items()}
        self.postive_data = [[], [], []]
        self.negative_data = [[], [], []]
        self.init()

    def init(self):
        r2e = self.r2e
        nega_sample_num = self.nega_sample_num

        def exists(h, t, r):
            return (h, t, r) in self.triple_set

        def _init_one(h, t, r):
            h_a = r2e[r]['h']
            t_a = r2e[r]['t']
            nega_h = random.sample(h_a, min(nega_sample_num + 1, len(h_a)))
            nega_t = random.sample(t_a, min(nega_sample_num + 1, len(t_a)))
            nega_h = [hh for hh in nega_h if not exists(hh, t, r)][:nega_sample_num]
            nega_t = [tt for tt in nega_t if not exists(h, tt, r)][:nega_sample_num]
            while len(nega_h) < nega_sample_num:
                hh = random.choice(self.hs)
                if not exists(hh, t, r):
                    nega_h.append(hh)
            while len(nega_t) < nega_sample_num:
                tt = random.choice(self.ts)
                if not exists(h, tt, r):
                    nega_t.append(tt)
            nega_h = nega_h + len(nega_h) * [h]
            nega_t = len(nega_t) * [t] + nega_t
            return nega_h, nega_t

        self.postive_data = [[], [], []]
        self.negative_data = [[], [], []]
        for h, t, r in self.triples:
            nega_h, nega_t = _init_one(h, t, r)
            self.negative_data[0] += nega_h
            self.negative_data[1] += nega_t
            self.negative_data[2] += [r] * len(nega_h)
            self.postive_data[0] += [h] * len(nega_h)
            self.postive_data[1] += [t] * len(nega_h)
            self.postive_data[2] += [r] * len(nega_h)
        return self

    def __len__(self):
        return len(self.postive_data[0])

    def __getitem__(self, idx):
        pos_h = self.postive_data[0][idx]
        pos_t = self.postive_data[1][idx]
        pos_r = self.postive_data[2][idx]
        neg_h = self.negative_data[0][idx]
        neg_t = self.negative_data[1][idx]
        neg_r = self.negative_data[2][idx]
        h_list = torch.tensor([pos_h, neg_h], dtype=torch.int64)
        t_list = torch.tensor([pos_t, neg_t], dtype=torch.int64)
        r_list = torch.tensor([pos_r, neg_r], dtype=torch.int64)
        return h_list, t_list, r_list

    def get_all(self):
        h_all = torch.tensor(list(zip(self.postive_data[0], self.negative_data[0])), dtype=torch.int64)
        t_all = torch.tensor(list(zip(self.postive_data[1], self.negative_data[1])), dtype=torch.int64)
        r_all = torch.tensor(list(zip(self.postive_data[2], self.negative_data[2])), dtype=torch.int64)
        return h_all, t_all, r_all