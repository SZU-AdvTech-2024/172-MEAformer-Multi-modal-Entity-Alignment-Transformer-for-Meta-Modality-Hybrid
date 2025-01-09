import json
import random, pickle
import time

from .triple_graph import TripleGraph


def read_mapping_i_name(path):
    def _parser(lines):
        for idx, line in enumerate(lines):
            i, name = line.strip().split('\t')
            lines[idx] = (name, int(i))
        return dict(lines)
    # lambda lines: dict([line.strip().split('\t') for line in lines])
    return read_file(path, _parser)
def read_mapping_name_i(path):
    def _parser(lines):
        for idx, line in enumerate(lines):
            name, i = line.strip().split('\t')
            lines[idx] = (name, int(i))
        return dict(lines)
    # lambda lines: dict([line.strip().split('\t') for line in lines])
    return read_file(path, _parser)


def read_triples(path):
    """
    triple pattern: (head_id, tail_id, relation_id)
    """
    return read_file(path, lambda lines: [tuple([int(item) for item in line.strip().split('\t')]) for line in lines])


def read_seeds(path):
    return read_file(path, lambda lines: [tuple([int(item) for item in line.strip().split('\t')]) for line in lines])


def read_rules(path, relation2id):
    def _read_rules(lines):
        lines = [json.loads(line) for line in lines]
        for i in range(len(lines)):
            premises, hypothesis, conf = lines[i]
            premises = tuple([tuple([head, tail, relation2id[relation]])
                              for head, tail, relation in premises])
            hypothesis = tuple(
                [hypothesis[0], hypothesis[1], relation2id[hypothesis[2]]])
            lines[i] = (premises, hypothesis, float(conf))
        return lines

    return read_file(path, _read_rules)


def read_file(path, parse_func):
    num = -1
    with open(path, 'r', encoding='utf8') as f:
        line = f.readline().strip()
        if line.isdigit():
            num = int(line)
        else:
            f.seek(0)
        lines = f.readlines()

    lines = parse_func(lines)

    if len(lines) != num and num >= 0:
        print_time_info('File: %s has corruptted, data_num: %d/%d.' %
                        (path, num, len(lines)))
        raise ValueError()
    return lines




def print_time_info(string, end='\n', dash_top=False, dash_bot=False, file=None):
    times = str(time.strftime('%Y-%m-%d %H:%M:%S',
                              time.localtime(time.time())))
    string = "[%s] %s" % (times, str(string))
    if dash_top:
        print(len(string) * '-', file=file)
    print(string, end=end, file=file)
    if dash_bot:
        print(len(string) * '-', file=file)
# 以上是数据处理
# ================================================================= #
# ================================================================= #
# ================================================================= #
def construct_entity_pair_for_bootstrap(sr_rel, tg_rel, rel_seeds):
    """
    rel_seeds = [(1,2),(3,2),(1,3)]
    sr_seeds, tg_seeds = list(zip(*rel_seeds))
    sr_seeds : (1, 3, 1)


    return : the rel triplet after filter
    """
    sr_seeds, tg_seeds = list(zip(*rel_seeds))
    sr_seeds = set(sr_seeds)
    tg_seeds = set(tg_seeds)
    # filter the ont aligned
    sr_rel = [rel for rel in sr_rel if rel not in sr_seeds]
    tg_rel = [rel for rel in tg_rel if rel not in tg_seeds]
    random.shuffle(sr_rel)
    random.shuffle(tg_rel)
    length = min(len(sr_rel), len(tg_rel))
    rel_seeds_for_bootstrap = list(zip(sr_rel[:length], tg_rel[:length]))
    return rel_seeds_for_bootstrap


def dict_union(dict1, dict2):
    """
    use it with careful
    """
    dict1 = {key: value for key, value in dict1.items()}
    for key, value in dict2.items():
        dict1[key] = value
    return dict1


def _check(ori, new, num):
    if len(ori) != len(new):
        print_time_info('Check failed %d.' % num, dash_top=True)
        raise ValueError()


def _load_languge(directory, language):
    relation2id = read_mapping_name_i(
        directory / ('relation2id_' + language + '.txt'))
    rules = read_rules(directory / 'AMIE' /
                       ('rule_for_triples_' + language + '.txt'), relation2id)
    if language == 'fr' :
        language = str(1)
    else:
        language = str(2)
    entity2id = read_mapping_i_name(directory / ('ent_ids_' + language))

    id2entity = {i: entity for entity, i in entity2id.items()}
    id2relation = {i: relation for relation, i in relation2id.items()}
    return id2entity, id2relation, rules


def _load_seeds(directory):
    relation_seeds = read_seeds(directory / 'relation_seeds.txt')
    # try:
    #     entity_seeds = read_seeds(directory / 'entity_seeds.txt')
    #     random.shuffle(entity_seeds)
    #     train_entity_seeds = entity_seeds[:int(len(entity_seeds) * train_seeds_ratio)]
    #     test_entity_seeds = entity_seeds[int(len(entity_seeds) * train_seeds_ratio):]
    # except FileNotFoundError:
    #     train_entity_seeds = read_seeds(directory / 'train_entity_seeds.txt')
    #     test_entity_seeds = read_seeds(directory / 'test_entity_seeds.txt')
    #     random.shuffle(train_entity_seeds)
    #     random.shuffle(test_entity_seeds)
    # return train_entity_seeds, test_entity_seeds, relation_seeds
    return relation_seeds

def print_triple(triple, id2entity, id2relation, end='\n'):
    head, tail, relation = triple
    print_time_info(
        ' '.join([id2entity[head], id2relation[relation], id2entity[tail]]), end=end)


def print_rule(rule, id2relation):
    premises, hypothesis, conf = rule
    premises = [(premise[0], id2relation[premise[2]], premise[1])
                for premise in premises]
    hypothesis = [(hypothesis[0], id2relation[hypothesis[2]], hypothesis[1])]
    rule = premises + [['=>']] + hypothesis + [[str(conf)]]
    print_time_info('  '.join(' '.join(part) for part in rule))


def _print_new_rules(bi_new_rules, id2relation_sr, id2relation_tg):
    for language, rules in bi_new_rules.items():
        print_time_info(language, dash_top=True)
        for rule in rules[:20]:
        # for rule in random.choices(rules, k=20):
            if language == 'sr':
                print_rule(rule, id2relation_sr)
            else:
                print_rule(rule, id2relation_tg)


def _print_new_triple_confs(bi_new_triple_confs, id2entity_sr, id2entity_tg, id2relation_sr, id2relation_tg):
    for language, triple_confs in bi_new_triple_confs.items():
        print_time_info(language, dash_top=True)
        for triple in random.choices(list(triple_confs.keys()), k=10):
            conf = triple_confs[triple]
            if language == 'sr':
                print_triple(triple, id2entity_sr, id2relation_sr, end='')
            else:
                print_triple(triple, id2entity_tg, id2relation_tg, end='')
            print(' ', conf)


def get_relation2conf(rules):
    relation2conf = {}
    for premises, hypothesis, conf in rules:
        inferred_relation = hypothesis[2]
        if inferred_relation in relation2conf:
            relation2conf[inferred_relation].append(float(conf))
        else:
            relation2conf[inferred_relation] = [float(conf)]
    relation2conf = {relation: sum(confs) / len(confs)
                     for relation, confs in relation2conf.items()}
    return relation2conf


def get_relation2imp(triples, relation_num):
    """
    imp : important relations
    """
    relation2imp = {i: {'head': set(), 'tail': set()} for i in range(relation_num)}

    for head, tail, relation in triples:
        relation2imp[relation]['head'].add(head)
        relation2imp[relation]['tail'].add(tail)

    # [(0, 1), (1, 1), (2, 1), (3, 0.36363636363636365), (4, 0.9857142857142858), (5, 0.25), (6, 1), (7, 1), (8, 1), (9, 1) ...]

    relation2imp = {relation: min(1, len(ht['tail']) / len(ht['head'])) for relation, ht in relation2imp.items()}


    return relation2imp


def _rule_based_graph_completion(triple_graph_sr, triple_graph_tg, rules_sr, rules_tg, triple2id_sr, triple2id_tg):
    """
    triples = [(head, tail, relation)]
    return new [((head, tail, relation), conf)...]
    """
    print_time_info('Rule based graph completion started!')

    def __rule_based_graph_completion_ori(triple_graph, rules):
        triples = triple_graph.triples
        new_triple_confs = {}
        new_triple_premises = {}
        for rule in rules:
            # print('The rule is', rule)
            new_triple_conf_premises_candidates = triple_graph.inference_by_rule(rule)
            # i = 0
            for new_triple, conf, premises in new_triple_conf_premises_candidates:
                if new_triple in triples:
                    if new_triple not in new_triple_confs:
                        new_triple_confs[new_triple] = conf
                        new_triple_premises[new_triple] = premises
                        # i += 1
                    else:
                        ori_conf = new_triple_confs[new_triple]
                        if ori_conf < conf:
                            new_triple_confs[new_triple] = conf
                            new_triple_premises[new_triple] = premises
                            # i += 1
            # print(i, '-----------')
        return new_triple_confs, new_triple_premises

    def __rule_based_graph_completion(triple_graph, rules):
        triples = triple_graph.triples
        new_triple_confs = {}
        new_triple_premises = {}
        for rule in rules:
            # print('The rule is', rule)
            new_triple_conf_premises_candidates = triple_graph.inference_by_rule(rule)
            # i = 0
            for new_triple, conf, premises in new_triple_conf_premises_candidates:
                if not new_triple in triples:
                    if new_triple not in new_triple_confs:
                        new_triple_confs[new_triple] = conf
                        new_triple_premises[new_triple] = premises
                        # i += 1
                    else:
                        ori_conf = new_triple_confs[new_triple]
                        if ori_conf < conf:
                            new_triple_confs[new_triple] = conf
                            new_triple_premises[new_triple] = premises
                            # i += 1
            # print(i, '-----------')
        return new_triple_confs, new_triple_premises
    # triple_confs_* : (h,r,t): conf
    # new_triple_premises_* : (h,r,t): [premise, ...]
    #TODO:=========================================================================
    ori = True
    if ori:
        new_triple_confs_sr, new_triple_premises_sr = __rule_based_graph_completion_ori(
            triple_graph_sr, rules_sr)
        new_triple_confs_tg, new_triple_premises_tg = __rule_based_graph_completion_ori(
            triple_graph_tg, rules_tg)
    else:
        new_triple_confs_sr, new_triple_premises_sr = __rule_based_graph_completion(
            triple_graph_sr, rules_sr)
        new_triple_confs_tg, new_triple_premises_tg = __rule_based_graph_completion(
            triple_graph_tg, rules_tg)


    new_triple_premises_sr = {triple: [triple2id_sr[premise] for premise in premises] for triple, premises in
                              new_triple_premises_sr.items()}
    new_triple_premises_tg = {triple: [triple2id_tg[premise] for premise in premises] for triple, premises in
                              new_triple_premises_tg.items()}
    print_time_info('Rule based graph completion finished!')
    return new_triple_confs_sr, new_triple_confs_tg, new_triple_premises_sr, new_triple_premises_tg


def rule_transfer(rules_sr, rules_tg, relation_seeds):
    """
    目前仅能处理len(premises) <= 2的情况
    """
    rule2conf_sr = {(premises, hypothesis): conf for premises, hypothesis, conf in rules_sr}
    rule2conf_tg = {(premises, hypothesis): conf for premises, hypothesis, conf in rules_tg}

    def _rule_transfer(rule2conf_sr, rule2conf_tg, r2r):
        """
        r2r : is the seed rels
        """
        new_rules = []
        for rule, conf in rule2conf_sr.items():
            """
            (head, tail, relation2id[relation])
            """
            premises, hypothesis = rule
            # all relations
            relations = [premise[2] for premise in premises] + [hypothesis[2]]
            feasible = True
            for relation in relations:
                if not relation in r2r:
                    feasible = False
            if feasible:
                premises = tuple([(head, tail, r2r[relation])
                                  for head, tail, relation in premises])
                hypothesis = (hypothesis[0], hypothesis[1], r2r[hypothesis[2]])
                if (premises, hypothesis) not in rule2conf_tg:
                    if len(premises) == 1:
                        new_rules.append((premises, hypothesis, conf))
                    else:
                        # fixme: ???
                        premises = (premises[1], premises[0])
                        if premises not in rule2conf_tg:
                            new_rules.append((premises, hypothesis, conf))
        return new_rules

    r2r = dict((relation_sr, relation_tg)
               for relation_sr, relation_tg in relation_seeds)
    new_rules_tg = _rule_transfer(rule2conf_sr, rule2conf_tg, r2r)
    r2r = dict((relation_tg, relation_sr)
               for relation_sr, relation_tg in relation_seeds)
    new_rules_sr = _rule_transfer(rule2conf_tg, rule2conf_sr, r2r)
    return new_rules_sr, new_rules_tg


class CrossGraphCompletion(object):

    def __init__(self, directory, train_seeds_triplets, triplets, rule_transfer=True, graph_completion=True):
        """
        we followed the experiment setting of JAPE
        the folder under the directory JAPE/0_x contains the entity alignment dataset for train and test.
        """
        # assert train_seeds_ratio in {0.1, 0.2, 0.3, 0.4, 0.5}, print_time_info(
        #     'Not a legal train seeds ratio: %f.' % train_seeds_ratio, dash_bot=True)
        self.new_triple_premises_tg = {}
        self.new_triple_premises_sr = {}
        self.directory = directory
        self.rule_transfer = rule_transfer
        # self.train_seeds_ratio = train_seeds_ratio
        self.graph_completion = graph_completion
        language_sr, language_tg = directory.name.split('_')
        self.language_pair = {'sr': language_sr, 'tg': language_tg}

        self._entity_seeds = []
        self.bp_entity_seeds = []
        self.test_entity_seeds = []

        self._relation_seeds = []
        self.bp_relation_seeds = []
        self.test_relaiton_seeds = []  ## randomly initialized, used only for bootstrap


        self.triples_sr = [(a, c, b) for a, b, c in triplets[:105998]]
        self.triples_tg = [(a, c, b) for a, b, c in triplets[105998:]]
        self.triple2id_sr = {}
        self.triple2id_tg = {}

        self._new_triple_confs_sr = {}
        self._new_triple_confs_tg = {}
        self._new_triple_premises_sr = {}
        self._new_triple_premises_tg = {}

        self.bp_new_triple_confs_sr = {}
        self.bp_new_triple_confs_tg = {}
        self.bp_new_triple_premises_sr = {}
        self.bp_new_triple_premises_tg = {}

        self.rules_sr = []
        self.rules_tg = []
        self.rules_trans2_sr = []
        self.rules_trans2_tg = []
        self.id2entity_sr = {}
        self.id2entity_tg = {}
        self.id2relation_sr = {}
        self.id2relation_tg = {}

        # calculate the average PCA confidence of rules of which relation x is the tail
        # conf(r) = \frac{sum([pca\_conf | (premises, r, pca\_conf)\in rules])}{num((premises, r, pca\_conf)\in rules)}
        self.relation2conf_sr = {}
        self.relation2conf_tg = {}

        # calculate imp(r) = 1- min(\frac{num(head|(head, tail, r) \in triples)}{num(tail|(head, tail, r) \in triples)}, 1)
        self.relation2imp_sr = {}
        self.relation2imp_tg = {}

        self.triple_graph_sr = TripleGraph()
        self.triple_graph_tg = TripleGraph()

    def __getattribute__(self, name):
        if name in {'new_triple_confs_sr', 'new_triple_confs_tg', 'new_triple_premises_sr', 'new_triple_premises_tg'}:
            attr = object.__getattribute__(self, '_' + name)
            return dict_union(attr, object.__getattribute__(self, 'bp_' + name))
        if name in {'entity_seeds', 'relation_seeds'}:
            attr = object.__getattribute__(self, '_' + name)
            return attr + object.__getattribute__(self, 'bp_' + name)
        attr = object.__getattribute__(self, name)
        return attr

    def init(self):
        """
        1. 先加载实体元组
        2. 推理
        3. transfer
        4. 求出每种关系的重要程度
        """
        directory = self.directory
        # load from directory
        self._relation_seeds = _load_seeds(
            directory)
        #
        self.id2entity_sr, self.id2relation_sr, self.rules_sr = _load_languge(
            directory, self.language_pair['sr'])
        self.id2entity_tg, self.id2relation_tg, self.rules_tg = _load_languge(
            directory, self.language_pair['tg'])

        # self.test_relaiton_seeds = construct_entity_pair_for_bootstrap(list(self.id2relation_sr.keys()),
        #                                                                         list(self.id2relation_tg.keys()),
        #                                                                         self._relation_seeds)


        self.triples_sr = list(self.triples_sr)
        self.triples_tg = list(self.triples_tg)
        assert isinstance(self.triples_sr, list)
        assert isinstance(self.triples_tg, list)
        for i, triple in enumerate(self.triples_sr):
            self.triple2id_sr[triple] = i
        for i, triple in enumerate(self.triples_tg):
            self.triple2id_tg[triple] = i
        assert len(self.triple2id_sr) == len(self.triples_sr)
        assert len(self.triple2id_tg) == len(self.triples_tg)
        if self.graph_completion:
            self.rule_based_graph_completion()

    def rule_based_graph_completion(self):
        # rule transfer
        if self.rule_transfer:
            new_rules_sr, new_rules_tg = rule_transfer(
                self.rules_sr, self.rules_tg, self._relation_seeds)
            self.rules_sr += new_rules_sr
            self.rules_tg += new_rules_tg
            # fixme: what is the rules_trans2_sr
            self.rules_trans2_sr += new_rules_sr
            self.rules_trans2_tg += new_rules_tg
            bi_new_rules = {'sr': new_rules_sr, 'tg': new_rules_tg}
            self._print_result_log(bi_new_rules, 'rule_transfer', 'rule')
            _print_new_rules(bi_new_rules, self.id2relation_sr,
                             self.id2relation_tg)

        # load triple into TripleGraph
        self.triple_graph_load(self.triples_sr, self.triples_tg)

        # rule_based_graph_completion
        new_triple_confs_sr, new_triple_confs_tg, new_triple_premises_sr, new_triple_premises_tg = _rule_based_graph_completion(
            self.triple_graph_sr, self.triple_graph_tg, self.rules_sr, self.rules_tg, self.triple2id_sr,
            self.triple2id_tg)
        self._new_triple_confs_sr = dict_union(self._new_triple_confs_sr, new_triple_confs_sr)
        self._new_triple_confs_tg = dict_union(self._new_triple_confs_tg, new_triple_confs_tg)
        self._new_triple_premises_sr = dict_union(self._new_triple_premises_sr, new_triple_premises_sr)
        self._new_triple_premises_tg = dict_union(self._new_triple_premises_tg, new_triple_premises_tg)

        # concat
        bi_new_triple_confs = {
            'sr': new_triple_confs_sr, 'tg': new_triple_confs_tg}
        self._print_result_log(bi_new_triple_confs,
                               'rule_based_graph_completion', 'triple')

        # print(bi_new_triple_confs)
        # _print_new_triple_confs(bi_new_triple_confs, self.id2entity_sr,
        #                         self.id2entity_tg, self.id2relation_sr, self.id2relation_tg)


    def check(self):
        orig_triple_sr = {triple for triple in self.triples_sr if triple not in self._new_triple_confs_sr}
        orig_triple_tg = {triple for triple in self.triples_tg if triple not in self._new_triple_confs_tg}
        ori_pos_sr = {(h, t) for h, t, r in orig_triple_sr}
        ori_pos_tg = {(h, t) for h, t, r in orig_triple_tg}
        new_pos_sr = {(h, t) for h, t, r in self._new_triple_confs_sr}
        new_pos_tg = {(h, t) for h, t, r in self._new_triple_confs_tg}
        print('sr ori pos num:', len(ori_pos_sr), '; tg ori pos num:', len(ori_pos_tg))
        print('sr new pos num:', len(new_pos_sr), '; tg new pos num:', len(new_pos_tg))
        print('sr add pos num:', len(new_pos_sr.difference(ori_pos_sr)), '; tg add pos num:',
              len(new_pos_tg.difference(ori_pos_tg)))
        # exit()
    def bootstrap(self, new_entity_seeds, new_relation_seeds):
        self.bp_entity_seeds = new_entity_seeds
        self.bp_relation_seeds = new_relation_seeds

        print_time_info('BootStrap: new triple infer started!')
        new_rules_sr, new_rules_tg = rule_transfer(
            self.rules_sr, self.rules_tg, self._relation_seeds + new_relation_seeds)
        rules_sr = {(premises, hypothesis) for premises, hypothesis, conf in self.rules_sr}
        rules_tg = {(premises, hypothesis) for premises, hypothesis, conf in self.rules_tg}
        new_rules_sr = [(premises, hypothesis, conf) for premises, hypothesis, conf in new_rules_sr if
                        (premises, hypothesis) not in rules_sr]
        new_rules_tg = [(premises, hypothesis, conf) for premises, hypothesis, conf in new_rules_tg if
                        (premises, hypothesis) not in rules_tg]
        new_triple_confs_sr, new_triple_confs_tg, new_triple_premises_sr, new_triple_premises_tg = _rule_based_graph_completion(
            self.triple_graph_sr, self.triple_graph_tg, new_rules_sr, new_rules_tg, self.triple2id_sr,
            self.triple2id_tg)

        self.bp_new_triple_confs_sr = {triple: conf for triple, conf in new_triple_confs_sr.items() if
                                       triple not in self._new_triple_confs_sr}
        self.bp_new_triple_confs_tg = {triple: conf for triple, conf in new_triple_confs_tg.items() if
                                       triple not in self._new_triple_confs_tg}
        self.bp_new_triple_premises_sr = {triple: premises for triple, premises in new_triple_premises_sr.items() if
                                          triple not in self._new_triple_confs_sr}
        self.bp_new_triple_premises_tg = {triple: premises for triple, premises in new_triple_premises_tg.items() if
                                          triple not in self._new_triple_confs_tg}
        print_time_info(
            'BootStrap: sr new triple %d; tg new triple %d!' % (len(new_triple_confs_sr), len(new_triple_confs_tg)))
    def triple_graph_load(self, triples_sr, triples_tg):
        self.triple_graph_sr.load(triples_sr)
        self.triple_graph_tg.load(triples_tg)

    def save(self, directory):
        if not directory.exists():
            directory.mkdir()
        save_path = directory / 'cgc.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
        print_time_info('Successfully save cgc to %s.' % save_path)

    @classmethod
    def restore(cls, directory):
        load_path = directory / 'cgc.pkl'
        with open(load_path, 'rb') as f:
            new_one = pickle.load(f)
        print_time_info('Successfully loaded cgc from %s.' % load_path)
        return new_one

    def _print_result_log(self, bi_new_triples, method, data_name='triple'):
        print('------------------------------------------------------------')
        print_time_info('language_pair: ' +
                        '_'.join(self.language_pair.values()))
        print_time_info('Method: ' + method)
        for key, language in self.language_pair.items():
            print_time_info(
                language + ' new %s numbers: ' % data_name + str(len(bi_new_triples[key])))
        print('------------------------------------------------------------\n')
