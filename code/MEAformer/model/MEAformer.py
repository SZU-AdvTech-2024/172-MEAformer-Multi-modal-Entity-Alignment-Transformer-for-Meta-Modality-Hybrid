import types
import torch
import transformers
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
import pdb
import math
from .Tool_model import AutomaticWeightedLoss
from .MEAformer_tools import MultiModalEncoder
from .MEAformer_loss import CustomMultiLossLayer, icl_loss

from src.utils import pairwise_distances
import os.path as osp
import json

from rule.rule_loss import SpecialLossRule


"""
With loss:
=========
    MEAformer(
      (multimodal_encoder): MultiModalEncoder(
        (entity_emb): Embedding(39654, 300)
        (rel_fc): Linear(in_features=1000, out_features=300, bias=True)
        (att_fc): Linear(in_features=1000, out_features=300, bias=True)
        (img_fc): Linear(in_features=2048, out_features=300, bias=True)
        (name_fc): Linear(in_features=300, out_features=300, bias=True)
        (char_fc): Linear(in_features=1669, out_features=300, bias=True)
        (cross_graph_model): GAT(
          (layer_stack): ModuleList(
            (0-1): 2 x MultiHeadGraphAttention (300 -> 300) * 2 heads
          )
        )
        (fusion): MformerFusion(
          (fusion_layer): ModuleList(
            (0): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=300, out_features=300, bias=True)
                  (key): Linear(in_features=300, out_features=300, bias=True)
                  (value): Linear(in_features=300, out_features=300, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=300, out_features=300, bias=True)
                  (LayerNorm): LayerNorm((300,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=300, out_features=400, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=400, out_features=300, bias=True)
                (LayerNorm): LayerNorm((300,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
          )
        )
      )
      (multi_loss_layer): CustomMultiLossLayer()
      (criterion_cl): icl_loss()
      (criterion_cl_joint): icl_loss()
      (criterion_transe): SpecialLossRule(
        (criterion): MarginRankingLoss()
      )
      (criterion_rule): SpecialLossRule(
        (criterion): MarginRankingLoss()
      )
    )

w/o loss:
=========
    MEAformer(
      (multimodal_encoder): MultiModalEncoder(
        (entity_emb): Embedding(39654, 300)
        (rel_fc): Linear(in_features=1000, out_features=300, bias=True)
        (att_fc): Linear(in_features=1000, out_features=300, bias=True)
        (img_fc): Linear(in_features=2048, out_features=300, bias=True)
        (name_fc): Linear(in_features=300, out_features=300, bias=True)
        (char_fc): Linear(in_features=1669, out_features=300, bias=True)
        (cross_graph_model): GAT(
          (layer_stack): ModuleList(
            (0-1): 2 x MultiHeadGraphAttention (300 -> 300) * 2 heads
          )
        )
        (fusion): MformerFusion(
          (fusion_layer): ModuleList(
            (0): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=300, out_features=300, bias=True)
                  (key): Linear(in_features=300, out_features=300, bias=True)
                  (value): Linear(in_features=300, out_features=300, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=300, out_features=300, bias=True)
                  (LayerNorm): LayerNorm((300,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=300, out_features=400, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=400, out_features=300, bias=True)
                (LayerNorm): LayerNorm((300,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
          )
        )
      )
      (multi_loss_layer): CustomMultiLossLayer()
      (criterion_cl): icl_loss()
      (criterion_cl_joint): icl_loss()
    )

"""

class MEAformer(nn.Module):
    def __init__(self, kgs, args, sp_twin_adj):
        super().__init__()
        self.kgs = kgs
        self.sp_twin_adj = sp_twin_adj
        self.args = args
        self.dim = 300
        self.rule_gamma = 0.12  # margin for relation loss
        self.img_features = F.normalize(torch.FloatTensor(kgs["images_list"])).to(self.args.device)
        self.input_idx = kgs["input_idx"].to(self.args.device)
        self.adj = kgs["adj"].to(self.args.device)
        self.rel_features = torch.Tensor(kgs["rel_features"]).to(self.args.device)
        self.att_features = torch.Tensor(kgs["att_features"]).to(self.args.device)
        self.name_features = None
        self.char_features = None
        if kgs["name_features"] is not None:
            self.name_features = kgs["name_features"].to(self.args.device)
            self.char_features = kgs["char_features"].to(self.args.device)

        img_dim = self._get_img_dim(kgs)

        char_dim = kgs["char_features"].shape[1] if self.char_features is not None else 100

        self.multimodal_encoder = MultiModalEncoder(args=self.args,
                                                    ent_num=kgs["ent_num"],
                                                    img_feature_dim=img_dim,
                                                    sp_twin_adj= self.sp_twin_adj,
                                                    char_feature_dim=char_dim,
                                                    use_project_head=self.args.use_project_head,
                                                    attr_input_dim=kgs["att_features"].shape[1])

        self.multi_loss_layer = CustomMultiLossLayer(loss_num=6)  # 6
        self.criterion_cl = icl_loss(tau=self.args.tau, ab_weight=self.args.ab_weight, n_view=2)
        self.criterion_cl_joint = icl_loss(tau=self.args.tau, ab_weight=self.args.ab_weight, n_view=2, replay=self.args.replay, neg_cross_kg=self.args.neg_cross_kg)

        tmp = -1 * torch.ones(self.input_idx.shape[0], dtype=torch.int64).to(self.args.device)
        self.replay_matrix = torch.stack([self.input_idx, tmp], dim=1).to(self.args.device)
        self.replay_ready = 0
        self.idx_one = torch.ones(self.args.batch_size, dtype=torch.int64).to(self.args.device)
        self.idx_double = torch.cat([self.idx_one, self.idx_one]).to(self.args.device)
        self.last_num = 1000000000000
        #========================================================================================#
        if self.args.rule:
            print("#==========================rule load in the Model===============================#")
            self.criterion_transe = SpecialLossRule(self.rule_gamma, cuda= True if self.args.device == 'cuda' else False)
            self.criterion_rule = SpecialLossRule(self.rule_gamma, cuda= True if self.args.device == 'cuda' else False)

        # self.idx_one = np.ones(self.args.batch_size, dtype=np.int64)
        ###################################################################
    #     self.rule_infer = True
    #
    # def rule(self, rules_data, transe_tv, ent_embedding, rel_embedding):
    #     # trans_e_score shape = [num, dim]
    #     # r_h shape = [num, 1], r_r shape = [num, 2], premises shape = [num, 2]
    #     r_h, r_t, r_r, premises = rules_data
    #     pad_value = torch.tensor([[1.0]])
    #     if self.is_cuda:
    #         pad_value = pad_value.to(self.args.device)
    #     transe_tv = torch.cat((transe_tv, pad_value), dim=0)  # for padding
    #     rule_score = self.trans_e(ent_embedding, rel_embedding, (r_h, r_t, r_r))
    #     rule_score = self.truth_value(rule_score)
    #     f1_score = transe_tv[premises[:, 0]]
    #     f2_score = transe_tv[premises[:, 1]]
    #     rule_score = 1 + f1_score * f2_score * (rule_score - 1)
    #     return rule_score
    def forward(self, batch, rules_data_sr, rules_data_tg, triples):
        gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb, hidden_states = self.joint_emb_generat(only_joint=False)
        gph_emb_hid, rel_emb_hid, att_emb_hid, img_emb_hid, name_emb_hid, char_emb_hid, joint_emb_hid = self.generate_hidden_emb(hidden_states)
        if self.args.rule:
            # 39654x300
            # 39654x300
            # 221720
            transe_tv = self.truth_value(self.trans_e(gph_emb, rel_emb, triples))
            # tg_transe_tv = self.truth_value(self.trans_e(gph_emb[19661:], rel_emb[19661:], triples[105998:]))
            # transe_tv = torch.cat((sr_transe_tv, tg_transe_tv), dim=0)
            # sr_rule_tv = self.rule(rules_data_sr, sr_transe_tv[:, :1], gph_emb[:, 0], rel_emb[:, 0])
            # tg_rule_tv = self.rule(rules_data_tg, tg_transe_tv[:, 1:], gph_emb[:, 1], rel_emb[:, 1])
            # rule_tv = torch.cat((sr_rule_tv, tg_rule_tv), dim=0)
            # print(len(rules_data_sr), type(rules_data_sr))
            # rules_data = rules_data_sr + rules_data_tg
            # r_h, r_t, r_r, premises = rules_data_sr
            # r_h_, r_t_, r_r_, premises_ = rules_data_tg
            # rules_data = [r_h+ r_h_, r_t + r_t_ , r_r+r_r_, premises+premises_]
            rules_data = tuple(torch.cat((a_i, b_i)) for a_i, b_i in zip(rules_data_sr, rules_data_tg))
            # todo :
            rule_tv = self.rule(rules_data, transe_tv, gph_emb, rel_emb)

        if self.args.replay:
            batch = torch.tensor(batch, dtype=torch.int64).to(self.args.device)
            all_ent_batch = torch.cat([batch[:, 0], batch[:, 1]])
            if not self.replay_ready:
                loss_joi, l_neg, r_neg = self.criterion_cl_joint(joint_emb, batch)
            else:
                neg_l = self.replay_matrix[batch[:, 0], self.idx_one[:batch.shape[0]]]
                neg_r = self.replay_matrix[batch[:, 1], self.idx_one[:batch.shape[0]]]
                neg_l_set = set(neg_l.tolist())
                neg_r_set = set(neg_r.tolist())
                all_ent_set = set(all_ent_batch.tolist())
                neg_l_list = list(neg_l_set - all_ent_set)
                neg_r_list = list(neg_r_set - all_ent_set)
                neg_l_ipt = torch.tensor(neg_l_list, dtype=torch.int64).to(self.args.device)
                neg_r_ipt = torch.tensor(neg_r_list, dtype=torch.int64).to(self.args.device)
                loss_joi, l_neg, r_neg = self.criterion_cl_joint(joint_emb, batch, neg_l_ipt, neg_r_ipt)

            index = (
                all_ent_batch,
                self.idx_double[:batch.shape[0] * 2],
            )
            new_value = torch.cat([l_neg, r_neg]).to(self.args.device)

            self.replay_matrix = self.replay_matrix.index_put(index, new_value)
            if self.replay_ready == 0:
                num = torch.sum(self.replay_matrix < 0)
                if num == self.last_num:
                    self.replay_ready = 1
                    print("-----------------------------------------")
                    print("begin replay!")
                    print("-----------------------------------------")
                else:
                    self.last_num = num
        else:
            loss_joi = self.criterion_cl_joint(joint_emb, batch)

        in_loss = self.inner_view_loss(gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, batch)
        out_loss = self.inner_view_loss(gph_emb_hid, rel_emb_hid, att_emb_hid, img_emb_hid, name_emb_hid, char_emb_hid, batch)
        if self.args.rule:
            # 14309506 - with sf 0.3 Ra
            rule_loss = self.criterion_transe(transe_tv) + self.criterion_rule(rule_tv)
            loss_all = loss_joi + in_loss + out_loss + rule_loss
            loss_dic = {"joint_Intra_modal": loss_joi.item(), "Intra_modal": in_loss.item(), "rule": rule_loss.item()}
        else:
            loss_all = loss_joi + in_loss + out_loss
            loss_dic = {"joint_Intra_modal": loss_joi.item(), "Intra_modal": in_loss.item()}



        # loss_dic = {"joint_Intra_modal": loss_joi.item(), "Intra_modal": in_loss.item()}
        output = {"loss_dic": loss_dic, "emb": joint_emb}
        return loss_all, output

    def trans_e(self, ent_embedding, rel_embedding, triples_data):
        h_list, t_list, r_list = triples_data
        h = ent_embedding[h_list]
        t = ent_embedding[t_list]
        r = rel_embedding[r_list]
        score = h + r - t
        return score
    def normalize(self):
        self.entity_embedding.normalize()
        self.relation_embedding.normalize()



    def truth_value(self, score):
        score = torch.norm(score, p=1, dim=-1)
        return 1 - score / 3 / math.sqrt(self.dim)  # (3 * math.sqrt(self.dim))
    def \
              rule(self, rules_data, transe_tv, ent_embedding, rel_embedding):
        # trans_e_score shape = [num, dim]
        # r_h shape = [num, 1], r_r shape = [num, 2], premises shape = [num, 2]
        r_h, r_t, r_r, premises = rules_data
        if transe_tv.shape[1] == 2:
            pad_value = torch.tensor([[1.0, 1.0]])
        else:
            pad_value = torch.tensor([[1.0]])


        # if self.is_cuda:
        #     pad_value = pad_value.cuda()
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
    def generate_hidden_emb(self, hidden):
        gph_emb = F.normalize(hidden[:, 0, :].squeeze(1))
        rel_emb = F.normalize(hidden[:, 1, :].squeeze(1))
        att_emb = F.normalize(hidden[:, 2, :].squeeze(1))
        img_emb = F.normalize(hidden[:, 3, :].squeeze(1))
        if hidden.shape[1] >= 6:
            name_emb = F.normalize(hidden[:, 4, :].squeeze(1))
            char_emb = F.normalize(hidden[:, 5, :].squeeze(1))
            joint_emb = torch.cat([gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb], dim=1)
        else:
            name_emb, char_emb = None, None
            loss_name, loss_char = None, None
            joint_emb = torch.cat([gph_emb, rel_emb, att_emb, img_emb], dim=1)

        return gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, joint_emb

    def inner_view_loss(self, gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, train_ill):
        # pdb.set_trace()
        loss_GCN = self.criterion_cl(gph_emb, train_ill) if gph_emb is not None else 0
        loss_rel = self.criterion_cl(rel_emb, train_ill) if rel_emb is not None else 0
        loss_att = self.criterion_cl(att_emb, train_ill) if att_emb is not None else 0
        loss_img = self.criterion_cl(img_emb, train_ill) if img_emb is not None else 0
        loss_name = self.criterion_cl(name_emb, train_ill) if name_emb is not None else 0
        loss_char = self.criterion_cl(char_emb, train_ill) if char_emb is not None else 0

        total_loss = self.multi_loss_layer([loss_GCN, loss_rel, loss_att, loss_img, loss_name, loss_char])
        return total_loss

    # --------- necessary ---------------

    def joint_emb_generat(self, only_joint=True):
        gph_emb, img_emb, rel_emb, att_emb, \
            name_emb, char_emb, joint_emb, hidden_states, weight_norm = self.multimodal_encoder(self.input_idx,
                                                                                                self.adj,
                                                                                                self.img_features,
                                                                                                self.rel_features,
                                                                                                self.att_features,
                                                                                                self.name_features,
                                                                                                self.char_features)
        if only_joint:
            return joint_emb, weight_norm
        else:
            return gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb, hidden_states

    # --------- share ---------------

    def _get_img_dim(self, kgs):
        if isinstance(kgs["images_list"], list):
            img_dim = kgs["images_list"][0].shape[1]
        elif isinstance(kgs["images_list"], np.ndarray) or torch.is_tensor(kgs["images_list"]):
            img_dim = kgs["images_list"].shape[1]
        return img_dim

    def Iter_new_links(self, epoch, left_non_train, final_emb, right_non_train, new_links=[]):
        if len(left_non_train) == 0 or len(right_non_train) == 0:
            return new_links
        distance_list = []
        for i in np.arange(0, len(left_non_train), 1000):
            d = pairwise_distances(final_emb[left_non_train[i:i + 1000]], final_emb[right_non_train])
            distance_list.append(d)
        distance = torch.cat(distance_list, dim=0)
        preds_l = torch.argmin(distance, dim=1).cpu().numpy().tolist()
        preds_r = torch.argmin(distance.t(), dim=1).cpu().numpy().tolist()
        del distance_list, distance, final_emb
        if (epoch + 1) % (self.args.semi_learn_step * 5) == self.args.semi_learn_step:
            new_links = [(left_non_train[i], right_non_train[p]) for i, p in enumerate(preds_l) if preds_r[p] == i]
        else:
            new_links = [(left_non_train[i], right_non_train[p]) for i, p in enumerate(preds_l) if (preds_r[p] == i) and ((left_non_train[i], right_non_train[p]) in new_links)]

        return new_links

    def data_refresh(self, logger, train_ill, test_ill_, left_non_train, right_non_train, new_links=[]):
        if len(new_links) != 0 and (len(left_non_train) != 0 and len(right_non_train) != 0):
            new_links_select = new_links
            train_ill = np.vstack((train_ill, np.array(new_links_select)))
            num_true = len([nl for nl in new_links_select if nl in test_ill_])
            # remove from left/right_non_train
            for nl in new_links_select:
                left_non_train.remove(nl[0])
                right_non_train.remove(nl[1])

            if self.args.rank == 0:
                logger.info(f"#new_links_select:{len(new_links_select)}")
                logger.info(f"train_ill.shape:{train_ill.shape}")
                logger.info(f"#true_links: {num_true}")
                logger.info(f"true link ratio: {(100 * num_true / len(new_links_select)):.1f}%")
                logger.info(f"#entity not in train set: {len(left_non_train)} (left) {len(right_non_train)} (right)")

            new_links = []
        else:
            logger.info("len(new_links) is 0")

        return left_non_train, right_non_train, train_ill, new_links
