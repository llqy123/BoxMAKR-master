import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from layers import Dense, CrossCompressUnit, AttentionUnit
from hyperrectangular import HGE, HGE_KGE
torch.set_printoptions(threshold=np.inf)

class HyperR:
    def __init__(self, min_embed, max_embed):
        self.min_embed = min_embed
        self.max_embed = max_embed
        self.delta_embed = max_embed - min_embed


class HGKR_model(nn.Module):
    def __init__(self, args, n_user, n_item, n_entity, n_relation, use_inner_product=True):
        super(HGKR_model, self).__init__()

        ### Modeling section
        self.args = args
        self.n_user = n_user
        self.n_item = n_item
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.use_inner_product = use_inner_product
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Init embeddings
        self.user_embeddings_lookup = nn.Embedding(self.n_user, self.args.dim)
        self.item_embeddings_lookup = nn.Embedding(self.n_item, self.args.dim)
        self.entity_embeddings_lookup = nn.Embedding(self.n_entity, self.args.dim)
        self.relation_embeddings_lookup = nn.Embedding(self.n_relation, self.args.dim)

        self.user_mlp = nn.Sequential()
        self.tail_mlp = nn.Sequential()
        self.relation_mlp = nn.Sequential()
        self.cc_unit = nn.Sequential()
        for i_cnt in range(self.args.L):
            self.user_mlp.add_module('user_mlp{}'.format(i_cnt),
                                     Dense(self.args.dim, self.args.dim))
            self.tail_mlp.add_module('tail_mlp{}'.format(i_cnt),
                                     Dense(self.args.dim, self.args.dim))
            self.relation_mlp.add_module('relation_mlp{}'.format(i_cnt),
                                         Dense(self.args.dim, self.args.dim))
            # Interaction unit
            if self.args.attention:
                self.cc_unit.add_module('cc_unit{}'.format(i_cnt),
                                        AttentionUnit(self.args.dim, self.args.channel))
            else:
                self.cc_unit.add_module('cc_unit{}'.format(i_cnt),
                                        CrossCompressUnit(self.args.dim))

        ### Prediction section
        self.kge_pred_mlp = Dense(self.args.dim * 2, self.args.dim)
        self.kge_mlp = nn.Sequential()
        for i_cnt in range(self.args.H - 1):
            self.kge_mlp.add_module('kge_mlp{}'.format(i_cnt),
                                    Dense(self.args.dim * 2, self.args.dim * 2))
        if self.use_inner_product == False:
            self.rs_pred_mlp = Dense(self.args.dim * 2, 1)
            self.rs_mlp = nn.Sequential()
            for i_cnt in range(self.args.H - 1):
                self.rs_mlp.add_module('rs_mlp{}'.format(i_cnt),
                                       Dense(self.args.dim * 2, self.args.dim * 2))

        self.hge = HGE(self.args.dim)
        self.hge_kge = HGE_KGE(self.args.dim, self.device, self.n_relation)

    def forward(self, user_indices=None, item_indices=None, head_indices=None,
                relation_indices=None, tail_indices=None):

        ### Modeling section
        if user_indices is not None:
            self.user_indices = user_indices
        if item_indices is not None:
            self.item_indices = item_indices
        if head_indices is not None:
            self.head_indices = head_indices
        if relation_indices is not None:
            self.relation_indices = relation_indices
        if tail_indices is not None:
            self.tail_indices = tail_indices

        # Initial embedding of items and header entities
        self.item_embeddings = self.item_embeddings_lookup(self.item_indices)
        self.head_embeddings = self.entity_embeddings_lookup(self.head_indices)

        # Input the initial embedding of the item and head entities into the interaction unit,
        # get the representation of the item and head entities after the interaction.
        self.item_embeddings, self.head_embeddings = self.cc_unit([self.item_embeddings, self.head_embeddings])

        ### Prediction section
        if user_indices is not None:
            ## Recommended Module
            # Obtain the initial embedding of the user
            # the representation of the user through a multi-layer MLP
            self.user_embeddings = self.user_embeddings_lookup(self.user_indices)
            self.user_embeddings = self.user_mlp(self.user_embeddings)

            # Calculate the user's predicted probability of the items
            if self.use_inner_product:
                # [batch_size]
                if self.args.hge == False:
                    self.scores = torch.sum(self.user_embeddings * self.item_embeddings, 1)
                else: # Calculated using the hge method
                    self.scores, self.user_embeddings, self.item_embeddings = self.hge(self.user_embeddings,
                                                                                            self.item_embeddings)
                # print(self.scores)
            else:
                # [batch_size, dim * 2]
                self.user_item_concat = torch.cat([self.user_embeddings, self.item_embeddings], 1)
                self.user_item_concat = self.rs_mlp(self.user_item_concat)
                # [batch_size]
                self.scores = torch.squeeze(self.rs_pred_mlp(self.user_item_concat))

            self.scores_normalized = torch.sigmoid(self.scores)
            outputs = [self.user_embeddings, self.item_embeddings, self.scores, self.scores_normalized]

        if relation_indices is not None:
            ## KGE module
            # The head entity has been interacted with in the recommendation module,
            # and here we get the representation of the head entity after the interaction.
            self.tail_embeddings = self.entity_embeddings_lookup(self.tail_indices)
            self.relation_embeddings = self.relation_embeddings_lookup(self.relation_indices)
            self.relation_embeddings = self.relation_mlp(self.relation_embeddings)
            self.tail_embeddings = self.tail_mlp(self.tail_embeddings)

            # Calculating the predicted probability
            # [batch_size, dim * 2]
            if self.args.hge == False:
                self.head_relation_concat = torch.cat([self.head_embeddings, self.relation_embeddings], 1)
                self.head_relation_concat = self.kge_mlp(self.head_relation_concat)
                # [batch_size, 1]
                self.tail_pred = self.kge_pred_mlp(self.head_relation_concat)
                self.tail_pred = torch.sigmoid(self.tail_pred)
                self.scores_kge = torch.sigmoid(torch.sum(self.tail_embeddings * self.tail_pred, 1))
            else:
                self.scores_kge, self.head_embeddings, self.tail_embeddings = self.hge_kge(self.head_embeddings,
                                                                                                self.relation_indices,
                                                                                                self.tail_embeddings)
            self.rmse = 0
            # self.rmse = torch.mean(
            #     torch.sqrt(torch.sum(torch.pow(self.tail_embeddings -
            #                                    self.tail_pred, 2), 1) / self.args.dim))
            outputs = [self.head_embeddings, self.tail_embeddings, self.scores_kge, self.rmse]



        return outputs


class HGKR(object):
    def __init__(self, args, n_users, n_items, n_entities,
                 n_relations):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._parse_args(n_users, n_items, n_entities, n_relations)
        self._build_model()
        self._build_loss()
        self._build_ops()

    def _parse_args(self, n_users, n_items, n_entities, n_relations):
        self.n_user = n_users
        self.n_item = n_items
        self.n_entity = n_entities
        self.n_relation = n_relations

    def _build_model(self):
        print("Build models")
        self.HGKR_model = HGKR_model(self.args, self.n_user, self.n_item, self.n_entity, self.n_relation)
        self.HGKR_model = self.HGKR_model.to(self.device, non_blocking=True)
        for m in self.HGKR_model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                torch.nn.init.xavier_uniform_(m.weight)
        # for param in self.HGKR_model.parameters():
        #     param.requires_grad = True

    def _build_loss(self):
        self.sigmoid_BCE = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

    def _build_ops(self):
        self.optimizer_rs = torch.optim.Adam(self.HGKR_model.parameters(),
                                             lr=self.args.lr_rs)
        self.optimizer_kge = torch.optim.Adam(self.HGKR_model.parameters(),
                                              lr=self.args.lr_kge)

    def _inference_rs(self, inputs):
        # Inputs
        self.user_indices = inputs[:, 0].long().to(self.device,
                                                   non_blocking=True)
        self.item_indices = inputs[:, 1].long().to(self.device,
                                                   non_blocking=True)
        labels = inputs[:, 2].float().to(self.device)
        self.head_indices = inputs[:, 1].long().to(self.device,
                                                   non_blocking=True)

        # Inference
        outputs = self.HGKR_model(user_indices=self.user_indices,
                                 item_indices=self.item_indices,
                                 head_indices=self.head_indices,
                                 relation_indices=None,
                                 tail_indices=None)

        user_embeddings, item_embeddings, scores, scores_normalized = outputs
        return user_embeddings, item_embeddings, scores, scores_normalized, labels

    def _inference_kge(self, inputs):
        # Inputs
        self.item_indices = inputs[:, 0].long().to(self.device,
                                                   non_blocking=True)
        self.head_indices = inputs[:, 0].long().to(self.device,
                                                   non_blocking=True)
        self.relation_indices = inputs[:, 1].long().to(self.device,
                                                       non_blocking=True)
        self.tail_indices = inputs[:, 2].long().to(self.device,
                                                   non_blocking=True)

        # Inference
        outputs = self.HGKR_model(user_indices=None,
                                 item_indices=self.item_indices,
                                 head_indices=self.head_indices,
                                 relation_indices=self.relation_indices,
                                 tail_indices=self.tail_indices)

        head_embeddings, tail_embeddings, scores_kge, rmse = outputs
        return head_embeddings, tail_embeddings, scores_kge, rmse

    def l2_loss(self, inputs):
        return torch.sum(inputs ** 2) / 2

    def loss_rs(self, user_embeddings, item_embeddings, scores, labels):
        # scores_for_signll = torch.cat([1-self.sigmoid(scores).unsqueeze(1),
        #                                self.sigmoid(scores).unsqueeze(1)], 1)
        # base_loss_rs = torch.mean(self.nll_loss(scores_for_signll, labels))
        base_loss_rs = torch.mean(self.sigmoid_BCE(scores, labels))
        l2_loss_rs = self.l2_loss(user_embeddings) + self.l2_loss(item_embeddings)
        for name, param in self.HGKR_model.named_parameters():
            if param.requires_grad and ('embeddings_lookup' not in name) \
                    and (('rs' in name) or ('cc_unit' in name) or ('user' in name)) \
                    and ('weight' in name):
                l2_loss_rs = l2_loss_rs + self.l2_loss(param)
        loss_rs = base_loss_rs + l2_loss_rs * self.args.l2_weight
        return loss_rs, base_loss_rs, l2_loss_rs

    def loss_kge(self, scores_kge, head_embeddings, tail_embeddings):
        base_loss_kge = -scores_kge
        l2_loss_kge = self.l2_loss(head_embeddings) + self.l2_loss(tail_embeddings)
        for name, param in self.HGKR_model.named_parameters():
            if param.requires_grad and ('embeddings_lookup' not in name) \
                    and (('kge' in name) or ('tail' in name) or ('cc_unit' in name)) \
                    and ('weight' in name):
                l2_loss_kge = l2_loss_kge + self.l2_loss(param)
        # Note: L2 regularization will be done by weight_decay of pytorch optimizer
        # print(base_loss_kge.shape, l2_loss_kge.shape)
        loss_kge = base_loss_kge + l2_loss_kge * self.args.l2_weight
        return loss_kge, base_loss_kge, l2_loss_kge

    def train_rs(self, inputs, show_grad=False, glob_step=None):
        self.HGKR_model.train()
        user_embeddings, item_embeddings, scores, _, labels = self._inference_rs(inputs)
        loss_rs, base_loss_rs, l2_loss_rs = self.loss_rs(user_embeddings, item_embeddings, scores, labels)

        self.optimizer_rs.zero_grad()
        loss_rs.backward()

        # if show_grad:
        #     plot_grad_flow(self.HGKR_model.named_parameters(),
        #                    "grad_plot/rs_grad_step{}".format(glob_step))

        self.optimizer_rs.step()
        loss_rs.detach()
        user_embeddings.detach()
        item_embeddings.detach()
        scores.detach()
        labels.detach()

        return loss_rs, base_loss_rs, l2_loss_rs

    def train_kge(self, inputs, show_grad=False, glob_step=None):
        self.HGKR_model.train()
        head_embeddings, tail_embeddings, scores_kge, rmse = self._inference_kge(inputs)
        loss_kge, base_loss_kge, l2_loss_kge = self.loss_kge(scores_kge, head_embeddings, tail_embeddings)

        self.optimizer_kge.zero_grad()
        loss_kge.sum().backward()
        # if show_grad:
        #     plot_grad_flow(self.HGKR_model.named_parameters(),
        #                    "grad_plot/kge_grad_step{}".format(glob_step))

        self.optimizer_kge.step()
        loss_kge.detach()
        head_embeddings.detach()
        tail_embeddings.detach()
        scores_kge.detach()
        # rmse.detach()
        return rmse, loss_kge.sum(), base_loss_kge.sum(), l2_loss_kge

    def eval(self, inputs):
        self.HGKR_model.eval()
        inputs = torch.from_numpy(inputs)
        user_embeddings, item_embeddings, _, scores, labels = self._inference_rs(inputs)
        labels = labels.to("cpu").detach().numpy()
        scores = scores.to("cpu").detach().numpy()

        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))

        return auc, acc

    def topk_eval(self, user_list, train_record, test_record, item_set, k_list):
        print("Eval TopK")
        precision_list = {k: [] for k in k_list}
        recall_list = {k: [] for k in k_list}

        # Calculate the topk for each user
        for user in user_list:
            test_item_list = list(item_set - train_record[user])
            item_score_map = dict()
            scores = self._get_scores(np.array([user] * len(test_item_list)),
                                      np.array(test_item_list),
                                      np.array(test_item_list))
            items = np.array(test_item_list)
            for item, score in zip(items, scores):
                item_score_map[item] = score
            item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
            item_sorted = [i[0] for i in item_score_pair_sorted]
            for k in k_list:
                hit_num = len(set(item_sorted[:k]) & test_record[user])
                precision_list[k].append(hit_num / k)
                recall_list[k].append(hit_num / len(test_record[user]))
        precision = [np.mean(precision_list[k]) for k in k_list]
        recall = [np.mean(recall_list[k]) for k in k_list]
        f1 = [2 / (1 / precision[i] + 1 / recall[i]) for i in range(len(k_list))]

        return precision, recall, f1

    def _get_scores(self, user, item_list, head_list):
        # Inputs
        user = torch.from_numpy(user)
        item_list = torch.from_numpy(item_list)
        head_list = torch.from_numpy(head_list)
        self.user_indices = user.long().to(self.device)
        self.item_indices = item_list.long().to(self.device)
        self.head_indices = head_list.long().to(self.device)

        self.HGKR_model.eval()
        # outputs = self.HGKR_model(self.user_indices, self.item_indices,
        #                          self.head_indices, self.relation_indices,
        #                          self.tail_indices)
        outputs = self.HGKR_model(self.user_indices, self.item_indices,
                                 self.head_indices, None, None)
        user_embeddings, item_embeddings, _, scores = outputs
        return scores
