import torch
import torch.nn as nn
# from param import *
# from tensor_dataloader import *
import torch.nn.functional as F
from torch.distributions import uniform
import copy
import numpy as np


class HyperR:
    def __init__(self, min_embed, max_embed):
        self.min_embed = min_embed
        self.max_embed = max_embed
        self.delta_embed = max_embed - min_embed


class HGE(nn.Module):
    def __init__(self, dim):
        super(HGE, self).__init__()

        self.dim = dim

        self.gumbel_beta = 0.01
        self.euler_gamma = 0.57721566490153286060
        self.alpha = 1e-16
        self.clamp_min = 0.0
        self.clamp_max = 1e10
        # self.REL_VOCAB_SIZE = params.REL_VOCAB_SIZE

    def forward(self, user_embeddings, item_embeddings):

        user_hyperr = self.get_hyperr(user_embeddings)
        item_hyperr = self.get_hyperr(item_embeddings)

        intersection_hyperr = self.intersection(user_hyperr, item_hyperr)

        log_intersection = self.log_volumes(intersection_hyperr)

        # Use the conditional probability as the confidence level of the prediction
        log_prob = log_intersection / self.log_volumes(item_hyperr)
        pos_predictions = log_prob

        # Return the confidence level of the model prediction
        # print(pos_predictions.shape)
        return pos_predictions, user_embeddings, item_embeddings

    def intersection(self, hyperr1, hyperr2):
        intersections_min = self.gumbel_beta * torch.logsumexp(
            torch.stack((hyperr1.min_embed / self.gumbel_beta, hyperr2.min_embed / self.gumbel_beta)),
            0
        )
        intersections_min = torch.max(
            intersections_min,
            torch.max(hyperr1.min_embed, hyperr2.min_embed)
        )
        intersections_max = - self.gumbel_beta * torch.logsumexp(
            torch.stack((-hyperr1.max_embed / self.gumbel_beta, -hyperr2.max_embed / self.gumbel_beta)),
            0
        )
        intersections_max = torch.min(
            intersections_max,
            torch.min(hyperr1.max_embed, hyperr2.max_embed)
        )

        intersection_hyperr = HyperR(intersections_min, intersections_max)
        return intersection_hyperr

    def log_volumes(self, hyperr, temp=1., gumbel_beta=1., scale=1.):
        eps = torch.finfo(hyperr.min_embed.dtype).tiny

        if isinstance(scale, float):
            s = torch.tensor(scale)
        else:
            s = scale

        log_vol = torch.sum(
            torch.log(
                F.softplus(hyperr.delta_embed - 2 * self.euler_gamma * self.gumbel_beta, beta=temp).clamp_min(eps)
            ),
            dim=-1
        ) + torch.log(s)

        return log_vol

    def get_hyperr(self, embeddings):
        min_rep = embeddings[:, :self.dim // 2]  # batchsize * embedding_size
        delta_rep = embeddings[:, self.dim // 2:]
        max_rep = min_rep + torch.exp(delta_rep)
        hyperr = HyperR(min_rep, max_rep)
        return hyperr


class HGE_KGE(nn.Module):
    def __init__(self, dim, device, n_relation):
        super(HGE_KGE, self).__init__()

        self.dim = dim
        self.n_relation = n_relation
        self.rel_dim = dim // 2
        self.gumbel_beta = 0.01
        self.euler_gamma = 0.57721566490153286060
        self.alpha = 1e-16
        self.clamp_min = 0.0
        self.clamp_max = 1e10

        rel_trans_for_head = torch.empty(self.n_relation, self.rel_dim)
        rel_scale_for_head = torch.empty(self.n_relation, self.rel_dim)
        torch.nn.init.normal_(rel_trans_for_head, mean=0, std=1e-4)
        torch.nn.init.normal_(rel_scale_for_head, mean=1, std=0.2)
        rel_trans_for_tail = torch.empty(self.n_relation, self.rel_dim)
        rel_scale_for_tail = torch.empty(self.n_relation, self.rel_dim)
        torch.nn.init.normal_(rel_trans_for_tail, mean=0, std=1e-4)
        torch.nn.init.normal_(rel_scale_for_tail, mean=1, std=0.2)

        self.rel_trans_for_head, self.rel_scale_for_head = nn.Parameter(rel_trans_for_head.to(device)), nn.Parameter(
            rel_scale_for_head.to(device))
        self.rel_trans_for_tail, self.rel_scale_for_tail = nn.Parameter(rel_trans_for_tail.to(device)), nn.Parameter(
            rel_scale_for_tail.to(device))

    def forward(self, head_embeddings, relation_idx, tail_embeddings):

        head = head_embeddings.clone()
        tail = tail_embeddings.clone()
        head_hyperr = self.transform_head_hyperr(head, relation_idx)
        tail_hyperr = self.transform_tail_hyperr(tail, relation_idx)

        intersection_hyperr = self.intersection(head_hyperr, tail_hyperr)

        log_intersection = self.log_volumes(intersection_hyperr)

        # Use the conditional probability as the confidence level of the prediction
        log_prob = log_intersection / self.log_volumes(tail_hyperr)
        pos_predictions = log_prob

        # Return the confidence level of the model prediction
        # print(pos_predictions.shape)
        return pos_predictions, head_embeddings, tail_embeddings

    def get_hyperr(self, embeddings):
        min_rep = embeddings[:, :self.dim // 2]  # batchsize * embedding_size
        delta_rep = embeddings[:, self.dim // 2:]
        max_rep = min_rep + torch.exp(delta_rep)
        hyperr = HyperR(min_rep, max_rep)
        return hyperr

    def transform_head_hyperr(self, head_embeddings, relation_idx):
        head_hyperr = self.get_hyperr(head_embeddings)

        # relu = nn.ReLU()

        translations = self.rel_trans_for_head[relation_idx]
        # scales = relu(self.rel_scale_for_head[relation_idx])
        scales = self.rel_scale_for_head[relation_idx]

        # affine transformation
        head_hyperr.min_embed += translations
        head_hyperr.delta_embed *= scales
        head_hyperr.max_embed = head_hyperr.min_embed + head_hyperr.delta_embed

        return head_hyperr

    def transform_tail_hyperr(self, tail_embeddings, relation_idx):
        tail_hyperr = self.get_hyperr(tail_embeddings)

        translations = self.rel_trans_for_tail[relation_idx]
        scales = self.rel_scale_for_tail[relation_idx]

        # affine transformation
        tail_hyperr.min_embed += translations
        tail_hyperr.delta_embed *= scales
        tail_hyperr.max_embed = tail_hyperr.min_embed + tail_hyperr.delta_embed

        return tail_hyperr

    def intersection(self, hyperr1, hyperr2):
        intersections_min = self.gumbel_beta * torch.logsumexp(
            torch.stack((hyperr1.min_embed / self.gumbel_beta, hyperr2.min_embed / self.gumbel_beta)),
            0
        )
        intersections_min = torch.max(
            intersections_min,
            torch.max(hyperr1.min_embed, hyperr2.min_embed)
        )
        intersections_max = - self.gumbel_beta * torch.logsumexp(
            torch.stack((-hyperr1.max_embed / self.gumbel_beta, -hyperr2.max_embed / self.gumbel_beta)),
            0
        )
        intersections_max = torch.min(
            intersections_max,
            torch.min(hyperr1.max_embed, hyperr2.max_embed)
        )

        intersection_hyperr = HyperR(intersections_min, intersections_max)
        return intersection_hyperr

    def log_volumes(self, hyperr, temp=1., gumbel_beta=1., scale=1.):
        eps = torch.finfo(hyperr.min_embed.dtype).tiny

        if isinstance(scale, float):
            s = torch.tensor(scale)
        else:
            s = scale

        log_vol = torch.sum(
            torch.log(
                F.softplus(hyperr.delta_embed - 2 * self.euler_gamma * self.gumbel_beta, beta=temp).clamp_min(eps)
            ),
            dim=-1
        ) + torch.log(s)

        return log_vol
