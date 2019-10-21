# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM

INFINITY_NUMBER = 1e12

class ASGCN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ASGCN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gcn_lstm = DynamicLSTM(2*opt.hidden_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*2*opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)

        self.gate1 = nn.Sequential(nn.Linear(opt.hidden_dim*2, opt.hidden_dim*2), nn.Tanh(),
                                   nn.Linear(opt.hidden_dim * 2, opt.hidden_dim * 2), nn.Sigmoid())
        self.gate2 = nn.Sequential(nn.Linear(opt.hidden_dim*2, opt.hidden_dim*2), nn.Tanh(),
                                   nn.Linear(opt.hidden_dim * 2, opt.hidden_dim * 2), nn.Sigmoid())

    def get_mask(self, x, aspect_double_idx, reverse=False):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i,0]):
                if reverse:
                    mask[i].append(1)
                else:
                    mask[i].append(0)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                if reverse:
                    mask[i].append(0)
                else:
                    mask[i].append(1)
            for j in range(aspect_double_idx[i,1]+1, seq_len):
                if reverse:
                    mask[i].append(1)
                else:
                    mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device)
        return mask.byte()

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, text_len)
        mask = self.get_mask(text_out, aspect_double_idx)
        sent = torch.max(text_out.masked_fill(mask, -INFINITY_NUMBER), 1)[0]
        mask = self.get_mask(text_out, aspect_double_idx, reverse=True)
        aspect = torch.max(text_out.masked_fill(mask, -INFINITY_NUMBER), 1)[0]
        output = self.fc(torch.cat([aspect, sent], dim=1))


        gate1 = self.gate1(aspect).repeat(1, text_out.shape[1]).view(text_out.shape)
        # gate2 = self.gate2(aspect).repeat(1, text_out.shape[1]).view(text_out.shape)
        
        return output



        # target = torch.max(text_out.masked_fill(mask.byte(), -INFINITY_NUMBER), 1)[0]
        # gate1 = self.gate1(target).repeat(1, text_out.shape[1]).view(text_out.shape)
        # gate2 = self.gate2(target).repeat(1, text_out.shape[1]).view(text_out.shape)
        # gcn1 = self.gc1(text_out, adj)
        # x = gcn1 * gate1
        # y = gcn1 * gate2
        # x1 = torch.max(x.masked_fill(text_mask.byte(), -INFINITY_NUMBER), 1)[0]
        # y1 = torch.max(y.masked_fill(text_mask.byte(), -INFINITY_NUMBER), 1)[0]
        # sf1 = nn.Softmax(1)
        # xy = (sf1(x1)*sf1(y1)).sum(1).mean()
        # x = self.gc2(x, adj) * gate2
        # x_sent = torch.max(x.masked_fill(text_mask.byte(), -INFINITY_NUMBER), 1)[0]
        # output = self.fc(torch.cat([x_sent, target], dim=1))
        # output_w = self.fc(torch.cat([x, target.repeat(1,x.shape[1]).view(x.shape)], dim=2))
        # sf2 = nn.Softmax(2)
        # output_p = sf2(output.repeat(1,text_indices.shape[1]).view(text_indices.shape[0], text_indices.shape[1], -1))
        # output_w_p = sf2(output_w)
        # scores = (output_p * torch.log(output_p/output_w_p+EPSILON)).sum(2)
        # scores_p = sf1(scores)
        # dist_to_target_p = sf1(dist_to_target.float())
        # kl = (dist_to_target_p * torch.log(dist_to_target_p / scores_p + EPSILON)).sum(1).mean()
        # return output, xy, kl