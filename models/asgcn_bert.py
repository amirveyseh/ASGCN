# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class ASGCN_BERT(nn.Module):
    def __init__(self, opt):
        super(ASGCN_BERT, self).__init__()
        self.opt = opt
        self.bert_embedding = BertModel.from_pretrained(self.opt.bert_version, output_hidden_states=True,
                                                        output_attentions=True)
        if 'base' in self.opt.bert_version:
            self.bert_dim = 768
        elif 'large' in self.opt.bert_version:
            self.bert_dim = 1024

        self.gc1 = GraphConvolution(self.bert_dim * self.opt.num_last_layer_bert, 2 * opt.hidden_dim)
        self.gc2 = GraphConvolution(2 * opt.hidden_dim, self.bert_dim * self.opt.num_last_layer_bert)
        self.fc = nn.Linear(self.bert_dim * self.opt.num_last_layer_bert, opt.polarities_dim)
        self.bert_dropout = nn.Dropout(opt.drop_rate)

    def embed(self, bert_in_indices, bert_out_indices):
        '''
        :param bert_in_indices.shape = [batch size, bert seq len]
        :param bert_out_indices.shape = [batch size, original seq len]
        :return:
        '''
        last_hidden_embedding, _, all_hidden_states, all_attentions = self.bert_embedding(bert_in_indices)
        # last_hidden_embedding.shape = [batch size, seq len, bert_dim]
        word_embeddings = all_hidden_states[0]  # [batch size, seq len, bert_dim]
        hidden_embeddings = list(all_hidden_states[1:])  # num_layers x [batch size, seq len, bert_dim]
        all_out_embeddings = []
        batch_size, seq_len, bert_dim = last_hidden_embedding.shape
        selected_layers = list(range(len(hidden_embeddings)))[-self.opt.num_last_layer_bert:]
        for sent_id in range(batch_size):
            out_embeddings = torch.cat([hidden_embeddings[layer_id][sent_id][bert_out_indices[sent_id]]
                                       for layer_id in selected_layers], dim=1) # [seq len, bert_dim x num last layers]
            all_out_embeddings.append(out_embeddings)
        outputs = torch.stack(all_out_embeddings, dim=0)  # [batch size, original seq len, bert_dim]
        return outputs

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i, 0]):
                weight[i].append(1 - (aspect_double_idx[i, 0] - j) / context_len)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i, 1] + 1, text_len[i]):
                weight[i].append(1 - (j - aspect_double_idx[i, 1]) / context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).to(self.opt.device)
        return weight * x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i, 0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i, 1] + 1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device)
        return mask * x

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj, bert_in_indices, bert_out_indices = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len + aspect_len - 1).unsqueeze(1)], dim=1)
        text = self.embed(bert_in_indices, bert_out_indices)  # [batch size, seq len, embed size]
        text_out = self.bert_dropout(text)

        x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        x = self.mask(x, aspect_double_idx)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1)  # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output
