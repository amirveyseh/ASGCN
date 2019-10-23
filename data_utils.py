# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
from transformers import BertTokenizer
import unidecode


def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            try:
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
            except:
                continue
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), type)
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors ...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[1, :] = np.random.uniform(-1 / np.sqrt(embed_dim), 1 / np.sqrt(embed_dim), (1, embed_dim))
        fname = './glove/glove.840B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix


class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v: k for k, v in word2idx.items()}

    def fit_on_text(self, text):
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text):
        text = text.lower()
        words = text.split()
        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return sequence


class ABSADataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ABSADatesetReader_Bert:
    @staticmethod
    def __read_text__(fnames):
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "
        return text

    @staticmethod
    def match_wordpiece_indices(ori_sentence, bert_tokenizer):
        ori_sentence = ori_sentence.lower()
        ori_tokens = ori_sentence.split()
        bert_tokens = bert_tokenizer.tokenize('[CLS] ' + ori_sentence + ' [SEP]')

        bert_in_indices = bert_tokenizer.convert_tokens_to_ids(bert_tokens)
        bert_out_indices = []
        ori_index = 0
        bert_index = 0
        while ori_index < len(ori_tokens) and bert_index < len(bert_tokens):
            ori_token = ori_tokens[ori_index]
            bert_token = bert_tokens[bert_index]
            convert_token = bert_tokenizer.tokenize(ori_token)[0]
            if convert_token != '[UNK]':
                if bert_token[:2] != '##' and (
                        ori_token == bert_token[:len(ori_token)] or ori_token[:len(bert_token)] == bert_token):
                    bert_out_indices.append(bert_index)
                    ori_index += 1
                else:
                    ori_token = unidecode.unidecode(ori_token)
                    if bert_token[:2] != '##' and (
                            ori_token == bert_token[:len(ori_token)] or ori_token[:len(bert_token)] == bert_token):
                        bert_out_indices.append(bert_index)
                        ori_index += 1
            elif bert_token == '[UNK]':
                bert_out_indices.append(bert_index)
                ori_index += 1
            bert_index += 1

        if len(bert_out_indices) != len(ori_tokens):
            bert_tokens = ['[CLS]'] + ori_tokens + ['[SEP]']
            bert_in_indices = bert_tokenizer.convert_tokens_to_ids(bert_tokens)
            bert_out_indices = list(range(len(bert_in_indices)))[1:-1]
            
        assert len(bert_out_indices) == len(ori_tokens)
        return bert_in_indices, bert_out_indices

    @staticmethod
    def __read_data__(fname, tokenizer, bert_tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        fin = open(fname + '.graph', 'rb')
        idx2gragh = pickle.load(fin)
        fin.close()

        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()

            text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            context_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_indices = tokenizer.text_to_sequence(text_left)
            # ********** for BERT ************
            bert_in_indices, bert_out_indices = ABSADatesetReader_Bert.match_wordpiece_indices(
                ori_sentence='{} {} {}'.format(text_left, aspect, text_right),
                bert_tokenizer=bert_tokenizer
            )
            # bert_in_indices = bert_tokenizer.convert_tokens_to_ids(
            #     tokens=[bert_tokenizer.cls_token] + '{} {} {}'.format(text_left, aspect, text_right).split() + [
            #         bert_tokenizer.sep_token])
            # bert_out_indices = [idx for idx in range(1, len(bert_in_indices) - 1)]
            # ********************************
            polarity = int(polarity) + 1
            dependency_graph = idx2gragh[i]

            data = {
                'text_indices': text_indices,
                'context_indices': context_indices,
                'aspect_indices': aspect_indices,
                'left_indices': left_indices,
                'polarity': polarity,
                'dependency_graph': dependency_graph,
                'bert_in_indices': bert_in_indices,
                'bert_out_indices': bert_out_indices
            }

            all_data.append(data)
        return all_data

    def __init__(self, dataset='twitter', opt=None):
        print("preparing {0} dataset ...".format(dataset))
        fname = {
            'twitter': {
                'train': './datasets/acl-14-short-data/train.raw',
                'test': './datasets/acl-14-short-data/test.raw'
            },
            'rest14': {
                'train': './datasets/semeval14/restaurant_train.raw',
                'test': './datasets/semeval14/restaurant_test.raw'
            },
            'lap14': {
                'train': './datasets/semeval14/laptop_train.raw',
                'test': './datasets/semeval14/laptop_test.raw'
            },
            'rest15': {
                'train': './datasets/semeval15/restaurant_train.raw',
                'test': './datasets/semeval15/restaurant_test.raw'
            },
            'rest16': {
                'train': './datasets/semeval16/restaurant_train.raw',
                'test': './datasets/semeval16/restaurant_test.raw'
            },

        }
        text = ABSADatesetReader_Bert.__read_text__([fname[dataset]['train'], fname[dataset]['test']])
        if os.path.exists(dataset + '_word2idx.pkl'):
            print("loading {0} tokenizer...".format(dataset))
            with open(dataset + '_word2idx.pkl', 'rb') as f:
                word2idx = pickle.load(f)
                tokenizer = Tokenizer(word2idx=word2idx)
        else:
            tokenizer = Tokenizer()
            tokenizer.fit_on_text(text)
            with open(dataset + '_word2idx.pkl', 'wb') as f:
                pickle.dump(tokenizer.word2idx, f)
        bert_tokenizer = BertTokenizer.from_pretrained(opt.bert_version)
        self.train_data = ABSADataset(
            ABSADatesetReader_Bert.__read_data__(fname[dataset]['train'], tokenizer, bert_tokenizer))
        self.test_data = ABSADataset(
            ABSADatesetReader_Bert.__read_data__(fname[dataset]['test'], tokenizer, bert_tokenizer))


class ABSADatesetReader:
    @staticmethod
    def __read_text__(fnames):
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "
        return text

    @staticmethod
    def __read_data__(fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        fin = open(fname + '.graph', 'rb')
        idx2gragh = pickle.load(fin)
        fin.close()

        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()

            text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            context_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_indices = tokenizer.text_to_sequence(text_left)
            polarity = int(polarity) + 1
            dependency_graph = idx2gragh[i]

            data = {
                'text_indices': text_indices,
                'context_indices': context_indices,
                'aspect_indices': aspect_indices,
                'left_indices': left_indices,
                'polarity': polarity,
                'dependency_graph': dependency_graph,
            }

            all_data.append(data)
        return all_data

    def __init__(self, dataset='twitter', embed_dim=300):
        print("preparing {0} dataset ...".format(dataset))
        fname = {
            'twitter': {
                'train': './datasets/acl-14-short-data/train.raw',
                'test': './datasets/acl-14-short-data/test.raw'
            },
            'rest14': {
                'train': './datasets/semeval14/restaurant_train.raw',
                'test': './datasets/semeval14/restaurant_test.raw'
            },
            'lap14': {
                'train': './datasets/semeval14/laptop_train.raw',
                'test': './datasets/semeval14/laptop_test.raw'
            },
            'rest15': {
                'train': './datasets/semeval15/restaurant_train.raw',
                'test': './datasets/semeval15/restaurant_test.raw'
            },
            'rest16': {
                'train': './datasets/semeval16/restaurant_train.raw',
                'test': './datasets/semeval16/restaurant_test.raw'
            },

        }
        text = ABSADatesetReader.__read_text__([fname[dataset]['train'], fname[dataset]['test']])
        if os.path.exists(dataset + '_word2idx.pkl'):
            print("loading {0} tokenizer...".format(dataset))
            with open(dataset + '_word2idx.pkl', 'rb') as f:
                word2idx = pickle.load(f)
                tokenizer = Tokenizer(word2idx=word2idx)
        else:
            tokenizer = Tokenizer()
            tokenizer.fit_on_text(text)
            with open(dataset + '_word2idx.pkl', 'wb') as f:
                pickle.dump(tokenizer.word2idx, f)
        self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim, dataset)
        self.train_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['train'], tokenizer))
        self.test_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['test'], tokenizer))
