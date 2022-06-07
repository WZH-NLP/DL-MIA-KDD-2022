import ast
import os
import pickle
from ast import literal_eval

import numpy as np
import random

from collections import Counter

import pandas as pd
import torch
from sklearn.utils import shuffle

from decorators import auto_init_args, lazy_execute
from config import UNK_IDX, UNK_WORD, EVAL_YEAR


class data_holder:
    @auto_init_args
    def __init__(self, train_data, dev_data, test_data, vocab):
        self.inv_vocab = {i: w for w, i in vocab.items()}


def load_data_from(fpath1):
    vector1 = []
    vector2 = []
    with open(fpath1, 'r') as f1:
        for lines in f1.readline():
            line1 = lines.strip().split('\t')
            vector1.append(line1[0])
            vector2.append(line1[1])
    return np.array(vector1), np.array(vector2)

class data_postprocess:
    @auto_init_args
    def __init__(self, vocabfile, filetohelps):
        self.vocabfile = vocabfile
        self.filetohelps = filetohelps
        self.newDict = {}
        with open(self.vocabfile, "rb") as fp:
            data = pickle.load(fp)
            vocabDict = data[1]
        for key, value in vocabDict.items():
            self.newDict[value] = key

    def postprocess1(self):
        DictMatchFiles = {}
        format1 = ['feature_vector', 'label']
        matchfiles = pd.read_csv(self.filetohelps, sep='\t', names=format1)
        for j in range(len(matchfiles)):
            vector = matchfiles.iloc[j, 0]
            vector = ast.literal_eval(vector)[0]
            vector = [float(v) for v in vector]
            label = matchfiles.iloc[j, 1]
            label = ast.literal_eval(label)[0]
            DictMatchFiles[str(vector)] = label
        return DictMatchFiles

    def postprocess2(self, filetoprocess, DictMatchFiles):
        sent = filetoprocess
        vector = []
        labels = []
        for i in range(sent.shape[0]):
            idx = sent[i].tolist()
            # print("idx:", idx)
            idx = list(map(int, idx))
            vec = [float(self.newDict[x]) for x in idx]
            label = DictMatchFiles[str(vec)]
            vector.append(vec)
            labels.append(label)
        return vector, labels



class data_processor:
    @auto_init_args
    def __init__(self, train_path, eval_path, experiment):
        self.expe = experiment

    def process(self):
        fn = "vocab_" + str(self.expe.config.vocab_size)
        vocab_file = os.path.join(self.expe.config.vocab_file, fn)
        train_data = self._load_sent(self.train_path)
        W, vocab = self._build_pretrain_vocab(train_data, file_name=vocab_file)
        self.expe.log.info("vocab_size: {}".format(len(vocab)))
        train_data = self._data_to_idx(train_data, vocab)
        data = data_holder(train_data=train_data, dev_data=None, test_data=None, vocab=vocab)
        return data, W

    # @lazy_execute("_load_from_pickle")
    def _load_sent(self, path):
        data_pair1 = []
        data_pair2 = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if len(line):
                    line = line.split('\t')
                    if len(line) == 2:
                        data_pair1.append(line[0].split(' '))
                        data_pair2.append(line[1].split(' '))
                    else:
                        self.expe.log.warning("unexpected data: " + line)
        assert len(data_pair1) == len(data_pair2)
        return data_pair1, data_pair2

    def _data_to_idx(self, data, vocab):
        idx_pair1 = []
        idx_pair2 = []
        for d1, d2 in zip(*data):
            s1 = [vocab.get(w) for w in d1]
            idx_pair1.append(s1)
            s2 = [vocab.get(w) for w in d2]
            idx_pair2.append(s2)
        return np.array(idx_pair1), np.array(idx_pair2)

    def _load_paragram_embedding(self, path):
        with open(path, encoding="latin-1") as fp:
            # word_vectors: word --> vector
            word_vectors = {}
            for line in fp:
                line = line.strip("\n").split(" ")
                word_vectors[line[0]] = np.array(list(map(float, line[1:])), dtype='float32')
        vocab_embed = word_vectors.keys()
        embed_dim = word_vectors[next(iter(vocab_embed))].shape[0]
        return word_vectors, vocab_embed, embed_dim

    def _create_vocab_from_data(self, data):
        vocab = {}
        index = 3
        for sent1, sent2 in zip(*data):
            for w in sent1 + sent2:
                if w not in vocab.keys():
                    vocab[w] = index
                    index += 1
        vocab[UNK_WORD] = UNK_IDX
        vocab["<bos>"] = 1
        vocab["<eos>"] = 2
        return vocab

    @lazy_execute("_load_from_pickle")
    def _build_pretrain_vocab(self, train_data):
        self.expe.log.info("loading embedding from: {}"
                           .format(self.expe.config.embed_file))
        word_vectors, vocab_embed, embed_dim = \
            self._load_paragram_embedding(self.expe.config.embed_file)
        vocab = self._create_vocab_from_data(train_data)

        W = np.random.uniform(size=(len(vocab), embed_dim)).astype('float32')
        n = 0
        # vocab_embed = word_vectors.keys()
        for w, i in vocab.items():
            if w in vocab_embed:
                W[i, :] = word_vectors[w]
                n += 1
        self.expe.log.info(
            "{}/{} vocabs are initialized with {} embeddings."
                .format(n, len(vocab), self.expe.config.embed_type))

        return W, vocab

    def _load_from_pickle(self, file_name):
        self.expe.log.info("loading from {}".format(file_name))
        with open(file_name, "rb") as fp:
            data = pickle.load(fp)
        return data


class batch_accumulator:
    def __init__(self, mega_batch, p_scramble, init_batch1, init_batch2):
        assert len(init_batch1) == len(init_batch2) == mega_batch
        self.p_scramble = p_scramble
        self.mega_batch = mega_batch
        self.pool = [init_batch1, init_batch2]

    def update(self, new_batch1, new_batch2):
        self.pool[0].pop(0)
        self.pool[1].pop(0)

        self.pool[0].append(new_batch1)
        self.pool[1].append(new_batch2)

        assert len(self.pool[0]) == len(self.pool[1]) == self.mega_batch

    def get_batch(self):
        data1 = np.concatenate(self.pool[0])
        data2 = np.concatenate(self.pool[1])

        max_len1 = max([len(sent) for sent in data1])
        max_len2 = max([len(sent) for sent in data2])

        input_data1 = \
            np.zeros((len(data1), max_len1)).astype("float32")
        input_mask1 = \
            np.zeros((len(data1), max_len1)).astype("float32")

        tgt_data1 = \
            np.zeros((len(data1), max_len1 + 2)).astype("float32")
        tgt_mask1 = \
            np.zeros((len(data1), max_len1 + 2)).astype("float32")

        input_data2 = \
            np.zeros((len(data2), max_len2)).astype("float32")
        input_mask2 = \
            np.zeros((len(data2), max_len2)).astype("float32")

        tgt_data2 = \
            np.zeros((len(data2), max_len2 + 2)).astype("float32")
        tgt_mask2 = \
            np.zeros((len(data2), max_len2 + 2)).astype("float32")

        for i, (sent1, sent2) in enumerate(zip(data1, data2)):
            if np.random.choice(
                    [True, False],
                    p=[self.p_scramble, 1 - self.p_scramble]).item():
                sent1 = np.random.permutation(sent1)
                sent2 = np.random.permutation(sent2)

            input_data1[i, :len(sent1)] = \
                np.asarray(list(sent1)).astype("float32")
            input_mask1[i, :len(sent1)] = 1.

            tgt_data1[i, :len(sent1) + 2] = \
                np.asarray([1] + list(sent1) + [2]).astype("float32")
            tgt_mask1[i, :len(sent1) + 2] = 1.

            input_data2[i, :len(sent2)] = \
                np.asarray(list(sent2)).astype("float32")
            input_mask2[i, :len(sent2)] = 1.

            tgt_data2[i, :len(sent2) + 2] = \
                np.asarray([1] + list(sent2) + [2]).astype("float32")
            tgt_mask2[i, :len(sent2) + 2] = 1.

        return input_data1, input_mask1, tgt_data1, tgt_mask1, \
               input_data2, input_mask2, tgt_data2, tgt_mask2


class bow_accumulator(batch_accumulator):
    def __init__(self, mega_batch, p_scramble, init_batch1, init_tgt1,
                 init_batch2, init_tgt2, vocab_size):
        assert len(init_batch1) == len(init_batch2) == mega_batch
        self.p_scramble = p_scramble
        self.mega_batch = mega_batch
        self.vocab_size = vocab_size
        self.pool = [init_batch1, init_tgt1, init_batch2, init_tgt2]

    def update(self, new_batch1, new_tgt1, new_batch2, new_tgt2):
        self.pool[0].pop(0)
        self.pool[1].pop(0)
        self.pool[2].pop(0)
        self.pool[3].pop(0)

        self.pool[0].append(new_batch1)
        self.pool[1].append(new_tgt1)
        self.pool[2].append(new_batch2)
        self.pool[3].append(new_tgt2)

        assert len(self.pool[0]) == len(self.pool[1]) == self.mega_batch

    def get_batch(self):
        data1 = np.concatenate(self.pool[0])
        data2 = np.concatenate(self.pool[2])

        tgt_data1 = np.concatenate(self.pool[1])
        tgt_data2 = np.concatenate(self.pool[3])

        max_len1 = max([len(sent) for sent in data1])
        max_len2 = max([len(sent) for sent in data2])

        input_data1 = \
            np.zeros((len(data1), max_len1)).astype("float32")
        input_mask1 = \
            np.zeros((len(data1), max_len1)).astype("float32")

        input_data2 = \
            np.zeros((len(data2), max_len2)).astype("float32")
        input_mask2 = \
            np.zeros((len(data2), max_len2)).astype("float32")

        for i, (sent1, sent2) in enumerate(zip(data1, data2)):
            if np.random.choice(
                    [True, False],
                    p=[self.p_scramble, 1 - self.p_scramble]).item():
                sent1 = np.random.permutation(sent1)
                sent2 = np.random.permutation(sent2)

            input_data1[i, :len(sent1)] = \
                np.asarray(list(sent1)).astype("float32")
            input_mask1[i, :len(sent1)] = 1.

            input_data2[i, :len(sent2)] = \
                np.asarray(list(sent2)).astype("float32")
            input_mask2[i, :len(sent2)] = 1.

        return input_data1, input_mask1, tgt_data1, \
               input_data2, input_mask2, tgt_data2


class minibatcher:
    @auto_init_args
    def __init__(self, data1, data2, batch_size, score_func,
                 shuffle, mega_batch, p_scramble, *args, **kwargs):
        self._reset()

    def __len__(self):
        return len(self.idx_pool)

    def _reset(self):
        self.pointer = 0
        idx_list = np.arange(len(self.data1))
        if self.shuffle:
            np.random.shuffle(idx_list)
        self.idx_pool = [idx_list[i: i + self.batch_size]
                         for i in range(0, len(self.data1), self.batch_size)]

        if self.mega_batch > 1:
            init_mega_ids = self.idx_pool[-self.mega_batch:]
            init_mega1, init_mega2 = [], []
            for idx in init_mega_ids:
                d1, d2 = self.data1[idx], self.data2[idx]
                init_mega1.append(d1)
                init_mega2.append(d2)
            self.mega_batcher = batch_accumulator(
                self.mega_batch, self.p_scramble, init_mega1, init_mega2)

    def _select_neg_sample(self, data, data_mask,
                           cand, cand_mask, ctgt, ctgt_mask, no_diag):
        score_matrix = self.score_func(
            data, data_mask, cand, cand_mask)

        if no_diag:
            diag_idx = np.arange(len(score_matrix))
            score_matrix[diag_idx, diag_idx] = -np.inf
        neg_idx = np.argmax(score_matrix, 1)

        neg_data = cand[neg_idx]
        neg_mask = cand_mask[neg_idx]

        tgt_data = ctgt[neg_idx]
        tgt_mask = ctgt_mask[neg_idx]

        max_neg_len = int(neg_mask.sum(-1).max())
        neg_data = neg_data[:, : max_neg_len]
        neg_mask = neg_mask[:, : max_neg_len]

        max_tgt_len = int(tgt_mask.sum(-1).max())
        tgt_data = tgt_data[:, : max_tgt_len]
        tgt_mask = tgt_mask[:, : max_tgt_len]

        assert neg_mask.sum(-1).max() == max_neg_len
        return score_matrix, neg_data, neg_mask, tgt_data, tgt_mask

    def _pad(self, data1, data2):
        assert len(data1) == len(data2)
        max_len1 = max([len(sent) for sent in data1])
        max_len2 = max([len(sent) for sent in data2])

        input_data1 = \
            np.zeros((len(data1), max_len1)).astype("float32")
        input_mask1 = \
            np.zeros((len(data1), max_len1)).astype("float32")
        tgt_data1 = \
            np.zeros((len(data1), max_len1 + 2)).astype("float32")
        tgt_mask1 = \
            np.zeros((len(data1), max_len1 + 2)).astype("float32")

        input_data2 = \
            np.zeros((len(data2), max_len2)).astype("float32")
        input_mask2 = \
            np.zeros((len(data2), max_len2)).astype("float32")
        tgt_data2 = \
            np.zeros((len(data2), max_len2 + 2)).astype("float32")
        tgt_mask2 = \
            np.zeros((len(data2), max_len2 + 2)).astype("float32")

        for i, (sent1, sent2) in enumerate(zip(data1, data2)):
            if np.random.choice(
                    [True, False],
                    p=[self.p_scramble, 1 - self.p_scramble]).item():
                sent1 = np.random.permutation(sent1)
                sent2 = np.random.permutation(sent2)

            input_data1[i, :len(sent1)] = \
                np.asarray(list(sent1)).astype("float32")
            input_mask1[i, :len(sent1)] = 1.

            tgt_data1[i, :len(sent1) + 2] = \
                np.asarray([1] + list(sent1) + [2]).astype("float32")
            tgt_mask1[i, :len(sent1) + 2] = 1.

            input_data2[i, :len(sent2)] = \
                np.asarray(list(sent2)).astype("float32")
            input_mask2[i, :len(sent2)] = 1.

            tgt_data2[i, :len(sent2) + 2] = \
                np.asarray([1] + list(sent2) + [2]).astype("float32")
            tgt_mask2[i, :len(sent2) + 2] = 1.

        if self.mega_batch > 1:
            cand1, cand_mask1, ctgt1, ctgt_mask1, \
            cand2, cand_mask2, ctgt2, ctgt_mask2 = \
                self.mega_batcher.get_batch()
            _, neg_data1, neg_mask1, ntgt1, ntgt_mask1 = \
                self._select_neg_sample(
                    input_data1, input_mask1, cand2,
                    cand_mask2, ctgt2, ctgt_mask2, False)
            _, neg_data2, neg_mask2, ntgt2, ntgt_mask2 = \
                self._select_neg_sample(
                    input_data2, input_mask2, cand1,
                    cand_mask1, ctgt1, ctgt_mask1, False)
            self.mega_batcher.update(data1, data2)

            return [input_data1, input_mask1, input_data2, input_mask2,
                    tgt_data1, tgt_mask1, tgt_data2, tgt_mask2,
                    neg_data1, neg_mask1, ntgt1, ntgt_mask1,
                    neg_data2, neg_mask2, ntgt2, ntgt_mask2]
        else:
            return [input_data1, input_mask1, input_data2, input_mask2,
                    tgt_data1, tgt_mask1, tgt_data2, tgt_mask2,
                    None, None, None, None,
                    None, None, None, None]

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.idx_pool):
            self._reset()
            raise StopIteration()

        idx = self.idx_pool[self.pointer]
        data1, data2 = self.data1[idx], self.data2[idx]
        self.pointer += 1
        return self._pad(data1, data2) + [idx]


class bow_minibatcher:
    @auto_init_args
    def __init__(self, data1, data2, vocab_size, batch_size,
                 score_func, shuffle, mega_batch, p_scramble,
                 *args, **kwargs):
        self._reset()

    def __len__(self):
        return len(self.idx_pool)

    def _reset(self):
        self.pointer = 0
        idx_list = np.arange(len(self.data1))
        if self.shuffle:
            np.random.shuffle(idx_list)
        self.idx_pool = [idx_list[i: i + self.batch_size]
                         for i in range(0, len(self.data1), self.batch_size)]

        if self.mega_batch > 1:
            init_mega_ids = self.idx_pool[-self.mega_batch:]
            init_mega1, init_mega2, init_tgt1, init_tgt2 = [], [], [], []
            for idx in init_mega_ids:
                d1, d2 = self.data1[idx], self.data2[idx]
                init_mega1.append(d1)
                init_mega2.append(d2)
                t1 = np.zeros((len(d1), self.vocab_size)).astype("float32")
                t2 = np.zeros((len(d2), self.vocab_size)).astype("float32")
                for i, (s1, s2) in enumerate(zip(d1, d2)):
                    t1[i, :] = np.bincount(s1, minlength=self.vocab_size)
                    t2[i, :] = np.bincount(s2, minlength=self.vocab_size)
                init_tgt1.append(t1)
                init_tgt2.append(t2)
            self.mega_batcher = bow_accumulator(
                self.mega_batch, self.p_scramble, init_mega1, init_tgt1,
                init_mega2, init_tgt2, self.vocab_size)

    def _select_neg_sample(self, data, data_mask,
                           cand, cand_mask, ctgt, no_diag):
        score_matrix = self.score_func(
            data, data_mask, cand, cand_mask)

        if no_diag:
            diag_idx = np.arange(len(score_matrix))
            score_matrix[diag_idx, diag_idx] = -np.inf
        neg_idx = np.argmax(score_matrix, 1)

        neg_data = cand[neg_idx]
        neg_mask = cand_mask[neg_idx]

        tgt_data = ctgt[neg_idx]

        max_neg_len = int(neg_mask.sum(-1).max())
        neg_data = neg_data[:, : max_neg_len]
        neg_mask = neg_mask[:, : max_neg_len]

        assert neg_mask.sum(-1).max() == max_neg_len
        return score_matrix, neg_data, neg_mask, tgt_data

    def _pad(self, data1, data2):
        assert len(data1) == len(data2)
        max_len1 = max([len(sent) for sent in data1])
        max_len2 = max([len(sent) for sent in data2])

        input_data1 = \
            np.zeros((len(data1), max_len1)).astype("float32")
        input_mask1 = \
            np.zeros((len(data1), max_len1)).astype("float32")
        tgt_data1 = \
            np.zeros((len(data1), self.vocab_size)).astype("float32")

        input_data2 = \
            np.zeros((len(data2), max_len2)).astype("float32")
        input_mask2 = \
            np.zeros((len(data2), max_len2)).astype("float32")
        tgt_data2 = \
            np.zeros((len(data2), self.vocab_size)).astype("float32")

        for i, (sent1, sent2) in enumerate(zip(data1, data2)):
            if np.random.choice(
                    [True, False],
                    p=[self.p_scramble, 1 - self.p_scramble]).item():
                sent1 = np.random.permutation(sent1)
                sent2 = np.random.permutation(sent2)

            input_data1[i, :len(sent1)] = \
                np.asarray(list(sent1)).astype("float32")
            input_mask1[i, :len(sent1)] = 1.
            tgt_data1[i, :] = np.bincount(sent1, minlength=self.vocab_size)

            input_data2[i, :len(sent2)] = \
                np.asarray(list(sent2)).astype("float32")
            input_mask2[i, :len(sent2)] = 1.

            tgt_data2[i, :] = np.bincount(sent2, minlength=self.vocab_size)

        if self.mega_batch > 1:
            cand1, cand_mask1, ctgt1, \
            cand2, cand_mask2, ctgt2 = \
                self.mega_batcher.get_batch()
            _, neg_data1, neg_mask1, ntgt1 = \
                self._select_neg_sample(
                    input_data1, input_mask1, cand2,
                    cand_mask2, ctgt2, False)
            _, neg_data2, neg_mask2, ntgt2 = \
                self._select_neg_sample(
                    input_data2, input_mask2, cand1,
                    cand_mask1, ctgt1, False)
            self.mega_batcher.update(data1, tgt_data1, data2, tgt_data2)

            return [input_data1, input_mask1, input_data2, input_mask2,
                    tgt_data1, tgt_data1, tgt_data2, tgt_data2,
                    neg_data1, neg_mask1, ntgt1, ntgt1,
                    neg_data2, neg_mask2, ntgt2, ntgt2]
        else:
            return [input_data1, input_mask1, input_data2, input_mask2,
                    tgt_data1, tgt_data1, tgt_data2, tgt_data2,
                    None, None, None, None,
                    None, None, None, None]

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.idx_pool):
            self._reset()
            raise StopIteration()

        idx = self.idx_pool[self.pointer]
        data1, data2 = self.data1[idx], self.data2[idx]
        self.pointer += 1
        return self._pad(data1, data2) + [idx]
